//! The OpenAI-compatible HTTP provider.
//!
//! [`OpenAiCompatibleProvider`] implements [`ChatProvider`] by sending
//! requests to any endpoint that speaks the OpenAI chat completions wire
//! format. Point it at a base URL and API key and it handles serialization,
//! SSE streaming, error mapping, and vendor extensions like
//! `reasoning_content`.

use futures::stream::{self, BoxStream, StreamExt};
use promptus_core::{
    ChatProvider, ChatRequest, ChatResponse, ModelInfo, ModelProvider, ProviderError, StreamEvent,
};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};

use crate::mapping;
use crate::wire::response;

/// Map a reqwest error to a `ProviderError`.
///
/// This lives here (not in `promptus_core`) because `reqwest` is a
/// dependency of this crate, not the core crate. The orphan rule prevents
/// a blanket `From<reqwest::Error>` impl.
fn map_reqwest_error(err: reqwest::Error) -> ProviderError {
    if err.is_timeout() || err.is_connect() {
        ProviderError::Network(err.to_string())
    } else {
        ProviderError::Other(err.to_string())
    }
}

/// An HTTP client for any provider that implements the OpenAI-compatible
/// chat completions wire format.
///
/// This includes OpenAI itself, but also Groq, DeepSeek, Together AI,
/// Xiaomi MiMo, Fireworks, Ollama, vLLM, and many others.
///
/// # Example
///
/// ```ignore
/// use promptus_openai::OpenAiCompatibleProvider;
///
/// let provider = OpenAiCompatibleProvider::new(
///     "https://api.groq.com/openai/v1",
///     "gsk_...",
/// );
/// ```
pub struct OpenAiCompatibleProvider {
    base_url: String,
    api_key: String,
    extra_headers: HeaderMap,
    http_client: reqwest::Client,
}

impl OpenAiCompatibleProvider {
    /// Create a new provider pointing at the given base URL with the given
    /// API key.
    ///
    /// The base URL should include the path prefix but **not** the
    /// `/chat/completions` suffix — that is appended automatically. For
    /// example, `"https://api.groq.com/openai/v1"` or
    /// `"http://localhost:11434/v1"`.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            // Strip trailing slash to avoid double-slash in the URL.
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key: api_key.into(),
            extra_headers: HeaderMap::new(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Add a custom header to every request this provider sends.
    ///
    /// Useful for providers that require additional headers (e.g.
    /// `X-Organization-Id`, custom auth tokens, or beta feature flags).
    ///
    /// # Panics
    ///
    /// Panics if the header name or value contains invalid ASCII characters.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let name = HeaderName::from_bytes(key.into().as_bytes()).expect("invalid header name");
        let val = HeaderValue::from_str(&value.into()).expect("invalid header value");
        self.extra_headers.insert(name, val);
        self
    }

    /// Use a pre-configured `reqwest::Client` instead of the default one.
    ///
    /// Useful for setting custom timeouts, proxies, or TLS configuration.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = client;
        self
    }

    fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    fn models_url(&self) -> String {
        format!("{}/models", self.base_url)
    }

    /// Build the common set of HTTP headers for every request.
    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .expect("API key contains invalid header characters"),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        for (key, value) in &self.extra_headers {
            headers.insert(key.clone(), value.clone());
        }
        headers
    }

    /// Execute a non-streaming chat completion request.
    async fn do_chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError> {
        let wire_req = mapping::request_to_wire(&request);
        let url = self.chat_url();

        let resp = self
            .http_client
            .post(&url)
            .headers(self.build_headers())
            .json(&wire_req)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        if !resp.status().is_success() {
            return Err(parse_error_response(resp).await);
        }

        let body: response::CreateChatCompletionResponse =
            resp.json().await.map_err(map_reqwest_error)?;

        mapping::response_from_wire(&body).map_err(ProviderError::Serialization)
    }

    /// Execute a streaming chat completion request.
    ///
    /// Returns a `BoxStream` of `StreamEvent`s. The stream emits events as
    /// they arrive via SSE, ending with a `StreamEvent::Finished`. Tool-call
    /// argument deltas are accumulated by index and emitted as individual
    /// `ToolCallDelta` events (the caller can use
    /// [`ToolCallAccumulator`] to reassemble them).
    async fn do_chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError> {
        let wire_req = mapping::request_to_wire(&request);
        let url = self.chat_url();

        let resp = self
            .http_client
            .post(&url)
            .headers(self.build_headers())
            .json(&wire_req)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        if !resp.status().is_success() {
            return Err(parse_error_response(resp).await);
        }

        let byte_stream = resp.bytes_stream();
        let event_stream = SseStream::new(byte_stream).flat_map(|result| {
            match result {
                Ok(data) => {
                    // The "[DONE]" sentinel signals the end of the stream.
                    if data.trim() == "[DONE]" {
                        return stream::iter(vec![]).boxed();
                    }
                    match serde_json::from_str::<response::CreateChatCompletionStreamResponse>(
                        &data,
                    ) {
                        Ok(chunk) => {
                            let events = mapping::stream_events_from_wire(&chunk);
                            stream::iter(events.into_iter().map(Ok)).boxed()
                        }
                        Err(e) => stream::iter(vec![Err(ProviderError::Serialization(format!(
                            "failed to parse stream chunk: {e}"
                        )))])
                        .boxed(),
                    }
                }
                Err(e) => stream::iter(vec![Err(e)]).boxed(),
            }
        });

        Ok(event_stream.boxed())
    }

    /// Fetch the list of available models from `GET /models`.
    async fn do_list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        let url = self.models_url();

        let resp = self
            .http_client
            .get(&url)
            .headers(self.build_headers())
            .send()
            .await
            .map_err(map_reqwest_error)?;

        if !resp.status().is_success() {
            return Err(parse_error_response(resp).await);
        }

        let body: response::ListModelsResponse = resp.json().await.map_err(map_reqwest_error)?;

        Ok(mapping::models_from_wire(&body))
    }
}

impl ChatProvider for OpenAiCompatibleProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError> {
        self.do_chat(request).await
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError> {
        self.do_chat_stream(request).await
    }
}

impl ModelProvider for OpenAiCompatibleProvider {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        self.do_list_models().await
    }
}

// ---------------------------------------------------------------------------
// SSE stream parser
// ---------------------------------------------------------------------------

/// Parses a byte stream into SSE `data:` payloads.
///
/// Handles the SSE wire format:
/// - Lines starting with `data: ` are extracted
/// - Multi-line data fields are concatenated
/// - Empty lines delimit events
/// - Lines starting with `:` are comments and are ignored
struct SseStream<S> {
    inner: S,
    buffer: String,
}

impl<S, B, E> SseStream<S>
where
    S: futures::Stream<Item = Result<B, E>> + Unpin,
    B: AsRef<[u8]>,
    E: std::fmt::Display,
{
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: String::new(),
        }
    }

    /// Try to extract one complete SSE event from the buffer.
    ///
    /// Returns `Some(data)` if a complete event was found (the data after
    /// `data: `), or `None` if the buffer doesn't yet contain a full event.
    fn try_extract_event(&mut self) -> Option<String> {
        // SSE events are delimited by a blank line (\n\n).
        if let Some(pos) = self.buffer.find("\n\n") {
            let event_block = self.buffer[..pos].to_owned();
            self.buffer.drain(..=pos + 1);

            let mut data = String::new();
            for line in event_block.lines() {
                if let Some(rest) = line.strip_prefix("data: ") {
                    if !data.is_empty() {
                        data.push('\n');
                    }
                    data.push_str(rest);
                }
                // Ignore comments (lines starting with :) and other fields
                // (event:, id:, retry:).
            }

            if data.is_empty() {
                // No data field found — skip this event
                None
            } else {
                Some(data)
            }
        } else {
            None
        }
    }
}

impl<S, B, E> futures::Stream for SseStream<S>
where
    S: futures::Stream<Item = Result<B, E>> + Unpin,
    B: AsRef<[u8]>,
    E: std::fmt::Display,
{
    type Item = Result<String, ProviderError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        loop {
            // Try to extract an event from the existing buffer.
            if let Some(data) = self.try_extract_event() {
                return std::task::Poll::Ready(Some(Ok(data)));
            }

            // Need more data — poll the inner stream.
            match std::pin::Pin::new(&mut self.inner).poll_next(cx) {
                std::task::Poll::Ready(Some(Ok(chunk))) => {
                    match std::str::from_utf8(chunk.as_ref()) {
                        Ok(text) => self.buffer.push_str(text),
                        Err(e) => {
                            return std::task::Poll::Ready(Some(Err(
                                ProviderError::Serialization(format!(
                                    "invalid UTF-8 in SSE stream: {e}"
                                )),
                            )));
                        }
                    }
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    return std::task::Poll::Ready(Some(Err(ProviderError::Network(
                        e.to_string(),
                    ))));
                }
                std::task::Poll::Ready(None) => {
                    // Stream ended — flush any remaining buffer as a final event.
                    if !self.buffer.trim().is_empty() {
                        let mut data = String::new();
                        for line in self.buffer.lines() {
                            if let Some(rest) = line.strip_prefix("data: ") {
                                if !data.is_empty() {
                                    data.push('\n');
                                }
                                data.push_str(rest);
                            }
                        }
                        self.buffer.clear();
                        if !data.is_empty() {
                            return std::task::Poll::Ready(Some(Ok(data)));
                        }
                    }
                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Error response parsing
// ---------------------------------------------------------------------------

/// Parse an HTTP error response into a `ProviderError`.
///
/// Tries to extract a structured error message from the response body. If
/// parsing fails (the body isn't valid JSON or has an unexpected shape), falls
/// back to including the raw body text in the error.
async fn parse_error_response(resp: reqwest::Response) -> ProviderError {
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap_or_default();

    // Try to parse as {"error": {"message": "...", "code": "..."}}
    if let Ok(err_resp) = serde_json::from_str::<response::ErrorResponse>(&body)
        && let Some(err) = err_resp.error
    {
        let message = err.message.unwrap_or_else(|| "unknown error".to_owned());
        let code = err.code.unwrap_or_default();
        return ProviderError::Http {
            status,
            message: if code.is_empty() {
                message
            } else {
                format!("[{code}] {message}")
            },
            body: Some(body),
        };
    }

    ProviderError::Http {
        status,
        message: format!("HTTP {status}"),
        body: Some(body),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_construction() {
        let p = OpenAiCompatibleProvider::new("https://api.example.com/v1", "sk-test");
        assert_eq!(p.base_url, "https://api.example.com/v1");
        assert_eq!(p.chat_url(), "https://api.example.com/v1/chat/completions");
        assert_eq!(p.models_url(), "https://api.example.com/v1/models");
    }

    #[test]
    fn provider_strips_trailing_slash() {
        let p = OpenAiCompatibleProvider::new("https://api.example.com/v1/", "sk-test");
        assert_eq!(p.base_url, "https://api.example.com/v1");
    }

    #[test]
    fn provider_with_custom_headers() {
        let p = OpenAiCompatibleProvider::new("https://api.example.com/v1", "sk-test")
            .with_header("X-Custom", "value");
        assert!(p.extra_headers.contains_key("x-custom"));
    }

    #[test]
    fn parse_error_response_standard_format() {
        let json = r#"{"error":{"message":"Invalid API key","type":"invalid_request_error","code":"invalid_api_key"}}"#;
        let err: response::ErrorResponse = serde_json::from_str(json).unwrap();
        let inner = err.error.unwrap();
        assert_eq!(inner.message.unwrap(), "Invalid API key");
        assert_eq!(inner.code.unwrap(), "invalid_api_key");
    }
}
