//! Wire-format response structs for the OpenAI-compatible chat completions API.
//!
//! All fields use `#[serde(default)]` or `Option` so that providers that omit
//! or rename OpenAI-specific fields (like `service_tier` or
//! `system_fingerprint`) never cause a hard parse failure.

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Non-streaming response
// ---------------------------------------------------------------------------

/// A single chat completion response from `POST /chat/completions`.
///
/// Most providers return exactly one choice, but the spec allows `n > 1`.
/// We parse all choices and return the first one (index 0) since the core
/// API doesn't expose multi-choice responses yet.
#[derive(Debug, Deserialize)]
pub(crate) struct CreateChatCompletionResponse {
    pub choices: Vec<ResponseChoice>,
    /// The model that actually generated the response.
    #[serde(default)]
    pub model: String,
    /// Token usage statistics.
    #[serde(default)]
    pub usage: Option<CompletionUsage>,
    // Fields we parse but don't expose in the core API:
    // id, created, object, service_tier, system_fingerprint
}

/// A single choice in a non-streaming response.
#[derive(Debug, Deserialize)]
pub(crate) struct ResponseChoice {
    pub message: ResponseMessage,
    /// Why the model stopped. String like "stop", "length", "tool_calls".
    pub finish_reason: Option<String>,
}

/// The message content of a non-streaming response.
#[derive(Debug, Deserialize)]
pub(crate) struct ResponseMessage {
    #[allow(dead_code)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    /// Vendor extension: reasoning content from models like DeepSeek-R1.
    /// Exposed by Groq, Together, Fireworks, and others for reasoning models.
    /// Not part of the official OpenAI spec — absent on most providers.
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

// ---------------------------------------------------------------------------
// Streaming response
// ---------------------------------------------------------------------------

/// A single chunk in a streamed chat completion.
///
/// Each chunk contains one or more choices with delta content. The final
/// chunk (when `stream_options.include_usage` is true) may have an empty
/// `choices` array and carry `usage` instead.
#[derive(Debug, Deserialize)]
pub(crate) struct CreateChatCompletionStreamResponse {
    #[serde(default)]
    pub choices: Vec<StreamChoice>,
    /// Token usage — present only on the final chunk when
    /// `stream_options.include_usage` was requested.
    #[serde(default)]
    pub usage: Option<CompletionUsage>,
    // Fields we parse but don't expose: id, created, object, model,
    // service_tier, system_fingerprint
}

/// A choice in a streaming response chunk.
#[derive(Debug, Deserialize)]
pub(crate) struct StreamChoice {
    pub delta: StreamDelta,
    /// Non-null only on the final chunk for this choice.
    pub finish_reason: Option<String>,
}

/// The delta content of a streaming chunk.
///
/// Fields are all optional because different chunks carry different pieces
/// of the response — the first chunk may have `role`, subsequent chunks have
/// `content` fragments, and tool-call chunks have `tool_calls` fragments.
#[derive(Debug, Deserialize)]
pub(crate) struct StreamDelta {
    #[serde(default)]
    pub content: Option<String>,
    /// Vendor extension: reasoning content delta from reasoning models.
    /// Not part of the official OpenAI spec.
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<StreamToolCallDelta>>,
    // role, refusal — parsed but not exposed
}

// ---------------------------------------------------------------------------
// Tool calls
// ---------------------------------------------------------------------------

/// A complete tool call in a non-streaming response.
#[derive(Debug, Deserialize)]
pub(crate) struct ResponseToolCall {
    pub id: String,
    pub function: ResponseFunctionCall,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ResponseFunctionCall {
    pub name: String,
    /// JSON-encoded arguments string. May not be valid JSON if the model
    /// hallucinated a malformed call.
    pub arguments: String,
}

/// A tool-call delta in a streaming response.
///
/// Only the first chunk for a given `index` carries `id` and `function.name`.
/// Subsequent chunks only have `function.arguments` (a string fragment).
#[derive(Debug, Deserialize)]
pub(crate) struct StreamToolCallDelta {
    /// The index of this tool call in the response's tool_calls array.
    pub index: u32,
    /// The tool call ID (present only in the first chunk for this index).
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<StreamToolCallFunctionDelta>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamToolCallFunctionDelta {
    /// The function name (present only in the first chunk for this index).
    #[serde(default)]
    pub name: Option<String>,
    /// A fragment of the arguments JSON string.
    #[serde(default)]
    pub arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

/// Token usage statistics from the API.
///
/// The nested `prompt_tokens_details` and `completion_tokens_details` objects
/// carry breakdowns like `reasoning_tokens` and `cached_tokens`.
#[derive(Debug, Deserialize)]
pub(crate) struct CompletionUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
}

// ---------------------------------------------------------------------------
// Model listing response
// ---------------------------------------------------------------------------

/// Response from `GET /models`.
///
/// The OpenAI spec returns `{"object": "list", "data": [...]}`.
#[derive(Debug, Deserialize)]
pub(crate) struct ListModelsResponse {
    #[serde(default)]
    pub data: Vec<WireModel>,
}

/// A single model entry in the `/models` response.
#[derive(Debug, Deserialize)]
pub(crate) struct WireModel {
    /// The model identifier (e.g. `"gpt-4o"`).
    pub id: String,
    /// The entity that owns or provides this model.
    #[serde(default)]
    pub owned_by: Option<String>,
    /// Unix timestamp (seconds) when the model was created.
    #[serde(default)]
    pub created: Option<u64>,
}

// ---------------------------------------------------------------------------
// Error response
// ---------------------------------------------------------------------------

/// The error body returned by OpenAI-compatible providers.
///
/// Most providers return `{"error": {"message": "...", "type": "...",
/// "code": "..."}}`, but some return slightly different shapes. We parse
/// with liberal defaults so we never panic on an unexpected error format.
#[derive(Debug, Deserialize)]
pub(crate) struct ErrorResponse {
    /// The error body. Uses a custom deserializer because some providers
    /// return `"error": "string"` instead of `"error": {...}`, which would
    /// cause a hard parse failure with the default `Option<ErrorBody>`.
    #[serde(default, deserialize_with = "deserialize_lenient_error")]
    pub error: Option<ErrorBody>,
}

fn deserialize_lenient_error<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ErrorBody>, D::Error> {
    let val = serde_json::Value::deserialize(deserializer)?;
    match val {
        serde_json::Value::Object(_) => {
            serde_json::from_value(val).map_err(serde::de::Error::custom)
        }
        // Some providers return a plain string — wrap it as the message.
        serde_json::Value::String(s) => Ok(Some(ErrorBody {
            message: Some(s),
            code: None,
        })),
        _ => Ok(None),
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ErrorBody {
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_non_streaming_response() {
        let json = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help?",
                    "tool_calls": null
                },
                "finish_reason": "stop"
            }],
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 5},
                "completion_tokens_details": {"reasoning_tokens": 0}
            }
        }"#;

        let resp: CreateChatCompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.model, "gpt-4o");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("Hello! How can I help?")
        );
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));

        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, Some(5));
        assert_eq!(
            usage.completion_tokens_details.unwrap().reasoning_tokens,
            Some(0)
        );
    }

    #[test]
    fn parse_response_with_reasoning_content() {
        let json = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "reasoning_content": "Let me think step by step...",
                    "tool_calls": null
                },
                "finish_reason": "stop"
            }],
            "model": "deepseek-r1"
        }"#;

        let resp: CreateChatCompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0].message.reasoning_content.as_deref(),
            Some("Let me think step by step...")
        );
    }

    #[test]
    fn parse_response_with_tool_calls() {
        let json = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"NYC\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "model": "gpt-4o"
        }"#;

        let resp: CreateChatCompletionResponse = serde_json::from_str(json).unwrap();
        let tc = &resp.choices[0].message.tool_calls.as_ref().unwrap()[0];
        assert_eq!(tc.id, "call_abc123");
        assert_eq!(tc.function.name, "get_weather");
        assert_eq!(tc.function.arguments, "{\"location\":\"NYC\"}");
    }

    #[test]
    fn parse_streaming_content_chunk() {
        let json = r#"{
            "choices": [{
                "delta": {"content": "Hello"},
                "finish_reason": null
            }]
        }"#;

        let chunk: CreateChatCompletionStreamResponse = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello"));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn parse_streaming_reasoning_chunk() {
        let json = r#"{
            "choices": [{
                "delta": {"reasoning_content": "Let me think..."},
                "finish_reason": null
            }]
        }"#;

        let chunk: CreateChatCompletionStreamResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            chunk.choices[0].delta.reasoning_content.as_deref(),
            Some("Let me think...")
        );
    }

    #[test]
    fn parse_streaming_tool_call_chunk() {
        let json = r#"{
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_123",
                        "function": {"name": "get_weather", "arguments": "{\"loc"}
                    }]
                },
                "finish_reason": null
            }]
        }"#;

        let chunk: CreateChatCompletionStreamResponse = serde_json::from_str(json).unwrap();
        let tc = &chunk.choices[0].delta.tool_calls.as_ref().unwrap()[0];
        assert_eq!(tc.index, 0);
        assert_eq!(tc.id.as_deref(), Some("call_123"));
        assert_eq!(
            tc.function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
        assert_eq!(
            tc.function.as_ref().unwrap().arguments.as_deref(),
            Some("{\"loc")
        );
    }

    #[test]
    fn parse_streaming_final_chunk_with_usage() {
        let json = r#"{
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }"#;

        let chunk: CreateChatCompletionStreamResponse = serde_json::from_str(json).unwrap();
        assert!(chunk.choices.is_empty());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn parse_error_response() {
        let json = r#"{
            "error": {
                "message": "Invalid API key",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        }"#;

        let err: ErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            err.error.unwrap().message.as_deref(),
            Some("Invalid API key")
        );
    }

    #[test]
    fn parse_malformed_error_response() {
        // Some providers return non-standard error shapes — a plain string
        // instead of an object.
        let json = r#"{"error": "something went wrong"}"#;
        let err: ErrorResponse = serde_json::from_str(json).unwrap();
        let inner = err.error.unwrap();
        assert_eq!(inner.message.as_deref(), Some("something went wrong"));
    }

    #[test]
    fn parse_response_with_missing_optional_fields() {
        // Minimal response — many providers omit service_tier,
        // system_fingerprint, etc.
        let json = r#"{
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "model": "some-model"
        }"#;

        let resp: CreateChatCompletionResponse = serde_json::from_str(json).unwrap();
        assert!(resp.usage.is_none());
        assert!(resp.choices[0].message.tool_calls.is_none());
        assert!(resp.choices[0].message.reasoning_content.is_none());
    }

    #[test]
    fn parse_list_models_response() {
        let json = r#"{
            "object": "list",
            "data": [
                {"id": "gpt-4o", "object": "model", "created": 1700000000, "owned_by": "openai"},
                {"id": "gpt-3.5-turbo", "object": "model", "created": 1690000000, "owned_by": "openai"}
            ]
        }"#;

        let resp: ListModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].id, "gpt-4o");
        assert_eq!(resp.data[0].owned_by.as_deref(), Some("openai"));
        assert_eq!(resp.data[0].created, Some(1_700_000_000));
        assert_eq!(resp.data[1].id, "gpt-3.5-turbo");
    }

    #[test]
    fn parse_list_models_minimal() {
        // Some providers return minimal model entries.
        let json = r#"{"data": [{"id": "local-model"}]}"#;

        let resp: ListModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].id, "local-model");
        assert!(resp.data[0].owned_by.is_none());
        assert!(resp.data[0].created.is_none());
    }

    #[test]
    fn parse_list_models_empty() {
        let json = r#"{"object": "list", "data": []}"#;

        let resp: ListModelsResponse = serde_json::from_str(json).unwrap();
        assert!(resp.data.is_empty());
    }
}
