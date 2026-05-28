//! Output parsers — extract structured data from model responses.
//!
//! These are plain [`Runnable<ChatResponse, T>`] implementations, not a
//! separate parser trait. Compose them at the end of any chain that produces
//! a `ChatResponse`.

use promptus_core::ChatResponse;
use serde::de::DeserializeOwned;

use crate::error::CatenaError;
use crate::runnable::Runnable;

/// Extracts the text content from a [`ChatResponse`].
///
/// Returns an error (not a panic, not silent `""`) if `content` is `None` —
/// this happens when the model only returned tool calls with no text.
pub struct StrOutputParser;

impl Runnable<ChatResponse> for StrOutputParser {
    type Output = String;
    type Error = CatenaError;

    async fn invoke(&self, input: ChatResponse) -> Result<Self::Output, Self::Error> {
        input.content.ok_or_else(|| {
            CatenaError::MissingField(
                "model returned no text content (only tool calls?)".to_owned(),
            )
        })
    }
}

/// Parses the text content of a [`ChatResponse`] as JSON into type `T`.
///
/// On parse failure, the error includes the raw text so you can see exactly
/// what the model produced — essential for debugging malformed output.
pub struct JsonOutputParser<T: DeserializeOwned + Send + Sync> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DeserializeOwned + Send + Sync> JsonOutputParser<T> {
    /// Create a new JSON output parser.
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DeserializeOwned + Send + Sync> Default for JsonOutputParser<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DeserializeOwned + Send + Sync> Runnable<ChatResponse> for JsonOutputParser<T> {
    type Output = T;
    type Error = CatenaError;

    async fn invoke(&self, input: ChatResponse) -> Result<Self::Output, Self::Error> {
        let text = input.content.ok_or_else(|| {
            CatenaError::MissingField(
                "model returned no text content (only tool calls?)".to_owned(),
            )
        })?;
        serde_json::from_str(&text).map_err(|e| {
            // Retain the raw text in the error — it's essential for
            // debugging what the model actually produced.
            CatenaError::Parse(format!("JSON parse error: {e}\n  raw text: {text}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use promptus_core::FinishReason;

    fn make_response(content: Option<&str>) -> ChatResponse {
        ChatResponse {
            content: content.map(|s| s.to_owned()),
            reasoning_content: None,
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: None,
            model: "test".to_owned(),
        }
    }

    #[tokio::test]
    async fn str_parser_extracts_text() {
        let resp = make_response(Some("hello world"));
        let result = StrOutputParser.invoke(resp).await.unwrap();
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn str_parser_errors_on_none() {
        let resp = make_response(None);
        let result = StrOutputParser.invoke(resp).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no text content"));
    }

    #[derive(Debug, serde::Deserialize, PartialEq)]
    struct Point {
        x: i32,
        y: i32,
    }

    #[tokio::test]
    async fn json_parser_parses_object() {
        let resp = make_response(Some(r#"{"x": 1, "y": 2}"#));
        let result: Point = JsonOutputParser::new().invoke(resp).await.unwrap();
        assert_eq!(result, Point { x: 1, y: 2 });
    }

    #[tokio::test]
    async fn json_parser_errors_on_malformed() {
        let resp = make_response(Some("not json at all"));
        let result: Result<Point, _> = JsonOutputParser::new().invoke(resp).await;
        let err = result.unwrap_err().to_string();
        assert!(err.contains("JSON parse error"));
        // Raw text must be preserved in the error for debugging.
        assert!(err.contains("not json at all"));
    }

    #[tokio::test]
    async fn json_parser_errors_on_none() {
        let resp = make_response(None);
        let result: Result<Point, _> = JsonOutputParser::new().invoke(resp).await;
        assert!(result.is_err());
    }
}
