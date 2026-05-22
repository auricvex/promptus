//! Wire-format request structs for the OpenAI-compatible chat completions API.
//!
//! These are private to this crate — external consumers interact with
//! `promptus_core::ChatRequest` instead.

use serde::Serialize;

// ---------------------------------------------------------------------------
// Top-level request
// ---------------------------------------------------------------------------

/// The wire-format request body sent to `POST /chat/completions`.
///
/// Fields use `skip_serializing_if = "Option::is_none"` so only parameters
/// the caller actually set are included in the JSON payload. This keeps
/// requests minimal and avoids confusing providers that reject unknown fields.
#[derive(Debug, Serialize)]
pub(crate) struct CreateChatCompletionRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Tools are `serde_json::Value` to support both standard function tools
    /// (`{"type": "function", "function": {...}}`) and provider-specific tool
    /// types (e.g. `{"type": "web_search", ...}` from MiMo).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<WireToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<WireReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<WireResponseFormat>,
    /// Always `true` when streaming; omitted (defaults to false) otherwise.
    #[serde(skip_serializing_if = "is_false")]
    pub stream: bool,
    /// Request token usage in the final stream chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<WireStreamOptions>,
    /// Extra provider-specific fields merged verbatim into the request body.
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

fn is_false(b: &bool) -> bool {
    !b
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// A wire-format message. Role-specific fields are flattened so the JSON
/// shape matches what the API expects: `{"role": "user", "content": "..."}`.
///
/// We use a single struct with all possible fields rather than an enum
/// because serde's tagged-enum handling for this API is fragile across
/// providers — many providers are lenient about extra null fields but strict
/// about unexpected `type` tags.
#[derive(Debug, Serialize)]
pub(crate) struct WireMessage {
    pub role: &'static str,
    /// Content can be a plain string or an array of content parts. We
    /// serialize as `ContentValue` which handles both cases.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ContentValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Present only for `tool` role messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Present only for `assistant` role messages that requested tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<WireToolCall>>,
}

/// Content can be either a plain string or an array of content parts.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum ContentValue {
    Text(String),
    Parts(Vec<WireContentPart>),
}

/// A content part in the wire format.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub(crate) enum WireContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: WireImageUrl },
    #[serde(rename = "file")]
    File { file: WireFileSource },
}

#[derive(Debug, Serialize)]
pub(crate) struct WireImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct WireFileSource {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Tool calls (in assistant messages)
// ---------------------------------------------------------------------------

/// A tool call embedded in an assistant message.
#[derive(Debug, Serialize)]
pub(crate) struct WireToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: &'static str,
    pub function: WireFunctionCall,
}

#[derive(Debug, Serialize)]
pub(crate) struct WireFunctionCall {
    pub name: String,
    /// JSON-encoded arguments string. We serialize the `serde_json::Value`
    /// back to a string because the wire format expects a string, not an
    /// inline object.
    pub arguments: String,
}

// ---------------------------------------------------------------------------
// Tool choice
// ---------------------------------------------------------------------------

/// Wire-format tool choice. Simple string values (`auto`, `none`, `required`)
/// serialize as strings; named choices serialize as objects.
#[derive(Debug)]
pub(crate) enum WireToolChoice {
    Auto,
    None,
    Required,
    Named(WireNamedToolChoice),
}

#[derive(Debug, Serialize)]
pub(crate) struct WireNamedToolChoice {
    #[serde(rename = "type")]
    pub choice_type: &'static str,
    pub function: WireNamedToolChoiceFunction,
}

#[derive(Debug, Serialize)]
pub(crate) struct WireNamedToolChoiceFunction {
    pub name: String,
}

// Serialize the simple variants as strings.
impl Serialize for WireToolChoice {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Named(inner) => inner.serialize(serializer),
        }
    }
}

// ---------------------------------------------------------------------------
// Reasoning effort
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub(crate) struct WireReasoningEffort(pub &'static str);

// ---------------------------------------------------------------------------
// Response format
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub(crate) enum WireResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: WireJsonSchema },
}

#[derive(Debug, Serialize)]
pub(crate) struct WireJsonSchema {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

// ---------------------------------------------------------------------------
// Stream options
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub(crate) struct WireStreamOptions {
    /// Request a final chunk with usage statistics.
    pub include_usage: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_simple_request() {
        let req = CreateChatCompletionRequest {
            model: "gpt-4o".to_owned(),
            messages: vec![WireMessage {
                role: "user",
                content: Some(ContentValue::Text("hello".to_owned())),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            }],
            temperature: Some(0.7),
            top_p: None,
            max_tokens: None,
            stop: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            response_format: None,
            stream: false,
            stream_options: None,
            extra: serde_json::Value::Null,
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "gpt-4o");
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"], "hello");
        assert!((json["temperature"].as_f64().unwrap() - 0.7).abs() < 0.01);
        // top_p should not be present
        assert!(json.get("top_p").is_none());
        // stream should not be present (false is skipped)
        assert!(json.get("stream").is_none());
    }

    #[test]
    fn serialize_stream_request() {
        let req = CreateChatCompletionRequest {
            model: "gpt-4o".to_owned(),
            messages: vec![WireMessage {
                role: "user",
                content: Some(ContentValue::Text("hi".to_owned())),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            }],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            response_format: None,
            stream: true,
            stream_options: Some(WireStreamOptions {
                include_usage: true,
            }),
            extra: serde_json::Value::Null,
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["stream"], true);
        assert_eq!(json["stream_options"]["include_usage"], true);
    }

    #[test]
    fn serialize_tool_choice_variants() {
        let auto = serde_json::to_value(&WireToolChoice::Auto).unwrap();
        assert_eq!(auto, "auto");

        let none = serde_json::to_value(&WireToolChoice::None).unwrap();
        assert_eq!(none, "none");

        let required = serde_json::to_value(&WireToolChoice::Required).unwrap();
        assert_eq!(required, "required");

        let named = serde_json::to_value(WireToolChoice::Named(WireNamedToolChoice {
            choice_type: "function",
            function: WireNamedToolChoiceFunction {
                name: "get_weather".to_owned(),
            },
        }))
        .unwrap();
        assert_eq!(named["type"], "function");
        assert_eq!(named["function"]["name"], "get_weather");
    }

    #[test]
    fn serialize_tool_definitions() {
        // Tools are now plain JSON values, supporting both standard function
        // tools and provider-specific tool types.
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        })];

        let json = serde_json::to_value(&tools).unwrap();
        assert_eq!(json[0]["type"], "function");
        assert_eq!(json[0]["function"]["name"], "get_weather");
        assert_eq!(json[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn serialize_response_format_json_schema() {
        let fmt = WireResponseFormat::JsonSchema {
            json_schema: WireJsonSchema {
                name: "my_schema".to_owned(),
                description: None,
                schema: Some(serde_json::json!({"type": "object"})),
                strict: Some(true),
            },
        };
        let json = serde_json::to_value(&fmt).unwrap();
        assert_eq!(json["type"], "json_schema");
        assert_eq!(json["json_schema"]["name"], "my_schema");
        assert_eq!(json["json_schema"]["strict"], true);
    }

    #[test]
    fn serialize_reasoning_effort() {
        let effort = WireReasoningEffort("high");
        let json = serde_json::to_value(&effort).unwrap();
        assert_eq!(json, "high");
    }

    #[test]
    fn serialize_multimodal_content() {
        let msg = WireMessage {
            role: "user",
            content: Some(ContentValue::Parts(vec![
                WireContentPart::Text {
                    text: "What's in this image?".to_owned(),
                },
                WireContentPart::ImageUrl {
                    image_url: WireImageUrl {
                        url: "https://example.com/cat.jpg".to_owned(),
                        detail: Some("high".to_owned()),
                    },
                },
            ])),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        };

        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][1]["type"], "image_url");
        assert_eq!(json["content"][1]["image_url"]["detail"], "high");
    }

    #[test]
    fn serialize_extra_fields_flattened() {
        let req = CreateChatCompletionRequest {
            model: "groq/compound".to_owned(),
            messages: vec![WireMessage {
                role: "user",
                content: Some(ContentValue::Text("hello".to_owned())),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            }],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            response_format: None,
            stream: false,
            stream_options: None,
            extra: serde_json::json!({
                "compound_custom": {
                    "tools": {
                        "enabled_tools": ["web_search"]
                    }
                }
            }),
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "groq/compound");
        assert_eq!(
            json["compound_custom"]["tools"]["enabled_tools"][0],
            "web_search"
        );
    }

    #[test]
    fn serialize_extra_null_omitted() {
        let req = CreateChatCompletionRequest {
            model: "gpt-4o".to_owned(),
            messages: vec![WireMessage {
                role: "user",
                content: Some(ContentValue::Text("hello".to_owned())),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            }],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            response_format: None,
            stream: false,
            stream_options: None,
            extra: serde_json::Value::Null,
        };

        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("compound_custom").is_none());
    }
}
