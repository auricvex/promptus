//! Provider-agnostic domain types for chat completions.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

/// The author of a message in a conversation.
///
/// `Developer` is an OpenAI-specific refinement of `System` — some providers
/// treat them identically, others give developer messages higher authority.
/// Map to whichever role the target provider expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

// ---------------------------------------------------------------------------
// Content parts
// ---------------------------------------------------------------------------

/// A discrete piece of message content — text, an image, or a file.
///
/// Messages carry a `Vec<ContentPart>` so multi-modal inputs (text + image)
/// can be expressed naturally. Most messages contain a single `Text` part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ContentPart {
    /// Plain text content.
    Text(String),
    /// An image — either a URL or inline base64 data.
    Image(ImageSource),
    /// A file — either a provider-hosted file ID or inline base64 data.
    File(FileSource),
}

/// The source of an image content part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImageSource {
    /// A publicly accessible URL.
    Url(String),
    /// Inline base64-encoded image data with its MIME type.
    Base64 { media_type: String, data: String },
}

/// The source of a file content part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FileSource {
    /// A file previously uploaded to the provider, referenced by ID.
    FileId(String),
    /// Inline base64-encoded file data with a filename.
    Base64 { filename: String, data: String },
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// A single message in a conversation.
///
/// The `role` determines who authored the message. `content` holds the
/// message body as one or more content parts (text, images, files).
///
/// For `Role::Assistant` messages that requested tool calls, populate
/// `tool_calls`. For `Role::Tool` replies, `tool_call_id` must reference the
/// tool call being answered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
    /// The ID of the tool call this message responds to. Required when
    /// `role` is `Role::Tool`.
    pub tool_call_id: Option<String>,
    /// Tool calls requested by the assistant. Present only on
    /// `Role::Assistant` messages.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// An optional participant name (used for multi-party conversations).
    pub name: Option<String>,
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// A tool specification in a chat request.
///
/// The standard variant is [`Function`](Self::Function), which wraps a
/// [`ToolDefinition`] and produces the canonical
/// `{"type": "function", "function": {...}}` wire format.
///
/// For providers that support non-function tool types (e.g. Xiaomi MiMo's
/// `web_search`, or future custom tool types), use [`Raw`](Self::Raw) to
/// pass an arbitrary JSON object that will be included verbatim in the
/// `tools` array.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolSpec {
    /// A standard function tool definition.
    Function(ToolDefinition),
    /// An arbitrary tool object, serialized as-is into the wire format.
    ///
    /// Use this for provider-specific tool types that don't follow the
    /// `{"type": "function", "function": {...}}` shape.
    ///
    /// # Example (MiMo web search)
    ///
    /// ```ignore
    /// use serde_json::json;
    ///
    /// let web_search = ToolSpec::Raw(json!({
    ///     "type": "web_search",
    ///     "max_keyword": 3,
    ///     "force_search": true,
    ///     "limit": 1,
    ///     "user_location": {
    ///         "type": "approximate",
    ///         "country": "China",
    ///         "region": "Hubei",
    ///         "city": "Wuhan"
    ///     }
    /// }));
    /// ```
    Raw(serde_json::Value),
}

/// The definition of a standard function tool the model may call.
///
/// `parameters` is a JSON Schema object describing the function's input.
/// If omitted, the function takes no parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema describing the function's parameters.
    pub parameters: serde_json::Value,
    /// Whether the provider should enforce strict schema adherence for
    /// structured outputs.
    pub strict: Option<bool>,
}

impl From<ToolDefinition> for ToolSpec {
    fn from(def: ToolDefinition) -> Self {
        ToolSpec::Function(def)
    }
}

/// Controls which tool the model is allowed or forced to call.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    /// The model may decide whether to call a tool (the default when tools
    /// are present).
    Auto,
    /// The model must not call any tool.
    None,
    /// The model must call one or more tools.
    Required,
    /// Force the model to call a specific tool by name.
    Named(String),
}

/// A tool call requested by the model.
///
/// `arguments` is the model's raw output parsed into a `serde_json::Value`.
/// If the model emitted malformed JSON, `arguments` will be
/// `Value::String(raw)` rather than a parsed object — callers should handle
/// both cases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned ID for this tool call, used to match tool responses.
    pub id: String,
    /// The function name the model wants to invoke.
    pub name: String,
    /// The function arguments. May be a JSON object on success, or a
    /// `Value::String` if the model produced invalid JSON.
    pub arguments: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Reasoning
// ---------------------------------------------------------------------------

/// Constrains the effort a reasoning model spends on internal reasoning.
///
/// Not all providers or models support every level — consult the target
/// provider's documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    /// No reasoning (supported only by some newer models).
    None,
    Minimal,
    Low,
    Medium,
    High,
    /// Extended reasoning beyond `High` (provider-specific).
    XHigh,
}

// ---------------------------------------------------------------------------
// Response format
// ---------------------------------------------------------------------------

/// Constrains the model's output format.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseFormat {
    /// Plain text output (the default).
    Text,
    /// Force the model to produce valid JSON (no schema enforcement).
    JsonObject,
    /// Force the model to produce JSON conforming to a specific schema.
    JsonSchema {
        /// A name for the schema (used by the provider for caching/logging).
        name: String,
        /// The JSON Schema the output must conform to.
        schema: serde_json::Value,
        /// Whether to enforce strict schema adherence.
        strict: bool,
    },
}

// ---------------------------------------------------------------------------
// ChatRequest
// ---------------------------------------------------------------------------

/// A provider-agnostic chat completion request.
///
/// This is the input to [`ChatProvider::chat`](crate::ChatProvider::chat) and
/// [`ChatProvider::chat_stream`](crate::ChatProvider::chat_stream). It
/// captures everything a provider needs to generate a response — the model,
/// conversation history, tool definitions, and sampling parameters.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// The model identifier (e.g. `"gpt-4o"`, `"llama-3.3-70b-versatile"`).
    pub model: String,
    /// The conversation history.
    pub messages: Vec<Message>,
    /// Tools the model may call. Use [`ToolSpec::Function`] for standard
    /// function tools, or [`ToolSpec::Raw`] for provider-specific tool types
    /// (e.g. web search).
    pub tools: Option<Vec<ToolSpec>>,
    /// Controls which tool the model is allowed/forced to call.
    pub tool_choice: Option<ToolChoice>,
    /// Constrains reasoning effort for reasoning models.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Sampling temperature (0.0–2.0). Higher values produce more random
    /// output.
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold (0.0–1.0).
    pub top_p: Option<f32>,
    /// Maximum number of tokens to generate (including reasoning tokens for
    /// some providers).
    pub max_tokens: Option<u32>,
    /// Stop sequences — the model will stop generating when any of these
    /// strings appear in the output.
    pub stop: Option<Vec<String>>,
    /// Constrains the output format (text, JSON, or JSON with a schema).
    pub response_format: Option<ResponseFormat>,
    /// Whether to stream the response via server-sent events.
    pub stream: bool,
    /// Extra provider-specific fields merged verbatim into the request body.
    ///
    /// The value must be a JSON object (or `None`). Its keys are flattened
    /// into the top-level request, so you can pass vendor extensions like
    /// Groq's `compound_custom` without the core types knowing about them.
    ///
    /// ```ignore
    /// use serde_json::json;
    ///
    /// let extra = json!({
    ///     "compound_custom": {
    ///         "tools": {
    ///             "enabled_tools": ["web_search", "code_interpreter"]
    ///         }
    ///     }
    /// });
    /// ```
    pub extra: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// ChatResponse
// ---------------------------------------------------------------------------

/// A provider-agnostic chat completion response.
///
/// Returned by [`ChatProvider::chat`](crate::ChatProvider::chat). Contains
/// the model's text output (if any), any tool calls it requested, and
/// optional usage/token accounting.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// The model's text output, if the model produced text rather than tool
    /// calls.
    pub content: Option<String>,
    /// Internal reasoning text, exposed by some reasoning models (e.g.
    /// DeepSeek-R1). Vendor-specific — not all providers populate this.
    pub reasoning_content: Option<String>,
    /// Tool calls the model requested. Empty if the model produced text.
    pub tool_calls: Vec<ToolCall>,
    /// Why the model stopped generating.
    pub finish_reason: FinishReason,
    /// Token usage statistics, if the provider reports them.
    pub usage: Option<Usage>,
    /// The model that actually generated the response (may differ from the
    /// requested model if the provider substituted a fallback).
    pub model: String,
}

/// Why the model stopped generating tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// The model hit a natural stop point or a configured stop sequence.
    Stop,
    /// The model hit the `max_tokens` limit.
    Length,
    /// The model requested one or more tool calls.
    ToolCalls,
    /// Content was omitted due to a content filter.
    ContentFilter,
    /// A provider-specific reason not covered by the other variants.
    Other(String),
}

/// Token usage statistics for a single request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the generated completion.
    pub completion_tokens: u32,
    /// Total tokens (prompt + completion).
    pub total_tokens: u32,
    /// Tokens spent on internal reasoning (reasoning models only).
    pub reasoning_tokens: Option<u32>,
    /// Prompt tokens served from the provider's cache.
    pub cached_prompt_tokens: Option<u32>,
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// A single event in a streamed chat completion.
///
/// Providers emit a sequence of `StreamEvent`s that, when consumed in order,
/// reconstruct the full response. Content and tool-call deltas are additive —
/// concatenate them to build the final output.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A fragment of the model's text output.
    ContentDelta(String),
    /// A fragment of the model's internal reasoning (vendor-specific).
    ReasoningDelta(String),
    /// A fragment of a tool call. Tool call arguments arrive as string
    /// fragments keyed by `index`; only the first chunk for a given index
    /// carries `id` and `name`.
    ToolCallDelta {
        /// The index of this tool call in the response's tool_calls array.
        index: u32,
        /// The tool call ID (present only in the first chunk for this index).
        id: Option<String>,
        /// The function name (present only in the first chunk for this index).
        name: Option<String>,
        /// A fragment of the function arguments JSON string.
        arguments_delta: Option<String>,
    },
    /// The stream has finished. Carries the stop reason and optional usage.
    Finished {
        finish_reason: FinishReason,
        usage: Option<Usage>,
    },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

impl Message {
    /// Create a system message with plain text content.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentPart::Text(text.into())],
            tool_call_id: None,
            tool_calls: None,
            name: None,
        }
    }

    /// Create a developer message with plain text content.
    ///
    /// Developer messages are an OpenAI-specific refinement of system
    /// messages — some providers treat them identically.
    pub fn developer(text: impl Into<String>) -> Self {
        Self {
            role: Role::Developer,
            content: vec![ContentPart::Text(text.into())],
            tool_call_id: None,
            tool_calls: None,
            name: None,
        }
    }

    /// Create a user message with plain text content.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentPart::Text(text.into())],
            tool_call_id: None,
            tool_calls: None,
            name: None,
        }
    }

    /// Create an assistant message with plain text content.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentPart::Text(text.into())],
            tool_call_id: None,
            tool_calls: None,
            name: None,
        }
    }

    /// Create a tool response message.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentPart::Text(content.into())],
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
            name: None,
        }
    }

    /// Get the text content of this message by concatenating all text parts.
    ///
    /// Returns `None` if the message has no text parts.
    pub fn text(&self) -> Option<String> {
        let texts: Vec<&str> = self
            .content
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(texts.join(""))
        }
    }
}

impl FinishReason {
    /// Parse a wire-format finish reason string into the typed enum.
    ///
    /// Unknown strings are wrapped in `FinishReason::Other` rather than
    /// causing an error, since providers may add proprietary stop reasons.
    pub fn from_wire(s: &str) -> Self {
        match s {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "tool_calls" => Self::ToolCalls,
            "content_filter" => Self::ContentFilter,
            other => Self::Other(other.to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_constructors() {
        let sys = Message::system("be helpful");
        assert_eq!(sys.role, Role::System);
        assert_eq!(sys.text().as_deref(), Some("be helpful"));

        let usr = Message::user("hello");
        assert_eq!(usr.role, Role::User);

        let ast = Message::assistant("hi there");
        assert_eq!(ast.role, Role::Assistant);

        let tool = Message::tool("call_123", "{\"result\": 42}");
        assert_eq!(tool.role, Role::Tool);
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_123"));
    }

    #[test]
    fn finish_reason_from_wire() {
        assert_eq!(FinishReason::from_wire("stop"), FinishReason::Stop);
        assert_eq!(FinishReason::from_wire("length"), FinishReason::Length);
        assert_eq!(
            FinishReason::from_wire("tool_calls"),
            FinishReason::ToolCalls
        );
        assert_eq!(
            FinishReason::from_wire("content_filter"),
            FinishReason::ContentFilter
        );
        assert_eq!(
            FinishReason::from_wire("something_else"),
            FinishReason::Other("something_else".to_owned())
        );
    }

    #[test]
    fn message_text_concatenation() {
        let msg = Message {
            role: Role::User,
            content: vec![
                ContentPart::Text("hello ".to_owned()),
                ContentPart::Text("world".to_owned()),
            ],
            tool_call_id: None,
            tool_calls: None,
            name: None,
        };
        assert_eq!(msg.text().as_deref(), Some("hello world"));
    }

    #[test]
    fn message_text_no_parts() {
        let msg = Message {
            role: Role::User,
            content: vec![],
            tool_call_id: None,
            tool_calls: None,
            name: None,
        };
        assert!(msg.text().is_none());
    }
}
