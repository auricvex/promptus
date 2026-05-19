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
