//! Ergonomic builder for chat completion requests.

use futures::stream::BoxStream;
use promptus_core::{
    ChatRequest, ChatResponse, ContentPart, ImageSource, Message, ProviderError, ReasoningEffort,
    ResponseFormat, Role, StreamEvent, ToolChoice, ToolSpec,
};

use crate::client::PromptusClient;

/// An ergonomic builder for constructing and sending chat requests.
///
/// Obtained from [`PromptusClient::chat`]. Provides convenience methods for
/// common message patterns (system, user, assistant) while still allowing
/// full control via [`messages`](Self::messages) and
/// [`request`](Self::request).
///
/// # Example
///
/// ```ignore
/// let response = client
///     .chat("groq")
///     .model("llama-3.3-70b-versatile")
///     .system("You are a helpful assistant.")
///     .user("What is the meaning of life?")
///     .temperature(0.7)
///     .send()
///     .await?;
/// ```
pub struct ChatRequestBuilder<'a> {
    client: &'a PromptusClient,
    provider_name: String,
    model: Option<String>,
    messages: Vec<Message>,
    tools: Option<Vec<ToolSpec>>,
    tool_choice: Option<ToolChoice>,
    reasoning_effort: Option<ReasoningEffort>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    stop: Option<Vec<String>>,
    response_format: Option<ResponseFormat>,
    extra: Option<serde_json::Value>,
}

impl<'a> ChatRequestBuilder<'a> {
    pub(crate) fn new(client: &'a PromptusClient, provider_name: String) -> Self {
        Self {
            client,
            provider_name,
            model: None,
            messages: Vec::new(),
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            response_format: None,
            extra: None,
        }
    }

    /// Set the model to use for this request.
    ///
    /// This is required — the request will fail at send time if no model is
    /// set.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Add a system message to the conversation.
    pub fn system(mut self, text: impl Into<String>) -> Self {
        self.messages.push(Message::system(text));
        self
    }

    /// Add a developer message to the conversation.
    ///
    /// Developer messages are an OpenAI-specific refinement of system
    /// messages — some providers treat them identically.
    pub fn developer(mut self, text: impl Into<String>) -> Self {
        self.messages.push(Message::developer(text));
        self
    }

    /// Add a user message to the conversation.
    pub fn user(mut self, text: impl Into<String>) -> Self {
        self.messages.push(Message::user(text));
        self
    }

    /// Add an assistant message to the conversation.
    ///
    /// Useful for few-shot prompting or reconstructing conversation history.
    pub fn assistant(mut self, text: impl Into<String>) -> Self {
        self.messages.push(Message::assistant(text));
        self
    }

    /// Add a message with an image content part.
    ///
    /// This is a convenience method for the common case of a user message
    /// containing text and an image URL. For more complex multi-part
    /// messages, construct a [`Message`] directly and use
    /// [`messages`](Self::messages).
    pub fn user_with_image(
        mut self,
        text: impl Into<String>,
        image_url: impl Into<String>,
    ) -> Self {
        self.messages.push(Message {
            role: Role::User,
            content: vec![
                ContentPart::Text(text.into()),
                ContentPart::Image(ImageSource::Url(image_url.into())),
            ],
            tool_call_id: None,
            tool_calls: None,
            name: None,
        });
        self
    }

    /// Add a tool response message.
    ///
    /// Use this to respond to a tool call from the model. `tool_call_id`
    /// should match the ID from the model's tool call, and `result` is the
    /// tool's output as a string.
    pub fn tool_result(
        mut self,
        tool_call_id: impl Into<String>,
        result: impl Into<String>,
    ) -> Self {
        self.messages.push(Message::tool(tool_call_id, result));
        self
    }

    /// Replace all messages with the given list.
    ///
    /// Use this for full control over the conversation history when the
    /// convenience methods don't fit.
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    /// Register a tool the model may call.
    ///
    /// Accepts a [`ToolDefinition`] (which converts to `ToolSpec::Function`
    /// automatically) or a [`ToolSpec`] directly. Can be called multiple
    /// times to register multiple tools.
    pub fn tool(mut self, tool: impl Into<ToolSpec>) -> Self {
        self.tools.get_or_insert_with(Vec::new).push(tool.into());
        self
    }

    /// Register a provider-specific tool as raw JSON.
    ///
    /// Use this for tool types that don't follow the standard function tool
    /// shape, such as Xiaomi MiMo's `web_search`:
    ///
    /// ```ignore
    /// use serde_json::json;
    ///
    /// builder.raw_tool(json!({
    ///     "type": "web_search",
    ///     "max_keyword": 3,
    ///     "force_search": true,
    /// }))
    /// ```
    pub fn raw_tool(mut self, tool: serde_json::Value) -> Self {
        self.tools
            .get_or_insert_with(Vec::new)
            .push(ToolSpec::Raw(tool));
        self
    }

    /// Set the tool choice policy.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Set the reasoning effort level for reasoning models.
    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Set the sampling temperature (0.0–2.0).
    ///
    /// Higher values produce more random output; lower values are more
    /// deterministic.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the nucleus sampling threshold (0.0–1.0).
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set the maximum number of tokens to generate.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Add a stop sequence.
    ///
    /// Can be called multiple times to add multiple stop sequences.
    pub fn stop(mut self, sequence: impl Into<String>) -> Self {
        self.stop.get_or_insert_with(Vec::new).push(sequence.into());
        self
    }

    /// Set the response format (text, JSON, or JSON with a schema).
    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Set extra provider-specific fields merged into the request body.
    ///
    /// The value must be a JSON object. Its keys are flattened into the
    /// top-level request, so you can pass vendor extensions like Groq's
    /// `compound_custom` without the core types knowing about them.
    pub fn extra(mut self, extra: serde_json::Value) -> Self {
        self.extra = Some(extra);
        self
    }

    /// Override the entire request with a pre-built [`ChatRequest`].
    ///
    /// This replaces all builder state. Useful when you have a `ChatRequest`
    /// from serialization or another source and just want to send it through
    /// the client.
    pub fn request(mut self, request: ChatRequest) -> Self {
        self.model = Some(request.model);
        self.messages = request.messages;
        self.tools = request.tools;
        self.tool_choice = request.tool_choice;
        self.reasoning_effort = request.reasoning_effort;
        self.temperature = request.temperature;
        self.top_p = request.top_p;
        self.max_tokens = request.max_tokens;
        self.stop = request.stop;
        self.response_format = request.response_format;
        self.extra = request.extra;
        self
    }

    /// Build the [`ChatRequest`] without sending it.
    ///
    /// Useful for inspecting or serializing the request before sending.
    pub fn build_request(self) -> Result<ChatRequest, ProviderError> {
        let model = self
            .model
            .ok_or_else(|| ProviderError::InvalidRequest("model is required".to_owned()))?;

        Ok(ChatRequest {
            model,
            messages: self.messages,
            tools: self.tools,
            tool_choice: self.tool_choice,
            reasoning_effort: self.reasoning_effort,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            stop: self.stop,
            response_format: self.response_format,
            stream: false,
            extra: None,
        })
    }

    /// Send the chat request (non-streaming) and return the response.
    pub async fn send(self) -> Result<ChatResponse, ProviderError> {
        let provider = self.client.provider(&self.provider_name).ok_or_else(|| {
            ProviderError::InvalidRequest(format!(
                "no provider registered with name '{}'",
                self.provider_name
            ))
        })?;

        let request = self.build_request()?;
        provider.chat(request).await
    }

    /// Send the chat request as a stream and return a stream of events.
    ///
    /// The stream yields [`StreamEvent`]s — content deltas, reasoning
    /// deltas, tool-call deltas, and a final `Finished` event.
    pub async fn stream(
        self,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError> {
        let provider = self.client.provider(&self.provider_name).ok_or_else(|| {
            ProviderError::InvalidRequest(format!(
                "no provider registered with name '{}'",
                self.provider_name
            ))
        })?;

        let mut request = self.build_request()?;
        request.stream = true;
        provider.chat_stream(request).await
    }
}
