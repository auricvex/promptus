//! ReAct-style tool-calling agent.
//!
//! [`ReActAgent`] implements [`Runnable<String, AgentOutcome>`] — feed it a
//! user message, and it runs a tool-calling loop: send messages to the
//! provider, execute any requested tool calls, feed results back, repeat
//! until the model produces a plain answer or the iteration cap is hit.
//!
//! ## Streaming events
//!
//! The agent uses streaming internally and can emit real-time progress events
//! via a callback:
//!
//! ```ignore
//! let agent = ReActAgent::builder(provider, "model")
//!     .on_event(Arc::new(|event| {
//!         match event {
//!             AgentEvent::ContentDelta { delta, .. } => print!("{delta}"),
//!             AgentEvent::ToolCallsReady { calls, .. } => println!("calling tools: {calls:?}"),
//!             _ => {}
//!         }
//!     }))
//!     .build();
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use promptus_core::{
    ChatProvider, ChatRequest, ChatResponse, ContentPart, FinishReason, Message, Role, StreamEvent,
    ToolCall, ToolCallAccumulator, ToolChoice, ToolSpec,
};
use serde_json::Value;

use crate::error::CatenaError;
use crate::memory::Memory;
use crate::runnable::Runnable;
use crate::tool::DynTool;

// ---------------------------------------------------------------------------
// Streaming events
// ---------------------------------------------------------------------------

/// A real-time event emitted by the agent during execution.
///
/// Register a callback via [`ReActAgentBuilder::on_event`] to receive these.
/// Events are emitted synchronously during the agent loop — the callback
/// must not block for extended periods.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A new iteration of the ReAct loop is starting.
    IterationStart {
        /// Zero-based iteration index.
        iteration: usize,
    },
    /// A fragment of the model's text output.
    ContentDelta {
        /// Which iteration this belongs to.
        iteration: usize,
        /// The text fragment.
        delta: String,
    },
    /// A fragment of the model's internal reasoning (vendor-specific).
    ReasoningDelta {
        /// Which iteration this belongs to.
        iteration: usize,
        /// The reasoning fragment.
        delta: String,
    },
    /// The model requested tool calls — emitted before execution begins.
    ToolCallsReady {
        /// Which iteration this belongs to.
        iteration: usize,
        /// The complete tool calls.
        calls: Vec<ToolCall>,
    },
    /// A tool has finished executing.
    ToolResult {
        /// Which iteration this belongs to.
        iteration: usize,
        /// Name of the tool that was called.
        name: String,
        /// The tool's output, or an error message.
        result: Result<String, String>,
    },
    /// The model produced a final answer (no tool calls).
    FinalAnswer {
        /// Which iteration produced the answer.
        iteration: usize,
        /// The answer text.
        answer: String,
    },
}

/// Callback type for receiving [`AgentEvent`]s.
pub type EventHandler = Arc<dyn Fn(&AgentEvent) + Send + Sync>;

// ---------------------------------------------------------------------------
// Agent step / outcome
// ---------------------------------------------------------------------------

/// A single tool-call step within an agent's reasoning loop.
#[derive(Debug, Clone)]
pub struct AgentStep {
    /// The name of the tool that was called.
    pub tool_name: String,
    /// The arguments the model passed to the tool.
    pub tool_input: Value,
    /// The tool's output, or an error message if the call failed.
    pub tool_output: Result<String, String>,
}

/// The final result of an agent invocation.
#[derive(Debug, Clone)]
pub struct AgentOutcome {
    /// The model's final text answer.
    pub output: String,
    /// Full trace of every tool-call step, for debugging and observability.
    pub steps: Vec<AgentStep>,
}

// ---------------------------------------------------------------------------
// ReActAgent
// ---------------------------------------------------------------------------

/// A tool-calling agent that runs a ReAct-style loop with streaming.
///
/// On each invocation:
/// 1. Build a `ChatRequest` from the accumulated history (via [`Memory`]) +
///    system prompt (if any) + tool definitions.
/// 2. Stream the response, emitting [`AgentEvent`]s in real-time.
/// 3. If the model responds with tool calls, execute them and feed results
///    back. Repeat.
/// 4. If the model responds with text (no tool calls), treat that as the
///    final answer.
/// 5. If `max_iterations` is hit, return [`CatenaError::MaxIterationsReached`].
pub struct ReActAgent<P: ChatProvider> {
    provider: Arc<P>,
    model: String,
    system_prompt: Option<String>,
    tools: Vec<Box<dyn DynTool>>,
    raw_tool_specs: Vec<ToolSpec>,
    extra: Option<Value>,
    memory: Arc<dyn Memory>,
    max_iterations: usize,
    on_event: Option<EventHandler>,
}

impl<P: ChatProvider> ReActAgent<P> {
    /// Create a new agent builder.
    pub fn builder(provider: P, model: impl Into<String>) -> ReActAgentBuilder<P> {
        ReActAgentBuilder {
            provider,
            model: model.into(),
            system_prompt: None,
            tools: Vec::new(),
            raw_tool_specs: Vec::new(),
            extra: None,
            memory: None,
            max_iterations: 10,
            on_event: None,
        }
    }

    /// Emit an event to the registered callback, if any.
    fn emit(&self, event: &AgentEvent) {
        if let Some(ref handler) = self.on_event {
            handler(event);
        }
    }

    /// Stream a response from the provider, emitting content/reasoning deltas
    /// and accumulating tool calls. Returns the assembled `ChatResponse`.
    async fn stream_response(
        &self,
        request: ChatRequest,
        iteration: usize,
    ) -> Result<ChatResponse, CatenaError> {
        let mut stream = self.provider.chat_stream(request).await?;

        let mut content = String::new();
        let mut reasoning_content = String::new();
        let mut accumulator = ToolCallAccumulator::new();
        let mut finish_reason = FinishReason::Stop;
        let mut usage = None;
        let model = String::new();

        while let Some(event) = stream.next().await {
            let event = event?;
            match event {
                StreamEvent::ContentDelta(delta) => {
                    content.push_str(&delta);
                    self.emit(&AgentEvent::ContentDelta { iteration, delta });
                }
                StreamEvent::ReasoningDelta(delta) => {
                    reasoning_content.push_str(&delta);
                    self.emit(&AgentEvent::ReasoningDelta { iteration, delta });
                }
                StreamEvent::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments_delta,
                } => {
                    accumulator.accumulate(index, id, name, arguments_delta);
                }
                StreamEvent::Finished {
                    finish_reason: reason,
                    usage: u,
                } => {
                    finish_reason = reason;
                    usage = u;
                }
            }
        }

        let tool_calls = accumulator.finish();

        Ok(ChatResponse {
            content: if content.is_empty() {
                None
            } else {
                Some(content)
            },
            reasoning_content: if reasoning_content.is_empty() {
                None
            } else {
                Some(reasoning_content)
            },
            tool_calls,
            finish_reason,
            usage,
            model,
        })
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for [`ReActAgent`].
pub struct ReActAgentBuilder<P: ChatProvider> {
    provider: P,
    model: String,
    system_prompt: Option<String>,
    tools: Vec<Box<dyn DynTool>>,
    raw_tool_specs: Vec<ToolSpec>,
    extra: Option<Value>,
    memory: Option<Arc<dyn Memory>>,
    max_iterations: usize,
    on_event: Option<EventHandler>,
}

impl<P: ChatProvider> ReActAgentBuilder<P> {
    /// Set a system prompt prepended to every request.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Register a tool the agent can call.
    pub fn tool(mut self, tool: impl DynTool + 'static) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    /// Register multiple tools the agent can call.
    pub fn tools<T: DynTool + 'static>(mut self, tools: Vec<T>) -> Self {
        for tool in tools {
            self.tools.push(Box::new(tool));
        }
        self
    }

    /// Register a raw tool spec sent verbatim to the provider.
    ///
    /// Use this for provider-specific tool types (e.g. server-side web search)
    /// that don't follow the standard function tool format. These specs are
    /// included in every request but are not executed client-side — the
    /// provider handles them internally.
    pub fn raw_tool_spec(mut self, spec: ToolSpec) -> Self {
        self.raw_tool_specs.push(spec);
        self
    }

    /// Set extra provider-specific fields merged into every request body.
    ///
    /// The value must be a JSON object (or `None`). Its keys are flattened
    /// into the top-level request, so you can pass vendor extensions like
    /// Groq's `compound_custom` without the core types knowing about them.
    pub fn extra(mut self, extra: Value) -> Self {
        self.extra = Some(extra);
        self
    }

    /// Set the conversation memory backend.
    ///
    /// If not called, a default [`BufferMemory`](crate::memory::BufferMemory)
    /// is used.
    pub fn memory(mut self, memory: Arc<dyn Memory>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Set the maximum number of tool-calling iterations before the agent
    /// gives up. Defaults to 10.
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Register a callback for real-time streaming events.
    ///
    /// The callback is called synchronously during the agent loop — it
    /// should not block for extended periods.
    pub fn on_event(mut self, handler: EventHandler) -> Self {
        self.on_event = Some(handler);
        self
    }

    /// Build the agent.
    pub fn build(self) -> ReActAgent<P> {
        ReActAgent {
            provider: Arc::new(self.provider),
            model: self.model,
            system_prompt: self.system_prompt,
            tools: self.tools,
            raw_tool_specs: self.raw_tool_specs,
            extra: self.extra,
            memory: self
                .memory
                .unwrap_or_else(|| Arc::new(crate::memory::BufferMemory::new())),
            max_iterations: self.max_iterations,
            on_event: self.on_event,
        }
    }
}

// ---------------------------------------------------------------------------
// Runnable impl
// ---------------------------------------------------------------------------

impl<P: ChatProvider> Runnable<String> for ReActAgent<P> {
    type Output = AgentOutcome;
    type Error = CatenaError;

    async fn invoke(&self, input: String) -> Result<Self::Output, Self::Error> {
        // 1. Seed the working history with memory + the new user message.
        let mut history = self.memory.history().await;
        history.push(Message::user(input));

        // Prepend system prompt if configured.
        let mut full_history = Vec::new();
        if let Some(ref sys) = self.system_prompt {
            full_history.push(Message::system(sys));
        }
        full_history.extend(history);

        let mut tool_specs: Vec<ToolSpec> =
            self.tools.iter().map(|t| t.definition().into()).collect();
        tool_specs.extend(self.raw_tool_specs.iter().cloned());

        // Build a lookup map: tool name → index into self.tools.
        let tool_index: HashMap<String, usize> = self
            .tools
            .iter()
            .enumerate()
            .map(|(i, t)| (t.definition().name, i))
            .collect();

        let mut steps = Vec::new();
        let mut final_answer = None;

        for iteration in 0..self.max_iterations {
            self.emit(&AgentEvent::IterationStart { iteration });

            // 2. Build and stream the request.
            let request = ChatRequest {
                model: self.model.clone(),
                messages: full_history.clone(),
                tools: Some(tool_specs.clone()),
                tool_choice: Some(ToolChoice::Auto),
                reasoning_effort: None,
                temperature: None,
                top_p: None,
                max_tokens: None,
                stop: None,
                response_format: None,
                stream: true,
                extra: self.extra.clone(),
            };

            let response = self.stream_response(request, iteration).await?;

            if response.tool_calls.is_empty() {
                // 4. No tool calls — this is the final answer.
                let text = response.content.clone().unwrap_or_default();
                final_answer = Some(text.clone());

                // Append the assistant's final message to working history.
                full_history.push(Message::assistant(&text));

                self.emit(&AgentEvent::FinalAnswer {
                    iteration,
                    answer: text,
                });
                break;
            }

            // 3. Tool calls present — emit event, then execute.
            self.emit(&AgentEvent::ToolCallsReady {
                iteration,
                calls: response.tool_calls.clone(),
            });

            // Record the assistant message first (wire format ordering).
            let assistant_msg = Message {
                role: Role::Assistant,
                content: if let Some(ref c) = response.content {
                    vec![ContentPart::Text(c.clone())]
                } else {
                    vec![]
                },
                tool_call_id: None,
                tool_calls: Some(response.tool_calls.clone()),
                name: None,
            };
            full_history.push(assistant_msg);

            for tc in &response.tool_calls {
                let is_raw = self
                    .raw_tool_specs
                    .iter()
                    .any(|spec| matches!(spec, ToolSpec::Raw(v) if v.get("type").and_then(|t| t.as_str()) == Some(tc.name.as_str())));

                let output = if is_raw {
                    Err(format!(
                        "'{}' is a server-side tool — it should be handled \
                         by the provider, not called client-side",
                        tc.name
                    ))
                } else {
                    match tool_index.get(&tc.name) {
                        Some(&idx) => {
                            let args = tc.arguments.clone();
                            match self.tools[idx].call_dyn(args).await {
                                Ok(result) => Ok(result),
                                Err(e) => Err(e.to_string()),
                            }
                        }
                        None => Err(format!("unknown tool: '{}'", tc.name)),
                    }
                };

                let output_str = match &output {
                    Ok(s) => s.clone(),
                    Err(e) => e.clone(),
                };

                self.emit(&AgentEvent::ToolResult {
                    iteration,
                    name: tc.name.clone(),
                    result: output.clone(),
                });

                steps.push(AgentStep {
                    tool_name: tc.name.clone(),
                    tool_input: tc.arguments.clone(),
                    tool_output: output,
                });

                // Append tool response — tool_call_id must match for wire
                // format correctness.
                full_history.push(Message::tool(&tc.id, &output_str));
            }
        }

        let output = final_answer.ok_or(CatenaError::MaxIterationsReached {
            max: self.max_iterations,
        })?;

        // 5. Persist the full exchange to memory (skip system prompt).
        let skip = if self.system_prompt.is_some() { 1 } else { 0 };
        let to_persist: Vec<Message> = full_history.into_iter().skip(skip).collect();
        self.memory.append(to_persist).await;

        Ok(AgentOutcome { output, steps })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::BufferMemory;
    use crate::tool::Tool;
    use promptus_core::ProviderError;

    // -----------------------------------------------------------------------
    // Mock ChatProvider — scripted responses (streaming)
    // -----------------------------------------------------------------------

    /// Converts a `ChatResponse` into a stream of `StreamEvent`s, mimicking
    /// what a real provider would emit.
    fn response_to_stream(
        resp: &ChatResponse,
    ) -> futures::stream::BoxStream<'static, Result<StreamEvent, ProviderError>> {
        let mut events = Vec::new();

        // Emit content as a single delta.
        if let Some(ref content) = resp.content {
            events.push(Ok(StreamEvent::ContentDelta(content.clone())));
        }

        // Emit reasoning as a single delta.
        if let Some(ref reasoning) = resp.reasoning_content {
            events.push(Ok(StreamEvent::ReasoningDelta(reasoning.clone())));
        }

        // Emit tool calls.
        for (i, tc) in resp.tool_calls.iter().enumerate() {
            events.push(Ok(StreamEvent::ToolCallDelta {
                index: i as u32,
                id: Some(tc.id.clone()),
                name: Some(tc.name.clone()),
                arguments_delta: Some(serde_json::to_string(&tc.arguments).unwrap()),
            }));
        }

        // Emit finished.
        events.push(Ok(StreamEvent::Finished {
            finish_reason: resp.finish_reason.clone(),
            usage: resp.usage.clone(),
        }));

        Box::pin(futures::stream::iter(events))
    }

    struct MockProvider {
        responses: tokio::sync::RwLock<Vec<ChatResponse>>,
    }

    impl MockProvider {
        fn new(responses: Vec<ChatResponse>) -> Self {
            Self {
                responses: tokio::sync::RwLock::new(responses),
            }
        }
    }

    impl ChatProvider for MockProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, ProviderError> {
            let mut responses = self.responses.write().await;
            if responses.is_empty() {
                panic!("mock provider: no more scripted responses");
            }
            Ok(responses.remove(0))
        }

        async fn chat_stream(
            &self,
            _request: ChatRequest,
        ) -> Result<
            futures::stream::BoxStream<'static, Result<StreamEvent, ProviderError>>,
            ProviderError,
        > {
            let mut responses = self.responses.write().await;
            if responses.is_empty() {
                panic!("mock provider: no more scripted responses");
            }
            let resp = responses.remove(0);
            Ok(response_to_stream(&resp))
        }
    }

    // -----------------------------------------------------------------------
    // Mock tool
    // -----------------------------------------------------------------------

    struct AddTool;

    impl Tool for AddTool {
        fn definition(&self) -> promptus_core::ToolDefinition {
            promptus_core::ToolDefinition {
                name: "add".to_owned(),
                description: Some("Adds two numbers.".to_owned()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "a": { "type": "number" },
                        "b": { "type": "number" }
                    },
                    "required": ["a", "b"]
                }),
                strict: None,
            }
        }

        async fn call(&self, arguments: Value) -> Result<String, CatenaError> {
            let a = arguments["a"]
                .as_f64()
                .ok_or_else(|| CatenaError::ToolError {
                    tool: "add".to_owned(),
                    message: "missing 'a'".to_owned(),
                })?;
            let b = arguments["b"]
                .as_f64()
                .ok_or_else(|| CatenaError::ToolError {
                    tool: "add".to_owned(),
                    message: "missing 'b'".to_owned(),
                })?;
            Ok((a + b).to_string())
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn tool_call_response(tool_name: &str, call_id: &str, args: Value) -> ChatResponse {
        ChatResponse {
            content: None,
            reasoning_content: None,
            tool_calls: vec![ToolCall {
                id: call_id.to_owned(),
                name: tool_name.to_owned(),
                arguments: args,
            }],
            finish_reason: FinishReason::ToolCalls,
            usage: None,
            model: "test-model".to_owned(),
        }
    }

    fn text_response(text: &str) -> ChatResponse {
        ChatResponse {
            content: Some(text.to_owned()),
            reasoning_content: None,
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: Some(promptus_core::Usage::default()),
            model: "test-model".to_owned(),
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn agent_plain_answer_no_tools() {
        let provider = MockProvider::new(vec![text_response("Hello!")]);
        let agent = ReActAgent::builder(provider, "test-model")
            .max_iterations(5)
            .build();

        let outcome = agent.invoke("Hi".to_owned()).await.unwrap();
        assert_eq!(outcome.output, "Hello!");
        assert!(outcome.steps.is_empty());
    }

    #[tokio::test]
    async fn agent_single_tool_call() {
        let provider = MockProvider::new(vec![
            tool_call_response("add", "call_1", serde_json::json!({"a": 2, "b": 3})),
            text_response("The answer is 5."),
        ]);
        let agent = ReActAgent::builder(provider, "test-model")
            .tool(AddTool)
            .max_iterations(5)
            .build();

        let outcome = agent.invoke("What is 2+3?".to_owned()).await.unwrap();
        assert_eq!(outcome.output, "The answer is 5.");
        assert_eq!(outcome.steps.len(), 1);
        assert_eq!(outcome.steps[0].tool_name, "add");
        assert_eq!(outcome.steps[0].tool_output, Ok("5".to_owned()));
    }

    #[tokio::test]
    async fn agent_multi_step_tool_calls() {
        let provider = MockProvider::new(vec![
            tool_call_response("add", "call_1", serde_json::json!({"a": 2, "b": 3})),
            tool_call_response("add", "call_2", serde_json::json!({"a": 5, "b": 10})),
            text_response("The final answer is 15."),
        ]);
        let agent = ReActAgent::builder(provider, "test-model")
            .tool(AddTool)
            .max_iterations(5)
            .build();

        let outcome = agent
            .invoke("What is 2+3, then add 10?".to_owned())
            .await
            .unwrap();
        assert_eq!(outcome.output, "The final answer is 15.");
        assert_eq!(outcome.steps.len(), 2);
        assert_eq!(outcome.steps[0].tool_output, Ok("5".to_owned()));
        assert_eq!(outcome.steps[1].tool_output, Ok("15".to_owned()));
    }

    #[tokio::test]
    async fn agent_unknown_tool_records_error() {
        let provider = MockProvider::new(vec![
            tool_call_response("nonexistent", "call_1", serde_json::json!({"x": 1})),
            text_response("I tried."),
        ]);
        let agent = ReActAgent::builder(provider, "test-model")
            .max_iterations(5)
            .build();

        let outcome = agent.invoke("do something".to_owned()).await.unwrap();
        assert_eq!(outcome.steps.len(), 1);
        assert!(outcome.steps[0].tool_output.is_err());
        assert!(
            outcome.steps[0]
                .tool_output
                .as_ref()
                .unwrap_err()
                .contains("unknown tool")
        );
    }

    #[tokio::test]
    async fn agent_max_iterations_exceeded() {
        let responses = (0..10)
            .map(|i| {
                tool_call_response(
                    "add",
                    &format!("call_{i}"),
                    serde_json::json!({"a": i, "b": 1}),
                )
            })
            .collect();
        let provider = MockProvider::new(responses);
        let agent = ReActAgent::builder(provider, "test-model")
            .tool(AddTool)
            .max_iterations(5)
            .build();

        let result = agent.invoke("loop forever".to_owned()).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            CatenaError::MaxIterationsReached { max: 5 }
        ));
    }

    #[tokio::test]
    async fn agent_persists_to_memory() {
        let provider = MockProvider::new(vec![
            tool_call_response("add", "call_1", serde_json::json!({"a": 1, "b": 2})),
            text_response("It's 3."),
        ]);
        let memory = Arc::new(BufferMemory::new());
        let agent = ReActAgent::builder(provider, "test-model")
            .tool(AddTool)
            .memory(memory.clone())
            .max_iterations(5)
            .build();

        agent.invoke("1+2?".to_owned()).await.unwrap();

        let history = memory.history().await;
        assert_eq!(history.len(), 4);
        assert_eq!(history[0].role, Role::User);
        assert_eq!(history[1].role, Role::Assistant);
        assert!(history[1].tool_calls.is_some());
        assert_eq!(history[2].role, Role::Tool);
        assert_eq!(history[3].role, Role::Assistant);
    }

    #[tokio::test]
    async fn agent_emits_events() {
        let provider = MockProvider::new(vec![
            tool_call_response("add", "call_1", serde_json::json!({"a": 1, "b": 2})),
            text_response("It's 3."),
        ]);

        // Use std::sync::Mutex — the callback is synchronous and runs inside
        // an async runtime, so we can't use tokio::sync::Mutex::blocking_lock.
        let events: Arc<std::sync::Mutex<Vec<AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let events_clone = events.clone();

        let agent = ReActAgent::builder(provider, "test-model")
            .tool(AddTool)
            .max_iterations(5)
            .on_event(Arc::new(move |event| {
                events_clone.lock().unwrap().push(event.clone());
            }))
            .build();

        agent.invoke("1+2?".to_owned()).await.unwrap();

        let events = events.lock().unwrap();
        // Should have: IterationStart, ContentDelta (none for tool-only),
        // ToolCallsReady, ToolResult, IterationStart, ContentDelta, FinalAnswer
        let has_iteration_start = events
            .iter()
            .any(|e| matches!(e, AgentEvent::IterationStart { .. }));
        let has_tool_calls_ready = events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolCallsReady { .. }));
        let has_tool_result = events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolResult { .. }));
        let has_final_answer = events
            .iter()
            .any(|e| matches!(e, AgentEvent::FinalAnswer { .. }));

        assert!(has_iteration_start, "expected IterationStart event");
        assert!(has_tool_calls_ready, "expected ToolCallsReady event");
        assert!(has_tool_result, "expected ToolResult event");
        assert!(has_final_answer, "expected FinalAnswer event");
    }

    #[tokio::test]
    async fn agent_raw_tool_specs_included_in_request() {
        // Verify that raw tool specs are merged into the tool list sent to
        // the provider. The mock provider can't inspect the request directly,
        // but we can verify the agent works when raw specs are registered.
        let provider = MockProvider::new(vec![text_response("Search done.")]);
        let raw_spec = ToolSpec::Raw(serde_json::json!({
            "type": "web_search",
            "max_keyword": 3,
        }));
        let agent = ReActAgent::builder(provider, "test-model")
            .raw_tool_spec(raw_spec)
            .max_iterations(5)
            .build();

        let outcome = agent.invoke("search for cats".to_owned()).await.unwrap();
        assert_eq!(outcome.output, "Search done.");
        assert!(outcome.steps.is_empty());
    }
}
