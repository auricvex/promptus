//! Provider traits for chat completions.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use crate::error::ProviderError;
use crate::types::{ChatRequest, ChatResponse, ModelInfo, StreamEvent};

// ---------------------------------------------------------------------------
// ChatProvider — ergonomic trait for implementors
// ---------------------------------------------------------------------------

/// A provider that can generate chat completions.
///
/// Implement this trait to add support for a new LLM provider. The trait uses
/// native `async fn` (RPITIT) so implementors write normal async code without
/// boxing or macros.
///
/// The companion [`DynChatProvider`] trait provides object-safe dispatch for
/// the facade's provider registry — you never need to implement it manually;
/// a blanket impl covers anything that implements `ChatProvider`.
///
/// # Example
///
/// ```ignore
/// impl ChatProvider for MyProvider {
///     async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError> {
///         // Make HTTP call, parse response, return ChatResponse
///     }
///
///     async fn chat_stream(
///         &self,
///         request: ChatRequest,
///     ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError> {
///         // Make streaming HTTP call, return stream of events
///     }
/// }
/// ```
pub trait ChatProvider: Send + Sync {
    /// Generate a single chat completion (non-streaming).
    fn chat(
        &self,
        request: ChatRequest,
    ) -> impl Future<Output = Result<ChatResponse, ProviderError>> + Send;

    /// Generate a streamed chat completion.
    ///
    /// Returns a stream of [`StreamEvent`]s that, when consumed in order,
    /// reconstruct the full response. The stream should yield a
    /// `StreamEvent::Finished` as its last event.
    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> impl Future<
        Output = Result<
            futures::stream::BoxStream<'static, Result<StreamEvent, ProviderError>>,
            ProviderError,
        >,
    > + Send;
}

// ---------------------------------------------------------------------------
// ModelProvider — trait for providers that can list available models
// ---------------------------------------------------------------------------

/// A provider that can enumerate its available models.
///
/// Not all providers support model listing — this trait is separate from
/// [`ChatProvider`] so implementors only commit to capabilities they actually
/// have. Providers that support both (e.g. OpenAI, Groq) implement both
/// traits.
///
/// # Example
///
/// ```ignore
/// impl ModelProvider for MyProvider {
///     async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
///         // Call GET /models, parse, return Vec<ModelInfo>
///     }
/// }
/// ```
pub trait ModelProvider: Send + Sync {
    /// List the models available from this provider.
    ///
    /// Returns a list of [`ModelInfo`] descriptors. The set of fields
    /// populated on each descriptor depends on the provider — `id` is always
    /// present; `owned_by` and `created` are optional.
    fn list_models(&self) -> impl Future<Output = Result<Vec<ModelInfo>, ProviderError>> + Send;
}

// ---------------------------------------------------------------------------
// DynModelProvider — object-safe trait for type-erased dispatch
// ---------------------------------------------------------------------------

/// Object-safe version of [`ModelProvider`] that uses boxed futures.
///
/// This trait exists solely to enable `Box<dyn DynModelProvider>` in
/// registries. A blanket impl automatically implements it for any
/// `T: ModelProvider + 'static`, so provider authors never write this by
/// hand.
pub trait DynModelProvider: Send + Sync {
    /// List the models available from this provider.
    fn list_models(&self) -> BoxFut<'_, Result<Vec<ModelInfo>, ProviderError>>;
}

/// Blanket impl: anything implementing [`ModelProvider`] automatically
/// implements [`DynModelProvider`].
impl<T: ModelProvider + 'static> DynModelProvider for T {
    fn list_models(&self) -> BoxFut<'_, Result<Vec<ModelInfo>, ProviderError>> {
        Box::pin(ModelProvider::list_models(self))
    }
}

// ---------------------------------------------------------------------------
// DynChatProvider — object-safe trait for type-erased dispatch
// ---------------------------------------------------------------------------

/// A boxed future returned by [`DynChatProvider`] methods.
pub type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Object-safe version of [`ChatProvider`] that uses boxed futures.
///
/// This trait exists solely to enable `Box<dyn DynChatProvider>` in the
/// facade's provider registry. A blanket impl automatically implements it
/// for any `T: ChatProvider + 'static`, so provider authors never write
/// this by hand.
pub trait DynChatProvider: Send + Sync {
    /// Generate a single chat completion (non-streaming).
    fn chat(&self, request: ChatRequest) -> BoxFut<'_, Result<ChatResponse, ProviderError>>;

    /// Generate a streamed chat completion.
    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> BoxFut<
        '_,
        Result<
            futures::stream::BoxStream<'static, Result<StreamEvent, ProviderError>>,
            ProviderError,
        >,
    >;
}

/// Blanket impl: anything implementing [`ChatProvider`] automatically
/// implements [`DynChatProvider`].
impl<T: ChatProvider + 'static> DynChatProvider for T {
    fn chat(&self, request: ChatRequest) -> BoxFut<'_, Result<ChatResponse, ProviderError>> {
        Box::pin(ChatProvider::chat(self, request))
    }

    fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> BoxFut<
        '_,
        Result<
            futures::stream::BoxStream<'static, Result<StreamEvent, ProviderError>>,
            ProviderError,
        >,
    > {
        Box::pin(ChatProvider::chat_stream(self, request))
    }
}

// ---------------------------------------------------------------------------
// Tool-call delta accumulator (shared helper for streaming)
// ---------------------------------------------------------------------------

/// Accumulates streamed tool-call deltas into complete [`ToolCall`]s.
///
/// OpenAI-compatible providers stream tool call arguments as string fragments
/// keyed by `index`. Only the first chunk for a given index carries `id` and
/// `name`. This helper collects those fragments and produces the final
/// `ToolCall` values.
///
/// # Usage
///
/// ```ignore
/// let mut acc = ToolCallAccumulator::new();
/// for event in stream {
///     if let StreamEvent::ToolCallDelta { index, id, name, arguments_delta } = event {
///         acc.accumulate(index, id, name, arguments_delta);
///     }
/// }
/// let tool_calls = acc.finish();
/// ```
#[derive(Debug, Default)]
pub struct ToolCallAccumulator {
    calls: HashMap<u32, PendingToolCall>,
}

#[derive(Debug, Default)]
struct PendingToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl ToolCallAccumulator {
    /// Create a new, empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a tool-call delta into the accumulator.
    ///
    /// `index` identifies which tool call this delta belongs to. `id` and
    /// `name` should be `Some` only for the first delta of a given index.
    /// `arguments_delta` is a string fragment to append to that tool call's
    /// arguments.
    pub fn accumulate(
        &mut self,
        index: u32,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: Option<String>,
    ) {
        let entry = self.calls.entry(index).or_default();
        if let Some(id) = id {
            entry.id = Some(id);
        }
        if let Some(name) = name {
            entry.name = Some(name);
        }
        if let Some(delta) = arguments_delta {
            entry.arguments.push_str(&delta);
        }
    }

    /// Consume the accumulator and produce the final list of tool calls.
    ///
    /// Tool calls are returned in index order. Arguments are parsed as JSON;
    /// if parsing fails (the model emitted malformed JSON), the raw string is
    /// wrapped in `serde_json::Value::String` so callers always get a `Value`.
    pub fn finish(self) -> Vec<crate::types::ToolCall> {
        let mut entries: Vec<_> = self.calls.into_iter().collect();
        entries.sort_by_key(|(i, _)| *i);

        entries
            .into_iter()
            .map(|(_, pending)| {
                let arguments = serde_json::from_str(&pending.arguments)
                    .unwrap_or(serde_json::Value::String(pending.arguments));
                crate::types::ToolCall {
                    id: pending.id.unwrap_or_default(),
                    name: pending.name.unwrap_or_default(),
                    arguments,
                }
            })
            .collect()
    }
}

/// A provider registry that stores heterogeneous providers behind
/// `Box<dyn DynChatProvider>`.
///
/// Used by the facade crate to hold named providers.
pub struct ProviderRegistry {
    providers: HashMap<String, Box<dyn DynChatProvider>>,
}

impl ProviderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Register a provider under the given name.
    ///
    /// If a provider with the same name already exists, it is replaced.
    pub fn register(&mut self, name: impl Into<String>, provider: Box<dyn DynChatProvider>) {
        self.providers.insert(name.into(), provider);
    }

    /// Get a reference to a named provider.
    ///
    /// Returns `None` if no provider is registered under that name.
    pub fn get(&self, name: &str) -> Option<&dyn DynChatProvider> {
        self.providers.get(name).map(|p| p.as_ref())
    }

    /// List the names of all registered providers.
    pub fn names(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_accumulator_basic() {
        let mut acc = ToolCallAccumulator::new();
        acc.accumulate(
            0,
            Some("call_1".to_owned()),
            Some("get_weather".to_owned()),
            Some(r#"{"loc"#.to_owned()),
        );
        acc.accumulate(0, None, None, Some(r#"ation":"NYC"}"#.to_owned()));

        let calls = acc.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, serde_json::json!({"location": "NYC"}));
    }

    #[test]
    fn tool_call_accumulator_multiple() {
        let mut acc = ToolCallAccumulator::new();
        acc.accumulate(
            0,
            Some("call_a".to_owned()),
            Some("fn_a".to_owned()),
            Some(r#"{"x":1}"#.to_owned()),
        );
        acc.accumulate(
            1,
            Some("call_b".to_owned()),
            Some("fn_b".to_owned()),
            Some(r#"{"y":2}"#.to_owned()),
        );

        let calls = acc.finish();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "fn_a");
        assert_eq!(calls[1].name, "fn_b");
    }

    #[test]
    fn tool_call_accumulator_malformed_json() {
        let mut acc = ToolCallAccumulator::new();
        acc.accumulate(
            0,
            Some("call_x".to_owned()),
            Some("fn_x".to_owned()),
            Some("not valid json {{{".to_owned()),
        );

        let calls = acc.finish();
        assert_eq!(calls.len(), 1);
        // Malformed JSON falls back to Value::String
        assert_eq!(
            calls[0].arguments,
            serde_json::Value::String("not valid json {{{".to_owned())
        );
    }

    #[test]
    fn tool_call_accumulator_empty() {
        let acc = ToolCallAccumulator::new();
        assert!(acc.finish().is_empty());
    }

    #[test]
    fn provider_registry_basics() {
        use crate::types::{ChatRequest, ChatResponse, StreamEvent};

        struct DummyProvider;
        impl ChatProvider for DummyProvider {
            async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, ProviderError> {
                unimplemented!()
            }
            async fn chat_stream(
                &self,
                _request: ChatRequest,
            ) -> Result<
                futures::stream::BoxStream<'static, Result<StreamEvent, ProviderError>>,
                ProviderError,
            > {
                unimplemented!()
            }
        }

        let mut reg = ProviderRegistry::new();
        reg.register("test", Box::new(DummyProvider));
        assert!(reg.get("test").is_some());
        assert!(reg.get("missing").is_none());
        assert_eq!(reg.names(), vec!["test"]);
    }

    #[test]
    fn model_provider_dyn_dispatch() {
        use crate::types::ModelInfo;

        struct DummyModelProvider;
        impl ModelProvider for DummyModelProvider {
            async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
                Ok(vec![ModelInfo {
                    id: "test-model".to_owned(),
                    owned_by: Some("test".to_owned()),
                    created: None,
                }])
            }
        }

        // Verify blanket impl works: ModelProvider → DynModelProvider
        let dyn_ref: &dyn DynModelProvider = &DummyModelProvider;
        // We can't call async in a sync test, but we can verify the trait
        // object compiles and the type is correct.
        let _ = dyn_ref;
    }
}
