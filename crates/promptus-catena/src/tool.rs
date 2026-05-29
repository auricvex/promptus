//! The `Tool` trait — something an agent can call at runtime.
//!
//! Distinct from `promptus_core::ToolDefinition` (which only *describes* a
//! tool for the wire format) — this trait *executes* it.

use promptus_core::BoxFut;
use promptus_core::ToolDefinition;
use serde_json::Value;

use crate::error::CatenaError;

// ---------------------------------------------------------------------------
// Tool trait
// ---------------------------------------------------------------------------

/// Something an agent can call.
///
/// [`definition()`](Tool::definition) is what gets sent to the model so it
/// knows the tool exists and its argument schema. [`call()`](Tool::call) is
/// what actually runs when the model decides to invoke it.
pub trait Tool: Send + Sync {
    /// Describe this tool for the model — name, description, parameter
    /// schema.
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with the given JSON arguments.
    ///
    /// Returns the tool's output as a string, which will be sent back to
    /// the model as a `Role::Tool` message.
    fn call(
        &self,
        arguments: Value,
    ) -> impl std::future::Future<Output = Result<String, CatenaError>> + Send;
}

// ---------------------------------------------------------------------------
// DynTool — object-safe adapter
// ---------------------------------------------------------------------------

/// Object-safe version of [`Tool`] for heterogeneous tool registries.
///
/// A blanket impl covers anything implementing `Tool`, so you never write
/// this by hand.
pub trait DynTool: Send + Sync {
    /// Describe this tool for the model.
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool, returning a boxed future.
    fn call_dyn(&self, arguments: Value) -> BoxFut<'_, Result<String, CatenaError>>;
}

/// Blanket impl: anything implementing [`Tool`] automatically implements
/// [`DynTool`].
impl<T: Tool> DynTool for T {
    fn definition(&self) -> ToolDefinition {
        Tool::definition(self)
    }

    fn call_dyn(&self, arguments: Value) -> BoxFut<'_, Result<String, CatenaError>> {
        Box::pin(Tool::call(self, arguments))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct EchoTool;

    impl Tool for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "echo".to_owned(),
                description: Some("Echoes the input back.".to_owned()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "text": { "type": "string" }
                    },
                    "required": ["text"]
                }),
                strict: None,
            }
        }

        async fn call(&self, arguments: Value) -> Result<String, CatenaError> {
            let text = arguments["text"]
                .as_str()
                .ok_or_else(|| CatenaError::ToolError {
                    tool: "echo".to_owned(),
                    message: "missing 'text' argument".to_owned(),
                })?;
            Ok(text.to_owned())
        }
    }

    #[tokio::test]
    async fn tool_call_succeeds() {
        let tool = EchoTool;
        let result = tool.call(json!({"text": "hello"})).await.unwrap();
        assert_eq!(result, "hello");
    }

    #[tokio::test]
    async fn tool_call_missing_arg() {
        let tool = EchoTool;
        let result = tool.call(json!({})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn dyn_tool_dispatch() {
        let tool: &dyn DynTool = &EchoTool;
        assert_eq!(tool.definition().name, "echo");
        let result = tool.call_dyn(json!({"text": "yo"})).await.unwrap();
        assert_eq!(result, "yo");
    }
}
