//! Error types for the Catena orchestration layer.
//!
//! [`CatenaError`] unifies all failure modes across Catena runnables,
//! agents, and tools into a single enum.

/// All errors produced by Catena runnables, agents, and tools.
#[derive(Debug, thiserror::Error)]
pub enum CatenaError {
    /// The underlying LLM provider returned an error.
    #[error("provider error: {0}")]
    Provider(#[from] promptus_core::ProviderError),

    /// A tool invocation failed.
    #[error("tool error in '{tool}': {message}")]
    ToolError { tool: String, message: String },

    /// A prompt template could not be rendered.
    #[error("template error: {0}")]
    Template(String),

    /// The model's output could not be parsed into the expected type.
    #[error("parse error: {0}")]
    Parse(String),

    /// The agent hit its iteration cap without producing a final answer.
    #[error("agent exceeded max iterations ({max})")]
    MaxIterationsReached { max: usize },

    /// A required field was missing from the model's response.
    #[error("missing field: {0}")]
    MissingField(String),

    /// The agent could not find a tool matching the model's request.
    #[error("unknown tool: '{0}'")]
    UnknownTool(String),

    /// A catch-all for errors that don't fit the other variants.
    #[error("{0}")]
    Other(String),
}
