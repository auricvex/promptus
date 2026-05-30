//! # Promptus Catena
//!
//! **Catena** (Latin: "chain") is Promptus's orchestration layer — composable
//! chains, conversation memory, and tool-calling agents.
//!
//! Everything in Catena composes through the [`Runnable`] trait: prompt
//! templates, LLM calls, output parsers, memory wrappers, and agents are all
//! `Runnable` implementations that wire together with `.then()` or `|`.
//!
//! ## Quick example
//!
//! ```ignore
//! use promptus_catena::prelude::*;
//!
//! let chain = PromptTemplate::user("Summarize: {{ text }}")
//!     .then(llm_call)
//!     .then(StrOutputParser);
//!
//! let summary = chain.invoke(vars).await?;
//! ```
//!
//! ## Modules
//!
//! - [`runnable`]: The [`Runnable`](runnable::Runnable) trait and
//!   [`Sequence`](runnable::Sequence) composition.
//! - [`template`]: [`PromptTemplate`](template::PromptTemplate) for
//!   Jinja2-style prompt rendering.
//! - [`parsers`]: [`StrOutputParser`](parsers::StrOutputParser) and
//!   [`JsonOutputParser`](parsers::JsonOutputParser).
//! - [`memory`]: The [`Memory`](memory::Memory) trait, plus
//!   [`BufferMemory`](memory::BufferMemory) and
//!   [`WindowMemory`](memory::WindowMemory).
//! - [`tool`]: The [`Tool`](tool::Tool) trait for agent-callable tools.
//! - [`agent`]: [`ReActAgent`](agent::ReActAgent) — a tool-calling agent
//!   with a ReAct-style loop.
//! - [`error`]: [`CatenaError`](error::CatenaError) — all errors in one
//!   enum.

pub mod agent;
pub mod error;
pub mod memory;
pub mod parsers;
pub mod runnable;
pub mod template;
pub mod tool;

/// Convenience re-imports.
pub mod prelude {
    pub use crate::agent::{AgentEvent, AgentOutcome, AgentStep, EventHandler, ReActAgent};
    pub use crate::error::CatenaError;
    pub use crate::memory::{BufferMemory, Memory, WindowMemory};
    pub use crate::parsers::{JsonOutputParser, StrOutputParser};
    pub use crate::runnable::{Runnable, Sequence};
    pub use crate::template::{PromptTemplate, TemplateVars};
    pub use crate::tool::{DynTool, Tool};
}
