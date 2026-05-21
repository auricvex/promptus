//! Provider-agnostic types, traits, and errors for the Promptus LLM client library.
//!
//! This crate defines the domain model that all provider implementations share.
//! It has no knowledge of any specific provider's wire format, HTTP client, or
//! serialization scheme — those live in provider crates like `promptus-openai`.
//!
//! # Key types
//!
//! - [`ChatRequest`] / [`ChatResponse`] — the input and output of a chat
//!   completion call.
//! - [`Message`] / [`ContentPart`] — conversation messages and their content.
//! - [`ToolDefinition`] / [`ToolCall`] — tool (function) calling support.
//! - [`StreamEvent`] — events produced during a streamed response.
//! - [`ChatProvider`] — the trait provider crates implement.
//! - [`ProviderError`] — the unified error type.

mod error;
mod traits;
mod types;

pub use error::ProviderError;
pub use traits::{BoxFut, ChatProvider, DynChatProvider, ProviderRegistry, ToolCallAccumulator};
pub use types::{
    ChatRequest, ChatResponse, ContentPart, FileSource, FinishReason, ImageSource, Message,
    ReasoningEffort, ResponseFormat, Role, StreamEvent, ToolCall, ToolChoice, ToolDefinition,
    ToolSpec, Usage,
};
