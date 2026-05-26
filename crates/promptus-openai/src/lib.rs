//! OpenAI-compatible chat completions provider for Promptus.
//!
//! This crate implements [`promptus_core::ChatProvider`] for any endpoint
//! that speaks the OpenAI chat completions wire format — the de facto
//! standard used by OpenAI, Groq, DeepSeek, Together AI, Xiaomi MiMo,
//! Fireworks, Ollama, vLLM, and many others.
//!
//! # Quick start
//!
//! ```ignore
//! use promptus_openai::OpenAiCompatibleProvider;
//!
//! let provider = OpenAiCompatibleProvider::new(
//!     "https://api.groq.com/openai/v1",
//!     std::env::var("GROQ_API_KEY").unwrap(),
//! );
//! ```
//!
//! # Vendor extensions
//!
//! This crate handles provider-specific extensions gracefully:
//!
//! - **`reasoning_content`**: DeepSeek-R1 and other reasoning models expose
//!   their internal reasoning as a sibling field on the assistant message.
//!   This is mapped to [`promptus_core::StreamEvent::ReasoningDelta`] during
//!   streaming and [`promptus_core::ChatResponse::reasoning_content`] for
//!   non-streaming responses.
//!
//! - **Unknown/missing fields**: The wire-format structs use liberal
//!   `#[serde(default)]` so providers that omit or rename OpenAI-specific
//!   fields (like `service_tier`, `system_fingerprint`, or `store`) never
//!   cause a hard parse failure.

mod mapping;
mod provider;
pub(crate) mod wire;

pub use provider::OpenAiCompatibleProvider;
