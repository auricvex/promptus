//! Promptus — a provider-agnostic LLM client library for Rust.
//!
//! Promptus gives applications a single API for talking to LLMs from any
//! provider. Register one or more providers, then send chat requests by name.
//!
//! # Quick start
//!
//! ```ignore
//! use promptus::{PromptusClient, OpenAiCompatibleProvider};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = PromptusClient::builder()
//!         .provider(
//!             "groq",
//!             OpenAiCompatibleProvider::new(
//!                 "https://api.groq.com/openai/v1",
//!                 std::env::var("GROQ_API_KEY")?,
//!             ),
//!         )
//!         .build();
//!
//!     let response = client
//!         .chat("groq")
//!         .model("llama-3.3-70b-versatile")
//!         .system("You are a helpful assistant.")
//!         .user("What is the capital of France?")
//!         .send()
//!         .await?;
//!
//!     println!("{}", response.content.unwrap_or_default());
//!     Ok(())
//! }
//! ```
//!
//! # Streaming
//!
//! ```ignore
//! use futures::StreamExt;
//!
//! let mut stream = client
//!     .chat("groq")
//!     .model("llama-3.3-70b-versatile")
//!     .user("Tell me a story.")
//!     .stream()
//!     .await?;
//!
//! while let Some(event) = stream.next().await {
//!     match event? {
//!         promptus::StreamEvent::ContentDelta(text) => print!("{text}"),
//!         promptus::StreamEvent::Finished { .. } => println!(),
//!         _ => {}
//!     }
//! }
//! ```
//!
//! # Tool calling
//!
//! ```ignore
//! use promptus::{ToolDefinition, ToolChoice};
//! use serde_json::json;
//!
//! let weather_tool = ToolDefinition {
//!     name: "get_weather".to_owned(),
//!     description: Some("Get the current weather for a location.".to_owned()),
//!     parameters: json!({
//!         "type": "object",
//!         "properties": {
//!             "location": { "type": "string" }
//!         },
//!         "required": ["location"]
//!     }),
//!     strict: None,
//! };
//!
//! let response = client
//!     .chat("groq")
//!     .model("llama-3.3-70b-versatile")
//!     .user("What's the weather in NYC?")
//!     .tool(weather_tool)
//!     .send()
//!     .await?;
//!
//! if !response.tool_calls.is_empty() {
//!     // Process tool calls, then send tool results back...
//! }
//! ```

mod builder;
mod client;

pub use builder::ChatRequestBuilder;
pub use client::{PromptusClient, PromptusClientBuilder};

// Re-export the orchestration layer so consumers don't need to add
// `promptus-catena` as a direct dependency.
pub use promptus_catena as catena;
pub use promptus_catena::prelude::*;

// Re-export the provider implementation so consumers don't need to add
// `promptus-openai` as a direct dependency.
pub use promptus_openai::OpenAiCompatibleProvider;

// Re-export all core types so consumers only need `use promptus::*`.
pub use promptus_core::{
    BoxFut, ChatProvider, ContentPart, DynChatProvider, FileSource, FinishReason, ImageSource,
    Message, ProviderError, ProviderRegistry, ReasoningEffort, ResponseFormat, Role, StreamEvent,
    ToolCall, ToolCallAccumulator, ToolChoice, ToolDefinition, ToolSpec, Usage,
};

// Re-export the core crate itself for consumers who need to reference it
// explicitly (e.g. in trait bounds).
pub use promptus_core as core;
