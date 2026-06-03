//! Streaming chat completion example.
//!
//! Demonstrates receiving a streamed response and printing content deltas
//! as they arrive.
//!
//! Run with:
//!   GROQ_API_KEY=your_key cargo run -p promptus --example streaming

use futures::StreamExt;
use promptus::{OpenAiCompatibleProvider, PromptusClient, StreamEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("GROQ_API_KEY").expect("set GROQ_API_KEY to run this example");

    let client = PromptusClient::builder()
        .provider(
            "groq",
            OpenAiCompatibleProvider::new("https://api.groq.com/openai/v1", api_key),
        )
        .build();

    let mut stream = client
        .chat("groq")
        .model("llama-3.3-70b-versatile")
        .user("Write a haiku about programming.")
        .stream()
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ContentDelta(text) => {
                print!("{text}");
            }
            StreamEvent::ReasoningDelta(text) => {
                // Reasoning models (e.g. DeepSeek-R1) emit internal reasoning.
                eprint!("[reasoning] {text}");
            }
            StreamEvent::Finished {
                finish_reason,
                usage,
            } => {
                println!("\n\nFinished: {finish_reason:?}");
                if let Some(usage) = usage {
                    println!(
                        "Tokens — prompt: {}, completion: {}, total: {}",
                        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                    );
                }
            }
            StreamEvent::ToolCallDelta { .. } => {
                // Not expected in this example — no tools registered.
            }
        }
    }

    Ok(())
}
