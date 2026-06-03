//! Basic non-streaming chat completion example.
//!
//! Demonstrates registering a provider and sending a simple user message.
//!
//! Run with:
//!   GROQ_API_KEY=your_key cargo run -p promptus --example basic_chat

use promptus::{OpenAiCompatibleProvider, PromptusClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("GROQ_API_KEY").expect("set GROQ_API_KEY to run this example");

    let client = PromptusClient::builder()
        .provider(
            "groq",
            OpenAiCompatibleProvider::new("https://api.groq.com/openai/v1", api_key),
        )
        .build();

    let response = client
        .chat("groq")
        .model("llama-3.3-70b-versatile")
        .system("You are a helpful assistant. Be concise.")
        .user("What is the capital of France?")
        .temperature(0.7)
        .send()
        .await?;

    println!("Response: {}", response.content.unwrap_or_default());

    if let Some(usage) = response.usage {
        println!(
            "Tokens — prompt: {}, completion: {}, total: {}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }

    Ok(())
}
