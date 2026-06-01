//! Web search example: an agent using Groq's compound model with server-side
//! web search enabled.
//!
//! Demonstrates passing provider-specific fields (like Groq's
//! `compound_custom`) via the `extra` builder method. The compound model
//! handles web search, code interpretation, and website visits server-side —
//! no client-side tool execution needed.
//!
//! Run with:
//!   GROQ_API_KEY=your_key cargo run -p promptus-catena --example web_search

use std::sync::Arc;

use promptus_catena::prelude::*;
use promptus_openai::OpenAiCompatibleProvider;
use serde_json::json;

// ---------------------------------------------------------------------------
// Event handler — prints real-time progress
// ---------------------------------------------------------------------------

fn print_event(event: &AgentEvent) {
    match event {
        AgentEvent::IterationStart { iteration } => {
            println!("\n--- Iteration {iteration} ---");
        }
        AgentEvent::ContentDelta { delta, .. } => {
            print!("{delta}");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        AgentEvent::ReasoningDelta { delta, .. } => {
            print!("\x1b[90m{delta}\x1b[0m");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        AgentEvent::ToolCallsReady { calls, .. } => {
            println!();
            for tc in calls {
                println!("  🔧 Calling {}({})", tc.name, tc.arguments);
            }
        }
        AgentEvent::ToolResult { name, result, .. } => match result {
            Ok(output) => println!("  ✅ {name} → {output}"),
            Err(e) => println!("  ❌ {name} → ERROR: {e}"),
        },
        AgentEvent::FinalAnswer { answer, .. } => {
            if !answer.is_empty() {
                println!();
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("GROQ_API_KEY").expect("set GROQ_API_KEY to run this example");

    let provider = OpenAiCompatibleProvider::new("https://api.groq.com/openai/v1", api_key);

    // Groq's compound model uses `compound_custom` to enable server-side
    // tools. We pass it via `extra` — the keys are flattened into the
    // top-level request body.
    let compound_custom = json!({
        "compound_custom": {
            "tools": {
                "enabled_tools": [
                    "web_search",
                    "code_interpreter",
                    "visit_website"
                ]
            }
        }
    });

    let agent = ReActAgent::builder(provider, "groq/compound")
        .system_prompt(
            "You are a helpful research assistant. Use web search to find \
             current information when needed.",
        )
        .extra(compound_custom)
        .max_iterations(10)
        .on_event(Arc::new(print_event))
        .build();

    println!("User: What are the latest developments in Rust 2024 edition?");
    let outcome = agent
        .invoke("What are the latest developments in Rust 2024 edition?".to_owned())
        .await?;
    println!("\nFinal answer: {}", outcome.output);

    Ok(())
}
