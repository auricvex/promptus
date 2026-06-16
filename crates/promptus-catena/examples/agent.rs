//! Agent example: a ReAct-style tool-calling agent with streaming events.
//!
//! Demonstrates creating tools, registering them with a `ReActAgent`, and
//! running a multi-step tool-calling conversation with real-time progress
//! events (content deltas, tool calls, results).
//!
//! Run with:
//!   GROQ_API_KEY=your_key cargo run -p promptus-catena --example agent

use std::sync::Arc;

use promptus_catena::prelude::*;
use promptus_openai::OpenAiCompatibleProvider;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// A tool that adds two numbers.
struct AddTool;

impl Tool for AddTool {
    fn definition(&self) -> promptus_core::ToolDefinition {
        promptus_core::ToolDefinition {
            name: "add".to_owned(),
            description: Some("Add two numbers together.".to_owned()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "a": { "type": "number", "description": "First number" },
                    "b": { "type": "number", "description": "Second number" }
                },
                "required": ["a", "b"]
            }),
            strict: None,
        }
    }

    async fn call(&self, args: Value) -> Result<String, CatenaError> {
        let a = args["a"].as_f64().ok_or_else(|| CatenaError::ToolError {
            tool: "add".to_owned(),
            message: "missing 'a'".to_owned(),
        })?;
        let b = args["b"].as_f64().ok_or_else(|| CatenaError::ToolError {
            tool: "add".to_owned(),
            message: "missing 'b'".to_owned(),
        })?;
        Ok((a + b).to_string())
    }
}

/// A tool that multiplies two numbers.
struct MultiplyTool;

impl Tool for MultiplyTool {
    fn definition(&self) -> promptus_core::ToolDefinition {
        promptus_core::ToolDefinition {
            name: "multiply".to_owned(),
            description: Some("Multiply two numbers together.".to_owned()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "a": { "type": "number", "description": "First number" },
                    "b": { "type": "number", "description": "Second number" }
                },
                "required": ["a", "b"]
            }),
            strict: None,
        }
    }

    async fn call(&self, args: Value) -> Result<String, CatenaError> {
        let a = args["a"].as_f64().ok_or_else(|| CatenaError::ToolError {
            tool: "multiply".to_owned(),
            message: "missing 'a'".to_owned(),
        })?;
        let b = args["b"].as_f64().ok_or_else(|| CatenaError::ToolError {
            tool: "multiply".to_owned(),
            message: "missing 'b'".to_owned(),
        })?;
        Ok((a * b).to_string())
    }
}

// ---------------------------------------------------------------------------
// Event handler — prints real-time progress
// ---------------------------------------------------------------------------

fn print_event(event: &AgentEvent) {
    match event {
        AgentEvent::IterationStart { iteration } => {
            println!("\n--- Iteration {iteration} ---");
        }
        AgentEvent::ContentDelta { delta, .. } => {
            // Print content tokens as they arrive (no newline — streaming).
            print!("{delta}");
            // Flush to ensure immediate output.
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        AgentEvent::ReasoningDelta { delta, .. } => {
            // Some models emit reasoning — show it in grey/muted.
            print!("\x1b[90m{delta}\x1b[0m");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        AgentEvent::ToolCallsReady { calls, .. } => {
            println!(); // Newline after any content.
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
                println!(); // Newline after streamed content.
            }
                println!(); // Newline after streamed content.
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("GROQ_API_KEY").expect("set GROQ_API_KEY to run this example");

    let provider = OpenAiCompatibleProvider::new("https://api.groq.com/openai/v1", api_key);

    let agent = ReActAgent::builder(provider, "meta-llama/llama-4-scout-17b-16e-instruct")
        .system_prompt("You are a helpful math assistant. Use tools to compute answers.")
        .tool(AddTool)
        .tool(MultiplyTool)
        .max_iterations(10)
        .on_event(Arc::new(print_event))
        .build();

    // First question — should trigger tool calls.
    println!("User: What is (3 + 5) * 12?");
    let outcome = agent.invoke("What is (3 + 5) * 12?".to_owned()).await?;
    println!("\nFinal answer: {}", outcome.output);

    // Second question — the agent has memory, so it remembers the context.
    println!("\nUser: Now add 100 to that result.");
    let outcome2 = agent
        .invoke("Now add 100 to that result.".to_owned())
        .await?;
    println!("\nFinal answer: {}", outcome2.output);

    Ok(())
}
