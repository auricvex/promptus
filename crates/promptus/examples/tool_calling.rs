//! Tool calling round-trip example.
//!
//! Demonstrates registering tools, sending a request that triggers a tool
//! call, and responding with the tool's result.
//!
//! Run with:
//!   GROQ_API_KEY=your_key cargo run -p promptus --example tool_calling

use promptus::{OpenAiCompatibleProvider, PromptusClient, ToolDefinition};
use serde_json::json;

/// A simple mock weather "tool" — returns a hardcoded response.
fn get_weather(location: &str) -> String {
    format!("The weather in {location} is 72°F and sunny.")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("GROQ_API_KEY").expect("set GROQ_API_KEY to run this example");

    let client = PromptusClient::builder()
        .provider(
            "groq",
            OpenAiCompatibleProvider::new("https://api.groq.com/openai/v1", api_key),
        )
        .build();

    let weather_tool = ToolDefinition {
        name: "get_weather".to_owned(),
        description: Some("Get the current weather for a given location.".to_owned()),
        parameters: json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }),
        strict: None,
    };

    // First request: ask a question that should trigger the tool.
    let response = client
        .chat("groq")
        .model("llama-3.3-70b-versatile")
        .user("What's the weather like in New York?")
        .tool(weather_tool)
        .send()
        .await?;

    if response.tool_calls.is_empty() {
        println!("Model responded with text (no tool call):");
        println!("{}", response.content.unwrap_or_default());
        return Ok(());
    }

    println!("Model requested tool calls:");
    for tc in &response.tool_calls {
        println!("  {}({})", tc.name, tc.arguments);
    }

    // Build the tool results and send them back.
    let mut builder = client
        .chat("groq")
        .model("llama-3.3-70b-versatile")
        .user("What's the weather like in New York?");

    // Re-add the assistant message with tool calls.
    // (In a real app you'd reconstruct the full conversation history.)
    let tool = ToolDefinition {
        name: "get_weather".to_owned(),
        description: Some("Get the current weather for a given location.".to_owned()),
        parameters: json!({
            "type": "object",
            "properties": {
                "location": { "type": "string" }
            },
            "required": ["location"]
        }),
        strict: None,
    };
    builder = builder.tool(tool);

    for tc in &response.tool_calls {
        // Extract the location from the arguments.
        let location = tc
            .arguments
            .get("location")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");

        let result = get_weather(location);
        println!("  -> Tool result: {result}");
        builder = builder.tool_result(&tc.id, result);
    }

    let final_response = builder.send().await?;
    println!("\nFinal response:");
    println!("{}", final_response.content.unwrap_or_default());

    Ok(())
}
