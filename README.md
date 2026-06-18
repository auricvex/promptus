# Promptus

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![CI](https://github.com/auricvex/promptus/actions/workflows/ci.yml/badge.svg)](https://github.com/auricvex/promptus/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)

**A provider-agnostic LLM client library for Rust.**

Promptus gives your application a single API for talking to large language
models from any provider that speaks the OpenAI chat completions wire format —
OpenAI, Groq, DeepSeek, Together AI, Ollama, vLLM, and more.

## Features

- **Provider-agnostic** — swap providers by changing a URL and API key, not
  your code.
- **Streaming** — real-time token-by-token streaming with content, reasoning,
  and tool-call deltas.
- **Tool calling** — define tools as JSON Schema, handle round-trips
  automatically.
- **Multi-modal** — send images and files alongside text in a single request.
- **Reasoning models** — first-class support for models that emit internal
  reasoning (DeepSeek-R1, etc.).
- **Composable chains** — build pipelines from templates, LLM calls, and
  output parsers using `.then()`.
- **ReAct agents** — tool-calling agents with memory, streaming events, and
  configurable iteration limits.
- **Vendor extensions** — pass provider-specific fields via the `extra` field
  without forking the library.

## Quick Start

Add Promptus to your project:

```toml
[dependencies]
promptus = "0.1"
tokio = { version = "1", features = ["full"] }
```

### Basic chat

```rust,no_run
use promptus::{OpenAiCompatibleProvider, PromptusClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = PromptusClient::builder()
        .provider(
            "groq",
            OpenAiCompatibleProvider::new(
                "https://api.groq.com/openai/v1",
                std::env::var("GROQ_API_KEY")?,
            ),
        )
        .build();

    let response = client
        .chat("groq")
        .model("llama-3.3-70b-versatile")
        .system("You are a helpful assistant.")
        .user("What is the capital of France?")
        .send()
        .await?;

    println!("{}", response.content.unwrap_or_default());
    Ok(())
}
```

### Streaming

```rust,no_run
use futures::StreamExt;
use promptus::{OpenAiCompatibleProvider, PromptusClient, StreamEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = PromptusClient::builder()
        .provider(
            "groq",
            OpenAiCompatibleProvider::new(
                "https://api.groq.com/openai/v1",
                std::env::var("GROQ_API_KEY")?,
            ),
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
            StreamEvent::ContentDelta(text) => print!("{text}"),
            StreamEvent::Finished { .. } => println!(),
            _ => {}
        }
    }
    Ok(())
}
```

### Composable chains (Catena)

```rust,ignore
use promptus::catena::prelude::*;

let chain = PromptTemplate::system_and_user(
    "You are a concise assistant.",
    "{{ question }}",
)
.then(llm)         // your ChatProvider implementation
.then(StrOutputParser);

let answer = chain.invoke(template_vars).await?;
```

### ReAct agent with tools

```rust,ignore
use promptus::catena::prelude::*;

let agent = ReActAgent::builder(provider, "llama-3.3-70b-versatile")
    .system_prompt("You are a helpful assistant. Use tools when needed.")
    .tool(MyTool)
    .max_iterations(10)
    .on_event(Arc::new(my_event_handler))
    .build();

let outcome = agent.invoke("What is (3 + 5) * 12?".to_owned()).await?;
println!("{}", outcome.output);
```

## Architecture

Promptus is a Cargo workspace with four crates:

```text
┌─────────────────────────────────────────────┐
│              promptus (facade)               │
│  PromptusClient · ChatRequestBuilder        │
├──────────┬──────────────┬───────────────────┤
│ promptus-│  promptus-   │   promptus-       │
│  openai  │   catena     │    core           │
│          │              │                   │
│ HTTP +   │ Chains,      │ Types, traits,    │
│ SSE +    │ templates,   │ errors            │
│ wire     │ memory,      │                   │
│ format   │ agents       │                   │
└──────────┴──────────────┴───────────────────┘
```

| Crate | Description |
|---|---|
| **`promptus`** | Public facade — re-exports everything. This is the crate users depend on. |
| **`promptus-core`** | Domain model: `ChatRequest`, `ChatResponse`, `ChatProvider` trait, error types. Zero provider-specific knowledge. |
| **`promptus-openai`** | HTTP provider for any OpenAI-compatible endpoint. Handles SSE streaming and wire format mapping. |
| **`promptus-catena`** | Orchestration layer: composable `Runnable` chains, `PromptTemplate`, output parsers, `Memory`, and `ReActAgent`. |

## Supported Providers

Any service that implements the OpenAI chat completions API works out of the
box:

- [OpenAI](https://platform.openai.com/)
- [Groq](https://groq.com/)
- [DeepSeek](https://platform.deepseek.com/)
- [Together AI](https://api.together.xyz/)
- [Ollama](https://ollama.com/) (local)
- [vLLM](https://github.com/vllm-project/vllm) (local)
- [OpenRouter](https://openrouter.ai/)
- Any other OpenAI-compatible endpoint

## Examples

Each crate includes runnable examples:

| Example | Crate | Description |
|---|---|---|
| [`basic_chat`](crates/promptus/examples/basic_chat.rs) | promptus | Non-streaming chat with system/user messages |
| [`streaming`](crates/promptus/examples/streaming.rs) | promptus | Streaming response with content/reasoning deltas |
| [`tool_calling`](crates/promptus/examples/tool_calling.rs) | promptus | Full tool-call round-trip |
| [`simple_chain`](crates/promptus-catena/examples/simple_chain.rs) | promptus-catena | Template → LLM → parser chain |
| [`agent`](crates/promptus-catena/examples/agent.rs) | promptus-catena | ReAct agent with tools and streaming events |
| [`web_search`](crates/promptus-catena/examples/web_search.rs) | promptus-catena | Server-side web search via vendor extensions |

Run an example:

```bash
GROQ_API_KEY=your_key cargo run -p promptus --example basic_chat
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines.

## License

This project is dual-licensed under the [MIT](LICENSE-MIT) and
[Apache 2.0](LICENSE-APACHE-2.0) licenses. You may choose either one.
