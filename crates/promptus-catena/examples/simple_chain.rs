//! Simple chain example: prompt template → LLM → string output.
//!
//! Demonstrates composing a `PromptTemplate`, an LLM call, and
//! `StrOutputParser` into a single chain using `.then()`.
//!
//! Run with:
//!   GROQ_API_KEY=your_key cargo run -p promptus-catena --example simple_chain

use promptus_catena::prelude::*;
use promptus_core::{ChatProvider, ChatRequest, ChatResponse};
use promptus_openai::OpenAiCompatibleProvider;

/// Wraps any `ChatProvider` as a `Runnable<ChatResponse, ChatResponse>` so it
/// can slot into a chain between a prompt template and an output parser.
struct LlmCall<P: ChatProvider> {
    provider: P,
    model: String,
}

impl<P: ChatProvider> LlmCall<P> {
    fn new(provider: P, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

impl<P: ChatProvider> Runnable<Vec<promptus_core::Message>> for LlmCall<P> {
    type Output = ChatResponse;
    type Error = CatenaError;

    async fn invoke(
        &self,
        messages: Vec<promptus_core::Message>,
    ) -> Result<Self::Output, Self::Error> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            response_format: None,
            stream: false,
            extra: None,
        };
        Ok(self.provider.chat(request).await?)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("GROQ_API_KEY").expect("set GROQ_API_KEY to run this example");

    let provider = OpenAiCompatibleProvider::new("https://api.groq.com/openai/v1", api_key);
    let llm = LlmCall::new(provider, "llama-3.3-70b-versatile");

    // Compose: template → LLM → string parser
    let chain = PromptTemplate::system_and_user(
        "You are a concise assistant. Respond in one sentence.",
        "{{ question }}",
    )
    .then(llm)
    .then(StrOutputParser);

    let mut vars = TemplateVars::new();
    vars.insert(
        "question".to_owned(),
        serde_json::Value::String("What is Rust?".to_owned()),
    );

    let answer = chain.invoke(vars).await?;
    println!("Answer: {answer}");

    Ok(())
}
