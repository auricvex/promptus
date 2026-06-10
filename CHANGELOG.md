# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-18

### Added

- **promptus-core**: Domain model with `ChatRequest`, `ChatResponse`,
  `ChatProvider` trait, `StreamEvent`, `ToolCall`, `ToolDefinition`, and
  `ProviderError`.
- **promptus-openai**: HTTP provider for OpenAI-compatible endpoints with SSE
  streaming, tool calling, and vendor extension support (e.g.
  `reasoning_content`).
- **promptus-catena**: Composable orchestration layer with `Runnable` chains,
  `PromptTemplate` (Jinja2-style via minijinja), `StrOutputParser`,
  `JsonOutputParser`, `BufferMemory`, `WindowMemory`, `Tool` trait, and
  `ReActAgent` with streaming events.
- **promptus** (facade): `PromptusClient` with builder pattern,
  `ChatRequestBuilder` for ergonomic request construction, and re-exports of
  all public types.
- Examples for basic chat, streaming, tool calling, composable chains, agents,
  and web search.
- Unit tests across all crates.
- Dual MIT / Apache 2.0 license.
