//! Conversions between `promptus_core` domain types and wire-format structs.

use promptus_core::{
    ChatRequest, ChatResponse, ContentPart, FileSource, FinishReason, ImageSource, Message,
    ModelInfo, ReasoningEffort, ResponseFormat, Role, StreamEvent, ToolCall, ToolChoice,
    ToolDefinition, ToolSpec, Usage,
};

use crate::wire::request::{
    ContentValue, CreateChatCompletionRequest, WireContentPart, WireFileSource, WireFunctionCall,
    WireImageUrl, WireMessage, WireNamedToolChoice, WireNamedToolChoiceFunction,
    WireReasoningEffort, WireResponseFormat, WireStreamOptions, WireToolCall, WireToolChoice,
};
use crate::wire::response::{CompletionUsage, CreateChatCompletionStreamResponse};

// ---------------------------------------------------------------------------
// Core -> Wire (request mapping)
// ---------------------------------------------------------------------------

/// Convert a `promptus_core::ChatRequest` into the wire-format request body.
pub(crate) fn request_to_wire(req: &ChatRequest) -> CreateChatCompletionRequest {
    CreateChatCompletionRequest {
        model: req.model.clone(),
        messages: req.messages.iter().map(message_to_wire).collect(),
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        stop: req.stop.clone(),
        tools: req
            .tools
            .as_ref()
            .map(|t| t.iter().map(tool_spec_to_wire).collect()),
        tool_choice: req.tool_choice.as_ref().map(tool_choice_to_wire),
        reasoning_effort: req.reasoning_effort.map(reasoning_effort_to_wire),
        response_format: req.response_format.as_ref().map(response_format_to_wire),
        stream: req.stream,
        stream_options: if req.stream {
            Some(WireStreamOptions {
                include_usage: true,
            })
        } else {
            None
        },
        extra: req.extra.clone().unwrap_or(serde_json::Value::Null),
    }
}

fn role_to_wire(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn message_to_wire(msg: &Message) -> WireMessage {
    let content = if msg.content.is_empty() {
        None
    } else if msg.content.len() == 1 {
        // Single text part — serialize as a plain string for cleaner JSON.
        match &msg.content[0] {
            ContentPart::Text(t) => Some(ContentValue::Text(t.clone())),
            other => Some(ContentValue::Parts(vec![content_part_to_wire(other)])),
        }
    } else {
        Some(ContentValue::Parts(
            msg.content.iter().map(content_part_to_wire).collect(),
        ))
    };

    WireMessage {
        role: role_to_wire(msg.role),
        content,
        name: msg.name.clone(),
        tool_call_id: msg.tool_call_id.clone(),
        tool_calls: msg.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|tc| WireToolCall {
                    id: tc.id.clone(),
                    call_type: "function",
                    function: WireFunctionCall {
                        name: tc.name.clone(),
                        // Serialize Value back to a string — the wire format
                        // expects arguments as a JSON-encoded string, not an
                        // inline object.
                        arguments: serde_json::to_string(&tc.arguments)
                            .unwrap_or_else(|_| "{}".to_owned()),
                    },
                })
                .collect()
        }),
    }
}

fn content_part_to_wire(part: &ContentPart) -> WireContentPart {
    match part {
        ContentPart::Text(t) => WireContentPart::Text { text: t.clone() },
        ContentPart::Image(src) => WireContentPart::ImageUrl {
            image_url: match src {
                ImageSource::Url(url) => WireImageUrl {
                    url: url.clone(),
                    detail: None,
                },
                ImageSource::Base64 { media_type, data } => WireImageUrl {
                    url: format!("data:{media_type};base64,{data}"),
                    detail: None,
                },
            },
        },
        ContentPart::File(src) => WireContentPart::File {
            file: match src {
                FileSource::FileId(id) => WireFileSource {
                    filename: None,
                    file_data: None,
                    file_id: Some(id.clone()),
                },
                FileSource::Base64 { filename, data } => WireFileSource {
                    file_data: Some(format!("data:application/octet-stream;base64,{data}")),
                    filename: Some(filename.clone()),
                    file_id: None,
                },
            },
        },
    }
}

fn tool_spec_to_wire(tool: &ToolSpec) -> serde_json::Value {
    match tool {
        ToolSpec::Function(def) => function_tool_to_wire(def),
        ToolSpec::Raw(value) => value.clone(),
    }
}

fn function_tool_to_wire(tool: &ToolDefinition) -> serde_json::Value {
    let mut function = serde_json::json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    });
    // Only include `strict` when set — some providers (e.g. Groq) reject null.
    if let Some(strict) = tool.strict {
        function["strict"] = serde_json::Value::Bool(strict);
    }
    serde_json::json!({
        "type": "function",
        "function": function,
    })
}

fn tool_choice_to_wire(choice: &ToolChoice) -> WireToolChoice {
    match choice {
        ToolChoice::Auto => WireToolChoice::Auto,
        ToolChoice::None => WireToolChoice::None,
        ToolChoice::Required => WireToolChoice::Required,
        ToolChoice::Named(name) => WireToolChoice::Named(WireNamedToolChoice {
            choice_type: "function",
            function: WireNamedToolChoiceFunction { name: name.clone() },
        }),
    }
}

fn reasoning_effort_to_wire(effort: ReasoningEffort) -> WireReasoningEffort {
    WireReasoningEffort(match effort {
        ReasoningEffort::None => "none",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::XHigh => "xhigh",
    })
}

fn response_format_to_wire(fmt: &ResponseFormat) -> WireResponseFormat {
    match fmt {
        ResponseFormat::Text => WireResponseFormat::Text,
        ResponseFormat::JsonObject => WireResponseFormat::JsonObject,
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => WireResponseFormat::JsonSchema {
            json_schema: crate::wire::request::WireJsonSchema {
                name: name.clone(),
                description: None,
                schema: Some(schema.clone()),
                strict: Some(*strict),
            },
        },
    }
}

// ---------------------------------------------------------------------------
// Wire -> Core (response mapping)
// ---------------------------------------------------------------------------

/// Convert the first choice of a wire-format response into a `ChatResponse`.
pub(crate) fn response_from_wire(
    resp: &crate::wire::response::CreateChatCompletionResponse,
) -> Result<ChatResponse, String> {
    let choice = resp
        .choices
        .first()
        .ok_or("response contained no choices")?;

    let content = choice.message.content.clone();
    let reasoning_content = choice.message.reasoning_content.clone();

    let tool_calls: Vec<ToolCall> = choice
        .message
        .tool_calls
        .as_ref()
        .map(|calls| {
            calls
                .iter()
                .map(|tc| {
                    let arguments = serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(serde_json::Value::String(tc.function.arguments.clone()));
                    ToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments,
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let finish_reason = choice
        .finish_reason
        .as_deref()
        .map(FinishReason::from_wire)
        .unwrap_or(FinishReason::Other("unknown".to_owned()));

    let usage = resp.usage.as_ref().map(usage_from_wire);

    Ok(ChatResponse {
        content,
        reasoning_content,
        tool_calls,
        finish_reason,
        usage,
        model: resp.model.clone(),
    })
}

// ---------------------------------------------------------------------------
// Wire -> Core (stream event mapping)
// ---------------------------------------------------------------------------

/// Convert a wire-format stream chunk into zero or more `StreamEvent`s.
///
/// A single chunk may yield multiple events (e.g. content delta + tool-call
/// deltas). Returns an empty vec for chunks with no meaningful content (e.g.
/// the final usage-only chunk).
pub(crate) fn stream_events_from_wire(
    chunk: &CreateChatCompletionStreamResponse,
) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    for choice in &chunk.choices {
        // Content delta
        if let Some(content) = &choice.delta.content
            && !content.is_empty()
        {
            events.push(StreamEvent::ContentDelta(content.clone()));
        }

        // Reasoning delta (vendor extension)
        if let Some(reasoning) = &choice.delta.reasoning_content
            && !reasoning.is_empty()
        {
            events.push(StreamEvent::ReasoningDelta(reasoning.clone()));
        }

        // Tool call deltas
        if let Some(tool_calls) = &choice.delta.tool_calls {
            for tc in tool_calls {
                events.push(StreamEvent::ToolCallDelta {
                    index: tc.index,
                    id: tc.id.clone(),
                    name: tc.function.as_ref().and_then(|f| f.name.clone()),
                    arguments_delta: tc.function.as_ref().and_then(|f| f.arguments.clone()),
                });
            }
        }

        // Finished event — only on the last chunk for this choice
        if let Some(reason) = &choice.finish_reason {
            let usage = chunk.usage.as_ref().map(usage_from_wire);
            events.push(StreamEvent::Finished {
                finish_reason: FinishReason::from_wire(reason),
                usage,
            });
        }
    }

    events
}

fn usage_from_wire(usage: &CompletionUsage) -> Usage {
    Usage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
        reasoning_tokens: usage
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens),
        cached_prompt_tokens: usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens),
    }
}

// ---------------------------------------------------------------------------
// Wire -> Core (model listing mapping)
// ---------------------------------------------------------------------------

/// Convert a wire-format model list response into `Vec<ModelInfo>`.
pub(crate) fn models_from_wire(resp: &crate::wire::response::ListModelsResponse) -> Vec<ModelInfo> {
    resp.data
        .iter()
        .map(|m| ModelInfo {
            id: m.id.clone(),
            owned_by: m.owned_by.clone(),
            created: m.created,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use promptus_core::{ChatRequest, ContentPart, ImageSource, Role};
    use serde_json::json;

    fn basic_request() -> ChatRequest {
        ChatRequest {
            model: "gpt-4o".to_owned(),
            messages: vec![Message::user("hello")],
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
        }
    }

    #[test]
    fn roundtrip_simple_request() {
        let req = basic_request();
        let wire = request_to_wire(&req);
        let json = serde_json::to_value(&wire).unwrap();

        assert_eq!(json["model"], "gpt-4o");
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"], "hello");
    }

    #[test]
    fn request_with_system_and_developer_messages() {
        let req = ChatRequest {
            model: "gpt-4o".to_owned(),
            messages: vec![
                Message::developer("You are a helpful assistant."),
                Message::system("Be concise."),
                Message::user("Hi"),
            ],
            ..basic_request()
        };
        let wire = request_to_wire(&req);
        let json = serde_json::to_value(&wire).unwrap();

        assert_eq!(json["messages"][0]["role"], "developer");
        assert_eq!(json["messages"][1]["role"], "system");
        assert_eq!(json["messages"][2]["role"], "user");
    }

    #[test]
    fn request_with_image_content() {
        let req = ChatRequest {
            messages: vec![Message {
                role: Role::User,
                content: vec![
                    ContentPart::Text("What's in this image?".to_owned()),
                    ContentPart::Image(ImageSource::Url("https://example.com/cat.jpg".to_owned())),
                ],
                tool_call_id: None,
                tool_calls: None,
                name: None,
            }],
            ..basic_request()
        };
        let wire = request_to_wire(&req);
        let json = serde_json::to_value(&wire).unwrap();

        // Multi-part content should be an array
        assert!(json["messages"][0]["content"].is_array());
        assert_eq!(json["messages"][0]["content"][0]["type"], "text");
        assert_eq!(json["messages"][0]["content"][1]["type"], "image_url");
    }

    #[test]
    fn request_with_base64_image() {
        let req = ChatRequest {
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentPart::Image(ImageSource::Base64 {
                    media_type: "image/png".to_owned(),
                    data: "iVBORw0KGgo=".to_owned(),
                })],
                tool_call_id: None,
                tool_calls: None,
                name: None,
            }],
            ..basic_request()
        };
        let wire = request_to_wire(&req);
        let json = serde_json::to_value(&wire).unwrap();

        let url = json["messages"][0]["content"][0]["image_url"]["url"]
            .as_str()
            .unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn request_with_tool_definitions() {
        let req = ChatRequest {
            tools: Some(vec![ToolSpec::Function(ToolDefinition {
                name: "get_weather".to_owned(),
                description: Some("Get the weather for a location".to_owned()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }),
                strict: Some(true),
            })]),
            tool_choice: Some(ToolChoice::Auto),
            ..basic_request()
        };
        let wire = request_to_wire(&req);
        let json = serde_json::to_value(&wire).unwrap();

        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(json["tools"][0]["function"]["strict"], true);
        assert_eq!(json["tool_choice"], "auto");
    }

    #[test]
    fn request_with_tool_strict_none_omitted() {
        let req = ChatRequest {
            tools: Some(vec![ToolSpec::Function(ToolDefinition {
                name: "add".to_owned(),
                description: Some("Add numbers".to_owned()),
                parameters: json!({"type": "object"}),
                strict: None,
            })]),
            ..basic_request()
        };
        let wire = request_to_wire(&req);
        let json = serde_json::to_value(&wire).unwrap();

        // `strict` should be absent, not null — some providers reject null.
        assert!(json["tools"][0]["function"].get("strict").is_none());
    }

    #[test]
    fn request_with_named_tool_choice() {
        let req = ChatRequest {
            tool_choice: Some(ToolChoice::Named("get_weather".to_owned())),
            ..basic_request()
        };
        let wire = request_to_wire(&req);
        let json = serde_json::to_value(&wire).unwrap();

        assert_eq!(json["tool_choice"]["type"], "function");
        assert_eq!(json["tool_choice"]["function"]["name"], "get_weather");
    }

    #[test]
    fn request_with_reasoning_effort() {
        let req = ChatRequest {
            reasoning_effort: Some(ReasoningEffort::High),
            ..basic_request()
        };
        let wire = request_to_wire(&req);
        let json = serde_json::to_value(&wire).unwrap();

        assert_eq!(json["reasoning_effort"], "high");
    }

    #[test]
    fn request_with_response_format_json_schema() {
        let req = ChatRequest {
            response_format: Some(ResponseFormat::JsonSchema {
                name: "my_schema".to_owned(),
                schema: json!({"type": "object", "properties": {"x": {"type": "integer"}}}),
                strict: true,
            }),
            ..basic_request()
        };
        let wire = request_to_wire(&req);
        let json = serde_json::to_value(&wire).unwrap();

        assert_eq!(json["response_format"]["type"], "json_schema");
        assert_eq!(json["response_format"]["json_schema"]["name"], "my_schema");
    }

    #[test]
    fn request_with_stream_options() {
        let req = ChatRequest {
            stream: true,
            ..basic_request()
        };
        let wire = request_to_wire(&req);
        let json = serde_json::to_value(&wire).unwrap();

        assert_eq!(json["stream"], true);
        assert_eq!(json["stream_options"]["include_usage"], true);
    }

    #[test]
    fn request_with_extra_fields() {
        let req = ChatRequest {
            extra: Some(json!({
                "compound_custom": {
                    "tools": {
                        "enabled_tools": ["web_search"]
                    }
                }
            })),
            ..basic_request()
        };
        let wire = request_to_wire(&req);
        let json = serde_json::to_value(&wire).unwrap();

        // Extra fields should be flattened into the top-level request.
        assert_eq!(
            json["compound_custom"]["tools"]["enabled_tools"][0],
            "web_search"
        );
    }

    #[test]
    fn request_with_extra_none_omitted() {
        let req = basic_request();
        let wire = request_to_wire(&req);
        let json = serde_json::to_value(&wire).unwrap();

        // No extra fields — compound_custom should not appear.
        assert!(json.get("compound_custom").is_none());
    }

    #[test]
    fn request_with_tool_response_message() {
        let req = ChatRequest {
            messages: vec![
                Message::user("What's the weather?"),
                Message {
                    role: Role::Assistant,
                    content: vec![],
                    tool_call_id: None,
                    tool_calls: Some(vec![ToolCall {
                        id: "call_123".to_owned(),
                        name: "get_weather".to_owned(),
                        arguments: json!({"location": "NYC"}),
                    }]),
                    name: None,
                },
                Message::tool("call_123", "72°F and sunny"),
            ],
            ..basic_request()
        };
        let wire = request_to_wire(&req);
        let json = serde_json::to_value(&wire).unwrap();

        // Assistant message should have tool_calls
        let assistant = &json["messages"][1];
        assert_eq!(assistant["role"], "assistant");
        assert_eq!(assistant["tool_calls"][0]["id"], "call_123");
        assert_eq!(
            assistant["tool_calls"][0]["function"]["name"],
            "get_weather"
        );

        // Tool message should have tool_call_id
        let tool = &json["messages"][2];
        assert_eq!(tool["role"], "tool");
        assert_eq!(tool["tool_call_id"], "call_123");
    }

    #[test]
    fn parse_and_map_non_streaming_response() {
        let json = r#"{
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "tool_calls": null
                },
                "finish_reason": "stop"
            }],
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2}
            }
        }"#;

        let wire: crate::wire::response::CreateChatCompletionResponse =
            serde_json::from_str(json).unwrap();
        let resp = response_from_wire(&wire).unwrap();

        assert_eq!(resp.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.finish_reason, FinishReason::Stop);
        assert_eq!(resp.model, "gpt-4o");
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.reasoning_tokens, Some(2));
        assert_eq!(usage.cached_prompt_tokens, Some(3));
    }

    #[test]
    fn parse_and_map_streaming_events() {
        let chunks = vec![
            r#"{"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"content":" world"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#,
        ];

        let mut all_events = Vec::new();
        for chunk_json in chunks {
            let chunk: CreateChatCompletionStreamResponse =
                serde_json::from_str(chunk_json).unwrap();
            all_events.extend(stream_events_from_wire(&chunk));
        }

        // Should have: "Hello", " world", Finished
        let content: Vec<&str> = all_events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, vec!["Hello", " world"]);

        let finished = all_events.iter().find_map(|e| match e {
            StreamEvent::Finished {
                finish_reason,
                usage,
            } => Some((finish_reason.clone(), usage.clone())),
            _ => None,
        });
        assert!(finished.is_some());
        let (reason, usage) = finished.unwrap();
        assert_eq!(reason, FinishReason::Stop);
        assert_eq!(usage.unwrap().total_tokens, 7);
    }

    #[test]
    fn map_list_models_response() {
        let json = r#"{
            "object": "list",
            "data": [
                {"id": "gpt-4o", "object": "model", "created": 1700000000, "owned_by": "openai"},
                {"id": "local-model", "object": "model"}
            ]
        }"#;

        let wire: crate::wire::response::ListModelsResponse = serde_json::from_str(json).unwrap();
        let models = models_from_wire(&wire);

        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "gpt-4o");
        assert_eq!(models[0].owned_by.as_deref(), Some("openai"));
        assert_eq!(models[0].created, Some(1_700_000_000));
        assert_eq!(models[1].id, "local-model");
        assert!(models[1].owned_by.is_none());
        assert!(models[1].created.is_none());
    }

    #[test]
    fn parse_streaming_tool_call_events() {
        let chunks = vec![
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"NYC\"}"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        ];

        let mut all_events = Vec::new();
        for chunk_json in chunks {
            let chunk: CreateChatCompletionStreamResponse =
                serde_json::from_str(chunk_json).unwrap();
            all_events.extend(stream_events_from_wire(&chunk));
        }

        let deltas: Vec<_> = all_events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments_delta,
                } => Some((*index, id.clone(), name.clone(), arguments_delta.clone())),
                _ => None,
            })
            .collect();

        assert_eq!(deltas.len(), 3);
        // First delta: id and name present
        assert_eq!(deltas[0].0, 0);
        assert_eq!(deltas[0].1.as_deref(), Some("call_1"));
        assert_eq!(deltas[0].2.as_deref(), Some("get_weather"));
        // Subsequent deltas: only arguments
        assert!(deltas[1].1.is_none());
        assert!(deltas[1].2.is_none());
        assert_eq!(deltas[1].3.as_deref(), Some("{\"loc"));
    }
}
