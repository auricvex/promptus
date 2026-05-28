//! Prompt templates — render structured prompts into messages.
//!
//! Uses [minijinja](https://docs.rs/minijinja) for Jinja2-like syntax that
//! LangChain users will already recognize. Templates produce
//! `Vec<promptus_core::Message>` so they compose naturally with the rest of
//! the chain via [`Runnable`].

use std::collections::HashMap;

use promptus_core::{ContentPart, Message, Role};
use serde_json::Value;

use crate::error::CatenaError;
use crate::runnable::Runnable;

/// Variables passed to a prompt template. A simple string-keyed map of
/// JSON values — matches how template variables arrive in practice.
pub type TemplateVars = HashMap<String, Value>;

/// A prompt template that renders Jinja2-style syntax into a list of
/// messages.
///
/// The template body uses `{{ variable }}` for interpolation. The simplest
/// usage is a single user message template:
///
/// ```ignore
/// let tmpl = PromptTemplate::user("Summarize the following:\n{{ text }}");
/// let messages = tmpl.invoke(vars).await?;
/// ```
///
/// For multi-message prompts (system + user, few-shot examples, etc.), use
/// [`PromptTemplate::from_messages`] with role-annotated segments.
pub struct PromptTemplate {
    segments: Vec<TemplateSegment>,
}

struct TemplateSegment {
    role: Role,
    template: String,
}

impl PromptTemplate {
    /// Create a template that produces a single user message.
    pub fn user(template: impl Into<String>) -> Self {
        Self {
            segments: vec![TemplateSegment {
                role: Role::User,
                template: template.into(),
            }],
        }
    }

    /// Create a template that produces a single system message.
    pub fn system(template: impl Into<String>) -> Self {
        Self {
            segments: vec![TemplateSegment {
                role: Role::System,
                template: template.into(),
            }],
        }
    }

    /// Create a template that produces a system message followed by a user
    /// message — the most common chat prompt shape.
    pub fn system_and_user(system: impl Into<String>, user: impl Into<String>) -> Self {
        Self {
            segments: vec![
                TemplateSegment {
                    role: Role::System,
                    template: system.into(),
                },
                TemplateSegment {
                    role: Role::User,
                    template: user.into(),
                },
            ],
        }
    }

    /// Create a template from explicit role-annotated segments.
    ///
    /// Each tuple is `(Role, template_string)`. Useful for few-shot prompts
    /// or other multi-message layouts.
    pub fn from_segments(segments: Vec<(Role, impl Into<String>)>) -> Self {
        Self {
            segments: segments
                .into_iter()
                .map(|(role, tmpl)| TemplateSegment {
                    role,
                    template: tmpl.into(),
                })
                .collect(),
        }
    }

    /// Render the template with the given variables, producing messages.
    fn render(&self, vars: &TemplateVars) -> Result<Vec<Message>, CatenaError> {
        let mut messages = Vec::with_capacity(self.segments.len());
        for seg in &self.segments {
            // minijinja::Environment is cheap to create for one-off renders.
            // If this becomes a hot path, we could cache compiled templates.
            // Strict mode ensures missing variables produce an error rather
            // than silently rendering as empty strings.
            let mut env = minijinja::Environment::new();
            env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
            env.add_template("t", &seg.template)
                .map_err(|e| CatenaError::Template(e.to_string()))?;
            let tmpl = env
                .get_template("t")
                .map_err(|e| CatenaError::Template(e.to_string()))?;
            let rendered = tmpl
                .render(vars)
                .map_err(|e| CatenaError::Template(e.to_string()))?;

            messages.push(Message {
                role: seg.role,
                content: vec![ContentPart::Text(rendered)],
                tool_call_id: None,
                tool_calls: None,
                name: None,
            });
        }
        Ok(messages)
    }
}

impl Runnable<TemplateVars> for PromptTemplate {
    type Output = Vec<Message>;
    type Error = CatenaError;

    async fn invoke(&self, input: TemplateVars) -> Result<Self::Output, Self::Error> {
        self.render(&input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn single_user_template() {
        let tmpl = PromptTemplate::user("Hello, {{ name }}!");
        let mut vars = TemplateVars::new();
        vars.insert("name".to_owned(), Value::String("World".to_owned()));

        let messages = tmpl.invoke(vars).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[0].text().as_deref(), Some("Hello, World!"));
    }

    #[tokio::test]
    async fn system_and_user_template() {
        let tmpl = PromptTemplate::system_and_user(
            "You are a helpful assistant.",
            "Summarize: {{ text }}",
        );
        let mut vars = TemplateVars::new();
        vars.insert(
            "text".to_owned(),
            Value::String("Rust is great.".to_owned()),
        );

        let messages = tmpl.invoke(vars).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(
            messages[1].text().as_deref(),
            Some("Summarize: Rust is great.")
        );
    }

    #[tokio::test]
    async fn multi_segment_template() {
        let tmpl = PromptTemplate::from_segments(vec![
            (Role::System, "You are {{ who }}."),
            (Role::User, "Question: {{ q }}"),
            (Role::Assistant, "Let me think..."),
            (Role::User, "Follow up: {{ q2 }}"),
        ]);

        let mut vars = TemplateVars::new();
        vars.insert("who".to_owned(), Value::String("a tutor".to_owned()));
        vars.insert("q".to_owned(), Value::String("What is 2+2?".to_owned()));
        vars.insert("q2".to_owned(), Value::String("And why?".to_owned()));

        let messages = tmpl.invoke(vars).await.unwrap();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].text().as_deref(), Some("You are a tutor."));
        assert_eq!(messages[2].role, Role::Assistant);
    }

    #[tokio::test]
    async fn missing_variable_produces_error() {
        let tmpl = PromptTemplate::user("Hello, {{ missing }}!");
        let vars = TemplateVars::new();
        let result = tmpl.invoke(vars).await;
        assert!(result.is_err());
    }
}
