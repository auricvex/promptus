//! The Promptus client and its builder.

use std::collections::HashMap;

use promptus_core::{
    DynChatProvider, DynModelProvider, ModelInfo, ProviderError, ProviderRegistry,
};

use crate::builder::ChatRequestBuilder;

/// A multi-provider LLM client.
///
/// `PromptusClient` is the main entry point for the Promptus library. It
/// holds a registry of named providers and lets you send chat requests to
/// any of them by name.
///
/// # Example
///
/// ```ignore
/// use promptus::{PromptusClient, OpenAiCompatibleProvider};
///
/// let client = PromptusClient::builder()
///     .provider("groq", OpenAiCompatibleProvider::new(
///         "https://api.groq.com/openai/v1",
///         groq_key,
///     ))
///     .provider("openai", OpenAiCompatibleProvider::new(
///         "https://api.openai.com/v1",
///         openai_key,
///     ))
///     .build();
///
/// // Send a chat request to Groq
/// let response = client
///     .chat("groq")
///     .model("llama-3.3-70b-versatile")
///     .user("Hello!")
///     .send()
///     .await?;
/// ```
pub struct PromptusClient {
    registry: ProviderRegistry,
    model_providers: HashMap<String, Box<dyn DynModelProvider>>,
}

impl PromptusClient {
    /// Create a builder to configure and construct a `PromptusClient`.
    pub fn builder() -> PromptusClientBuilder {
        PromptusClientBuilder::new()
    }

    /// Start building a chat request targeting the named provider.
    ///
    /// Returns a [`ChatRequestBuilder`] that provides ergonomic methods for
    /// adding messages, tools, and parameters before sending.
    ///
    /// # Panics
    ///
    /// Panics at send time (not here) if no provider is registered under the
    /// given name.
    pub fn chat(&self, provider: &str) -> ChatRequestBuilder<'_> {
        ChatRequestBuilder::new(self, provider.to_owned())
    }

    /// Get a reference to the named provider.
    ///
    /// Returns `None` if no provider is registered under that name.
    pub fn provider(&self, name: &str) -> Option<&dyn DynChatProvider> {
        self.registry.get(name)
    }

    /// List the names of all registered providers.
    pub fn provider_names(&self) -> Vec<&str> {
        self.registry.names()
    }

    /// List the models available from the named provider.
    ///
    /// Returns `ProviderError::InvalidRequest` if no model provider is
    /// registered under the given name. Use
    /// [`PromptusClientBuilder::model_provider`] to register one.
    pub async fn list_models(&self, provider: &str) -> Result<Vec<ModelInfo>, ProviderError> {
        let model_provider = self.model_providers.get(provider).ok_or_else(|| {
            ProviderError::InvalidRequest(format!("no model provider registered as '{provider}'"))
        })?;
        model_provider.list_models().await
    }
}

/// Builder for constructing a [`PromptusClient`].
///
/// Register providers with [`provider`](Self::provider) then call
/// [`build`](Self::build) to create the client.
pub struct PromptusClientBuilder {
    registry: ProviderRegistry,
    model_providers: HashMap<String, Box<dyn DynModelProvider>>,
}

impl PromptusClientBuilder {
    fn new() -> Self {
        Self {
            registry: ProviderRegistry::new(),
            model_providers: HashMap::new(),
        }
    }

    /// Register a provider under the given name.
    ///
    /// The name is used later in [`PromptusClient::chat`] to select which
    /// provider handles the request. If a provider with the same name is
    /// already registered, it is replaced.
    pub fn provider(
        mut self,
        name: impl Into<String>,
        provider: impl DynChatProvider + 'static,
    ) -> Self {
        self.registry.register(name, Box::new(provider));
        self
    }

    /// Register a model-listing provider under the given name.
    ///
    /// This enables [`PromptusClient::list_models`] for the given provider
    /// name. A provider that implements both [`DynChatProvider`] and
    /// [`DynModelProvider`] (like [`OpenAiCompatibleProvider`](crate::OpenAiCompatibleProvider))
    /// must be registered separately for each capability.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use promptus::{PromptusClient, OpenAiCompatibleProvider};
    ///
    /// let provider = OpenAiCompatibleProvider::new(url, key);
    /// let client = PromptusClient::builder()
    ///     .provider("groq", OpenAiCompatibleProvider::new(url, key))
    ///     .model_provider("groq", OpenAiCompatibleProvider::new(url, model_key))
    ///     .build();
    ///
    /// let models = client.list_models("groq").await?;
    /// ```
    pub fn model_provider(
        mut self,
        name: impl Into<String>,
        provider: impl DynModelProvider + 'static,
    ) -> Self {
        self.model_providers.insert(name.into(), Box::new(provider));
        self
    }

    /// Build the `PromptusClient`.
    pub fn build(self) -> PromptusClient {
        PromptusClient {
            registry: self.registry,
            model_providers: self.model_providers,
        }
    }
}
