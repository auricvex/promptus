//! Provider-agnostic error types for Promptus.

/// Errors returned by a [`ChatProvider`](crate::ChatProvider) implementation.
///
/// Variants cover the failure modes common across LLM providers: HTTP transport
/// errors, response deserialization failures, invalid requests, and
/// provider-specific issues that don't fit the other categories.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// The HTTP request failed or the server returned a non-success status.
    ///
    /// Carries the HTTP status code (if available), the provider's error
    /// message, and optionally the raw response body for debugging.
    #[error("HTTP error (status {status}): {message}")]
    Http {
        status: u16,
        message: String,
        body: Option<String>,
    },

    /// The provider's response could not be deserialized into the expected
    /// type.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// The request was invalid — missing required fields, conflicting options,
    /// or rejected by the provider's validation before a model was invoked.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// A network-level failure (DNS, TLS, connection refused, timeout, etc.).
    #[error("network error: {0}")]
    Network(String),

    /// Catch-all for errors that don't fit the other variants.
    #[error("{0}")]
    Other(String),
}

impl From<serde_json::Error> for ProviderError {
    fn from(err: serde_json::Error) -> Self {
        ProviderError::Serialization(err.to_string())
    }
}
