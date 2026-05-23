//! Wire-format types for the OpenAI-compatible chat completions API.
//!
//! These structs are private to this crate and map 1:1 with the JSON bodies
//! sent to and received from the API. External consumers interact with the
//! provider-agnostic types from `promptus_core` instead.

pub(crate) mod request;
pub(crate) mod response;
