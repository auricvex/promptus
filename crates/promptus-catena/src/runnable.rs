//! The `Runnable` trait — the composable unit of async work in Catena.
//!
//! Everything in Catena (prompt templates, LLM calls, output parsers, memory
//! wrappers, agents) implements `Runnable` so they can all be wired together
//! with [`.then()`](Runnable::then).
//!
//! ## Composition
//!
//! ```ignore
//! let chain = step_a.then(step_b).then(step_c);
//! let result = chain.invoke(input).await?;
//! ```
//!
//! ## `|` sugar (not available as a blanket impl)
//!
//! The `BitOr` operator (`a | b`) would be sugar over `.then()`, but a
//! blanket `impl BitOr<B> for A where A: Runnable<Input>, B: Runnable<A::Output>`
//! hits Rust's orphan/coherence rules when `Input` is unconstrained — the
//! compiler can't prove the projection `A::Output` is well-formed in the
//! impl's where clause. Use `.then()` as the primary composition method;
//! individual types can still implement `BitOr` for their specific input
//! type if desired.

use std::future::Future;

use promptus_core::BoxFut;

// ---------------------------------------------------------------------------
// Runnable
// ---------------------------------------------------------------------------

/// A composable unit of async work: takes an `Input`, produces
/// `Result<Self::Output, Self::Error>`.
///
/// Chain runnables with [`.then()`](Runnable::then):
///
/// ```ignore
/// let chain = step_a.then(step_b);
/// let result = chain.invoke(input).await?;
/// ```
pub trait Runnable<Input>: Send + Sync {
    /// The type produced on success.
    type Output;
    /// The error type.
    type Error;

    /// Run once on a single input.
    fn invoke(
        &self,
        input: Input,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;

    /// Run over many inputs concurrently.
    ///
    /// The default implementation calls [`invoke`](Runnable::invoke) on each
    /// input via `try_join_all`. Override only if the underlying operation
    /// has a real batch API worth using instead.
    fn batch(
        &self,
        inputs: Vec<Input>,
    ) -> impl Future<Output = Result<Vec<Self::Output>, Self::Error>> + Send
    where
        Input: Send,
        Self::Output: Send,
        Self::Error: Send,
    {
        async move {
            let futures = inputs.into_iter().map(|i| self.invoke(i));
            futures::future::try_join_all(futures).await
        }
    }

    /// Chain this runnable with another: run `self`, then feed the output
    /// to `next`.
    ///
    /// This is the primary composition method. Produces a [`Sequence`].
    fn then<Next>(self, next: Next) -> Sequence<Self, Next>
    where
        Self: Sized,
        Next: Runnable<Self::Output, Error = Self::Error>,
    {
        Sequence {
            first: self,
            second: next,
        }
    }
}

// ---------------------------------------------------------------------------
// DynRunnable — object-safe adapter (for future use)
// ---------------------------------------------------------------------------

/// Object-safe version of [`Runnable`] using boxed futures.
///
/// This exists for the specific spots that need type erasure — heterogeneous
/// tool registries, pipelines assembled at runtime from config, etc.
///
/// A blanket impl from `Runnable` is not possible here because `Input` is
/// a generic parameter (Rust's orphan rules require it to be constrained
/// by the implementing type). For Phase 1, concrete dyn-compatible adapters
/// (like [`DynTool`](crate::tool::DynTool) and
/// [`Memory`](crate::memory::Memory)) cover the dynamic-dispatch use cases.
/// This trait is provided for downstream crates that need to erase the
/// `Runnable` type at a specific, concrete `Input` type.
pub trait DynRunnable<Input>: Send + Sync {
    /// The type produced on success.
    type Output;
    /// The error type.
    type Error;

    /// Run once, returning a boxed future for object-safe dispatch.
    fn invoke_dyn(&self, input: Input) -> BoxFut<'_, Result<Self::Output, Self::Error>>;
}

// ---------------------------------------------------------------------------
// Sequence
// ---------------------------------------------------------------------------

/// Runs two runnables in series: the first's output feeds into the second.
///
/// Created by [`Runnable::then`]. Typically not constructed directly.
pub struct Sequence<A, B> {
    first: A,
    second: B,
}

impl<A, B> Sequence<A, B> {
    /// Create a new sequence. Prefer `.then()` instead.
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A, B, Input> Runnable<Input> for Sequence<A, B>
where
    A: Runnable<Input>,
    B: Runnable<A::Output, Error = A::Error>,
    Input: Send,
    A::Output: Send,
{
    type Output = B::Output;
    type Error = A::Error;

    async fn invoke(&self, input: Input) -> Result<Self::Output, Self::Error> {
        let intermediate = self.first.invoke(input).await?;
        self.second.invoke(intermediate).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // A trivial runnable that doubles a number.
    struct Doubler;

    impl Runnable<i64> for Doubler {
        type Output = i64;
        type Error = String;

        async fn invoke(&self, input: i64) -> Result<Self::Output, Self::Error> {
            Ok(input * 2)
        }
    }

    // A trivial runnable that adds 10.
    struct AddTen;

    impl Runnable<i64> for AddTen {
        type Output = i64;
        type Error = String;

        async fn invoke(&self, input: i64) -> Result<Self::Output, Self::Error> {
            Ok(input + 10)
        }
    }

    // A runnable that converts i64 to String.
    struct ToStr;

    impl Runnable<i64> for ToStr {
        type Output = String;
        type Error = String;

        async fn invoke(&self, input: i64) -> Result<Self::Output, Self::Error> {
            Ok(input.to_string())
        }
    }

    #[tokio::test]
    async fn sequence_two_steps() {
        // 5 → doubler → 10 → add_ten → 20
        let chain = Doubler.then(AddTen);
        let result = chain.invoke(5_i64).await.unwrap();
        assert_eq!(result, 20);
    }

    #[tokio::test]
    async fn sequence_three_steps() {
        // 5 → doubler → 10 → add_ten → 20 → to_str → "20"
        let chain = Doubler.then(AddTen).then(ToStr);
        let result = chain.invoke(5_i64).await.unwrap();
        assert_eq!(result, "20");
    }

    #[tokio::test]
    async fn batch_default() {
        let results = Doubler.batch(vec![1_i64, 2, 3]).await.unwrap();
        assert_eq!(results, vec![2, 4, 6]);
    }

    #[tokio::test]
    async fn sequence_preserves_error() {
        struct Fails;

        impl Runnable<i64> for Fails {
            type Output = i64;
            type Error = String;

            async fn invoke(&self, _input: i64) -> Result<Self::Output, Self::Error> {
                Err("boom".to_owned())
            }
        }

        let chain = Doubler.then(Fails);
        let result = chain.invoke(5_i64).await;
        assert_eq!(result, Err("boom".to_owned()));
    }
}
