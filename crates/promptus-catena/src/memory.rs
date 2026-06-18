//! Conversation memory — read/write access to message history.
//!
//! The [`Memory`] trait abstracts over storage backends. Phase 1 ships two
//! in-process implementations:
//!
//! - [`BufferMemory`]: keeps the full transcript, unbounded.
//! - [`WindowMemory`]: keeps only the last *N* messages, trimming oldest on
//!   append.
//!
//! Both are cheaply `Clone`-able (internal `Arc`) so the same instance can
//! be shared across a chain and an agent loop within one session.
//!
//! The trait uses [`BoxFut`](promptus_core::BoxFut) return types (rather
//! than RPITIT) so it is object-safe and can be used as `Arc<dyn Memory>`.

use std::sync::Arc;

use promptus_core::{BoxFut, Message};
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// Memory trait
// ---------------------------------------------------------------------------

/// Conversation history a chain or agent can read from and append to.
///
/// Uses boxed futures (`BoxFut`) so the trait is object-safe — callers can
/// hold `Arc<dyn Memory>` and swap implementations at runtime.
///
/// Implementations may be in-process (Phase 1) or backed by external
/// storage (future work) — callers shouldn't need to care which.
pub trait Memory: Send + Sync {
    /// Read the current message history.
    fn history(&self) -> BoxFut<'_, Vec<Message>>;

    /// Append messages to the history.
    fn append(&self, messages: Vec<Message>) -> BoxFut<'_, ()>;

    /// Clear all history.
    fn clear(&self) -> BoxFut<'_, ()>;
}

// ---------------------------------------------------------------------------
// BufferMemory — keeps everything
// ---------------------------------------------------------------------------

/// In-memory conversation history that retains all messages.
///
/// Clone-cheap (shared `Arc` internally) so the same memory instance can
/// be passed to both a chain and an agent executor.
#[derive(Clone)]
pub struct BufferMemory {
    messages: Arc<RwLock<Vec<Message>>>,
}

impl BufferMemory {
    /// Create an empty buffer memory.
    pub fn new() -> Self {
        Self {
            messages: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create a buffer memory seeded with existing messages.
    ///
    /// Useful for restoring a persisted transcript on startup.
    pub fn from_history(messages: Vec<Message>) -> Self {
        Self {
            messages: Arc::new(RwLock::new(messages)),
        }
    }
}

impl Default for BufferMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory for BufferMemory {
    fn history(&self) -> BoxFut<'_, Vec<Message>> {
        Box::pin(async { self.messages.read().await.clone() })
    }

    fn append(&self, new_messages: Vec<Message>) -> BoxFut<'_, ()> {
        Box::pin(async { self.messages.write().await.extend(new_messages) })
    }

    fn clear(&self) -> BoxFut<'_, ()> {
        Box::pin(async { self.messages.write().await.clear() })
    }
}

// ---------------------------------------------------------------------------
// WindowMemory — keeps only the last N messages
// ---------------------------------------------------------------------------

/// In-memory conversation history that retains only the last `window_size`
/// messages. When the window is full, oldest messages are dropped on append.
///
/// The window counts individual messages, not conversation turns (a turn may
/// consist of a user message, an assistant message, and several tool-call
/// round-trips). This keeps the implementation simple and predictable.
#[derive(Clone)]
pub struct WindowMemory {
    messages: Arc<RwLock<Vec<Message>>>,
    window_size: usize,
}

impl WindowMemory {
    /// Create a window memory that keeps the last `window_size` messages.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is 0.
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "window_size must be > 0");
        Self {
            messages: Arc::new(RwLock::new(Vec::new())),
            window_size,
        }
    }
}

impl Memory for WindowMemory {
    fn history(&self) -> BoxFut<'_, Vec<Message>> {
        Box::pin(async { self.messages.read().await.clone() })
    }

    fn append(&self, new_messages: Vec<Message>) -> BoxFut<'_, ()> {
        let window = self.window_size;
        Box::pin(async move {
            let mut messages = self.messages.write().await;
            messages.extend(new_messages);
            // Trim oldest messages to stay within the window.
            let len = messages.len();
            if len > window {
                messages.drain(..len - window);
            }
        })
    }

    fn clear(&self) -> BoxFut<'_, ()> {
        Box::pin(async { self.messages.write().await.clear() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn buffer_memory_append_and_history() {
        let mem = BufferMemory::new();
        assert!(mem.history().await.is_empty());

        mem.append(vec![Message::user("hello"), Message::assistant("hi")])
            .await;
        let history = mem.history().await;
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].text().as_deref(), Some("hello"));
        assert_eq!(history[1].text().as_deref(), Some("hi"));
    }

    #[tokio::test]
    async fn buffer_memory_clear() {
        let mem = BufferMemory::new();
        mem.append(vec![Message::user("hello")]).await;
        assert_eq!(mem.history().await.len(), 1);

        mem.clear().await;
        assert!(mem.history().await.is_empty());
    }

    #[tokio::test]
    async fn buffer_memory_clone_shares_state() {
        let mem1 = BufferMemory::new();
        let mem2 = mem1.clone();

        mem1.append(vec![Message::user("shared")]).await;
        assert_eq!(mem2.history().await.len(), 1);
        assert_eq!(mem2.history().await[0].text().as_deref(), Some("shared"));
    }

    #[tokio::test]
    async fn window_memory_trims_old() {
        let mem = WindowMemory::new(3);

        mem.append(vec![
            Message::user("a"),
            Message::assistant("b"),
            Message::user("c"),
        ])
        .await;
        assert_eq!(mem.history().await.len(), 3);

        // Adding one more should drop "a".
        mem.append(vec![Message::assistant("d")]).await;
        let history = mem.history().await;
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].text().as_deref(), Some("b"));
        assert_eq!(history[2].text().as_deref(), Some("d"));
    }

    #[tokio::test]
    async fn window_memory_batch_trim() {
        let mem = WindowMemory::new(2);

        // Append 4 messages at once — only the last 2 should survive.
        mem.append(vec![
            Message::user("1"),
            Message::assistant("2"),
            Message::user("3"),
            Message::assistant("4"),
        ])
        .await;
        let history = mem.history().await;
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].text().as_deref(), Some("3"));
        assert_eq!(history[1].text().as_deref(), Some("4"));
    }

    #[tokio::test]
    async fn window_memory_clear() {
        let mem = WindowMemory::new(5);
        mem.append(vec![Message::user("x")]).await;
        mem.clear().await;
        assert!(mem.history().await.is_empty());
    }

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn window_memory_zero_panics() {
        WindowMemory::new(0);
    }
}
