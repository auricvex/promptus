# Contributing to Promptus

Thank you for your interest in contributing! This document explains how to get
set up, make changes, and submit them.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/promptus.git
   cd promptus
   ```
3. **Install Rust** (stable channel is required):
   ```bash
   rustup show active-toolchain   # should show "stable"
   ```
4. **Verify** the project builds and tests pass:
   ```bash
   cargo test --workspace
   ```

## Development Workflow

1. Create a branch from `main`:
   ```bash
   git checkout -b my-feature
   ```
2. Make your changes.
3. Run the full verification suite before committing:
   ```bash
   cargo fmt --check
   cargo clippy --workspace --all-targets
   cargo test --workspace
   ```
4. Commit with a clear message (see below).
5. Push and open a pull request against `main`.

## Code Style

- **Formatting** is enforced by `rustfmt.toml` — run `cargo fmt` to
  auto-format.
- **Linting** is enforced by `clippy.toml` — run `cargo clippy` to check.
- No `#[allow(...)]` without a comment explaining why.
- No `unwrap()` or `expect()` outside of `#[cfg(test)]` code.
- All public items must have doc comments.

## Commit Messages

Use clear, descriptive commit messages:

```
short summary (50 chars or less)

Optional longer description wrapped at 72 characters. Explain the motivation
for the change and how it differs from the previous behavior.
```

- Use the imperative mood: "Add feature" not "Added feature".
- Reference issues with `#123` when relevant.

## Pull Request Checklist

Before submitting a PR, make sure:

- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy --workspace --all-targets` passes
- [ ] `cargo test --workspace` passes
- [ ] New public items have doc comments
- [ ] Non-trivial logic has tests
- [ ] The PR description explains *what* and *why*

## Reporting Bugs

Open a GitHub issue with:

- A clear title and description.
- Steps to reproduce (a minimal code snippet is ideal).
- Expected vs. actual behavior.
- Your Rust version (`rustc --version`) and OS.

## Adding a New Provider

To add support for a new LLM provider:

1. Create a new crate `crates/promptus-<name>`.
2. Implement the `ChatProvider` trait from `promptus-core`.
3. Add the crate to the workspace `Cargo.toml`.
4. Re-export it from the `promptus` facade crate.
5. Add an example demonstrating usage.

## License

By contributing, you agree that your contributions will be dual-licensed
under MIT and Apache 2.0.
