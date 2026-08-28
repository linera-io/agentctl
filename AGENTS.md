# AGENTS.md

Instructions for AI agents working on this repository.

This file routes; it is not a second source of truth. **On any conflict, the linked file wins.**

- **Module map** — `src/lib.rs`, then the `mod.rs` of each subtree (`brain/`, `ui/`, `terminals/`, `providers/`). The compiler keeps those current, so no prose copy is maintained.
- **Setup, contribution guidelines, design decisions** — [docs/contributing.md](docs/contributing.md).
- **Before every commit** — `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`, and `cargo test --all-targets`. CI runs all three and fails on any of them.
- **Config keys, auto-rules, event hooks, Claude Code integration** — [docs/configuration.md](docs/configuration.md).
- **CLI modes, status detection, brain gate, plugin components** — [docs/reference.md](docs/reference.md).
- **Terminal backends** — [docs/terminal-support.md](docs/terminal-support.md).
- **Symptoms and known defects** — [docs/troubleshooting.md](docs/troubleshooting.md) and [docs/known-bugs/](docs/known-bugs/).
