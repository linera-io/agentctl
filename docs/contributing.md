# Contributing

Contributions are welcome.

## Setup

```bash
git clone https://github.com/mercurialsolo/claudectl.git
cd claudectl
cargo build
cargo test --all-targets
```

## Before Submitting

```bash
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

## Guidelines

- **No new dependencies** without strong justification — the project stays lightweight
- **Test behavior, not implementation** — focus on what the code does
- **Match existing patterns** — look at similar code before writing new code
- **Keep commits atomic** — one logical change per commit
- **Adding a config field** — add the `RawConfig` field, the `Config` default, and a branch in `Config::apply`, all in `config.rs`, plus the CLI flag in `main.rs`. A missing `apply` branch silently ignores the setting.
- **Changing status inference** — status detection carries extensive tests; update them in the same change.
- **Plugin hook scripts fail open** — the scripts in `claude-plugin/hooks/scripts/` exit 0 when the binary is missing or the gate is off, so a broken hook never blocks Claude Code. Keep that property.

Not all contributions are code. Hooks, docs, config presets, terminal compatibility fixes, and packaging help are all valuable.

## Architecture

The module map is `src/lib.rs`, then the `mod.rs` of each subtree (`brain/`, `ui/`, `terminals/`, `providers/`). The compiler keeps those current, so they are not mirrored here.

## Design Decisions

Rationale the code does not state on its own:

- **Native `ps` over the `sysinfo` crate** — keeps the dependency set and the binary small.
- **Multi-signal status inference** — CPU usage, JSONL events, and timestamps are combined; no single signal is authoritative.
- **Incremental JSONL parsing** — `monitor.rs` tracks a per-session file offset and never rereads a whole transcript.
- **Refresh runs off the render thread** — the TUI owns a 2-worker `tokio` runtime built in `main.rs`; refresh I/O is dispatched through `tokio::task::spawn_blocking`, so the existing blocking `std::fs` and `std::process` call sites stay unchanged.
- **Shared state is `Arc<RwLock<Arc<AppData>>>`** — each frame takes an `Arc` snapshot while refresh swaps the inner `Arc` on completion, so reads never block on writes.
- **Deny-first rule evaluation** — a deny rule wins over any approve or brain suggestion regardless of config order; among non-deny rules, the first match in config order wins.
- **Brain decisions stay local** — decision logs and few-shot examples never leave the machine.

## Reporting Issues

Found a bug? [Open an issue](https://github.com/mercurialsolo/claudectl/issues/new) with `claudectl --version`, your terminal (`echo $TERM_PROGRAM`), and steps to reproduce.
