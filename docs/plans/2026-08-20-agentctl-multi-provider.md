# Agentctl Multi-Provider Control Plane Implementation Plan

> **For Codex:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Turn claudectl into agentctl, a single Claude Code and Codex control
plane with the existing TUI, full provider-aware supervision, and safe
externally sandboxed Codex full-access mode.

**Architecture:** Keep the existing session data shape during the migration,
add provider identity and adapter dispatch, then move Claude-only behavior into
its adapter. Use Codex rollout files for passive discovery and the official
App Server JSONL protocol for managed live state and controls. Preserve legacy
commands and paths as read/launch aliases. Treat `~/.agents` as the canonical
shared home and render thin Claude/Codex adapters from it; never directly share
mutable provider-owned configuration or transcript files.

**Tech Stack:** Rust 2024, ratatui, crossterm, serde/serde_json, clap, existing
Tokio blocking-worker runtime, Codex App Server JSON-RPC over stdio proxy.

---

No task below includes a commit step because repository instructions require
explicit user authorization before commits.

### Task 1: Canonical shared agent home

**Files:**
- Create: `src/shared_home.rs`
- Modify: `src/lib.rs`
- Modify: `src/config.rs`
- Create: `tests/shared_home_tests.rs`
- Create: `docs/shared-agent-home.md`

1. Write failing tests for the canonical `~/.agents` layout covering global
   instructions, skills, durable memory, MCP registry, hook/workflow registry,
   provider adapters, and unified session index.
2. Write failing adoption tests using a temporary home with existing Claude
   instructions/memory/skills and Codex configuration. Require a dry-run plan,
   canonical-first resolution, no deletion, and idempotent reconciliation.
3. Implement pure discovery and reconciliation planning before any filesystem
   writes. Classify paths as shared source, generated adapter, native cache, or
   immutable transcript evidence.
4. Implement atomic adapter rendering with backups when replacing an existing
   generated adapter. Refuse to replace an independently edited native file and
   report drift instead.
5. Adopt the existing Claude memory graph in place through an explicit shared
   reference; do not copy or move the 1,747 notes. Make Codex's instruction
   adapter point at the same graph and tool paths.
6. Add provider-neutral MCP/hook schemas and deterministic Claude/Codex render
   plans. Keep credentials out of the canonical files; reference environment or
   native credential stores.
7. Run shared-home tests twice to prove idempotence and document migration,
   rollback, ownership boundaries, and conflict handling.

### Task 2: Product identity with backward compatibility

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/lib.rs`
- Modify: `src/main.rs`
- Modify: `src/config.rs`
- Modify: `src/helpers.rs`
- Modify: `src/init.rs`
- Modify: `src/bin/claudectl-hook.rs`
- Test: co-located tests in `src/config.rs`, `src/helpers.rs`, and `src/init.rs`

1. Write failing tests that the canonical product name is `agentctl`, the
   canonical config/data roots use `agentctl`, and legacy claudectl files are
   read when the canonical file is absent.
2. Run only those tests and confirm failures mention the old canonical name.
3. Rename the package/library and add `agentctl` plus `claudectl` compatibility
   binaries that compile the same entry point.
4. Implement canonical-first, legacy-second path resolution and idempotent
   hook migration; never delete legacy data.
5. Run focused tests, then `cargo check --all-targets`.

### Task 3: Provider-neutral session identity

**Files:**
- Create: `src/provider.rs`
- Modify: `src/session.rs`
- Modify: `src/lib.rs`
- Modify: `src/demo.rs`
- Modify: `tests/integration_tests.rs`

1. Write failing tests for `AgentProvider::{Claude,Codex}` labels, executable
   names, transcript roots, and serde round trips.
2. Write a failing test that every constructed session has a provider and that
   legacy sessions default to Claude.
3. Add `provider: AgentProvider` to the session record. Rename the canonical
   type to `AgentSession` and temporarily export `ClaudeSession` as a deprecated
   compatibility alias so consumers can migrate in small green steps.
4. Update fixtures and constructors without changing existing Claude behavior.
5. Run session, demo, and integration tests.

### Task 4: Provider column in the unchanged TUI

**Files:**
- Modify: `src/ui/table.rs`
- Modify: `src/ui/detail.rs`
- Modify: `src/app.rs`
- Test: co-located tests in `src/ui/table.rs`

1. Write failing row/header tests requiring an `Agent` column between PID and
   Name, with compact `Claude` and `Codex` values.
2. Confirm the tests fail on the current 14-column table.
3. Add the column, width constraint, detail-panel provider line, and provider
   search text while preserving all keyboard behavior and layout priorities.
4. Run UI and app tests.

### Task 5: Provider adapter and Claude extraction

**Files:**
- Create: `src/providers/mod.rs`
- Create: `src/providers/claude.rs`
- Modify: `src/discovery.rs`
- Modify: `src/process.rs`
- Modify: `src/transcript.rs`
- Modify: `src/terminals/mod.rs`
- Modify: `src/commands.rs`
- Test: existing co-located tests plus `tests/integration_tests.rs`

1. Write failing contract tests for provider discovery, transcript parsing,
   launch args, resume args, rename support, and process matching.
2. Add a synchronous `AgentProviderAdapter` interface whose results are plain
   data and whose side effects remain in thin wrappers.
3. Move/wrap current Claude behavior behind `ClaudeProvider` without changing
   observable output.
4. Dispatch discovery, parsing, and terminal actions by `AgentProvider`.
5. Run all existing Claude-focused tests to prove parity.

### Task 6: Passive Codex rollout and process discovery

**Files:**
- Create: `src/providers/codex.rs`
- Create: `src/providers/codex_rollout.rs`
- Modify: `src/process.rs`
- Modify: `src/discovery.rs`
- Modify: `src/monitor.rs`
- Test: fixtures under `tests/fixtures/codex/` and co-located module tests

1. Add minimal redacted rollout fixtures for `session_meta`, `turn_context`,
   user/assistant messages, token events, compaction, names, active tools, and
   errors.
2. Write failing parser tests for cwd, thread id/name, model, tokens, context,
   last activity, pending approval, and status.
3. Write failing process tests for `codex`, `codex resume <id>`, and
   `codex --remote <endpoint>` without misclassifying unrelated commands.
4. Implement incremental rollout parsing and bounded recursive discovery under
   `$CODEX_HOME/sessions`; correlate process cwd/start time when argv lacks an
   id.
5. Merge Codex and Claude discoveries by `(provider, session_id)` and run
   focused parser/discovery tests.

### Task 7: Codex launch, resume, rename, focus, and input

**Files:**
- Modify: `src/providers/codex.rs`
- Modify: `src/terminals/mod.rs`
- Modify: terminal backends under `src/terminals/`
- Modify: `src/commands.rs`
- Modify: `src/app.rs`
- Test: co-located terminal/provider/app tests

1. Write failing command-builder tests for new, resume, remote, prompt, cwd,
   and externally-sandboxed full-access modes, including shell escaping.
2. Write a failing dispatch test proving Claude and Codex receive different
   launch/resume commands while focus/input reuse the same terminal target.
3. Implement provider-aware commands and reuse the existing terminal bridge.
4. Implement rename through App Server when managed; use the Codex CLI/thread
   store fallback only for passive sessions.
5. Run terminal, command, and app tests.

### Task 8: Codex App Server client and live event model

**Files:**
- Create: `src/providers/codex_app_server.rs`
- Modify: `src/providers/codex.rs`
- Modify: `src/app.rs`
- Modify: `src/session.rs`
- Test: fake JSONL server tests in `src/providers/codex_app_server.rs`

1. Write failing protocol tests for initialize/initialized, request-id
   correlation, `thread/list`, `thread/read`, `thread/resume`,
   `thread/name/set`, `turn/start`, status notifications, and approval events.
2. Use a fake child process with pipes; do not mock the protocol parser.
3. Implement a blocking worker that spawns `codex app-server proxy --sock ...`
   (or direct `codex app-server --stdio` for agentctl-owned instances), sends
   JSONL requests, and streams notifications to the existing Tokio channel.
4. Overlay App Server state on passive rollout/process state, preferring live
   server status without hiding passive sessions when the server is down.
5. Run protocol and app merge tests.

### Task 9: Approval, budget, health, orchestration, and brain parity

**Files:**
- Modify: `src/rules.rs`
- Modify: `src/health.rs`
- Modify: `src/history.rs`
- Modify: `src/orchestrator.rs`
- Modify: `src/brain/context.rs`
- Modify: `src/brain/engine.rs`
- Modify: `src/brain/mailbox.rs`
- Modify: `src/brain/decisions.rs`
- Test: existing module tests and `tests/integration_tests.rs`

1. Write failing tests showing provider identity in rule/brain context and
   provider-correct approval, deny, terminate, route, spawn, and resume actions.
2. Add provider as an optional rule predicate and persisted history dimension.
3. Translate Codex App Server approval requests/responses to the existing rule
   and brain decision model; preserve deny-first behavior.
4. Generalize health and budget calculations around provider-supplied usage and
   context limits; explicitly mark unavailable cost data instead of inventing
   estimates.
5. Let orchestration tasks select `provider`, defaulting legacy task files to
   Claude, and dispatch launches/routes accordingly.
6. Run focused rule, health, history, orchestration, and brain tests.

### Task 10: Externally sandboxed full-access mode

**Files:**
- Modify: `src/config.rs`
- Modify: `src/sandbox_registry.rs`
- Modify: `src/terminals/sandbox_terminal_bridge.rs`
- Modify: `src/providers/codex.rs`
- Modify: `docs/configuration.md`
- Modify: `docs/terminal-support.md`
- Test: co-located config/registry/bridge/provider tests

1. Write failing tests that full-access launch is rejected without the explicit
   sandbox marker, rejects a host/out-of-bound endpoint, and succeeds with a
   validated sandbox-local endpoint.
2. Add Codex provider configuration for App Server mode, endpoint/socket,
   external-sandbox marker, and permission mode.
3. Start or connect to the App Server inside the sandbox; launch host-visible
   `codex --remote` through the existing bridge using the shared endpoint.
4. Ensure loss of the marker or endpoint causes fail-closed ordinary Codex
   permissions, never silent host full access.
5. Run sandbox/bridge/provider tests.

### Task 11: Complete rename, migration docs, and verification

**Files:**
- Modify: `README.md`, `AGENTS.md`, `CLAUDE.md`, `CHANGELOG.md`
- Modify: `docs/**/*.md`, `plugin/**`, `scripts/**`, `.github/**`
- Modify: assets and install metadata that encode the old product name
- Keep: deliberate compatibility references to `claudectl`

1. Write a failing repository identity check that rejects unintended canonical
   `claudectl` names while allowing documented compatibility references.
2. Update documentation, examples, hooks, install paths, badges, and metadata
   to agentctl; document legacy migration and both providers.
3. Regenerate the indxr index.
4. Run `cargo fmt --check`, `cargo test`, `cargo clippy --all-targets -- -D warnings`,
   debug/release builds, CLI help smoke tests for both names, and App Server fake
   integration tests.
5. Inspect the final structural diff and list any deliberate remaining
   `claudectl` references.
6. Only after user authorization, rename the GitHub repository and update
   remotes/distribution metadata; GitHub redirects preserve old clone URLs.
