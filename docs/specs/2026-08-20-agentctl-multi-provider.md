# Spec: Agentctl Multi-Provider Control Plane

## Problem

`claudectl` is an effective local control plane, but its identity, discovery,
transcript parser, process model, launch commands, state paths, and UI assume
Claude Code. The user now works across Claude Code and Codex and needs the same
session supervision, terminal control, budgets, health checks, orchestration,
and local brain for both. Codex must also be able to run unsandboxed internally
when the entire runtime is contained by the user's external sandbox.

## Approach

Rename the product and primary binary to `agentctl`, while shipping a
`claudectl` compatibility binary and reading legacy configuration/state paths.
Introduce an `AgentProvider` discriminator and provider adapter boundary around
discovery, transcript parsing, launch/resume/rename, approval, and control.
Preserve the existing session/TUI behavior during migration; add a compact
`Agent` column showing `Claude` or `Codex`.

Make `~/.agents` the canonical, provider-neutral agent home. It owns shared
instructions, skills, durable memories, MCP definitions, hook definitions,
workflow definitions, and the unified session index. Claude Code and Codex get
thin generated adapters that reference or render this shared state into the
native files each provider expects. Agentctl reconciles adapters
deterministically and reports drift; users do not manually import setup between
providers.

Provider-owned mutable files are not shared directly. Agentctl must not symlink
Claude settings JSON to Codex TOML, either provider's native memory database, or
raw transcript files because both providers may rewrite them concurrently.
Instead, the provider-neutral registry is the source of truth and agentctl
renders provider-specific configuration. Native transcripts remain immutable
evidence in their original format and are indexed into a unified catalog with
provider, native session id, cwd, title, and handoff lineage.

The Claude adapter wraps today's pointer/process/JSONL/hook implementation. The
Codex adapter passively discovers standalone CLI sessions from process metadata
and rollout files, and actively manages sessions through `codex app-server`.
Agentctl communicates with the managed daemon through `codex app-server proxy`
over JSONL stdio, avoiding a new WebSocket dependency. Host-visible Codex TUIs
connect to the same daemon with `codex --remote`; the existing sandbox terminal
bridge launches/focuses the host window.

Sandbox full-access is fail-closed. Agentctl may request Codex
`danger-full-access`/no approvals only when an explicit agentctl sandbox marker
is present and the configured App Server endpoint is inside that sandbox.
Outside that boundary it preserves the user's ordinary Codex permissions.

## Validated Assumptions

- The current claudectl TUI and workflows remain the product baseline.
- Claude and Codex are equal providers, not separate dashboards.
- Scope includes discovery, focus/input, launch/resume/rename, approvals,
  budgets, health, orchestration, and local-brain decisions.
- The provider is visible as a table column.
- `agentctl` is the new name; `claudectl` remains a compatibility alias.
- The App Server can live inside the external sandbox while a host TUI connects
  through a shared Unix socket or an explicitly configured local endpoint.
- Existing user data and oversized imported transcripts are preserved.
- `~/.agents` is the source of truth for shared setup; provider-native files
  are adapters or caches, not separate user-managed copies.
- Existing Claude memories are adopted in place first, then migrated only with
  an explicit, reversible operation. No memory file is deleted during adoption.

## Success Criteria

- [ ] `agentctl` and the legacy `claudectl` command open the same dashboard.
- [ ] Existing claudectl configuration/state is discovered and migrated without
      deletion; new writes use agentctl paths.
- [ ] Claude and Codex consume one canonical instruction/skill/memory/MCP/hook
      setup from `~/.agents`, with deterministic adapters and drift reporting.
- [ ] The existing Claude memory graph is available to Codex without copying
      1,747 notes into a second native memory store.
- [ ] Provider-native config and transcript files remain independently writable
      and cannot corrupt one another.
- [ ] Claude and Codex sessions render together with a correct provider column.
- [ ] Standalone Codex sessions are discoverable and resumable.
- [ ] Managed Codex sessions expose live status, approvals, names, turns, and
      controls through App Server events and requests.
- [ ] Focus, send input, launch, resume, rename, terminate, route, and spawn
      dispatch through the selected session's provider.
- [ ] Budgets, health checks, orchestration, and brain context include provider
      identity and do not apply Claude-only assumptions to Codex.
- [ ] External-sandbox full-access mode fails closed when its marker or endpoint
      validation is absent.
- [ ] Both oversized imported Claude sessions have preserved audit transcripts
      and usable Codex continuation threads.
- [ ] Formatting, unit/integration tests, clippy, build, and indxr regeneration
      pass from a clean worktree.

## Not Doing (and why)

- Managing Codex cloud-only jobs or IDE-only sessions — this design targets
  locally observable CLI/App Server sessions.
- General remote Internet-facing App Server hosting — local Unix sockets and
  explicitly configured sandbox-local endpoints cover the damage-control use
  case without adding TLS/auth infrastructure to agentctl.
- Removing legacy claudectl paths or command aliases — compatibility prevents a
  flag-day migration.
- Replacing Claude's and Codex's native transcript formats with a new format —
  the unified index and explicit handoffs provide portability without losing
  native resumability or audit history.

## Open Questions

- Rename the GitHub repository only after the compatibility build and migration
  verification are complete; GitHub's redirect can then preserve old remotes.
