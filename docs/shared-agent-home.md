# The shared agent home (`~/.agents`)

One place for the state a person accumulates — instructions, skills, durable
memory, MCP and hook definitions, workflows, a unified session index — with each
provider getting a thin generated adapter pointing back at it. Swap the harness
and the accumulated context survives untouched; that is the whole point.

## Layout

```
~/.agents/
├── instructions/global.md     shared instructions, rendered into each provider
├── skills/                    Agent Skills, read by every provider that supports them
├── memory/                    durable notes; an existing graph is adopted, not copied
├── config/mcp.toml            provider-neutral MCP registry
├── config/hooks.toml          provider-neutral hook registry
├── workflows/                 workflow definitions
├── sessions/index.jsonl       unified session index across providers
├── adapters/                  per-provider adapter declarations
└── plugins/                   NOT OURS — see below
```

## Which products?

`PROVIDERS` in `shared_home.rs` is the single list — each row gives a product's
config directory, its global-instructions file, and where it keeps transcripts
and any adoptable memory graph. Adapter rendering, ownership classification and
memory discovery all iterate it, so supporting another product is a row rather
than an edit in five places. A test iterates the table and asserts each of those
three behaviours, so a row that would need a special case fails there.

## Ownership

`PathOwnership` is the whole model, and every write decision derives from it:

| Class | Meaning | Written by |
| --- | --- | --- |
| `SharedSource` | user-authored, canonical | the user, and agentctl on their behalf |
| `GeneratedAdapter` | rendered from shared source into a provider's native location | agentctl |
| `ProviderNative` | the provider's own mutable state | the provider only |
| `TranscriptEvidence` | immutable session records | the provider only, never rewritten |
| `External` | everything else | nobody here |

**`~/.agents/plugins` is `ProviderNative`, despite living inside the shared
home.** codex-cli discovers and rewrites `~/.agents/plugins/marketplace.json`
implicitly — `~/.agents` is OpenAI's own convention, not one this project
invented. Claiming it would put agentctl and Codex in a write race over the same
file. Verified against codex-cli 0.148.0.

## Reconcile: observe → plan → apply

Split into three so the middle step is pure:

- **`observe()`** is the only function that reads the filesystem before `apply`.
- **`plan()`** takes an `Observed` and returns a `Plan`. No I/O, so the rules are
  testable against a constructed input and a plan can be shown to the user
  before anything is written.
- **`apply()`** executes. Writes are atomic — temp file then rename — so a crash
  mid-render cannot leave a provider reading half a file.

**A plan cannot delete.** There is no `Action::Remove`; the guarantee is in the
type, not in a runtime check. Adoption of an existing setup is the common case,
and a reconcile that runs on every hook event and *can* delete is one bad
reading away from taking a user's own instructions with it. If removal is ever
wanted it belongs in a separate, explicit operation.

**Applying a plan and re-planning yields nothing.** That idempotence is what
makes it safe on every hook event, and it is asserted directly.

## When do you re-run it?

Whenever the shared content changes. Running it when nothing has changed prints
"up to date" and does nothing, so it is safe on a hook, in a shell alias, or by
hand.

Edit `~/.agents/instructions/global.md` and the next `--apply` pushes it to
every provider's adapter. That propagation is the point of the tool.

## Drift

Each render records the bytes it wrote, under `adapters/`. That stamp is what
separates *we wrote this and the source has since moved on* — ours to re-render
— from *a human changed it* — theirs, reported and never overwritten.

Without the stamp the two are indistinguishable: both show an adapter whose
content differs from the shared source. Treating them alike breaks whichever one
you guess wrong, and guessing "drift" breaks the main use case, because then
editing the shared instructions reaches nobody.

A file we never rendered is theirs by the same rule — no stamp, no claim on it,
even where we would have put ours. Silently replacing it is how someone loses
instructions they wrote themselves.

Replacing an adapter we *do* own still writes a `.bak` first, with a numeric
suffix so two renders in the same second cannot have the second discard the
first backup.

## Adopting an existing memory graph

An established Claude auto-memory directory is adopted **in place**, by
reference. Copying would duplicate every note and fork its history the moment
either side is written. Discovery deliberately returns nothing when more than
one candidate exists — adopting the wrong project's graph points every provider
at someone else's notes, while a missed adoption is a no-op the user can
correct.

## Credentials

`McpRegistry::validated` refuses any env value that is not a `${VAR}`
reference, naming the server and the variable. This registry is shared,
version-controllable, and rendered into every provider's native config, so one
literal secret here becomes a secret in all of them at once.

It refuses rather than redacting, for the same reason the rest of this codebase
does: a redacting writer fails open on the one shape it did not anticipate, and
does it silently.

Servers are sorted at construction, so the same set always yields the same order
and a future reconcile diff means a real change rather than reordering noise.

## Not yet implemented

- Rendering the MCP and hook registries into each product's native config.
  Validation is wired in; the writers are not. Each product wants a different
  file format, so they belong in `PROVIDERS` alongside the rest, and a renderer
  named after one vendor is the shape this deliberately avoids.
- Populating `sessions/index.jsonl` — that arrives with provider discovery.
- Migrating memory *out* of a provider's tree. Adoption is by reference only;
  an explicit, reversible move is deliberately a separate operation.
