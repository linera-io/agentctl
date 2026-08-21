# Recovering sandbox sessions that vanished from the TUI

Live sessions disappear from the dashboard while their processes keep running.
This has happened repeatedly. Every recovery so far has been improvised. This is
the procedure.

**Nothing is ever lost when this happens.** The processes are alive, the
transcripts are intact, and the terminals still work. What is lost is one file's
contents: this sandbox's slice of `sandbox-sessions.json`.

## Why it happens

The registry slice for a sandbox is written by `record_hook_event`, which
reconciles **only when a hook fires inside that sandbox**. Three properties
combine badly:

1. **A reconcile is wholesale.** It replaces the sandbox's entire slice with
   whatever the live scan returned.
2. **A live scan can return empty for reasons that are not "no sessions"** — a
   failed process-table read, pointers already gone, a hook firing during
   teardown. Before the `EmptySlice::Freeze` guard, that emptied the slice.
3. **Idle sessions fire no hooks.** Once a slice is empty, nothing rewrites it
   until someone interacts with a session in that sandbox. Sessions idle for
   days stay invisible for days.

The pid cannot be recovered from outside the sandbox. It exists only as the
filename of `~/.claude/sessions/<pid>.json`, inside the user+mount namespace
each `sc` session creates for itself (`sandbox-bootstrap`, `unshare --user
--mount`). Each session gets its *own* namespace with exactly one pointer in it,
so even `sbx exec` into the right sandbox reaches at most one of them.

**A missing pid is fine.** `from_registry_entry` maps it to `0`, and
`collector_says_gone` explicitly treats an absent pid as "no evidence" — which is
what keeps the row on screen. Tab-to-session keys on `host_terminal_id` and
`host_tty`, not the pid. Do not invent a pid to make the column look right: a
fabricated pid makes `collector_says_gone` trust the collector's zero reading and
filters every row straight back off.

## Diagnose (30 seconds, host)

```bash
# 1. What does the registry think?
python3 -c "import json;d=json.load(open('$HOME/.local/share/claudectl/sandbox-sessions.json'));[print(k,len(v)) for k,v in d['sandboxes'].items()]"

# 2. What is actually running? This is the ground truth.
ps -eo pid,ppid,lstart,args | grep -E '[c]laude|[s]c '

# 3. Are the sandboxes up?
sbx ls
```

If (2) shows more sessions than (1), the registry lost them. Proceed.

## Recover (host)

Every field needed to rebuild an entry is in the `sbx exec` argv that (2)
printed. For each session:

| Registry field     | Where it comes from                       |
| ------------------ | ----------------------------------------- |
| `session_id`       | `--resume <id>`                           |
| sandbox key        | `-e SANDBOX_NAME=linera-agent-…`          |
| `host_tty`         | `-e SANDBOX_HOST_TTY=/dev/ttysNNN`        |
| `host_terminal_id` | `-e SANDBOX_HOST_TERMINAL_ID=…`           |
| `transcript`       | `~/.claude/projects/*/<session_id>.jsonl` |
| `cwd`              | `-w /Users/<you>`                         |
| `pid`              | leave `null` — see above                  |

A session launched with an initial prompt rather than `--resume` has no id in
its argv. Those are new sessions, so their `SessionStart` hook has already
registered them; they are not the ones that go missing.

Back up first, then write only the sandboxes you are repairing — other slices
must be left alone:

```bash
cp ~/.local/share/claudectl/sandbox-sessions.json{,.bak-$(date +%F-%H%M)}
```

## Rules that make recovery safe

These are the mistakes that turned previous recoveries into long ones.

- **Never delete an entry because a probe found nothing.** "I could not see it"
  is not "it is not running". Repair scripts must be additive-only.
- **Never key entries to a sandbox name `sbx` does not report.** `foreign_sessions`
  filters every row through `running_sandbox_filter`, so an invented key renders
  nothing at all.
- **Never use `ps` from inside a sandbox to judge another sandbox.** Sandboxes
  are separate VMs; that `ps` cannot see them, and its silence means nothing.
- **Ask for host `ps` output first.** It is the only complete view, and it
  carries sandbox, session id, tty and terminal id in one line.

## Verify

Rows appear in the TUI, and tab-to-session reaches the right window. The pid
column showing `0` for foreign rows is expected, not a failure.
