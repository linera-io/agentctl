use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use crate::session::{ClaudeSession, RawSession};

fn sessions_dir() -> PathBuf {
    dirs_home().join(".claude").join("sessions")
}

fn dirs_home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

pub fn projects_dir() -> PathBuf {
    dirs_home().join(".claude").join("projects")
}

/// The live sessions to mirror into the restore registry: every session with a
/// running process. Idle time is irrelevant — if the process is alive, the
/// session is kept, forever.
///
/// This only READS the session-pointer JSONs — claudectl never deletes a user's
/// session files. A session is a live process, not a timestamp: liveness means
/// its pid is a live `claude` process in this namespace (the snapshot check
/// inside [`assemble_sessions`]), degrading to bare `kill -0` if `ps` fails.
pub fn live_sessions() -> Vec<ClaudeSession> {
    try_live_sessions().unwrap_or_default()
}

/// Like [`live_sessions`], but `None` when the sessions directory could not be
/// read at all — as distinct from `Some(vec![])` for a directory that is
/// readable and genuinely empty.
///
/// The reaper needs that distinction before it prunes: an empty scan means
/// "every session closed" (fair to prune), but a *failed* scan must never be
/// read as that, or a transient FS error would delete every live session's
/// restore entry. An unreadable/unparseable individual pointer file still just
/// skips that one session (a torn mid-write read is momentary, and the reaper's
/// two-scan window forgives it); only a failure to enumerate the directory at
/// all is reported as `None`.
pub fn try_live_sessions() -> Option<Vec<ClaudeSession>> {
    let entries = fs::read_dir(sessions_dir()).ok()?;
    let mut raw_sessions = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(raw) = serde_json::from_str::<RawSession>(&content) else {
            continue;
        };
        raw_sessions.push(raw);
    }
    Some(snapshot_and_assemble(
        raw_sessions,
        crate::process::live_claude_procs,
        registry_slice_for_current_scope(),
        &pid_alive,
        &proc_cwd,
        &transcript_session_name,
        DeadPointers::Drop,
    ))
}

/// What a scan does with a pointer file whose pid is dead.
enum DeadPointers {
    /// Keep the row — the display turns it into a 30s Finished tombstone
    /// (`fetch_and_enrich` marks pids absent from `ps` as Finished).
    Keep,
    /// Drop it — the live set (registry writer, restore) must never contain
    /// a dead session.
    Drop,
}

/// Take the process-table snapshot AFTER the pointers have been read, then
/// assemble. A session that starts mid-scan is then in the snapshot — its
/// pointer may be missed this pass, but the ps supplement still includes it.
/// The reverse order would drop a just-started session entirely for one scan
/// (pointer read, pid not yet in the snapshot). Both wrappers go through this
/// one function so the ordering exists in exactly one place, and a test pins
/// it by injecting a recording snapshot closure after pre-read pointers.
fn snapshot_and_assemble(
    raw_pointers: Vec<RawSession>,
    take_snapshot: impl FnOnce()
        -> Option<std::collections::HashMap<u32, crate::process::LiveClaudeProc>>,
    registry: Vec<crate::sandbox_registry::SessionEntry>,
    pid_alive_probe: &impl Fn(u32) -> bool,
    resolve_cwd: &impl Fn(u32) -> Option<String>,
    resolve_name: &impl Fn(&str, &str) -> Option<String>,
    dead_pointers: DeadPointers,
) -> Vec<ClaudeSession> {
    let procs = take_snapshot();
    assemble_sessions(
        raw_pointers,
        procs.as_ref(),
        registry,
        pid_alive_probe,
        resolve_cwd,
        resolve_name,
        dead_pointers,
    )
}

/// Pure composition of one discovery pass — THE seam where every source
/// meets: pointer files (already read), the process-table snapshot, and the
/// restore-registry slice. All I/O stays in the thin wrappers
/// ([`try_live_sessions`], [`scan_sessions`]), so each regression scenario of
/// the pipeline is testable with plain fixtures: this is where the
/// pointerless-recovery, cross-namespace-leak, cwd-repair, and
/// snapshot-ordering guarantees actually live.
///
/// Liveness: with a snapshot, a pid counts as live only if it is a *claude*
/// process in this namespace right now — rejecting recycled pids and, through
/// a shared/seeded `~/.claude`, other namespaces' pointers colliding with
/// unrelated local pids. Without a snapshot (`ps` failed) it degrades to the
/// injected `pid_alive_probe` (bare `kill -0` in production) — a transient
/// `ps` failure must never shrink the result below the pointer scan.
fn assemble_sessions(
    raw_pointers: Vec<RawSession>,
    procs: Option<&std::collections::HashMap<u32, crate::process::LiveClaudeProc>>,
    registry: Vec<crate::sandbox_registry::SessionEntry>,
    pid_alive_probe: &impl Fn(u32) -> bool,
    resolve_cwd: &impl Fn(u32) -> Option<String>,
    resolve_name: &impl Fn(&str, &str) -> Option<String>,
    dead_pointers: DeadPointers,
) -> Vec<ClaudeSession> {
    let is_live = |pid: u32| match procs {
        Some(map) => map.contains_key(&pid),
        None => pid_alive_probe(pid),
    };
    let mut sessions = Vec::new();
    for raw in raw_pointers {
        let keep = if is_live(raw.pid) {
            true
        } else {
            match dead_pointers {
                // A pointer whose pid is alive but NOT a claude process here
                // is foreign (recycled pid, or another namespace's session
                // colliding with an unrelated local process) — never a
                // tombstone. A genuinely dead pid is tombstone material.
                DeadPointers::Keep => !pid_alive_probe(raw.pid),
                DeadPointers::Drop => false,
            }
        };
        if keep {
            sessions.push(ClaudeSession::from_raw(raw));
        }
    }
    merge_pointerless_live(&mut sessions, registry, &is_live, resolve_cwd);
    extend_with_ps_live(&mut sessions, procs, resolve_cwd, resolve_name);
    sessions
}

/// Re-add restore-registry sessions that are still process-alive but have NO
/// live pointer file, deduped against what the pointer scan already found.
///
/// Claude Code writes `~/.claude/sessions/<pid>.json` at SessionStart and
/// removes it at session-end events (auto-compact, `/clear`, a remote-control
/// detach/reattach) *even while the process keeps running*. So a pointer-only
/// scan silently under-reports live sessions: a long-running session that has
/// hit any of those events becomes invisible for the rest of its life. The
/// restore registry — written while the pointer still existed, and pruned only
/// by the reaper once the owning terminal dies — still holds those sessions, so
/// we bring back every entry whose recorded pid is alive right now.
///
/// This single supplement reaches all three consumers of the scan, which is why
/// it lives here rather than in the display alone:
///   1. the live display ([`scan_sessions`]);
///   2. the restore-registry writer (`record_hook_event` → `record_live_sessions`
///      → [`live_sessions`]), so the host merge and — critically — the sandbox
///      `replace_sandbox_slice` (which otherwise *drops* a pointer-less session
///      on the next in-sandbox hook) keep it;
///   3. `--restore-sessions` / `--restore-sbx-sessions`, which read that registry.
///
/// The registry slice is chosen by the sandbox marker (see
/// [`registry_slice_for_current_scope`]) so an in-sandbox scan supplements
/// from that sandbox's slice (sandbox-namespace pids) and a host scan from
/// the local registry (host pids) — liveness is always evaluated in the
/// caller's own namespace, matching the pids it is checking. A failure to
/// read the registry degrades to no supplement (never fewer sessions than the
/// pointer scan), so it can never turn a readable scan into a failed one.
///
/// Appends each `registry` entry that is (a) not already among `sessions`
/// (deduped by `session_id`), (b) has a recorded pid, and (c) is alive per
/// `is_alive`. All side-effecting inputs are injected, so this is
/// deterministic to unit-test.
fn merge_pointerless_live(
    sessions: &mut Vec<ClaudeSession>,
    registry: Vec<crate::sandbox_registry::SessionEntry>,
    is_alive: &impl Fn(u32) -> bool,
    resolve_cwd: &impl Fn(u32) -> Option<String>,
) {
    let seen: HashSet<String> = sessions.iter().map(|s| s.session_id.clone()).collect();
    let used_pids: HashSet<u32> = sessions.iter().map(|s| s.pid).collect();
    // One process = one row. Claude Code rotates `sessionId` under a running
    // process (/clear, compaction), so a polluted slice can hold several
    // entries all claiming the same live pid — one per superseded
    // conversation (2026-07-28: one tab showed as three rows, each stealing
    // a different transcript). Keep the newest claim per pid: greatest
    // `started_at_ms`; on a tie the EARLIEST record wins — both registry
    // writers emit freshest-first (live scan before retained history), so
    // earlier in the file means fresher. Skip pids the scan already
    // represents.
    let mut winners: std::collections::HashMap<
        u32,
        (u64, usize, crate::sandbox_registry::SessionEntry),
    > = std::collections::HashMap::new();
    for (index, entry) in registry.into_iter().enumerate() {
        if seen.contains(&entry.session_id) {
            continue;
        }
        let Some(pid) = entry.pid else { continue };
        if used_pids.contains(&pid) || !is_alive(pid) {
            continue;
        }
        let is_newer = winners
            .get(&pid)
            .is_none_or(|(started, _, _)| entry.started_at_ms > *started);
        if is_newer {
            winners.insert(pid, (entry.started_at_ms, index, entry));
        }
    }
    let mut picked: Vec<(u64, usize, crate::sandbox_registry::SessionEntry)> =
        winners.into_values().collect();
    picked.sort_by_key(|(_, index, _)| *index);
    for (_, _, entry) in picked {
        let pid = entry.pid.expect("winners hold pid-bearing entries only");
        // A registry entry recorded with an empty cwd would otherwise be
        // immortal: this supplemented session (carrying the empty cwd) is
        // what the next hook re-records, so the bad value round-trips
        // forever, starving transcript resolution and the terminal-switch
        // cwd fallback. Repair it from the live process; the next registry
        // write then persists the repaired value. A recorded non-empty cwd
        // is never second-guessed.
        let cwd = if entry.cwd.is_empty() {
            resolve_cwd(pid).unwrap_or_default()
        } else {
            entry.cwd
        };
        sessions.push(ClaudeSession::from_raw(RawSession {
            pid,
            session_id: entry.session_id,
            cwd,
            started_at: entry.started_at_ms,
            name: entry.name,
            name_source: None,
        }));
    }
}

/// The restore-registry slice matching the current scan's scope: the current
/// sandbox's slice when running inside a sandbox, else the host-local registry.
fn registry_slice_for_current_scope() -> Vec<crate::sandbox_registry::SessionEntry> {
    match crate::sandbox_registry::current_sandbox() {
        Some(name) => crate::sandbox_registry::load()
            .sandboxes
            .remove(&name)
            .unwrap_or_default(),
        None => crate::sandbox_registry::load_local().sessions,
    }
}

/// Canonical transcript path for a session:
/// `~/.claude/projects/<cwd-slug>/<session-id>.jsonl`. Recorded in the restore
/// registry so restore can check resumability.
pub fn transcript_path(session_id: &str, cwd: &str) -> PathBuf {
    projects_dir()
        .join(cwd_to_slug(cwd))
        .join(format!("{session_id}.jsonl"))
}

pub fn scan_sessions() -> Vec<ClaudeSession> {
    let dir = sessions_dir();
    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut raw_sessions = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }

        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                crate::logger::log(
                    "WARN",
                    &format!("session file read error: {}: {e}", path.display()),
                );
                continue;
            }
        };

        match serde_json::from_str::<RawSession>(&content) {
            Ok(raw) => raw_sessions.push(raw),
            Err(e) => {
                crate::logger::log(
                    "WARN",
                    &format!("session file parse error: {}: {e}", path.display()),
                );
            }
        }
    }

    // JSONL paths are resolved later by resolve_jsonl_paths() after
    // command_args are populated; supplemented sessions get status/CPU from
    // the same later passes, so re-added sessions show with real data.
    snapshot_and_assemble(
        raw_sessions,
        crate::process::live_claude_procs,
        registry_slice_for_current_scope(),
        &pid_alive,
        &proc_cwd,
        &transcript_session_name,
        DeadPointers::Keep,
    )
}

/// Third discovery source, after the pointer scan and the registry
/// supplement: any live `claude` process in this namespace that neither
/// source covered. This is the backstop that survives the worst case —
/// Claude Code deleted the pointer file mid-session AND the registry entry
/// is gone (e.g. an old claudectl's replace-style sandbox slice write
/// dropped it) — the session still shows, and the next registry write
/// re-records it, healing the registry from the process table.
///
/// Identity: the `--resume <uuid>` argument when present (the id the process
/// was resumed under — also how later passes find its transcript); sessions
/// launched without `--resume` whose pointer is already gone get an empty
/// session_id, which the registry writer skips (nothing to `--resume` at
/// restore time) but the display shows normally.
///
/// Name: `resolve_name(session_id, cwd)` — production reads the transcript's
/// last `custom-title`/`agent-name` record. This is the moment a session's
/// registry entry was just lost (that's the only way a `--resume` session
/// lands here), so whatever name we record now is what the registry keeps;
/// without recovery it would be recorded name-less and the title lost until
/// the session restarts. Only consulted for sessions with a real uuid — an
/// empty session_id has no locatable transcript and is skipped by the
/// registry writer anyway. One-shot by construction: the next hook write puts
/// the session back in the registry, after which the registry supplement
/// covers it and this pass no longer sees it.
fn extend_with_ps_live(
    sessions: &mut Vec<ClaudeSession>,
    procs: Option<&std::collections::HashMap<u32, crate::process::LiveClaudeProc>>,
    resolve_cwd: &impl Fn(u32) -> Option<String>,
    resolve_name: &impl Fn(&str, &str) -> Option<String>,
) {
    let Some(procs) = procs else { return };
    let seen_pids: HashSet<u32> = sessions.iter().map(|s| s.pid).collect();
    // Session ids already claimed — by covered sessions AND by rows added in
    // this very loop. A `--resume` uuid can legitimately appear on several
    // live processes (duplicate resume waves happen); only the FIRST claimant
    // gets it as identity. Later claimants still get a visible row (a live
    // claude process must never be hidden) but with an empty id, so the
    // registry — keyed by session id — records exactly one entry per uuid and
    // restore can never double-spawn it.
    let mut seen_ids: HashSet<String> = sessions
        .iter()
        .map(|s| s.session_id.clone())
        .filter(|id| !id.is_empty())
        .collect();

    // Sort for deterministic output order across scans.
    let mut uncovered: Vec<(&u32, &crate::process::LiveClaudeProc)> = procs
        .iter()
        .filter(|(pid, _)| !seen_pids.contains(pid))
        .collect();
    uncovered.sort_by_key(|(pid, _)| **pid);

    let mut found = Vec::new();
    for (&pid, proc_info) in uncovered {
        let session_id = extract_resume_uuid(&proc_info.args)
            .filter(|token| crate::process::looks_like_uuid(token))
            .filter(|id| seen_ids.insert(id.clone()))
            .unwrap_or_default();
        let cwd = resolve_cwd(pid).unwrap_or_default();
        let name = if session_id.is_empty() {
            None
        } else {
            resolve_name(&session_id, &cwd)
        };
        let mut session = ClaudeSession::from_raw(RawSession {
            pid,
            session_id,
            cwd,
            started_at: proc_info.started_at_ms,
            name,
            name_source: None,
        });
        session.command_args = proc_info.args.clone();
        found.push(session);
    }
    sessions.extend(found);
}

/// Production `resolve_name` for [`extend_with_ps_live`]: the transcript's
/// own name record. See [`crate::transcript::last_session_name`].
fn transcript_session_name(session_id: &str, cwd: &str) -> Option<String> {
    crate::transcript::last_session_name(&transcript_path(session_id, cwd))
}

/// Working directory of a live process, in the caller's namespace.
#[cfg(target_os = "linux")]
fn proc_cwd(pid: u32) -> Option<String> {
    fs::read_link(format!("/proc/{pid}/cwd"))
        .ok()
        .map(|p| p.to_string_lossy().into_owned())
}

/// Working directory of a live process via `lsof` (macOS has no /proc).
/// Invoked only for sessions BOTH the pointer scan and the registry missed —
/// zero in the usual steady state; ~50ms per scan per session stuck in that
/// double-miss state (possible for a session with no known `--resume` id,
/// which the registry writer skips).
///
/// The absolute `/usr/sbin/lsof` comes first: with `env_clear()` there is no
/// `PATH`, and execvp's fallback default path (`/usr/bin:/bin`) does not
/// include `/usr/sbin`, where macOS installs lsof — a bare `"lsof"` never
/// resolves (verified empirically). The bare name is the fallback for other
/// Unixes that land here with lsof elsewhere on the default path.
#[cfg(not(target_os = "linux"))]
fn proc_cwd(pid: u32) -> Option<String> {
    let pid_arg = pid.to_string();
    let args = ["-a", "-p", pid_arg.as_str(), "-d", "cwd", "-Fn"];
    let output = std::process::Command::new("/usr/sbin/lsof")
        .args(args)
        .env_clear()
        .output()
        .or_else(|_| {
            std::process::Command::new("lsof")
                .args(args)
                .env_clear()
                .output()
        })
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .find_map(|line| line.strip_prefix('n'))
        .map(str::to_string)
}

/// Resolve JSONL paths for sessions. Must be called AFTER command_args are populated
/// (i.e., after fetch_ps_data), so we can use --resume UUIDs for correct mapping.
pub fn resolve_jsonl_paths(sessions: &mut [ClaudeSession]) {
    for session in sessions.iter_mut() {
        let slug = cwd_to_slug(&session.cwd);
        let project_dir = projects_dir().join(&slug);

        // Priority 1: Try the session's own ID in the expected project dir
        let own_path = project_dir.join(format!("{}.jsonl", session.session_id));
        if own_path.exists() {
            session.jsonl_path = Some(own_path);
            continue;
        }

        // Priority 2: Try the --resume UUID from command args
        if let Some(resume_id) = extract_resume_uuid(&session.command_args) {
            let resume_path = project_dir.join(format!("{resume_id}.jsonl"));
            if resume_path.exists() {
                session.jsonl_path = Some(resume_path);
                continue;
            }
        }

        // Priority 3: Search ALL project directories for a JSONL matching the
        // session ID. This handles cwd encoding mismatches between claudectl
        // and Claude Code (e.g., symlink resolution, path normalization
        // differences).
        if !session.session_id.is_empty()
            && let Some(found) = search_all_projects_for_session(&session.session_id)
        {
            crate::logger::log(
                "DEBUG",
                &format!(
                    "session {}: slug mismatch — found JSONL via project scan: {}",
                    session.session_id,
                    found.display()
                ),
            );
            session.jsonl_path = Some(found);
            continue;
        }

        // Priority 4 (last resort, identity-less rows only): a project dir
        // holding EXACTLY ONE transcript. "Most recently modified .jsonl"
        // used to stand here; in a cwd shared by many sessions — even across
        // machines, since ~/.claude is one shared mount — it wired rows onto
        // whichever transcript happened to be written last, stealing that
        // session's telemetry AND (through the monitor's title recovery) its
        // name (2026-07-28). Rows WITH a session id never guess: their own
        // file is the only correct answer (priorities 1-3), and since a
        // bound path is final for the row's lifetime, binding a foreign file
        // while their own doesn't exist yet (fresh rotation) would be a
        // permanent mis-wire. They stay unresolved and retry every pass
        // until their file appears. An id-less row is itself writing a
        // transcript in this dir, so a dir with exactly one is its own —
        // modulo the first seconds before the file exists, which is why
        // this stays last.
        if session.session_id.is_empty()
            && let Some(only) = sole_jsonl(&project_dir)
        {
            session.jsonl_path = Some(only);
            continue;
        }

        crate::logger::log(
            "DEBUG",
            &format!(
                "session {}: no JSONL found (slug={}, project_dir_exists={})",
                session.session_id,
                slug,
                project_dir.exists()
            ),
        );
    }
}

/// Search all directories under ~/.claude/projects/ for a JSONL file matching the session ID.
/// This is a fallback when the cwd-based slug doesn't match the actual directory on disk.
fn search_all_projects_for_session(session_id: &str) -> Option<PathBuf> {
    let filename = format!("{session_id}.jsonl");
    let base = projects_dir();
    let entries = fs::read_dir(&base).ok()?;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let candidate = path.join(&filename);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Extract the UUID from a --resume argument in command args.
fn extract_resume_uuid(command_args: &str) -> Option<String> {
    let marker = "--resume ";
    let start = command_args.find(marker)? + marker.len();
    let rest = &command_args[start..];
    // Take until whitespace — could be a UUID or a named session
    let token: String = rest.chars().take_while(|c| !c.is_whitespace()).collect();
    if token.is_empty() {
        return None;
    }
    // Strip surrounding quotes
    let token = token.trim_matches('"').trim_matches('\'');
    Some(token.to_string())
}

/// The project directory's transcript, iff it holds exactly one. More than
/// one candidate means any pick is a guess — see the Priority-4 comment in
/// [`resolve_jsonl_paths`].
fn sole_jsonl(dir: &PathBuf) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut only: Option<PathBuf> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        if only.is_some() {
            return None;
        }
        only = Some(path);
    }
    only
}

/// Feature #29: Scan for subagent task .jsonl files.
/// Claude Code spawns sub-agents whose files live in:
///   /tmp/claude-{uid}/{project_slug}/{sessionId}/tasks/
pub fn scan_subagents(sessions: &mut [ClaudeSession]) {
    let uid = unsafe { libc::getuid() };
    let tmp_base = PathBuf::from(format!("/tmp/claude-{uid}"));

    if !tmp_base.exists() {
        for session in sessions.iter_mut() {
            session.active_subagent_count = 0;
            session.active_subagent_jsonl_paths.clear();
        }
        return;
    }

    for session in sessions.iter_mut() {
        let slug = cwd_to_slug(&session.cwd);
        let tasks_dir = tmp_base.join(&slug).join(&session.session_id).join("tasks");

        if !tasks_dir.exists() {
            session.active_subagent_count = 0;
            session.active_subagent_jsonl_paths.clear();
            continue;
        }

        let mut jsonls = Vec::new();
        collect_subagent_jsonls(&tasks_dir, &mut jsonls);
        jsonls.sort();
        session.active_subagent_count = jsonls.len();
        session.active_subagent_jsonl_paths = jsonls;
    }
}

fn collect_subagent_jsonls(dir: &PathBuf, jsonls: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_subagent_jsonls(&path, jsonls);
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
            jsonls.push(path);
        }
    }
}

/// Resolve git worktree identity for each session (for conflict detection).
/// Sessions in different worktrees of the same repo get different IDs.
/// Runs `git rev-parse --show-toplevel` once per unique cwd.
pub fn resolve_worktree_ids(sessions: &mut [ClaudeSession]) {
    // Cache results to avoid running git multiple times for the same cwd
    let mut cache: std::collections::HashMap<String, String> = std::collections::HashMap::new();

    for session in sessions.iter_mut() {
        if session.worktree_id.is_some() {
            continue;
        }
        let id = if let Some(cached) = cache.get(&session.cwd) {
            cached.clone()
        } else {
            let resolved = std::process::Command::new("git")
                .args(["rev-parse", "--show-toplevel"])
                .current_dir(&session.cwd)
                .output()
                .ok()
                .and_then(|o| {
                    if o.status.success() {
                        String::from_utf8(o.stdout)
                            .ok()
                            .map(|s| s.trim().to_string())
                    } else {
                        None
                    }
                })
                // Fall back to cwd if not a git repo
                .unwrap_or_else(|| session.cwd.clone());
            cache.insert(session.cwd.clone(), resolved.clone());
            resolved
        };
        session.worktree_id = Some(id);
    }
}

fn cwd_to_slug(cwd: &str) -> String {
    let trimmed = cwd.trim_end_matches('/');
    if trimmed.is_empty() {
        return "-".to_string();
    }
    trimmed.replace('/', "-")
}

/// Whether `pid` is a running process. `kill(pid, 0)` returns 0 for a live
/// process; pid 0 is rejected since it would signal the whole process group.
/// Is a process with this pid running right now? (`kill -0`; start-time-blind.)
pub(crate) fn pid_alive(pid: u32) -> bool {
    pid != 0 && unsafe { libc::kill(pid as libc::pid_t, 0) } == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pid_alive_true_for_self_false_for_zero_and_bogus() {
        assert!(pid_alive(std::process::id()));
        assert!(!pid_alive(0));
        // A very high pid is almost certainly not a running process.
        assert!(!pid_alive(2_000_000_000));
    }

    fn entry(session_id: &str, pid: Option<u32>) -> crate::sandbox_registry::SessionEntry {
        crate::sandbox_registry::SessionEntry {
            session_id: session_id.to_string(),
            cwd: "/Users/ndr/work".to_string(),
            transcript: String::new(),
            started_at_ms: 1_784_900_000_000,
            name: Some(format!("name-{session_id}")),
            pid,
            owner_pid: None,
            owner_started_at: None,
        }
    }

    fn session(session_id: &str, pid: u32) -> ClaudeSession {
        ClaudeSession::from_raw(RawSession {
            pid,
            session_id: session_id.to_string(),
            cwd: "/Users/ndr/work".to_string(),
            started_at: 1_784_900_000_000,
            name: None,
            name_source: None,
        })
    }

    #[test]
    fn merge_adds_alive_pointerless_registry_session() {
        let mut sessions = Vec::new();
        merge_pointerless_live(
            &mut sessions,
            vec![entry("s1", Some(4242))],
            &|_| true,
            &|_| None,
        );
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, "s1");
        assert_eq!(sessions[0].pid, 4242);
        // Recorded fields survive the SessionEntry -> RawSession -> ClaudeSession hop.
        assert_eq!(sessions[0].session_name, "name-s1");
        assert_eq!(sessions[0].cwd, "/Users/ndr/work");
    }

    #[test]
    fn merge_dedups_against_sessions_already_from_the_pointer_scan() {
        // s1 is already present (its pointer file was found); the registry copy
        // must not double-add it, even though it too is alive.
        let mut sessions = vec![session("s1", 111)];
        merge_pointerless_live(
            &mut sessions,
            vec![entry("s1", Some(111)), entry("s2", Some(222))],
            &|_| true,
            &|_| None,
        );
        let ids: Vec<&str> = sessions.iter().map(|s| s.session_id.as_str()).collect();
        assert_eq!(ids, vec!["s1", "s2"], "s1 kept once, s2 added");
    }

    #[test]
    fn merge_repairs_empty_recorded_cwd_from_live_process() {
        // Regression: a registry entry once recorded with an empty cwd
        // round-tripped forever (supplement → re-record → supplement…),
        // permanently starving transcript resolution and the terminal-switch
        // cwd fallback. The supplement must repair it from the live process —
        // and never second-guess a non-empty recorded cwd.
        let mut sessions = Vec::new();
        let mut empty_cwd = entry("repair-me", Some(100));
        empty_cwd.cwd = String::new();
        let recorded = entry("keep-me", Some(200)); // cwd "/Users/ndr/work"
        merge_pointerless_live(
            &mut sessions,
            vec![empty_cwd, recorded],
            &|_| true,
            &|pid| (pid == 100).then(|| "/Users/ndr/live".to_string()),
        );
        assert_eq!(sessions[0].cwd, "/Users/ndr/live", "empty cwd repaired");
        assert_eq!(
            sessions[1].cwd, "/Users/ndr/work",
            "recorded cwd not second-guessed"
        );
    }

    #[test]
    fn merge_skips_dead_pids_and_missing_pids() {
        let mut sessions = Vec::new();
        let alive = 4242u32;
        merge_pointerless_live(
            &mut sessions,
            vec![
                entry("dead", Some(9001)),
                entry("no-pid", None),
                entry("alive", Some(alive)),
            ],
            &move |pid| pid == alive,
            &|_| None,
        );
        let ids: Vec<&str> = sessions.iter().map(|s| s.session_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["alive"],
            "only the alive, pid-bearing entry is added"
        );
    }

    fn proc_map(
        entries: &[(u32, &str)],
    ) -> std::collections::HashMap<u32, crate::process::LiveClaudeProc> {
        entries
            .iter()
            .map(|(pid, args)| {
                (
                    *pid,
                    crate::process::LiveClaudeProc {
                        started_at_ms: 1_784_900_000_000,
                        args: args.to_string(),
                    },
                )
            })
            .collect()
    }

    #[test]
    fn ps_extend_adds_uncovered_claude_proc_with_resume_uuid_as_id() {
        let mut sessions = Vec::new();
        let uuid = "3c20ad09-d564-429d-868c-d3a67dcace79";
        let procs = proc_map(&[(
            4084,
            &format!("--dangerously-skip-permissions --resume {uuid}"),
        )]);
        extend_with_ps_live(&mut sessions, Some(&procs), &|_| None, &|_, _| None);
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].pid, 4084);
        assert_eq!(sessions[0].session_id, uuid);
        assert_eq!(sessions[0].started_at, 1_784_900_000_000);
        assert!(
            sessions[0].command_args.contains("--resume"),
            "args stored so later passes can resolve the transcript"
        );
    }

    #[test]
    fn ps_extend_dedups_by_pid_and_demotes_covered_session_ids_to_empty() {
        let uuid = "3c20ad09-d564-429d-868c-d3a67dcace79";
        // s1 covers pid 111; the registry-supplemented session covers `uuid`
        // under a different pid (session id rotation keeps the same process).
        let mut sessions = vec![session("s1", 111), session(uuid, 222)];
        let procs = proc_map(&[
            (111, "--resume other-name"),
            (333, &format!("--resume {uuid}")),
            (444, "some prompt text"),
        ]);
        extend_with_ps_live(&mut sessions, Some(&procs), &|_| None, &|_, _| None);
        let rows: Vec<(u32, &str)> = sessions
            .iter()
            .map(|s| (s.pid, s.session_id.as_str()))
            .collect();
        assert_eq!(
            rows,
            vec![(111, "s1"), (222, uuid), (333, ""), (444, "")],
            "pid 111 deduped by pid; 333 stays visible but cannot claim an id \
             another live session already holds; 444 added with unknown id"
        );
    }

    #[test]
    fn ps_extend_gives_duplicate_resume_uuids_to_only_the_first_proc() {
        // Duplicate resume waves leave several live `claude --resume X`
        // processes. Only one may carry X as identity, or the registry gets
        // two id-X entries and restore double-spawns X.
        let uuid = "3c20ad09-d564-429d-868c-d3a67dcace79";
        let mut sessions = Vec::new();
        let args = format!("--resume {uuid}");
        let procs = proc_map(&[(100, &args), (200, &args)]);
        extend_with_ps_live(&mut sessions, Some(&procs), &|_| None, &|_, _| None);
        let ids: Vec<&str> = sessions.iter().map(|s| s.session_id.as_str()).collect();
        assert_eq!(
            ids,
            vec![uuid, ""],
            "first proc claims the uuid, the second stays visible without it"
        );
    }

    #[test]
    fn ps_extend_without_resume_uuid_gets_empty_session_id() {
        let mut sessions = Vec::new();
        // `--resume my-named-session` is a name, not a uuid — identity unknown.
        let procs = proc_map(&[(555, "--resume my-named-session")]);
        extend_with_ps_live(&mut sessions, Some(&procs), &|_| None, &|_, _| None);
        assert_eq!(sessions.len(), 1);
        assert_eq!(
            sessions[0].session_id, "",
            "non-uuid resume token must not be recorded as a session id"
        );
    }

    #[test]
    fn ps_extend_is_deterministic_and_noop_without_snapshot() {
        let mut sessions = Vec::new();
        extend_with_ps_live(&mut sessions, None, &|_| None, &|_, _| None);
        assert!(sessions.is_empty(), "no ps snapshot -> no additions");

        let procs = proc_map(&[(30, ""), (10, ""), (20, "")]);
        extend_with_ps_live(&mut sessions, Some(&procs), &|_| None, &|_, _| None);
        let pids: Vec<u32> = sessions.iter().map(|s| s.pid).collect();
        assert_eq!(pids, vec![10, 20, 30], "added in sorted pid order");
    }

    // ------------------------------------------------------------------
    // Composition-seam regression scenarios. Each replays a real incident
    // (2026-07-24) end-to-end through assemble_sessions — the pipeline where
    // pointer files, the ps snapshot, and the registry meet. If any of these
    // fail, do NOT weaken the assertion: each one guards a way sessions were
    // actually lost, leaked, or frozen in production.
    // ------------------------------------------------------------------

    fn raw_pointer(session_id: &str, pid: u32) -> RawSession {
        RawSession {
            pid,
            session_id: session_id.into(),
            cwd: "/Users/ndr".into(),
            started_at: 1_784_900_000_000,
            name: None,
            name_source: None,
        }
    }

    fn assembled_ids(sessions: &[ClaudeSession]) -> Vec<(u32, String)> {
        sessions
            .iter()
            .map(|s| (s.pid, s.session_id.clone()))
            .collect()
    }

    #[test]
    fn regression_one_row_per_live_pid_across_rotated_sids() {
        // 2026-07-28: Claude Code rotated sessionId under a running process
        // (/clear), the registry held one entry per sid all claiming the same
        // live pid, and the pointerless re-add (sid-deduped only) turned one
        // tab into multiple rows — each then stealing a different transcript.
        // Same started_at (one process) and production file order (both
        // writers emit freshest-first): the FIRST record is the current
        // conversation and must win the row.
        let current = entry("bbbb-current", Some(91));
        let superseded = entry("aaaa-superseded", Some(91));
        let procs = proc_map(&[(91, "")]);
        let sessions = assemble_sessions(
            Vec::new(),
            Some(&procs),
            vec![current, superseded],
            &|_| false,
            &|_| Some("/Users/ndr".into()),
            &|_, _| None,
            DeadPointers::Drop,
        );
        assert_eq!(
            assembled_ids(&sessions),
            vec![(91, "bbbb-current".to_string())],
            "one live pid must yield exactly one row — the current conversation"
        );
    }

    #[test]
    fn newest_started_entry_wins_the_pid_regardless_of_record_order() {
        // A strictly newer started_at outranks file position: a slice written
        // by an old binary can hold the fresher conversation later in the
        // file, and record order alone must not pick the stale one.
        let mut stale = entry("aaaa-stale", Some(93));
        stale.started_at_ms = 1_000;
        let mut fresh = entry("bbbb-fresh", Some(93));
        fresh.started_at_ms = 2_000;
        let procs = proc_map(&[(93, "")]);
        let sessions = assemble_sessions(
            Vec::new(),
            Some(&procs),
            vec![stale, fresh],
            &|_| false,
            &|_| Some("/Users/ndr".into()),
            &|_, _| None,
            DeadPointers::Drop,
        );
        assert_eq!(
            assembled_ids(&sessions),
            vec![(93, "bbbb-fresh".to_string())]
        );
    }

    #[test]
    fn pointer_row_blocks_registry_readd_for_the_same_pid() {
        // A pointer-backed row already represents the process; a superseded
        // sid's registry entry with the same pid must not add a second row.
        let raw = raw_pointer("cccc-pointer", 92);
        let stale = entry("dddd-superseded", Some(92));
        let procs = proc_map(&[(92, "")]);
        let sessions = assemble_sessions(
            vec![raw],
            Some(&procs),
            vec![stale],
            &|_| false,
            &|_| Some("/Users/ndr".into()),
            &|_, _| None,
            DeadPointers::Drop,
        );
        assert_eq!(
            assembled_ids(&sessions),
            vec![(92, "cccc-pointer".to_string())]
        );
    }

    #[test]
    fn regression_sid_bearing_row_never_binds_a_foreign_sole_transcript() {
        // 2026-07-28 review finding: a bound jsonl_path is final for the
        // row's lifetime, so a row WITH a session id must never bind the
        // dir's sole transcript while its own file doesn't exist yet (fresh
        // rotation) — that would be a permanent mis-wire. It stays
        // unresolved and binds its own file once it appears. An id-less row
        // may bind the sole file (it has no better identity), and a row
        // whose file lives under a different slug must find it via the
        // all-projects sid search BEFORE any last resort.
        let _lock = crate::sandbox_registry::tests::env_guard();
        let home = tempfile::tempdir().unwrap();
        let saved_home = std::env::var_os("HOME");
        // SAFETY: env access serialized by the held env lock.
        unsafe {
            std::env::set_var("HOME", home.path());
        }

        let shared = home.path().join(".claude/projects/-Users-ndr");
        std::fs::create_dir_all(&shared).unwrap();
        std::fs::write(shared.join("aaaa-foreign.jsonl"), "{}").unwrap();
        let other_slug = home.path().join(".claude/projects/-Users-ndr-work");
        std::fs::create_dir_all(&other_slug).unwrap();
        std::fs::write(other_slug.join("cccc-elsewhere.jsonl"), "{}").unwrap();

        let make = |sid: &str| {
            ClaudeSession::from_raw(RawSession {
                pid: 1,
                session_id: sid.into(),
                cwd: "/Users/ndr".into(),
                started_at: 0,
                name: None,
                name_source: None,
            })
        };
        let mut sessions = vec![make("bbbb-mine"), make(""), make("cccc-elsewhere")];
        resolve_jsonl_paths(&mut sessions);

        // SAFETY: same lock still held.
        unsafe {
            match &saved_home {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
        }

        assert_eq!(
            sessions[0].jsonl_path, None,
            "an identity-bearing row must not steal the sole foreign file"
        );
        assert_eq!(
            sessions[1].jsonl_path,
            Some(shared.join("aaaa-foreign.jsonl")),
            "an id-less row may bind the dir's sole transcript"
        );
        assert_eq!(
            sessions[2].jsonl_path,
            Some(other_slug.join("cccc-elsewhere.jsonl")),
            "the sid search across project dirs outranks the last resort"
        );
    }

    #[test]
    fn sole_jsonl_requires_exactly_one_candidate() {
        // The last-resort transcript association must never guess between
        // candidates — "newest in dir" cross-wired rows onto other sessions'
        // transcripts (2026-07-28), including across machines via the shared
        // ~/.claude mount.
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.path().to_path_buf();
        assert_eq!(sole_jsonl(&dir_path), None, "empty dir: nothing to bind");
        std::fs::write(dir_path.join("a.jsonl"), "{}").unwrap();
        assert_eq!(
            sole_jsonl(&dir_path),
            Some(dir_path.join("a.jsonl")),
            "a single candidate is unambiguous"
        );
        std::fs::write(dir_path.join("b.jsonl"), "{}").unwrap();
        assert_eq!(
            sole_jsonl(&dir_path),
            None,
            "two candidates: refuse to guess"
        );
    }

    #[test]
    fn regression_sessions_survive_total_pointer_and_registry_loss() {
        // Incident: CC deleted every pointer mid-life AND an old binary's
        // replace-style slice writes destroyed the registry — 30 live
        // sessions became invisible and unrestorable. The process table must
        // recover them, with `--resume` uuids as identity.
        let uuid = "3c20ad09-d564-429d-868c-d3a67dcace79";
        let procs = proc_map(&[(4084, &format!("--resume {uuid}"))]);
        let sessions = assemble_sessions(
            Vec::new(),
            Some(&procs),
            Vec::new(),
            &|_| false,
            &|_| Some("/Users/ndr".into()),
            &|_, _| None,
            DeadPointers::Drop,
        );
        assert_eq!(assembled_ids(&sessions), vec![(4084, uuid.to_string())]);
        assert_eq!(sessions[0].cwd, "/Users/ndr");
    }

    #[test]
    fn regression_ps_readd_recovers_title_from_transcript() {
        // 2026-07-28 title-blanking, the recovery half: when a session lands
        // in the ps supplement its registry entry was JUST lost, and whatever
        // this pass reports is what the next hook write records. The name must
        // be recovered from the transcript (the only surviving source), not
        // recorded as None.
        let uuid = "3c20ad09-d564-429d-868c-d3a67dcace79";
        let procs = proc_map(&[(4084, &format!("--resume {uuid}"))]);
        let sessions = assemble_sessions(
            Vec::new(),
            Some(&procs),
            Vec::new(), // registry slice: entry already pruned
            &|_| false,
            &|_| Some("/Users/ndr".into()),
            &|id, cwd| {
                assert_eq!(id, uuid);
                assert_eq!(cwd, "/Users/ndr");
                Some("resume-old-sessions-audit".into())
            },
            DeadPointers::Drop,
        );
        assert_eq!(
            sessions[0].session_name, "resume-old-sessions-audit",
            "ps rediscovery must recover the title from the transcript"
        );
    }

    #[test]
    fn regression_derived_placeholder_assembles_as_unnamed() {
        // 2026-07-28: Claude Code recreated a resumed session's pointer with
        // a fresh derived placeholder ("ndr-5e", nameSource:"derived").
        // Assembled verbatim it became the displayed title, and — as a Some
        // name — the next hook write overwrote the stored real /rename title
        // with it. Assembled as unnamed, the hook write records None and
        // backfill_missing_names keeps the stored title instead.
        let uuid = "d74ca77e-09ba-42cc-a148-290b6ed2ac98";
        let mut raw = raw_pointer(uuid, 2647641);
        raw.name = Some("ndr-5e".into());
        raw.name_source = Some("derived".into());
        let procs = proc_map(&[(2647641, &format!("--resume {uuid}"))]);
        let sessions = assemble_sessions(
            vec![raw],
            Some(&procs),
            Vec::new(),
            &|_| false,
            &|_| Some("/Users/ndr".into()),
            &|_, _| None,
            DeadPointers::Drop,
        );
        assert_eq!(sessions.len(), 1);
        assert!(
            sessions[0].session_name.is_empty(),
            "a derived placeholder must never assemble into a title"
        );
    }

    #[test]
    fn ps_readd_without_uuid_never_probes_for_a_name() {
        // No `--resume` uuid ⇒ no locatable transcript, and the registry
        // writer skips the session anyway. The resolver must not run — in
        // production it opens files, and this pass repeats every tick for
        // uuid-less sessions.
        let procs = proc_map(&[(77, "")]);
        let mut sessions = Vec::new();
        extend_with_ps_live(&mut sessions, Some(&procs), &|_| None, &|_, _| {
            panic!("resolve_name must not be called for an empty session_id")
        });
        assert_eq!(sessions.len(), 1);
        assert!(sessions[0].session_name.is_empty());
    }

    #[test]
    fn regression_pointerless_sessions_recover_from_registry() {
        // Incident: CC deleted pointers mid-life; the registry still knew the
        // sessions. They must appear with their recorded identity and never
        // be double-added by the ps supplement.
        let procs = proc_map(&[(111, ""), (222, "")]);
        let sessions = assemble_sessions(
            Vec::new(),
            Some(&procs),
            vec![entry("s1", Some(111)), entry("s2", Some(222))],
            &|_| false,
            &|_| None,
            &|_, _| None,
            DeadPointers::Drop,
        );
        assert_eq!(
            assembled_ids(&sessions),
            vec![(111, "s1".into()), (222, "s2".into())],
            "registry identities kept, ps supplement adds no duplicates"
        );
    }

    #[test]
    fn regression_foreign_pointer_never_enters_live_set_but_dead_ones_tombstone() {
        // Incident: a host session's pointer, visible in-sandbox, collided
        // with an unrelated in-namespace pid and got captured into the
        // sandbox registry. Alive-but-not-claude pointer pids must be
        // excluded everywhere; genuinely dead pids must still tombstone in
        // the display variant only.
        let pointers = vec![raw_pointer("foreign", 3824), raw_pointer("dead", 9001)];
        let procs = proc_map(&[]);
        let alive_probe = |pid: u32| pid == 3824; // 3824 alive (not claude), 9001 dead

        let live = assemble_sessions(
            pointers.clone(),
            Some(&procs),
            Vec::new(),
            &alive_probe,
            &|_| None,
            &|_, _| None,
            DeadPointers::Drop,
        );
        assert!(
            live.is_empty(),
            "live set: foreign excluded, dead excluded; got {:?}",
            assembled_ids(&live)
        );

        let display = assemble_sessions(
            pointers,
            Some(&procs),
            Vec::new(),
            &alive_probe,
            &|_| None,
            &|_, _| None,
            DeadPointers::Keep,
        );
        assert_eq!(
            assembled_ids(&display),
            vec![(9001, "dead".into())],
            "display: dead pointer tombstones, foreign one never shows"
        );
    }

    #[test]
    fn regression_ps_failure_never_shrinks_the_scan() {
        // A transient `ps` failure must degrade liveness to the injected
        // probe (kill -0 in production), not lose sessions — and must not
        // fabricate ps-discovered rows either.
        let sessions = assemble_sessions(
            vec![raw_pointer("p1", 100)],
            None,
            vec![entry("r1", Some(200))],
            &|pid| pid == 100 || pid == 200,
            &|_| None,
            &|_, _| None,
            DeadPointers::Drop,
        );
        assert_eq!(
            assembled_ids(&sessions),
            vec![(100, "p1".into()), (200, "r1".into())],
            "pointer + registry sessions survive on kill -0 alone"
        );
    }

    #[test]
    fn regression_just_started_session_is_never_invisible_for_a_scan() {
        // Incident-adjacent: the snapshot must be taken AFTER the pointer
        // read (the shared snapshot_and_assemble is the ONLY place either
        // wrapper takes it), so a session that starts mid-scan — pointer
        // missed, pid only in the later snapshot — is still discovered via
        // the ps supplement. The recording closure proves the snapshot is
        // taken lazily, inside the helper, with the pointers already fixed.
        let snapshot_taken = std::cell::Cell::new(false);
        let sessions = snapshot_and_assemble(
            Vec::new(), // pointer read already finished: session 555 missed it
            || {
                snapshot_taken.set(true);
                Some(proc_map(&[(555, "")]))
            },
            Vec::new(),
            &|_| false,
            &|_| Some("/Users/ndr/new".into()),
            &|_, _| None,
            DeadPointers::Keep,
        );
        assert!(snapshot_taken.get(), "helper must take the snapshot itself");
        assert_eq!(assembled_ids(&sessions), vec![(555, String::new())]);
        assert_eq!(sessions[0].cwd, "/Users/ndr/new");
    }

    #[test]
    fn regression_empty_recorded_cwd_is_repaired_in_the_full_pipeline() {
        // Incident: one bad heal recorded 29 sessions with empty cwd; the
        // supplement → re-record loop made it immortal. The assembled session
        // must carry the repaired cwd so the next registry write persists it.
        let procs = proc_map(&[(100, "")]);
        let mut poisoned = entry("poisoned", Some(100));
        poisoned.cwd = String::new();
        let sessions = assemble_sessions(
            Vec::new(),
            Some(&procs),
            vec![poisoned],
            &|_| false,
            &|pid| (pid == 100).then(|| "/Users/ndr".to_string()),
            &|_, _| None,
            DeadPointers::Drop,
        );
        // Identity asserted too: a mutation that DROPS the poisoned entry
        // (instead of repairing it) must fail here, not slip through because
        // a ps-backstop row happened to occupy index 0.
        assert_eq!(
            assembled_ids(&sessions),
            vec![(100, "poisoned".into())],
            "the recorded identity survives the repair"
        );
        assert_eq!(sessions[0].cwd, "/Users/ndr", "cwd repaired at the seam");
    }

    #[test]
    fn regression_duplicate_resume_wave_yields_one_identity_and_full_visibility() {
        // Incident-class: duplicate resume waves leave several live
        // `claude --resume X` processes. All must be visible; exactly one may
        // carry X (registry keyed by id; restore must not double-spawn).
        let uuid = "3c20ad09-d564-429d-868c-d3a67dcace79";
        let args = format!("--resume {uuid}");
        let procs = proc_map(&[(100, &args), (200, &args)]);
        let sessions = assemble_sessions(
            Vec::new(),
            Some(&procs),
            Vec::new(),
            &|_| false,
            &|_| None,
            &|_, _| None,
            DeadPointers::Drop,
        );
        let ids: Vec<&str> = sessions.iter().map(|s| s.session_id.as_str()).collect();
        assert_eq!(ids, vec![uuid, ""], "one identity, both rows visible");
    }

    #[test]
    fn transcript_path_uses_cwd_slug_and_session_id() {
        let path = transcript_path("abc-123", "/Users/ndr/work");
        let rendered = path.to_string_lossy();
        assert!(
            rendered.ends_with("/projects/-Users-ndr-work/abc-123.jsonl"),
            "{rendered}"
        );
    }

    #[test]
    fn slug_basic_path() {
        assert_eq!(cwd_to_slug("/Users/foo/bar"), "-Users-foo-bar");
    }

    #[test]
    fn slug_trailing_slash() {
        // Must strip trailing slash — otherwise slug ends with "-" and won't match disk
        assert_eq!(
            cwd_to_slug("/Users/foo/bar/"),
            "-Users-foo-bar",
            "trailing slash must be stripped before slugifying"
        );
    }

    #[test]
    fn slug_multiple_trailing_slashes() {
        assert_eq!(cwd_to_slug("/Users/foo/bar///"), "-Users-foo-bar");
    }

    #[test]
    fn slug_with_hyphens_in_name() {
        assert_eq!(
            cwd_to_slug("/Users/dev/data-platform-answers"),
            "-Users-dev-data-platform-answers"
        );
    }

    #[test]
    fn slug_root() {
        assert_eq!(cwd_to_slug("/"), "-");
    }

    #[test]
    fn slug_single_component() {
        assert_eq!(cwd_to_slug("/tmp"), "-tmp");
    }
}
