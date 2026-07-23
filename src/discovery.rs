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
/// session files. A session is a live process, not a timestamp, so liveness is
/// decided only by `kill -0` on its PID.
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
    let mut sessions = Vec::new();
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
        if pid_alive(raw.pid) {
            sessions.push(ClaudeSession::from_raw(raw));
        }
    }
    Some(sessions)
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

    let mut sessions = Vec::new();

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

        let raw: RawSession = match serde_json::from_str(&content) {
            Ok(r) => r,
            Err(e) => {
                crate::logger::log(
                    "WARN",
                    &format!("session file parse error: {}: {e}", path.display()),
                );
                continue;
            }
        };

        // JSONL path resolved later by resolve_jsonl_paths() after command_args are populated
        sessions.push(ClaudeSession::from_raw(raw));
    }

    sessions
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

        // Priority 3: Fall back to most recently modified .jsonl in the project dir
        if let Some(latest) = find_latest_jsonl(&project_dir) {
            session.jsonl_path = Some(latest);
            continue;
        }

        // Priority 4: Search ALL project directories for a JSONL matching the session ID.
        // This handles cwd encoding mismatches between claudectl and Claude Code
        // (e.g., symlink resolution, path normalization differences).
        if let Some(found) = search_all_projects_for_session(&session.session_id) {
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

/// Find the most recently modified .jsonl file in a project directory.
fn find_latest_jsonl(dir: &PathBuf) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut best: Option<(PathBuf, std::time::SystemTime)> = None;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let modified = entry.metadata().ok()?.modified().ok()?;
        if best.as_ref().is_none_or(|(_, t)| modified > *t) {
            best = Some((path, modified));
        }
    }

    best.map(|(p, _)| p)
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
