use crate::session::{ClaudeSession, SessionStatus};
use std::collections::{HashMap, HashSet};

/// Check which PIDs are alive and fetch TTY, CPU%, MEM, command args — all via `ps`.
/// No sysinfo dependency needed.
pub fn fetch_and_enrich(sessions: &mut [ClaudeSession]) {
    if sessions.is_empty() {
        return;
    }

    let pids: Vec<String> = sessions.iter().map(|s| s.pid.to_string()).collect();
    let pid_arg = pids.join(",");

    let output = std::process::Command::new("ps")
        .args(["-o", "pid=,tty=,%cpu=,rss=,command=", "-p", &pid_arg])
        .env_clear()
        .output();

    let output = match output {
        Ok(o) => o,
        Err(e) => {
            crate::logger::log("ERROR", &format!("ps command failed: {e}"));
            // ps failed — mark all as Finished (will show tombstone for 30s)
            for s in sessions.iter_mut() {
                s.status = SessionStatus::Finished;
                s.cpu_percent = 0.0;
            }
            return;
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Build a pid → session-index map once. Replaces the prior O(N²)
    // inner loop that scanned every session for every ps line.
    let pid_to_idx: HashMap<u32, usize> = sessions
        .iter()
        .enumerate()
        .map(|(i, s)| (s.pid, i))
        .collect();
    let mut alive_pids: HashSet<u32> = HashSet::new();

    for line in stdout.lines() {
        let trimmed = line.trim();
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() < 5 {
            continue;
        }
        let Ok(pid) = fields[0].parse::<u32>() else {
            continue;
        };
        let tty = fields[1].to_string();
        let cpu = fields[2].parse::<f32>().unwrap_or(0.0);
        let rss_kb = fields[3].parse::<f64>().unwrap_or(0.0);
        let mem_mb = rss_kb / 1024.0;
        let command = fields[4..].join(" ");

        // Only count this PID as alive if it's actually a claude process.
        // PIDs get reused on macOS — a dead claude session's PID may belong
        // to an unrelated process now. Match on argv0 basename, not a raw
        // substring: `claudectl`, `bash -lc '... claude ...'`, and
        // `grep claude` would all match a substring check.
        if !is_claude_process(&command) {
            continue;
        }

        alive_pids.insert(pid);

        let Some(&idx) = pid_to_idx.get(&pid) else {
            continue;
        };
        let session = &mut sessions[idx];

        // ps tty is invariant per pid (set at exec time, never changes), so
        // overwriting every tick is wasted work and would also clobber the
        // host-tty override below. Set once when empty.
        if session.tty.is_empty() {
            session.tty = tty;
        }
        // The terminal sidecar is written exactly once at session start by
        // sandbox-bootstrap-inner and is invariant for the lifetime of the
        // pid; reading it every tick was 40+ syscalls + JSON parses for no
        // information gain. `sidecar_loaded` flips on the first attempt
        // (success or absence) so we only do the I/O once per session.
        if !session.sidecar_loaded {
            if let Some(s) = read_terminal_sidecar(pid) {
                if let Some(host_tty) = s.host_tty {
                    session.tty = host_tty;
                }
                session.terminal_id = s.terminal_id;
                session.host_terminal_target = s.host_terminal_target;
            }
            session.sidecar_loaded = true;
        }

        // Resolve which terminal this session runs in, once, from its own
        // process environment — so claudectl can switch/input/approve a session
        // that lives in a different terminal than claudectl itself. Host-native
        // only: sandbox sessions carry a sidecar terminal_id/host target and are
        // routed via the bridge, so we leave their `terminal` as None.
        if !session.terminal_resolved {
            if session.terminal_id.is_none() && session.host_terminal_target.is_none() {
                session.terminal = crate::terminals::detect_terminal_for_pid(pid);
            }
            session.terminal_resolved = true;
        }
        session.mem_mb = mem_mb;

        // CPU smoothing: track last 3 readings, use average
        session.cpu_history.push(cpu);
        if session.cpu_history.len() > 3 {
            session.cpu_history.remove(0);
        }
        session.cpu_percent =
            session.cpu_history.iter().sum::<f32>() / session.cpu_history.len() as f32;

        // Extract args (everything after "claude")
        if let Some(idx) = command.find("claude") {
            let after_claude = &command[idx + 6..];
            session.command_args = after_claude.trim().to_string();
        }

        // Extract session name from --name or --resume
        let cmd_parts: Vec<&str> = command.split_whitespace().collect();
        extract_session_meta(&cmd_parts, session);
    }

    // Mark dead PIDs as Finished instead of removing them immediately.
    // They'll be displayed briefly so the user can see what exited.
    for session in sessions.iter_mut() {
        if !alive_pids.contains(&session.pid) {
            session.status = crate::session::SessionStatus::Finished;
            session.cpu_percent = 0.0;
        }
    }
}

/// True iff `command`'s argv0, after stripping any leading path, is exactly
/// `"claude"`. This excludes `claudectl`, `grep claude`, and
/// `bash -lc '... claude ...'`.
fn is_claude_process(command: &str) -> bool {
    claude_argv0_token_count(command).is_some()
}

/// How many whitespace-split tokens of `command` make up a claude argv0 —
/// `Some(n)` when the process is claude, `None` otherwise.
///
/// ps prints argv0 verbatim, so a binary under a space-containing directory
/// (`/Users/x/My Tools/claude`) splits across tokens. When the first token
/// alone isn't `claude` but looks like a path start, prefix tokens are
/// re-joined until the joined string's basename is exactly `claude` — i.e.
/// the component after the LAST `/` must be `claude` alone, so
/// `/usr/bin/env claude` ("env claude") and `grep claude` never match.
fn claude_argv0_token_count(command: &str) -> Option<usize> {
    let basename = |s: &str| s.rsplit('/').next().unwrap_or(s).to_string();
    let mut tokens = command.split_whitespace();
    let first = tokens.next()?;
    if basename(first) == "claude" {
        return Some(1);
    }
    if !(first.starts_with('/') || first.starts_with('~') || first.starts_with('.')) {
        return None;
    }
    let mut joined = String::from(first);
    for (extra, token) in tokens.take(7).enumerate() {
        joined.push(' ');
        joined.push_str(token);
        if basename(&joined) == "claude" {
            return Some(extra + 2);
        }
    }
    None
}

/// A live `claude` process visible in the caller's pid namespace, from one
/// `ps x` snapshot.
pub struct LiveClaudeProc {
    /// Unix epoch ms the process started, derived from ps `etime` (elapsed
    /// time), so no timezone-dependent `lstart` parsing is involved.
    pub started_at_ms: u64,
    /// Everything after argv0 on the command line (`--resume <id>`, prompt…).
    pub args: String,
}

/// Snapshot every live `claude` process in this pid namespace: pid →
/// [`LiveClaudeProc`]. `None` when `ps` itself failed — callers must degrade
/// to their previous liveness source (bare `kill -0`), never to "no sessions".
///
/// This is the ground truth the session scan anchors on: Claude Code deletes
/// its `~/.claude/sessions/<pid>.json` pointer on session-end events that
/// don't kill the process (auto-compact, `/clear`, remote-control attach), so
/// neither pointer files nor the restore registry can be trusted to cover the
/// live set. The process table always does, and reading it in the caller's
/// own namespace also makes the check sandbox-correct: a host pointer file
/// seen through the shared `~/.claude` mount can only pass if its pid is a
/// *claude* process here, not merely any process (pid collisions across
/// namespaces are common).
pub fn live_claude_procs() -> Option<HashMap<u32, LiveClaudeProc>> {
    // `x` (not `ax`): only the invoking user's processes, tty-less included.
    // The pointer scan is scoped to $HOME, so the process-table source must
    // stay scoped to the same user — `ax` would surface other users' claude
    // sessions on a shared machine.
    let output = std::process::Command::new("ps")
        .args(["x", "-o", "pid=,etime=,command="])
        .env_clear()
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut procs = HashMap::new();
    for line in stdout.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            continue;
        }
        let Ok(pid) = fields[0].parse::<u32>() else {
            continue;
        };
        let Some(elapsed_secs) = parse_etime(fields[1]) else {
            continue;
        };
        let command = fields[2..].join(" ");
        let Some(argv0_tokens) = claude_argv0_token_count(&command) else {
            continue;
        };
        let args = command
            .split_whitespace()
            .skip(argv0_tokens)
            .collect::<Vec<_>>()
            .join(" ");
        procs.insert(
            pid,
            LiveClaudeProc {
                started_at_ms: now_ms.saturating_sub(elapsed_secs * 1000),
                args,
            },
        );
    }
    Some(procs)
}

/// Parse ps `etime` (`[[dd-]hh:]mm:ss`) into elapsed seconds.
fn parse_etime(etime: &str) -> Option<u64> {
    let (days, clock) = match etime.split_once('-') {
        Some((d, rest)) => (d.parse::<u64>().ok()?, rest),
        None => (0, etime),
    };
    let parts: Vec<&str> = clock.split(':').collect();
    let (hours, minutes, seconds): (u64, u64, u64) = match parts.as_slice() {
        [h, m, s] => (h.parse().ok()?, m.parse().ok()?, s.parse().ok()?),
        [m, s] => (0, m.parse().ok()?, s.parse().ok()?),
        _ => return None,
    };
    Some(((days * 24 + hours) * 60 + minutes) * 60 + seconds)
}

struct TerminalSidecar {
    host_tty: Option<String>,
    terminal_id: Option<String>,
    /// Per-host-terminal connection target (kitty socket+window id, tmux
    /// socket+pane, wezterm pane id+optional socket). Populated by the
    /// agent-sandbox wrappers when the host runs a Linux desktop terminal.
    /// Absent on macOS-host sandboxes (which use osa-bridge instead) and
    /// on host-native claudectl runs.
    host_terminal_target: Option<crate::session::HostTerminalTarget>,
}

/// Read the per-session terminal sidecar written by the agent sandbox's
/// bootstrap (see tools/agent-sandbox/sbx-template/sandbox-bootstrap-inner).
/// Returns the HOST-side TTY + terminal-application id if present.
///
/// The sidecar lives at $HOME/.claude/sessions/<pid>.terminal.json. Only the
/// agent sandbox writes it; for non-sandbox claude sessions (host-native) the
/// file is absent and this returns None — the regular `ps` TTY stands.
fn read_terminal_sidecar(pid: u32) -> Option<TerminalSidecar> {
    let home = std::env::var_os("HOME")?;
    let path = std::path::PathBuf::from(home)
        .join(".claude")
        .join("sessions")
        .join(format!("{pid}.terminal.json"));
    let body = std::fs::read_to_string(&path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&body).ok()?;
    let trim = |v: &serde_json::Value, key: &str| -> Option<String> {
        v.get(key)
            .and_then(|x| x.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
    };
    let host_terminal_target = parse_host_terminal_target(&value, &trim);
    Some(TerminalSidecar {
        host_tty: trim(&value, "host_tty"),
        terminal_id: trim(&value, "terminal_id"),
        host_terminal_target,
    })
}

/// Pick the host-terminal target out of the sidecar JSON. The agent-sandbox
/// wrappers write whichever set of env vars the host terminal exported:
///   kitty   -> KITTY_WINDOW_ID + KITTY_LISTEN_ON
///   tmux    -> TMUX (socket,N,session-id) + TMUX_PANE
///   wezterm -> WEZTERM_PANE + (optional) WEZTERM_UNIX_SOCKET
/// We probe in that order. If multiple are present we trust kitty first
/// because it is the strongest single signal (a kitty window id is unique
/// even when nested in tmux).
fn parse_host_terminal_target(
    value: &serde_json::Value,
    trim: &dyn Fn(&serde_json::Value, &str) -> Option<String>,
) -> Option<crate::session::HostTerminalTarget> {
    // Sidecar JSON keys are lowercase (matches the existing host_tty /
    // terminal_id / terminal_type convention written by sandbox-bootstrap-inner).
    if let (Some(window_id), Some(listen_on)) = (
        trim(value, "kitty_window_id"),
        trim(value, "kitty_listen_on"),
    ) {
        return Some(crate::session::HostTerminalTarget::Kitty {
            socket: listen_on,
            window_id,
        });
    }
    if let (Some(tmux), Some(pane)) = (trim(value, "tmux"), trim(value, "tmux_pane")) {
        // $TMUX is "<socket>,<server-pid>,<session-id>"; we only need the
        // socket path (first field) for `tmux -S`. tmux(1) "ENVIRONMENT".
        let socket = tmux.split(',').next().unwrap_or(&tmux).to_string();
        return Some(crate::session::HostTerminalTarget::Tmux { socket, pane });
    }
    if let Some(pane_str) = trim(value, "wezterm_pane")
        && let Ok(pane_id) = pane_str.parse::<u64>()
    {
        return Some(crate::session::HostTerminalTarget::WezTerm {
            pane_id,
            unix_socket: trim(value, "wezterm_unix_socket"),
        });
    }
    None
}

fn extract_session_meta(cmd: &[&str], session: &mut ClaudeSession) {
    // If the session JSON already provided a name (via /rename or auto-name),
    // don't overwrite it from the process command line.
    let name_already_set = !session.session_name.is_empty();
    let mut i = 0;
    while i < cmd.len() {
        match cmd[i] {
            "--name" | "-n" if i + 1 < cmd.len() => {
                if !name_already_set {
                    session.session_name = cmd[i + 1].to_string();
                }
                i += 2;
                continue;
            }
            "--resume" | "-r" if i + 1 < cmd.len() => {
                let val = cmd[i + 1];
                if !name_already_set && !looks_like_uuid(val) {
                    session.session_name = val.to_string();
                }
                i += 2;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
}

pub(crate) fn looks_like_uuid(s: &str) -> bool {
    s.len() == 36
        && s.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
        && s.matches('-').count() == 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_claude_process_matches_bare_argv0() {
        assert!(is_claude_process("claude --dangerously-skip-permissions"));
    }

    #[test]
    fn is_claude_process_matches_absolute_path() {
        assert!(is_claude_process("/usr/local/bin/claude --resume foo"));
    }

    #[test]
    fn is_claude_process_rejects_claudectl() {
        assert!(!is_claude_process("claudectl --list"));
    }

    #[test]
    fn is_claude_process_rejects_shell_wrapping() {
        assert!(!is_claude_process(
            "bash -lc 'exec sandbox-bootstrap claude --resume foo'"
        ));
    }

    #[test]
    fn is_claude_process_rejects_grep_claude() {
        assert!(!is_claude_process("grep claude"));
    }

    #[test]
    fn parse_etime_all_forms() {
        assert_eq!(parse_etime("03:22"), Some(3 * 60 + 22));
        assert_eq!(parse_etime("01:02:03"), Some(3600 + 2 * 60 + 3));
        assert_eq!(
            parse_etime("2-01:02:03"),
            Some(2 * 86_400 + 3600 + 2 * 60 + 3)
        );
        assert_eq!(parse_etime("00:00"), Some(0));
        assert_eq!(parse_etime("garbage"), None);
        assert_eq!(parse_etime(""), None);
    }

    #[test]
    fn live_claude_procs_snapshot_contains_no_self() {
        // claudectl's own test binary is not argv0 `claude`, so a live
        // snapshot must never include our own pid (guards the argv0 filter
        // against regressing into a substring match).
        if let Some(procs) = live_claude_procs() {
            assert!(!procs.contains_key(&std::process::id()));
        }
    }
}
