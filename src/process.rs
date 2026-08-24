use crate::session::{AgentSession, SessionStatus};
use std::collections::{HashMap, HashSet};

/// How many enrich ticks a session without any sidecar source is re-probed
/// before absence is accepted as final (~1 minute at the 2s TUI interval).
const SIDECAR_PROBE_ATTEMPTS: u8 = 30;

/// Apply one sidecar probe result to the session. A successful probe is
/// cached forever (the sidecar is invariant per pid). Absence is NOT cached
/// on the first attempt: a first-tick probe can lose a race (registry
/// mid-write, transient /proc read failure under startup load), and a row
/// frozen without routing keeps Tab-switching broken for the TUI's whole
/// lifetime — observed live on a 30-session dashboard. Retried with a bounded
/// budget so host-native sessions (which never have a sidecar) still settle
/// to zero steady-state I/O.
fn apply_sidecar_probe(session: &mut AgentSession, sidecar: Option<TerminalSidecar>) {
    match sidecar {
        Some(s) => {
            if let Some(host_tty) = s.host_tty {
                session.tty = host_tty;
            }
            session.terminal_id = s.terminal_id;
            session.host_terminal_target = s.host_terminal_target;
            session.sidecar_loaded = true;
        }
        None => {
            session.sidecar_attempts = session.sidecar_attempts.saturating_add(1);
            if session.sidecar_attempts >= SIDECAR_PROBE_ATTEMPTS {
                session.sidecar_loaded = true;
            }
        }
    }
}

/// How long to wait before re-sampling CPU for sessions that have no baseline
/// yet. Long enough that `ps`'s whole-second `cputime` resolution yields a
/// usable delta for a busy process, short enough to be imperceptible in a
/// human-invoked command.
const CPU_BASELINE_GAP: std::time::Duration = std::time::Duration::from_millis(250);

/// Check which PIDs are alive and fetch TTY, CPU, MEM, command args — all via `ps`.
/// No sysinfo dependency needed.
pub fn fetch_and_enrich(sessions: &mut [AgentSession]) {
    if sessions.is_empty() {
        return;
    }
    enrich_from_ps(sessions);

    // A CPU *rate* needs two samples, and a session being seen for the first
    // time has only a baseline. The TUI would fill that in on its next tick,
    // but one-shot commands (`--json`, `--list`, `--summary`) never get one —
    // their CPU column would read "not known yet" forever. Take a second
    // reading for exactly those sessions.
    //
    // Costs one extra `ps` and `CPU_BASELINE_GAP` on the tick a session first
    // appears, and nothing at all once every session has a rate.
    if sessions
        .iter()
        .any(|s| s.cpu_rate_percent.is_none() && s.cpu_sample.is_some())
    {
        std::thread::sleep(CPU_BASELINE_GAP);
        resample_cpu(sessions);
    }
}

/// Re-read cumulative CPU time only, and difference it against the baseline
/// each session already carries.
///
/// Deliberately *not* a second `enrich_from_ps`: that pass also probes terminal
/// sidecars and consumes the bounded `sidecar_attempts` retry budget, so running
/// it twice per tick would halve the number of ticks a session gets to resolve
/// its terminal.
fn resample_cpu(sessions: &mut [AgentSession]) {
    let pids: Vec<String> = sessions.iter().map(|s| s.pid.to_string()).collect();
    let Ok(output) = std::process::Command::new("ps")
        .args(["-o", "pid=,cputime=", "-p", &pids.join(",")])
        .env_clear()
        .output()
    else {
        return;
    };
    let sampled_at_ms = now_ms();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let cputimes: HashMap<u32, f64> = stdout
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let pid = fields.next()?.parse::<u32>().ok()?;
            Some((pid, crate::cpu::parse_cputime_secs(fields.next()?)?))
        })
        .collect();

    for session in sessions.iter_mut() {
        let (Some(prev), Some(&cputime_secs)) = (session.cpu_sample, cputimes.get(&session.pid))
        else {
            continue;
        };
        let cur = crate::cpu::CpuSample {
            cputime_secs,
            sampled_at_ms,
        };
        session.cpu_rate_percent = crate::cpu::cpu_rate_percent(prev, cur);
        session.cpu_sample = Some(cur);
    }
}

fn enrich_from_ps(sessions: &mut [AgentSession]) {
    let pids: Vec<String> = sessions.iter().map(|s| s.pid.to_string()).collect();
    let pid_arg = pids.join(",");

    // `cputime` (cumulative CPU seconds), not `%cpu`: differencing two of these
    // across ticks is the only way to learn what a process is doing *now*.
    // See `crate::cpu` for the measurement, and why `%cpu` reported an idle
    // session as busy for hours.
    let output = std::process::Command::new("ps")
        .args(["-o", "pid=,tty=,cputime=,rss=,command=", "-p", &pid_arg])
        .env_clear()
        .output();

    let sampled_at_ms = now_ms();

    let output = match output {
        Ok(o) => o,
        Err(e) => {
            crate::logger::log("ERROR", &format!("ps command failed: {e}"));
            // ps failed — mark all as Finished (will show tombstone for 30s)
            for s in sessions.iter_mut() {
                s.status = SessionStatus::Finished;
                s.cpu_rate_percent = None;
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
        let cputime_secs = crate::cpu::parse_cputime_secs(fields[2]);
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
        // information gain, so a SUCCESSFUL probe is cached forever.
        if !session.sidecar_loaded {
            // The sidecar file can be gone (swept alongside a destroyed
            // registry, or its dir recreated) while the session lives on. The
            // same routing facts are still in the session process's own
            // environment — the sidecar was written FROM that environment at
            // bootstrap — so fall back to reading it there. Without this,
            // Tab-switching to a recovered session has no target.
            apply_sidecar_probe(
                session,
                read_terminal_sidecar(pid).or_else(|| sidecar_from_proc_env(pid)),
            );
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

        // Difference this cumulative reading against the previous tick's. The
        // first tick for a pid has nothing to difference and leaves the rate
        // `None` — unknown, which status inference refuses to read as "busy".
        let sample = cputime_secs.map(|cputime_secs| crate::cpu::CpuSample {
            cputime_secs,
            sampled_at_ms,
        });
        session.cpu_rate_percent = match (session.cpu_sample, sample) {
            (Some(prev), Some(cur)) => crate::cpu::cpu_rate_percent(prev, cur),
            _ => None,
        };
        if sample.is_some() {
            session.cpu_sample = sample;
        }

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
            // A dead process consumes nothing. This is a measurement, not an
            // absence of one, so it is 0.0 rather than None — and the stale
            // sample goes with it, so a recycled pid cannot be differenced
            // against its predecessor's counter.
            session.cpu_rate_percent = Some(0.0);
            session.cpu_sample = None;
        }
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// True iff `command`'s argv0, after stripping any leading path, is exactly
/// `"claude"`. This excludes `agentctl`/`claudectl`, `grep claude`, and
/// `bash -lc '... claude ...'`.
///
/// Shared with the reaper's cross-sandbox collector so both places decide
/// "this pid is a claude session" from one implementation — a `ps` row is the
/// liveness evidence in both, and two spellings of that test would be two
/// answers to whether a recycled pid is a live session.
pub(crate) fn is_claude_process(command: &str) -> bool {
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
/// The bootstrap writes `<pid>.terminal.json` to `$HOME/.claude/sessions/` —
/// but inside the sandbox that path is a per-exec-namespace bind mount of
/// `/var/lib/sandbox-sessions`, and execs that skip the bootstrap overlay
/// (notably the shell the TUI runs in) see the host directory instead, where
/// the sidecars don't exist. So probe the canonical sandbox dir first: it is
/// the real directory behind every overlay and is visible from any in-sandbox
/// namespace. It doesn't exist on hosts, where the `$HOME` path stands. For
/// non-sandbox claude sessions the file is absent everywhere and this returns
/// None — the regular `ps` TTY stands.
fn read_terminal_sidecar(pid: u32) -> Option<TerminalSidecar> {
    first_sidecar_in(&sidecar_candidate_dirs(), pid)
}

/// The HOST-side `(terminal_id, tty)` for a session running in this sandbox,
/// straight from its sidecar (falling back to the process environment the
/// sidecar was written from).
///
/// Deliberately re-probes rather than reading `AgentSession::tty`: that field
/// holds the host tty only when the sidecar probe happened to succeed, and the
/// container tty from `ps` otherwise. The two are indistinguishable after the
/// fact, so inferring from it would sometimes persist a *container* tty into
/// the registry labelled as a host one — a wrong routing key is worse than an
/// absent one, because the absent one still falls back to the cwd/title chain.
pub fn host_terminal_routing(pid: u32) -> (Option<String>, Option<String>) {
    // Log the INPUTS and which branch answered, not just the result. A silent
    // `None` here is indistinguishable from "the sidecar was fine but nothing
    // asked", and it is written into the registry as a plain null — so the
    // symptom surfaces much later, on the host, as "Tab went nowhere".
    let dirs = sidecar_candidate_dirs();
    if let Some(sidecar) = read_terminal_sidecar(pid) {
        crate::logger::log(
            "DEBUG",
            &format!(
                "routing: pid {pid} resolved from sidecar file (id={} tty={})",
                sidecar.terminal_id.as_deref().unwrap_or("-"),
                sidecar.host_tty.as_deref().unwrap_or("-")
            ),
        );
        return (sidecar.terminal_id, sidecar.host_tty);
    }
    if let Some(sidecar) = sidecar_from_proc_env(pid) {
        crate::logger::log(
            "DEBUG",
            &format!(
                "routing: pid {pid} resolved from /proc environ (id={} tty={})",
                sidecar.terminal_id.as_deref().unwrap_or("-"),
                sidecar.host_tty.as_deref().unwrap_or("-")
            ),
        );
        return (sidecar.terminal_id, sidecar.host_tty);
    }
    // Name the paths that were actually tried and whether each file existed,
    // so "no sidecar" can be told apart from "wrong directory" without
    // re-deriving the probe order by hand.
    let tried: Vec<String> = dirs
        .iter()
        .map(|dir| {
            let path = dir.join(format!("{pid}.terminal.json"));
            format!(
                "{}={}",
                path.display(),
                if path.exists() { "present" } else { "absent" }
            )
        })
        .collect();
    crate::logger::log(
        "WARN",
        &format!(
            "routing: pid {pid} unresolved — no host terminal id or tty; tried {}",
            tried.join(" ")
        ),
    );
    (None, None)
}

/// Sidecar search path, most authoritative first. The sandbox dir honors the
/// same `CLAUDECTL_SANDBOX_SESSIONS_DIR` override as the reaper — with one
/// caveat: the reaper interpolates the value into a bash script (where `~`
/// and `$VARS` expand), while this probe uses it as a literal path, so an
/// override must be an absolute literal path to work for both.
fn sidecar_candidate_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs = vec![std::path::PathBuf::from(
        crate::reaper::sandbox_sessions_dir(),
    )];
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(
            std::path::PathBuf::from(home)
                .join(".claude")
                .join("sessions"),
        );
    }
    dirs
}

fn first_sidecar_in(dirs: &[std::path::PathBuf], pid: u32) -> Option<TerminalSidecar> {
    dirs.iter()
        .find_map(|dir| read_terminal_sidecar_in(dir, pid))
}

fn read_terminal_sidecar_in(dir: &std::path::Path, pid: u32) -> Option<TerminalSidecar> {
    let path = dir.join(format!("{pid}.terminal.json"));
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

/// Rebuild the terminal sidecar from the session process's environment when
/// the sidecar FILE is gone. sandbox-bootstrap-inner exports these variables
/// into every session it launches and then writes the sidecar from them, so
/// the environment is the authoritative source the file merely caches.
/// Linux-only: sidecars exist only for sandbox sessions, and the sandbox is
/// Linux, where /proc/<pid>/environ is readable (claudectl runs as root
/// there).
#[cfg(target_os = "linux")]
fn sidecar_from_proc_env(pid: u32) -> Option<TerminalSidecar> {
    let raw = std::fs::read(format!("/proc/{pid}/environ")).ok()?;
    let mut vars: HashMap<String, String> = HashMap::new();
    for chunk in raw.split(|byte| *byte == 0) {
        let entry = String::from_utf8_lossy(chunk);
        if let Some((key, value)) = entry.split_once('=') {
            vars.insert(key.to_string(), value.to_string());
        }
    }
    sidecar_from_env_map(&vars)
}

#[cfg(not(target_os = "linux"))]
fn sidecar_from_proc_env(_pid: u32) -> Option<TerminalSidecar> {
    None
}

/// Pure core of [`sidecar_from_proc_env`]: map the bootstrap-exported
/// environment to sidecar fields. Probe order for the host-terminal target
/// mirrors [`parse_host_terminal_target`]: kitty first (strongest single
/// signal), then tmux, then wezterm.
#[cfg(target_os = "linux")]
fn sidecar_from_env_map(vars: &HashMap<String, String>) -> Option<TerminalSidecar> {
    let get = |key: &str| {
        vars.get(key)
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    let host_tty = get("SANDBOX_HOST_TTY");
    let terminal_id = get("SANDBOX_HOST_TERMINAL_ID");
    let host_terminal_target =
        if let (Some(window_id), Some(socket)) = (get("KITTY_WINDOW_ID"), get("KITTY_LISTEN_ON")) {
            Some(crate::session::HostTerminalTarget::Kitty { socket, window_id })
        } else if let (Some(tmux), Some(pane)) = (get("TMUX"), get("TMUX_PANE")) {
            let socket = tmux.split(',').next().unwrap_or(&tmux).to_string();
            Some(crate::session::HostTerminalTarget::Tmux { socket, pane })
        } else {
            get("WEZTERM_PANE")
                .and_then(|pane| pane.parse::<u64>().ok())
                .map(|pane_id| crate::session::HostTerminalTarget::WezTerm {
                    pane_id,
                    unix_socket: get("WEZTERM_UNIX_SOCKET"),
                })
        };
    if host_tty.is_none() && terminal_id.is_none() && host_terminal_target.is_none() {
        return None;
    }
    Some(TerminalSidecar {
        host_tty,
        terminal_id,
        host_terminal_target,
    })
}

fn extract_session_meta(cmd: &[&str], session: &mut AgentSession) {
    // If the session JSON already provided a name (via /rename or auto-name),
    // don't overwrite it from the process command line.
    let name_already_set = !session.session_name.is_empty();
    let mut i = 0;
    while i < cmd.len() {
        match cmd[i] {
            "--name" | "-n" if i + 1 < cmd.len() => {
                if !name_already_set {
                    session.session_name = cmd[i + 1].to_string();
                    // A user-typed CLI flag is an explicit choice, and it has
                    // no transcript record to re-assert it — without the flag
                    // a later scan-supplied name would overwrite it for good.
                    session.name_is_explicit = true;
                }
                i += 2;
                continue;
            }
            "--resume" | "-r" if i + 1 < cmd.len() => {
                let val = cmd[i + 1];
                if !name_already_set && !looks_like_uuid(val) {
                    session.session_name = val.to_string();
                    session.name_is_explicit = true;
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
    fn is_claude_process_rejects_our_own_binaries() {
        assert!(!is_claude_process("claudectl --list"));
        assert!(!is_claude_process("agentctl --list"));
    }

    #[test]
    fn is_claude_process_rejects_shell_wrapping() {
        assert!(!is_claude_process(
            "bash -lc 'exec sandbox-bootstrap claude --resume foo'"
        ));
    }

    #[test]
    fn regression_sidecar_found_in_canonical_sandbox_dir_without_overlay() {
        // 2026-07-28: the TUI's exec namespace lacked the ~/.claude/sessions
        // overlay, so sidecars only existed in /var/lib/sandbox-sessions.
        // Reading only $HOME/.claude/sessions left terminal_id unset and Tab
        // fell back to cwd matching, focusing an arbitrary same-cwd Ghostty
        // tab (or erroring for cwds with no host tab).
        let sandbox = tempfile::tempdir().unwrap();
        let home = tempfile::tempdir().unwrap();
        std::fs::write(
            sandbox.path().join("4242.terminal.json"),
            r#"{"host_tty":"/dev/ttys038","terminal_id":"7464E59F","terminal_type":"ghostty"}"#,
        )
        .unwrap();
        let dirs = vec![sandbox.path().to_path_buf(), home.path().to_path_buf()];
        let sidecar = first_sidecar_in(&dirs, 4242).expect("sidecar in sandbox dir must be found");
        assert_eq!(sidecar.terminal_id.as_deref(), Some("7464E59F"));
        assert_eq!(sidecar.host_tty.as_deref(), Some("/dev/ttys038"));
    }

    #[test]
    fn sidecar_falls_back_to_home_sessions_dir() {
        // Host-native macOS: /var/lib/sandbox-sessions doesn't exist and the
        // $HOME path is the only location.
        let sandbox = tempfile::tempdir().unwrap();
        let home = tempfile::tempdir().unwrap();
        std::fs::write(
            home.path().join("7.terminal.json"),
            r#"{"terminal_id":"ID-7"}"#,
        )
        .unwrap();
        let dirs = vec![sandbox.path().to_path_buf(), home.path().to_path_buf()];
        assert_eq!(
            first_sidecar_in(&dirs, 7).unwrap().terminal_id.as_deref(),
            Some("ID-7")
        );
    }

    #[test]
    fn sidecar_prefers_canonical_dir_when_both_exist() {
        // Inside a session namespace both paths resolve to the same directory;
        // if they ever diverge, the canonical sandbox dir must win.
        let sandbox = tempfile::tempdir().unwrap();
        let home = tempfile::tempdir().unwrap();
        std::fs::write(
            sandbox.path().join("9.terminal.json"),
            r#"{"terminal_id":"CANONICAL"}"#,
        )
        .unwrap();
        std::fs::write(
            home.path().join("9.terminal.json"),
            r#"{"terminal_id":"OVERLAY"}"#,
        )
        .unwrap();
        let dirs = vec![sandbox.path().to_path_buf(), home.path().to_path_buf()];
        assert_eq!(
            first_sidecar_in(&dirs, 9).unwrap().terminal_id.as_deref(),
            Some("CANONICAL")
        );
    }

    #[test]
    fn sidecar_absent_everywhere_returns_none() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        assert!(first_sidecar_in(&[a.path().to_path_buf(), b.path().to_path_buf()], 1).is_none());
    }

    #[test]
    fn regression_production_lookup_consults_the_sandbox_sessions_dir() {
        // Pins the WIRING, not just the helper: read_terminal_sidecar itself
        // must consult the env-resolved sandbox dir. Reverting it to the old
        // $HOME-only body would leave the first_sidecar_in tests green but
        // fail this one.
        let _lock = crate::sandbox_registry::tests::env_guard();
        let sandbox = tempfile::tempdir().unwrap();
        std::fs::write(
            sandbox.path().join("4291042.terminal.json"),
            r#"{"terminal_id":"WIRED"}"#,
        )
        .unwrap();
        // SAFETY: env access serialized by the held env lock.
        unsafe {
            std::env::set_var("CLAUDECTL_SANDBOX_SESSIONS_DIR", sandbox.path());
        }
        let result = read_terminal_sidecar(4291042);
        // SAFETY: same lock still held.
        unsafe {
            std::env::remove_var("CLAUDECTL_SANDBOX_SESSIONS_DIR");
        }
        let sidecar = result.expect("production lookup must consult the sandbox dir");
        assert_eq!(sidecar.terminal_id.as_deref(), Some("WIRED"));
    }

    #[test]
    fn is_claude_process_rejects_grep_claude() {
        assert!(!is_claude_process("grep claude"));
    }

    #[test]
    fn sidecar_probe_retries_absence_then_settles_and_caches_success() {
        // Regression: absence cached on the FIRST attempt froze a session
        // without terminal routing for the TUI's lifetime.
        let mut session = crate::session::AgentSession::from_raw(crate::session::RawSession {
            pid: 42,
            session_id: "s".into(),
            cwd: "/w".into(),
            started_at: 0,
            name: None,
            name_source: None,
        });

        apply_sidecar_probe(&mut session, None);
        assert!(
            !session.sidecar_loaded,
            "one miss must not settle the probe"
        );

        // A later success still lands and caches.
        apply_sidecar_probe(
            &mut session,
            Some(TerminalSidecar {
                host_tty: Some("/dev/ttys037".into()),
                terminal_id: Some("9B65C6AC".into()),
                host_terminal_target: None,
            }),
        );
        assert!(session.sidecar_loaded);
        assert_eq!(session.tty, "/dev/ttys037");
        assert_eq!(session.terminal_id.as_deref(), Some("9B65C6AC"));

        // Pure absence settles only after the bounded budget.
        let mut never = crate::session::AgentSession::from_raw(crate::session::RawSession {
            pid: 43,
            session_id: "n".into(),
            cwd: "/w".into(),
            started_at: 0,
            name: None,
            name_source: None,
        });
        for _ in 0..SIDECAR_PROBE_ATTEMPTS - 1 {
            apply_sidecar_probe(&mut never, None);
            assert!(!never.sidecar_loaded);
        }
        apply_sidecar_probe(&mut never, None);
        assert!(never.sidecar_loaded, "budget exhausted -> settled");
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

    #[cfg(target_os = "linux")]
    #[test]
    fn sidecar_from_env_map_recovers_routing_fields() {
        let vars: HashMap<String, String> = [
            ("SANDBOX_HOST_TTY", "/dev/ttys037"),
            (
                "SANDBOX_HOST_TERMINAL_ID",
                "9B65C6AC-B586-4943-ADB8-298D0759AF13",
            ),
            ("SANDBOX_NAME", "linera-agent"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let sidecar = sidecar_from_env_map(&vars).expect("host tty + terminal id recovered");
        assert_eq!(sidecar.host_tty.as_deref(), Some("/dev/ttys037"));
        assert_eq!(
            sidecar.terminal_id.as_deref(),
            Some("9B65C6AC-B586-4943-ADB8-298D0759AF13")
        );
        assert!(sidecar.host_terminal_target.is_none());

        let tmux: HashMap<String, String> =
            [("TMUX", "/tmp/tmux-0/default,42,3"), ("TMUX_PANE", "%7")]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
        let Some(crate::session::HostTerminalTarget::Tmux { socket, pane }) =
            sidecar_from_env_map(&tmux).and_then(|s| s.host_terminal_target)
        else {
            panic!("expected tmux target");
        };
        assert_eq!(socket, "/tmp/tmux-0/default");
        assert_eq!(pane, "%7");

        assert!(
            sidecar_from_env_map(&HashMap::new()).is_none(),
            "no routing vars -> no sidecar"
        );
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
