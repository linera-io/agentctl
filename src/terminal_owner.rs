//! Which terminal instance a Claude session belongs to — and whether that
//! terminal is still running.
//!
//! This is the datum the restore registry needs and never had. A session
//! disappears for two very different reasons, and only one of them is worth
//! restoring:
//!
//!   - **The user closed it** (`/exit`, Ctrl-D, ⌘W on the tab). The terminal
//!     application is still there afterwards. Restoring it later would be an
//!     unwanted resurrection.
//!   - **The terminal died under it** (Ghostty restart-to-update, a crash,
//!     `sbx rm`). Everything went at once, and bringing it all back is exactly
//!     what `--restore-sessions` is for.
//!
//! Neither the session's own exit nor its pointer file can tell those apart —
//! both look identical from inside the dying process. What separates them is
//! whether the *terminal application* outlived the session, so that is what we
//! record: the owning instance, identified by pid **plus** start time (pid
//! alone is recycled by the kernel and would eventually alias a stranger).
//!
//! ## Picking the owner
//!
//! Walk the parent chain up until the next hop would be init or a *session
//! manager* — a process that deliberately outlives the terminals it starts
//! (see `SESSION_MANAGERS`). The last process before that boundary is the owner.
//!
//! On macOS a Ghostty session is `claude → bash → login → ghostty`, with
//! `ghostty` parented to launchd, so the walk lands on the application process
//! shared by every one of its windows. On a systemd desktop the chain tops out
//! at the per-login `systemd --user`, which survives any terminal quit; stopping
//! below it lands on `gnome-terminal-server` instead — the process that dies
//! when the terminal quits but survives closing a single tab, which is exactly
//! the distinction we need. Over ssh it stops below sshd, on the login shell —
//! which dies with the connection but survives an `/exit` in the session above
//! it, the same distinction one level down.
//!
//! A session whose parent chain has already been reparented to init has no
//! distinguishable owner; `owner_of` returns `None` and callers treat it as
//! "unknown", never as "the terminal is gone".

use std::cell::OnceCell;
use std::collections::HashMap;
use std::process::Command;

/// Cap on parent-chain hops. Real chains are 3-5 deep; anything longer means a
/// corrupt table (or a cycle, which `ps` shouldn't produce but we don't bet the
/// hook loop on it).
const MAX_HOPS: usize = 64;

/// Ancestors that outlive the terminals hanging off them, so the walk stops
/// *below* rather than *at* them.
///
/// Without this the owner of every session on a systemd desktop would be the
/// per-login `systemd --user`, and of every ssh session the listening `sshd` —
/// processes that are still running long after the terminal died. They would
/// report "terminal still alive" for sessions that went down *with* their
/// terminal, and the reconcile would prune the restore set: the original bug,
/// reintroduced on Linux.
const SESSION_MANAGERS: [&str; 4] = ["systemd", "sshd", "launchd", "init"];

/// The terminal instance a session ran under.
///
/// `started_at` comes from `ps`'s `lstart` column, read under a pinned `LC_ALL`
/// **and** `TZ` so the string is byte-stable between the hook process that
/// records it and the reaper that later checks it — the reaper runs under
/// launchd with a near-empty environment, and a timezone difference alone would
/// make every comparison fail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalOwner {
    pub pid: u32,
    pub started_at: String,
}

/// One row of the process table.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProcessRow {
    parent: u32,
    started_at: String,
    command: String,
}

/// A snapshot of the process table.
///
/// Taken once and reused, so a reconcile costs a single `ps` no matter how many
/// sessions it inspects.
pub struct ProcessTable {
    rows: HashMap<u32, ProcessRow>,
}

impl ProcessTable {
    /// Read the current process table. `None` if `ps` could not be run — the
    /// callers all treat that as "no information" and leave the registry alone
    /// rather than guessing.
    pub fn snapshot() -> Option<Self> {
        let output = Command::new("ps")
            .args(["-A", "-o", "pid=,ppid=,lstart=,comm="])
            .env("LC_ALL", "C")
            .env("TZ", "UTC0")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        Some(ProcessTable {
            rows: parse_process_table(&String::from_utf8_lossy(&output.stdout)),
        })
    }

    /// The terminal instance owning `pid`, or `None` when the pid is gone or
    /// has no ancestor between it and the nearest session manager.
    pub fn owner_of(&self, pid: u32) -> Option<TerminalOwner> {
        resolve_owner(&self.rows, pid)
    }

    /// Is this exact terminal instance still running? Compares the start time
    /// too, so a recycled pid reads as gone rather than as the original.
    pub fn is_alive(&self, owner: &TerminalOwner) -> bool {
        self.rows
            .get(&owner.pid)
            .is_some_and(|row| row.started_at == owner.started_at)
    }
}

/// A process table sampled lazily, on first use.
///
/// Used by the hook path to *attribute* a live session to its terminal. Lazy
/// because the steady state needs no sample at all: while every registered
/// session is still live and already carries a recorded owner, there is nothing
/// to attribute, so the common hook event costs zero `ps` calls — one sample is
/// taken only when a session is met for the first time.
///
/// It deliberately offers no "is this owner alive" or "confirm over time" API.
/// Deciding a session is gone for good is the reaper's job, and the reaper must
/// sample *after* it has observed the session depart (see
/// [`crate::reaper`])— an ordering a shared lazy oracle can't enforce, so it
/// owns its samples directly rather than borrowing one from here.
pub struct OwnerCheck {
    table: OnceCell<Option<ProcessTable>>,
}

impl OwnerCheck {
    pub fn lazy() -> Self {
        OwnerCheck {
            table: OnceCell::new(),
        }
    }

    /// Resolve the terminal instance owning `pid`, sampling on first use.
    pub fn owner_of(&self, pid: u32) -> Option<TerminalOwner> {
        self.table
            .get_or_init(ProcessTable::snapshot)
            .as_ref()?
            .owner_of(pid)
    }
}

/// Parse `ps -A -o pid=,ppid=,lstart=,comm=` output.
///
/// `lstart` is fixed at five whitespace-separated tokens under `LC_ALL=C`
/// ("Wed Jul 22 16:05:11 2026"), so it can sit between the numeric columns and a
/// command that may itself contain spaces. Unparseable lines are skipped — a
/// partial table is still useful.
fn parse_process_table(ps_output: &str) -> HashMap<u32, ProcessRow> {
    const LSTART_FIELDS: usize = 5;
    let mut rows = HashMap::new();
    for line in ps_output.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 2 + LSTART_FIELDS {
            continue;
        }
        let (Ok(pid), Ok(parent)) = (fields[0].parse::<u32>(), fields[1].parse::<u32>()) else {
            continue;
        };
        rows.insert(
            pid,
            ProcessRow {
                parent,
                started_at: fields[2..2 + LSTART_FIELDS].join(" "),
                command: fields[2 + LSTART_FIELDS..].join(" "),
            },
        );
    }
    rows
}

/// Is this the kind of ancestor that outlives the terminals below it?
///
/// Compares the executable name only: `ps` reports paths (`/usr/lib/systemd/systemd`)
/// and login shells with a leading dash (`-bash`).
fn is_session_manager(command: &str) -> bool {
    let name = command
        .rsplit('/')
        .next()
        .unwrap_or(command)
        .trim_start_matches('-');
    SESSION_MANAGERS.contains(&name)
}

/// Walk `pid`'s parent chain to the last process before init or a session
/// manager.
///
/// `None` when the pid isn't in the table, when the chain starts at that
/// boundary already (an orphaned session owns nothing), or when it doesn't
/// terminate within [`MAX_HOPS`].
fn resolve_owner(rows: &HashMap<u32, ProcessRow>, pid: u32) -> Option<TerminalOwner> {
    let mut current = pid;
    for _ in 0..MAX_HOPS {
        let row = rows.get(&current)?;
        let stop = row.parent <= 1
            || rows
                .get(&row.parent)
                .is_none_or(|parent| is_session_manager(&parent.command));
        if stop {
            // `current` is the top-most attributable ancestor. If nothing sits
            // between the session and that boundary, there is no terminal to
            // attribute it to.
            if current == pid {
                return None;
            }
            return Some(TerminalOwner {
                pid: current,
                started_at: row.started_at.clone(),
            });
        }
        current = row.parent;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape `ps` reports for a Ghostty session on macOS, verified against
    /// Andre's box: claude -> bash -> login -> ghostty -> launchd(1).
    fn ghostty_table() -> HashMap<u32, ProcessRow> {
        parse_process_table(
            "\
27891 27446 Wed Jul 22 16:07:14 2026 claude
27446 27445 Wed Jul 22 16:06:52 2026 -/opt/homebrew/bin/bash
27445 15601 Wed Jul 22 16:06:52 2026 /usr/bin/login
15601     1 Wed Jul 22 16:05:11 2026 /Applications/Ghostty.app/Contents/MacOS/ghostty
",
        )
    }

    fn table(rows: HashMap<u32, ProcessRow>) -> ProcessTable {
        ProcessTable { rows }
    }

    fn owner(pid: u32, started_at: &str) -> TerminalOwner {
        TerminalOwner {
            pid,
            started_at: started_at.to_string(),
        }
    }

    // ---- parsing --------------------------------------------------------

    #[test]
    fn parses_lstart_and_command_around_each_other() {
        let rows = ghostty_table();
        let ghostty = rows.get(&15601).expect("ghostty row");
        assert_eq!(ghostty.started_at, "Wed Jul 22 16:05:11 2026");
        assert_eq!(
            ghostty.command,
            "/Applications/Ghostty.app/Contents/MacOS/ghostty"
        );
    }

    #[test]
    fn parses_a_command_containing_spaces() {
        let rows =
            parse_process_table("42 1 Wed Jul 22 16:05:11 2026 /Applications/My Term.app/x\n");
        assert_eq!(
            rows.get(&42).map(|r| r.command.as_str()),
            Some("/Applications/My Term.app/x")
        );
    }

    #[test]
    fn parses_a_single_digit_day() {
        // `ps` pads the day, producing a double space inside lstart.
        let rows = parse_process_table("42 1 Wed Jul  2 16:05:11 2026 ghostty\n");
        assert_eq!(
            rows.get(&42).map(|r| r.started_at.as_str()),
            Some("Wed Jul 2 16:05:11 2026")
        );
    }

    #[test]
    fn skips_rows_that_are_too_short_to_carry_a_start_time() {
        assert!(parse_process_table("42 1\nnotapid 1 Wed Jul 22 16:05:11 2026 x\n").is_empty());
    }

    // ---- owner resolution -----------------------------------------------

    #[test]
    fn owner_is_the_terminal_app_not_the_shell() {
        let owner = resolve_owner(&ghostty_table(), 27891).expect("owner");
        assert_eq!(owner.pid, 15601);
        assert_eq!(owner.started_at, "Wed Jul 22 16:05:11 2026");
    }

    #[test]
    fn sessions_in_the_same_terminal_share_an_owner() {
        // Two windows of one Ghostty: both walks must land on the app process,
        // or a quit would look like a per-window close for one of them.
        let mut rows = ghostty_table();
        for (pid, parent) in [(26559u32, 25954u32), (25954, 25913), (25913, 15601)] {
            rows.insert(
                pid,
                ProcessRow {
                    parent,
                    started_at: "Wed Jul 22 16:06:00 2026".into(),
                    command: "x".into(),
                },
            );
        }
        assert_eq!(
            resolve_owner(&rows, 26559).map(|o| o.pid),
            resolve_owner(&rows, 27891).map(|o| o.pid)
        );
    }

    #[test]
    fn walk_stops_below_systemd_user_on_a_linux_desktop() {
        // gnome-terminal-server dies with the terminal but survives closing one
        // tab. `systemd --user` survives everything, so owning the session with
        // it would report "terminal alive" forever and prune the restore set.
        let rows = parse_process_table(
            "\
900 880 Wed Jul 22 16:07:14 2026 claude
880 700 Wed Jul 22 16:06:52 2026 bash
700 500 Wed Jul 22 16:00:00 2026 /usr/libexec/gnome-terminal-server
500   1 Wed Jul 22 09:00:00 2026 /usr/lib/systemd/systemd
",
        );
        assert_eq!(resolve_owner(&rows, 900).map(|o| o.pid), Some(700));
    }

    #[test]
    fn walk_stops_below_sshd_over_ssh() {
        // sshd outlives every connection, so the walk stops below it — landing
        // on the login shell. That is the right answer here: the shell dies when
        // the connection drops (restore material) but survives `/exit` in the
        // session above it (a hand-close), which is the distinction we need.
        let rows = parse_process_table(
            "\
900 880 Wed Jul 22 16:07:14 2026 claude
880 700 Wed Jul 22 16:06:52 2026 -bash
700 600 Wed Jul 22 16:00:00 2026 /usr/sbin/sshd
600   1 Wed Jul 22 09:00:00 2026 /usr/sbin/sshd
",
        );
        assert_eq!(resolve_owner(&rows, 900).map(|o| o.pid), Some(880));
    }

    #[test]
    fn orphaned_session_has_no_owner() {
        // Reparented to init: nothing sits between it and the boundary, so we
        // must not claim the session owns itself.
        let rows = parse_process_table("999 1 Wed Jul 22 16:07:14 2026 claude\n");
        assert_eq!(resolve_owner(&rows, 999), None);
    }

    #[test]
    fn unknown_pid_has_no_owner() {
        assert_eq!(resolve_owner(&ghostty_table(), 12345), None);
    }

    #[test]
    fn cyclic_chain_terminates() {
        let rows = parse_process_table(
            "\
10 11 Wed Jul 22 16:07:14 2026 a
11 10 Wed Jul 22 16:07:14 2026 b
",
        );
        assert_eq!(resolve_owner(&rows, 10), None);
    }

    // ---- liveness --------------------------------------------------------

    #[test]
    fn is_alive_rejects_a_recycled_pid() {
        let table = table(ghostty_table());
        assert!(table.is_alive(&owner(15601, "Wed Jul 22 16:05:11 2026")));
        assert!(!table.is_alive(&owner(15601, "Tue Jul 21 09:00:00 2026")));
    }

    #[test]
    fn is_alive_rejects_a_pid_that_left_the_table() {
        assert!(!table(ghostty_table()).is_alive(&owner(4242, "Wed Jul 22 16:05:11 2026")));
    }

    #[test]
    fn a_dead_terminals_owner_reads_gone_after_it_exits() {
        // The signal the reaper acts on: once the terminal process is gone from
        // the table, its owner is not alive — this is what a post-settle sample
        // sees for a terminal that co-died with its sessions.
        assert!(!table(HashMap::new()).is_alive(&owner(15601, "Wed Jul 22 16:05:11 2026")));
    }

    #[test]
    fn a_failed_ps_yields_no_owner() {
        let blind = OwnerCheck::lazy();
        blind.table.set(None).ok().expect("fresh cell");
        assert_eq!(blind.owner_of(27891), None);
    }
}
