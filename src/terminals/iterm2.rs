use super::{applescript_cd_exec, run_osascript};
use crate::session::AgentSession;

/// Open a new iTerm2 window that runs `command` in `cwd`. `create window with
/// default profile` opens a window with a login shell; we then `write text` the
/// `cd … && exec …` so the shell's PATH resolves `claude`/`sc`.
pub fn spawn_window(cwd: &str, command: &str) -> Result<String, String> {
    run_osascript(&new_window_script(cwd, command))?;
    Ok("iTerm2".to_string())
}

fn new_window_script(cwd: &str, command: &str) -> String {
    format!(
        r#"
        tell application "iTerm2"
            set w to (create window with default profile)
            tell current session of w to write text {}
            activate
        end tell
        "#,
        applescript_cd_exec(cwd, command),
    )
}

pub fn switch(session: &AgentSession) -> Result<(), String> {
    let script = format!(
        r#"
        tell application "iTerm2"
            repeat with w in windows
                repeat with t in tabs of w
                    repeat with s in sessions of t
                        if tty of s contains "{tty}" then
                            select t
                            set index of w to 1
                            activate
                            return "ok"
                        end if
                    end repeat
                end repeat
            end repeat
            error "TTY not found in iTerm2"
        end tell
        "#,
        tty = session.tty
    );
    run_osascript(&script)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_window_script_creates_a_window_and_runs_the_command() {
        let script = new_window_script("/work/scylla", "sc --resume abc123");
        assert!(script.contains("create window with default profile"));
        assert!(script.contains("write text"));
        assert!(script.contains(r#"quoted form of "/work/scylla""#));
        assert!(script.contains("exec sc --resume abc123"));
    }

    #[test]
    fn new_window_script_escapes_quotes_in_cwd() {
        let script = new_window_script("/work/a\"b", "sc --resume x");
        assert!(script.contains(r#"quoted form of "/work/a\"b""#));
    }
}
