use crate::session::ClaudeSession;

/// Open a new tmux window running `command` in `cwd`, in the current session
/// (restore runs inside tmux — `detect_terminal` returns `Tmux` only when
/// `$TMUX` is set). tmux runs `command` via the shell; `cwd`/`command` are argv
/// tokens, and the restore path allowlist-validates the session id, so the
/// command carries no metacharacters.
pub fn spawn_window(cwd: &str, command: &str) -> Result<String, String> {
    let output = std::process::Command::new("tmux")
        .args(spawn_argv(cwd, command))
        .output()
        .map_err(|e| format!("tmux new-window failed: {e}"))?;

    if output.status.success() {
        Ok("tmux window".into())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

fn spawn_argv(cwd: &str, command: &str) -> Vec<String> {
    ["new-window", "-c", cwd, command]
        .map(String::from)
        .to_vec()
}

pub fn launch(cwd: &str, prompt: Option<&str>, resume: Option<&str>) -> Result<String, String> {
    let mut parts = vec!["claude".to_string()];
    parts.extend(
        super::build_claude_args(prompt, resume)
            .into_iter()
            .map(|arg| super::shell_escape(&arg)),
    );
    let command = parts.join(" ");

    let output = std::process::Command::new("tmux")
        .args(["new-window", "-c", cwd, &command])
        .output()
        .map_err(|e| format!("tmux new-window failed: {e}"))?;

    if output.status.success() {
        Ok("tmux window".into())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

pub fn switch(session: &ClaudeSession) -> Result<(), String> {
    // tmux can list panes with their TTY: `tmux list-panes -a -F '#{pane_tty} #{session_name}:#{window_index}.#{pane_index}'`
    let output = std::process::Command::new("tmux")
        .args([
            "list-panes",
            "-a",
            "-F",
            "#{pane_tty} #{session_name}:#{window_index}.#{pane_index}",
        ])
        .output()
        .map_err(|e| format!("tmux list-panes failed: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() == 2 && parts[0].contains(&session.tty) {
            let target = parts[1]; // e.g. "main:2.1"
            // Select the tmux window+pane
            let _ = std::process::Command::new("tmux")
                .args(["select-window", "-t", target])
                .output();
            let _ = std::process::Command::new("tmux")
                .args(["select-pane", "-t", target])
                .output();
            return Ok(());
        }
    }

    Err(format!("TTY {} not found in tmux panes", session.tty))
}

pub fn send_input(session: &ClaudeSession, text: &str) -> Result<(), String> {
    let output = std::process::Command::new("tmux")
        .args([
            "list-panes",
            "-a",
            "-F",
            "#{pane_tty} #{session_name}:#{window_index}.#{pane_index}",
        ])
        .output()
        .map_err(|e| format!("tmux failed: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() == 2 && parts[0].contains(&session.tty) {
            let _ = std::process::Command::new("tmux")
                .args(["send-keys", "-t", parts[1], text, ""])
                .output();
            return Ok(());
        }
    }

    Err("TTY not found in tmux".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_argv_opens_a_window_in_cwd_running_the_command() {
        let argv = spawn_argv("/work/scylla", "claude --resume abc123");
        assert_eq!(
            argv,
            ["new-window", "-c", "/work/scylla", "claude --resume abc123"]
        );
    }
}
