#![allow(dead_code)]

/// Configuration for an external agent (Codex, Aider, custom scripts).
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub name: String,
    pub agent_type: String,
    pub command: String,
    pub capabilities: Vec<String>,
    pub cwd: String,
}

impl AgentConfig {
    pub fn new(name: String) -> Self {
        Self {
            name,
            agent_type: "custom".into(),
            command: String::new(),
            capabilities: Vec::new(),
            cwd: ".".into(),
        }
    }

    /// Format as a compact line for the brain prompt.
    pub fn prompt_line(&self) -> String {
        let caps = if self.capabilities.is_empty() {
            "general".to_string()
        } else {
            self.capabilities.join(", ")
        };
        format!("- {}: {} ({})", self.name, caps, self.agent_type)
    }
}

/// Format all registered agents as a prompt section for the brain.
pub fn format_agents_prompt(agents: &[AgentConfig]) -> String {
    if agents.is_empty() {
        return String::new();
    }

    let mut lines = vec!["## Available Agents".to_string()];
    for agent in agents {
        lines.push(agent.prompt_line());
    }
    lines.join("\n")
}

/// Result of running an external agent.
#[derive(Debug, Clone)]
pub struct AgentResult {
    pub agent_name: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub success: bool,
}

/// Spawn an agent process with a prompt, capture stdout/stderr, and return the result.
/// This is blocking — call from a spawned thread.
pub fn run_agent(agent: &AgentConfig, prompt: &str) -> Result<AgentResult, String> {
    let full_command = format!("{} {}", agent.command, shell_escape(prompt));

    let output = std::process::Command::new("sh")
        .args(["-c", &full_command])
        .current_dir(&agent.cwd)
        .output()
        .map_err(|e| format!("failed to spawn agent '{}': {e}", agent.name))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code();

    // Log output to .claudectl-runs/agents/
    log_agent_output(&agent.name, &stdout, &stderr);

    Ok(AgentResult {
        agent_name: agent.name.clone(),
        stdout,
        stderr,
        exit_code,
        success: output.status.success(),
    })
}

/// Find an agent by name in the registry.
pub fn find_agent<'a>(agents: &'a [AgentConfig], name: &str) -> Option<&'a AgentConfig> {
    agents.iter().find(|a| a.name == name)
}

fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

fn log_agent_output(agent_name: &str, stdout: &str, stderr: &str) {
    let dir = crate::product::runs_dir(std::path::Path::new(".")).join("agents");
    let _ = std::fs::create_dir_all(&dir);

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if !stdout.is_empty() {
        let path = dir.join(format!("{agent_name}.{ts}.stdout.log"));
        let _ = std::fs::write(&path, stdout);
    }
    if !stderr.is_empty() {
        let path = dir.join(format!("{agent_name}.{ts}.stderr.log"));
        let _ = std::fs::write(&path, stderr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_config_defaults() {
        let a = AgentConfig::new("test".into());
        assert_eq!(a.name, "test");
        assert_eq!(a.agent_type, "custom");
        assert_eq!(a.cwd, ".");
        assert!(a.capabilities.is_empty());
    }

    #[test]
    fn prompt_line_format() {
        let mut a = AgentConfig::new("codex".into());
        a.agent_type = "codex".into();
        a.capabilities = vec!["code-review".into(), "refactoring".into()];
        let line = a.prompt_line();
        assert!(line.contains("codex"));
        assert!(line.contains("code-review, refactoring"));
    }

    #[test]
    fn format_agents_prompt_empty() {
        assert_eq!(format_agents_prompt(&[]), "");
    }

    #[test]
    fn format_agents_prompt_multiple() {
        let agents = vec![
            {
                let mut a = AgentConfig::new("codex".into());
                a.capabilities = vec!["review".into()];
                a
            },
            {
                let mut a = AgentConfig::new("aider".into());
                a.capabilities = vec!["implementation".into()];
                a
            },
        ];
        let output = format_agents_prompt(&agents);
        assert!(output.contains("Available Agents"));
        assert!(output.contains("codex"));
        assert!(output.contains("aider"));
    }

    #[test]
    fn parse_agent_config_from_fields() {
        let mut a = AgentConfig::new("test-agent".into());
        a.agent_type = "aider".into();
        a.command = "aider --yes".into();
        a.capabilities = vec!["debugging".into(), "implementation".into()];
        a.cwd = "/tmp/project".into();

        assert_eq!(a.command, "aider --yes");
        assert_eq!(a.capabilities.len(), 2);
    }
}
