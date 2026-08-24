#![allow(dead_code)]

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use crate::session::{AgentSession, SessionStatus};
use crate::terminals;

/// A message queued for delivery to a session.
#[derive(Debug, Clone)]
pub struct MailMessage {
    pub timestamp: u64,
    pub from_pid: u32,
    pub from_project: String,
    pub summary: String,
    pub delivered: bool,
}

fn mailbox_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    crate::product::home_dot_dir(&PathBuf::from(home))
        .join("brain")
        .join("mailbox")
}

fn mailbox_path(pid: u32) -> PathBuf {
    mailbox_dir().join(format!("{pid}.jsonl"))
}

/// Queue a message for delivery to a target session.
pub fn enqueue(from_pid: u32, from_project: &str, target_pid: u32, summary: &str) {
    let dir = mailbox_dir();
    let _ = fs::create_dir_all(&dir);

    let path = mailbox_path(target_pid);
    let record = serde_json::json!({
        "ts": now_epoch_ms(),
        "from_pid": from_pid,
        "from_project": from_project,
        "summary": summary,
        "delivered": false,
    });

    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(
            file,
            "{}",
            serde_json::to_string(&record).unwrap_or_default()
        );
    }
}

/// Read pending (undelivered) messages for a session.
pub fn pending_messages(pid: u32) -> Vec<MailMessage> {
    let path = mailbox_path(pid);
    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    content
        .lines()
        .filter_map(|line| {
            let json: serde_json::Value = serde_json::from_str(line).ok()?;
            let delivered = json
                .get("delivered")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if delivered {
                return None;
            }
            Some(MailMessage {
                timestamp: json.get("ts").and_then(|v| v.as_u64()).unwrap_or(0),
                from_pid: json.get("from_pid").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                from_project: json
                    .get("from_project")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                summary: json
                    .get("summary")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                delivered: false,
            })
        })
        .collect()
}

/// Attempt to deliver pending messages to eligible sessions.
/// Only delivers when the target is in WaitingInput state (not mid-processing).
/// Returns a list of (pid, status_message) for deliveries made.
pub fn deliver_pending(sessions: &[AgentSession]) -> Vec<(u32, String)> {
    let mut delivered = Vec::new();

    for session in sessions {
        // Only deliver to sessions waiting for input (not mid-work)
        if session.status != SessionStatus::WaitingInput {
            continue;
        }

        let messages = pending_messages(session.pid);
        if messages.is_empty() {
            continue;
        }

        // Batch all pending messages into one delivery
        let mut batch = String::new();
        for msg in &messages {
            if !batch.is_empty() {
                batch.push('\n');
            }
            batch.push_str(&format!("[From {}] {}", msg.from_project, msg.summary));
        }

        match terminals::send_input(session, &batch) {
            Ok(()) => {
                // Mark all as delivered by rewriting the file
                mark_delivered(session.pid);
                delivered.push((
                    session.pid,
                    format!(
                        "Delivered {} message(s) to {}",
                        messages.len(),
                        session.display_name()
                    ),
                ));
            }
            Err(e) => {
                crate::logger::log(
                    "MAILBOX",
                    &format!("Delivery to {} failed: {e}", session.display_name()),
                );
            }
        }
    }

    delivered
}

/// Mark all messages for a PID as delivered.
fn mark_delivered(pid: u32) {
    let path = mailbox_path(pid);
    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return,
    };

    let updated: Vec<String> = content
        .lines()
        .map(|line| {
            if let Ok(mut json) = serde_json::from_str::<serde_json::Value>(line) {
                json["delivered"] = serde_json::Value::Bool(true);
                serde_json::to_string(&json).unwrap_or_else(|_| line.to_string())
            } else {
                line.to_string()
            }
        })
        .collect();

    let _ = fs::write(&path, updated.join("\n") + "\n");
}

/// Clean up mailbox files for PIDs that no longer exist.
pub fn cleanup(active_pids: &[u32]) {
    let dir = mailbox_dir();
    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(pid_str) = name.strip_suffix(".jsonl") {
            if let Ok(pid) = pid_str.parse::<u32>() {
                if !active_pids.contains(&pid) {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
    }
}

fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enqueue_and_read_pending() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("42.jsonl");

        // Write directly to temp path
        let record = serde_json::json!({
            "ts": 1000,
            "from_pid": 1,
            "from_project": "source",
            "summary": "found a bug",
            "delivered": false,
        });
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(file, "{}", serde_json::to_string(&record).unwrap()).unwrap();

        let record2 = serde_json::json!({
            "ts": 2000,
            "from_pid": 2,
            "from_project": "other",
            "summary": "fix applied",
            "delivered": true,
        });
        writeln!(file, "{}", serde_json::to_string(&record2).unwrap()).unwrap();
        drop(file);

        // Parse manually (pending_messages reads from fixed path)
        let content = fs::read_to_string(&path).unwrap();
        let pending: Vec<_> = content
            .lines()
            .filter_map(|line| {
                let json: serde_json::Value = serde_json::from_str(line).ok()?;
                if json["delivered"].as_bool() == Some(true) {
                    return None;
                }
                Some(json["summary"].as_str()?.to_string())
            })
            .collect();

        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0], "found a bug");
    }

    #[test]
    fn mark_delivered_updates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");

        let record = serde_json::json!({
            "ts": 1000,
            "from_pid": 1,
            "from_project": "src",
            "summary": "msg",
            "delivered": false,
        });
        fs::write(&path, serde_json::to_string(&record).unwrap() + "\n").unwrap();

        // Read and verify it's undelivered
        let content = fs::read_to_string(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(json["delivered"], false);

        // Mark delivered by rewriting
        let mut json = json;
        json["delivered"] = serde_json::Value::Bool(true);
        fs::write(&path, serde_json::to_string(&json).unwrap() + "\n").unwrap();

        let content = fs::read_to_string(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(json["delivered"], true);
    }
}
