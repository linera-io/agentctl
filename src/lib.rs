#![allow(unknown_lints)]
#![allow(
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::io_other_error
)]

pub mod app;
pub mod brain;
pub mod config;
pub mod cpu;
pub mod demo;
pub mod discovery;
pub mod health;
pub mod helpers;
pub mod history;
pub mod hook_state;
pub mod hooks;
pub mod init;
pub mod launch;
pub mod logger;
pub mod models;
pub mod monitor;
pub mod orchestrator;
pub mod process;
pub mod reaper;
pub mod recorder;
pub mod rules;
pub mod sandbox_registry;
pub mod session;
/// Seam tests joining the hook writer to the dashboard renderer. Test-only:
/// every module it exercises is already public, this adds no surface.
#[cfg(test)]
mod session_lifecycle_tests;
pub mod session_recorder;
pub mod terminal_owner;
pub mod terminals;
pub mod theme;
pub mod transcript;
pub mod ui;
pub mod usage_ledger;
