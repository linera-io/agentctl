//! The product's own name, and the one it used to have.
//!
//! Every user-visible path is derived here rather than spelled out at each
//! call site, so the rename is one constant instead of a search-and-replace
//! across ten files. Paths resolve canonical-first and legacy-second: an
//! existing `claudectl` install keeps reading and writing exactly where it
//! already does, and nothing is ever moved or deleted on its behalf.

use std::path::{Path, PathBuf};

pub const NAME: &str = "agentctl";
pub const LEGACY_NAME: &str = "claudectl";

/// Pick between a canonical path and its legacy twin.
///
/// Canonical wins whenever it exists, so a migrated install never falls back.
/// Legacy wins only when it alone exists, which keeps an established install
/// writing in place instead of silently splitting its state across two roots.
/// Neither existing means a fresh install, which starts canonical.
pub fn resolve(canonical: PathBuf, legacy: PathBuf) -> PathBuf {
    if canonical.exists() {
        return canonical;
    }
    if legacy.exists() {
        return legacy;
    }
    canonical
}

/// `~/.agentctl`, or `~/.claudectl` if that is what this machine already has.
pub fn home_dot_dir(home: &Path) -> PathBuf {
    resolve(
        home.join(format!(".{NAME}")),
        home.join(format!(".{LEGACY_NAME}")),
    )
}

/// `~/.config/agentctl`, falling back to the legacy directory.
pub fn config_dir(home: &Path) -> PathBuf {
    let config = home.join(".config");
    resolve(config.join(NAME), config.join(LEGACY_NAME))
}

/// The per-project config file, `.agentctl.toml`, in the given directory.
pub fn project_config(dir: &Path) -> PathBuf {
    resolve(
        dir.join(format!(".{NAME}.toml")),
        dir.join(format!(".{LEGACY_NAME}.toml")),
    )
}

/// The orchestrator's per-project run directory.
pub fn runs_dir(dir: &Path) -> PathBuf {
    resolve(
        dir.join(format!(".{NAME}-runs")),
        dir.join(format!(".{LEGACY_NAME}-runs")),
    )
}

/// The product subdirectory inside an XDG data or state root.
pub fn data_subdir(root: &Path) -> PathBuf {
    resolve(root.join(NAME), root.join(LEGACY_NAME))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn the_canonical_product_name_is_agentctl() {
        assert_eq!(NAME, "agentctl");
        assert_eq!(LEGACY_NAME, "claudectl");
    }

    #[test]
    fn a_fresh_install_uses_canonical_paths() {
        let home = tmp();
        assert_eq!(home_dot_dir(home.path()), home.path().join(".agentctl"));
        assert_eq!(
            config_dir(home.path()),
            home.path().join(".config/agentctl")
        );
        assert_eq!(
            project_config(home.path()),
            home.path().join(".agentctl.toml")
        );
    }

    #[test]
    fn an_established_claudectl_install_is_read_in_place() {
        let home = tmp();
        std::fs::create_dir_all(home.path().join(".claudectl")).unwrap();
        std::fs::create_dir_all(home.path().join(".config/claudectl")).unwrap();
        std::fs::write(home.path().join(".claudectl.toml"), "").unwrap();

        assert_eq!(home_dot_dir(home.path()), home.path().join(".claudectl"));
        assert_eq!(
            config_dir(home.path()),
            home.path().join(".config/claudectl")
        );
        assert_eq!(
            project_config(home.path()),
            home.path().join(".claudectl.toml")
        );
    }

    #[test]
    fn canonical_wins_once_it_exists_so_a_migrated_install_never_falls_back() {
        let home = tmp();
        std::fs::create_dir_all(home.path().join(".claudectl")).unwrap();
        std::fs::create_dir_all(home.path().join(".agentctl")).unwrap();

        assert_eq!(home_dot_dir(home.path()), home.path().join(".agentctl"));
    }

    /// Reading legacy state must never imply moving or removing it.
    #[test]
    fn resolving_a_legacy_path_leaves_it_untouched() {
        let home = tmp();
        let legacy = home.path().join(".claudectl");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("parked.json"), "[]").unwrap();

        let resolved = home_dot_dir(home.path());

        assert_eq!(resolved, legacy);
        assert!(legacy.join("parked.json").exists(), "legacy data survives");
    }

    #[test]
    fn runs_and_data_dirs_follow_the_same_rule() {
        let project = tmp();
        assert_eq!(
            runs_dir(project.path()),
            project.path().join(".agentctl-runs")
        );
        std::fs::create_dir_all(project.path().join(".claudectl-runs")).unwrap();
        assert_eq!(
            runs_dir(project.path()),
            project.path().join(".claudectl-runs")
        );

        let data = tmp();
        assert_eq!(data_subdir(data.path()), data.path().join("agentctl"));
        std::fs::create_dir_all(data.path().join("claudectl")).unwrap();
        assert_eq!(data_subdir(data.path()), data.path().join("claudectl"));
    }
}
