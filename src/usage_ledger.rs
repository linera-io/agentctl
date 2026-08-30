//! Persistent usage ledger — append-only per-message token totals, sourced
//! from every Claude Code JSONL transcript on disk (including subagents).
//!
//! This exists because `history::record_session` only fires when claudectl
//! observes a session transition into `Finished` — a race window that misses
//! any session closed via a terminal-close/SIGHUP (Claude Code deletes its
//! own pointer file on exit, so the next tick drops the session before
//! claudectl can write a history row). The ledger side-steps that race by
//! reading directly from `~/.claude/projects/**/*.jsonl`, which Claude Code
//! retains effectively forever.
//!
//! Cost is computed at read time (not stored in the CSV) so a fix to
//! `models.rs` pricing retroactively corrects every historical summary, and
//! so the raw token counts remain usable for future "what-if" queries.
//!
//! Rows are keyed by `msg_key` (`<message.id>:<requestId>`) and deduplicated
//! on read. Two distinct sources of duplication make that necessary: one API
//! response is written as several JSONL lines that each repeat the same
//! cumulative `usage`, and a rewritten transcript is re-scanned from byte 0
//! and appended again in full. Counting rows instead of responses overstated
//! spend by 5-7x.
//!
//! Format:
//!   CSV: ~/.local/share/claudectl/usage_log.csv
//!     timestamp_ms,session_id,model,fresh_input,cache_read,cache_write,
//!     cache_write_1h,output,msg_key
//!   Offsets: ~/.local/share/claudectl/usage_offsets.json
//!     { "<jsonl-path>": { "last_byte": u64, "mtime_ms": u64 } }

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::models;
use crate::transcript::{TranscriptEvent, TranscriptRole, parse_line};

const LEDGER_BASENAME: &str = "usage_log.csv";
const OFFSETS_BASENAME: &str = "usage_offsets.json";
/// v2. Adds `cache_write_1h` (billed at 2x base, not the 5m 1.25x) and
/// `msg_key`, without which duplicate rows cannot be identified — see
/// `migrate_v1_ledger`.
const HEADER: &str = "timestamp_ms,session_id,model,fresh_input,cache_read,cache_write,cache_write_1h,output,msg_key";
const HEADER_V1: &str = "timestamp_ms,session_id,model,fresh_input,cache_read,cache_write,output";
const FIELD_COUNT: usize = 9;
/// Compact once the file exceeds this. Comfortably above a full retention
/// window of dense use (~232k rows / ~18 MB observed) so a healthy ledger is
/// never rewritten, but far below the runaway growth the append path caused.
static COMPACT_THRESHOLD_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(64 * 1024 * 1024);

fn compact_threshold() -> u64 {
    COMPACT_THRESHOLD_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Aggregated usage over a time window. Cost is computed from `model` at
/// read time using current `models.rs` pricing; historical pricing changes
/// therefore retroactively flow through.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UsageSummary {
    pub fresh_input: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub output: u64,
    pub cost_usd: f64,
    pub msg_count: u64,
}

impl UsageSummary {
    pub fn total_tokens(&self) -> u64 {
        self.fresh_input + self.cache_read + self.cache_write + self.output
    }
}

/// Result of a single `scan_and_append` invocation. Surfaced to the TUI so
/// the user can see "first scan indexed N messages" on startup.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScanReport {
    pub files_scanned: usize,
    pub files_updated: usize,
    pub rows_appended: u64,
}

fn ledger_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    crate::product::shared_state_root(&home)
}

fn projects_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".claude")
        .join("projects")
}

fn ledger_path() -> PathBuf {
    ledger_dir().join(LEDGER_BASENAME)
}

fn offsets_path() -> PathBuf {
    ledger_dir().join(OFFSETS_BASENAME)
}

#[derive(Debug, Clone, Default)]
struct FileOffset {
    last_byte: u64,
    mtime_ms: u64,
    /// True once we've drained this JSONL after observing its writer
    /// process exited. After draining a dead-writer file we know it can
    /// never grow again, so subsequent scans skip it entirely (no stat,
    /// no open). Drops the every-30-s scan from ~1700 ms (stat-ing every
    /// historical JSONL) to ~30 ms on a heavy `~/.claude/projects` tree.
    drained: bool,
}

type OffsetMap = HashMap<String, FileOffset>;

fn load_offsets_at(path: &Path) -> OffsetMap {
    let Ok(raw) = fs::read_to_string(path) else {
        return OffsetMap::new();
    };
    let Ok(value) = serde_json::from_str::<Value>(&raw) else {
        return OffsetMap::new();
    };
    let Some(obj) = value.as_object() else {
        return OffsetMap::new();
    };
    let mut out = OffsetMap::new();
    for (k, v) in obj {
        let last_byte = v.get("last_byte").and_then(|n| n.as_u64()).unwrap_or(0);
        let mtime_ms = v.get("mtime_ms").and_then(|n| n.as_u64()).unwrap_or(0);
        // Older offsets files won't have the `drained` field; default to
        // false so we re-scan once on upgrade and pick up the marker.
        let drained = v.get("drained").and_then(|n| n.as_bool()).unwrap_or(false);
        out.insert(
            k.clone(),
            FileOffset {
                last_byte,
                mtime_ms,
                drained,
            },
        );
    }
    out
}

fn save_offsets_at(path: &Path, offsets: &OffsetMap) {
    let mut obj = serde_json::Map::new();
    for (k, v) in offsets {
        let mut entry = serde_json::Map::new();
        entry.insert("last_byte".into(), Value::from(v.last_byte));
        entry.insert("mtime_ms".into(), Value::from(v.mtime_ms));
        entry.insert("drained".into(), Value::from(v.drained));
        obj.insert(k.clone(), Value::Object(entry));
    }
    let Ok(rendered) = serde_json::to_string(&Value::Object(obj)) else {
        return;
    };
    let _ = fs::write(path, rendered);
}

/// Read `~/.claude/sessions/*.json` pointer files and return the set of
/// `sessionId` values for currently-live Claude Code sessions. The set is
/// used by `scan_and_append` to gate stat-skipping on dead-writer files.
fn read_live_session_ids(sessions_dir: &Path) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    let Ok(entries) = fs::read_dir(sessions_dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // Skip the *.terminal.json sidecars and anything else non-pointer.
        let Some(ext) = path.extension() else {
            continue;
        };
        if ext != "json" {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.ends_with(".terminal.json") {
            continue;
        }
        let Ok(body) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&body) else {
            continue;
        };
        if let Some(sid) = value.get("sessionId").and_then(|v| v.as_str()) {
            out.insert(sid.to_string());
        }
    }
    out
}

/// Per-project-subdir cached file list. Walking
/// `~/.claude/projects` recursively from scratch takes ~500 ms on a
/// heavy tree (Andre's box has 36 project subdirs, 2k+ jsonls). Most
/// scans see no directory changes, so we keep a per-subdir cache keyed
/// by mtime: if a project subdir's mtime is unchanged since last walk,
/// its JSONL list can't have grown OR shrunk (POSIX mtime updates on
/// add/remove/rename of direct children) — reuse the cached list.
///
/// New sessions show up on the very next scan_and_append after their
/// project subdir's mtime moves (i.e. within ≤30 s of session start)
/// instead of waiting for a full re-walk timer. Steady-state cost: 1
/// read_dir of projects_root (~36 entries) + 36 stats = ~10 ms.
#[derive(Default)]
struct SubdirCache {
    last_mtime_ms: u64,
    files: Vec<PathBuf>,
}

type FileListCache = HashMap<PathBuf, SubdirCache>;

fn file_list_cache() -> &'static Mutex<FileListCache> {
    static CACHE: OnceLock<Mutex<FileListCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(FileListCache::new()))
}

fn dir_mtime_ms(path: &Path) -> u64 {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Build the JSONL file list using the per-project-subdir mtime cache.
/// Subdirs whose mtime is unchanged since last call return their cached
/// file list; changed (or new) subdirs are re-walked. Subdirs that
/// disappeared since last call are evicted from the cache.
fn find_jsonl_files_cached(root: &Path) -> Vec<PathBuf> {
    let mut all = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return all;
    };

    let mut cache = file_list_cache()
        .lock()
        .expect("file list cache mutex poisoned");
    let mut seen_subdirs: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();

    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(ft) = entry.file_type() else { continue };
        if !ft.is_dir() {
            // Top-level files. Bare `*.jsonl` directly under
            // projects_root is unusual but handle it just in case.
            if ft.is_file() && path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                all.push(path);
            }
            continue;
        }
        let mtime = dir_mtime_ms(&path);
        seen_subdirs.insert(path.clone());

        let dc = cache.entry(path.clone()).or_default();
        if dc.last_mtime_ms != mtime {
            let mut walked = Vec::new();
            walk_jsonls_into(&path, &mut walked);
            dc.files = walked;
            dc.last_mtime_ms = mtime;
        }
        all.extend(dc.files.iter().cloned());
    }

    // Project subdir was deleted since last scan ⇒ drop its cache
    // entry so memory doesn't accrete forever.
    cache.retain(|k, _| seen_subdirs.contains(k));

    all
}

fn walk_jsonls_into(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            let Ok(ft) = entry.file_type() else { continue };
            if ft.is_dir() {
                stack.push(p);
            } else if ft.is_file() && p.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                out.push(p);
            }
        }
    }
}

/// Single-stat helper: returns `(mtime_ms, len)` for the path in one
/// `fs::metadata()` call. Splitting these into separate calls (the older
/// `mtime_ms()` then a second `fs::metadata()` for `.len()`) doubled the
/// per-file syscall count in `scan_and_append`, which dominates tick
/// cost when there are thousands of JSONLs (each tick has to touch every
/// one to detect new bytes). Returns `None` when the file isn't
/// stat-able — caller treats that as "skip this file".
fn metadata_mtime_len(path: &Path) -> Option<(u64, u64)> {
    let m = fs::metadata(path).ok()?;
    let mtime = m
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_millis() as u64;
    Some((mtime, m.len()))
}

/// Session id carved from the JSONL filename stem. Works for both the
/// top-level `<uuid>.jsonl` and subagent `agent-*.jsonl` layouts — in the
/// latter case the string returned is the agent id, which is what we want
/// for attribution.
fn session_id_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// CSV-safe rendering of a model string. Model names are lowercase
/// alphanumerics + `-` in practice, but we still defensively strip commas
/// and newlines so a pathological entry can't corrupt the ledger.
fn csv_escape(raw: &str) -> String {
    raw.chars()
        .filter(|c| *c != ',' && *c != '\n' && *c != '\r')
        .collect()
}

/// Process-wide flag set while a background `scan_and_append` is in flight,
/// so the caller can avoid spawning a second thread before the first
/// finishes. The atomic is the cheapest correct synchronisation —
/// scan_and_append serialises against itself naturally via the file-list
/// cache mutex and offsets-file write, but we don't want to QUEUE work,
/// just skip a duplicate kick.
static SCAN_IN_FLIGHT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Fire-and-forget background scan. Returns immediately. If a previous
/// background scan is still running (a possibility if the projects tree
/// is huge or the disk is slow), drops this kick on the floor — the
/// next tick will try again.
///
/// The TUI's main thread never blocks on the JSONL walk this way;
/// freezes from a slow `scan_and_append` are eliminated regardless of
/// how long the scan takes. The in-memory ledger cache (read by
/// `load_summary` on the main thread) is updated as a side-effect when
/// the next `load_summary` reads the now-larger CSV.
pub fn scan_and_append_background() {
    use std::sync::atomic::Ordering;
    if SCAN_IN_FLIGHT
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    let work = || {
        let _ = scan_and_append();
        SCAN_IN_FLIGHT.store(false, Ordering::Release);
    };
    // Prefer tokio's blocking-pool thread when we're inside a runtime
    // (the TUI path goes through `runtime.block_on(run_tui)`), so the
    // pool can amortise OS thread creation across the long-running
    // process. Fall back to a fresh std::thread for one-shot CLI
    // commands and tests, which never set up a runtime.
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn_blocking(work);
    } else {
        std::thread::Builder::new()
            .name("ledger-scan".into())
            .spawn(work)
            // If thread spawn fails (PID limit, OS denial), clear the
            // flag so the next tick can retry instead of getting stuck
            // "in flight".
            .map_err(|_| SCAN_IN_FLIGHT.store(false, Ordering::Release))
            .ok();
    }
}

/// Scan every JSONL and append any new assistant `usage` blocks to the
/// ledger. Offsets persist across runs so subsequent scans are O(new bytes).
pub fn scan_and_append() -> ScanReport {
    scan_and_append_at(
        &projects_dir(),
        &ledger_path(),
        &offsets_path(),
        &dirs_home().join(".claude").join("sessions"),
    )
}

fn dirs_home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

/// Testable variant: explicit paths. Production wrapper computes paths from
/// `$HOME` and delegates here.
pub fn scan_and_append_at(
    projects_root: &Path,
    ledger: &Path,
    offsets_file: &Path,
    sessions_dir: &Path,
) -> ScanReport {
    if let Some(parent) = ledger.parent() {
        if fs::create_dir_all(parent).is_err() {
            return ScanReport::default();
        }
    }

    // Migration, append and compaction all rewrite or extend the same file and
    // must be exclusive against other scanners — including other processes.
    with_ledger_lock(ledger, || {
        scan_and_append_locked(projects_root, ledger, offsets_file, sessions_dir)
    })
    .unwrap_or_default()
}

fn scan_and_append_locked(
    projects_root: &Path,
    ledger: &Path,
    offsets_file: &Path,
    sessions_dir: &Path,
) -> ScanReport {
    migrate_v1_ledger(ledger, offsets_file);

    let needs_header = !ledger.exists();

    let Ok(ledger_file) = OpenOptions::new().create(true).append(true).open(ledger) else {
        return ScanReport::default();
    };
    let mut ledger_out = BufWriter::new(ledger_file);

    if needs_header {
        let _ = writeln!(ledger_out, "{HEADER}");
    }

    let mut offsets = load_offsets_at(offsets_file);
    let live = read_live_session_ids(sessions_dir);
    let files = find_jsonl_files_cached(projects_root);

    let mut report = ScanReport {
        files_scanned: files.len(),
        ..Default::default()
    };

    for jsonl in &files {
        let key = jsonl.display().to_string();
        let prev = offsets.get(&key).cloned().unwrap_or_default();
        let session_id = session_id_from_path(jsonl);
        let writer_alive = live.contains(&session_id);

        // Drained-skip: a JSONL whose writer process exited and which we
        // already drained on a prior scan can never grow again. Skip the
        // stat + open + read entirely. This is the dominant code path on
        // any box with many historical sessions — typically 99%+ of files.
        if !writer_alive && prev.drained {
            continue;
        }

        let Some((current_mtime, current_size)) = metadata_mtime_len(jsonl) else {
            continue;
        };

        // Fast skip: same mtime AND same size as last scan ⇒ no new
        // bytes can possibly exist. For dead writers also flip the
        // `drained` marker so the next scan can skip the stat above.
        if current_mtime == prev.mtime_ms && current_size == prev.last_byte {
            if !writer_alive && !prev.drained {
                offsets.insert(
                    key.clone(),
                    FileOffset {
                        last_byte: current_size,
                        mtime_ms: current_mtime,
                        drained: true,
                    },
                );
            }
            continue;
        }

        // Truncation / rewrite: fall back to full re-scan by resetting offset.
        let mut start = prev.last_byte;
        if current_size < prev.last_byte {
            start = 0;
        }
        if start == current_size {
            // mtime moved but bytes didn't — touch(1), partial overwrite
            // that landed exactly at the prior length, etc. Nothing new
            // to parse but we should still update the cached mtime so
            // the fast skip above kicks in next tick.
            offsets.insert(
                key.clone(),
                FileOffset {
                    last_byte: current_size,
                    mtime_ms: current_mtime,
                    drained: !writer_alive,
                },
            );
            continue;
        }

        let Ok(mut file) = File::open(jsonl) else {
            continue;
        };
        if start > 0 && file.seek(SeekFrom::Start(start)).is_err() {
            continue;
        }

        let reader = BufReader::new(&file);
        let sid = session_id_from_path(jsonl);
        let mut appended = 0u64;

        for line in reader.lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            let Some(TranscriptEvent::Message(msg)) = parse_line(&line) else {
                continue;
            };
            if msg.role != TranscriptRole::Assistant {
                continue;
            }
            let Some(usage) = msg.usage else { continue };
            if usage.input_tokens == 0
                && usage.cache_read_input_tokens == 0
                && usage.cache_creation_input_tokens == 0
                && usage.output_tokens == 0
            {
                continue;
            }
            let ts = msg.timestamp_ms.unwrap_or(current_mtime);
            let model = msg.model.as_deref().unwrap_or("");
            // A line with no message key can't be deduplicated against its
            // siblings, so key it by position instead — unique per line, which
            // preserves the pre-v2 "count every line" behaviour only for the
            // entries that genuinely carry no identity.
            let key = msg
                .message_key
                .clone()
                .unwrap_or_else(|| format!("{sid}#{ts}#{}", usage.output_tokens));
            let write_1h = usage.cache_creation_1h_input_tokens;
            let row = format!(
                "{},{},{},{},{},{},{},{},{}",
                ts,
                csv_escape(&sid),
                csv_escape(model),
                usage.input_tokens,
                usage.cache_read_input_tokens,
                usage.cache_creation_input_tokens.saturating_sub(write_1h),
                write_1h,
                usage.output_tokens,
                csv_escape(&key),
            );
            if writeln!(ledger_out, "{row}").is_ok() {
                appended += 1;
            }
        }

        if appended > 0 {
            report.files_updated += 1;
            report.rows_appended += appended;
        }

        offsets.insert(
            key,
            FileOffset {
                last_byte: current_size,
                mtime_ms: current_mtime,
                // We just performed a full read up to `current_size`. If
                // the writer is dead, future scans can skip this file
                // entirely via the drained-skip above.
                drained: !writer_alive,
            },
        );
    }

    let _ = ledger_out.flush();
    drop(ledger_out);
    save_offsets_at(offsets_file, &offsets);

    if fs::metadata(ledger).is_ok_and(|m| m.len() > compact_threshold()) {
        match compact_ledger(ledger) {
            Ok(kept) => crate::logger::log(
                "INFO",
                &format!("usage ledger: compacted to {kept} rows inside the retention window"),
            ),
            Err(e) => crate::logger::log("WARN", &format!("usage ledger: compaction failed: {e}")),
        }
    }

    report
}

/// Aggregate ledger rows whose timestamp falls in `[since_ms, now)`. Pass
/// `since_ms == 0` for the full-history total. Cost is computed per row
/// using current `models::resolve` prices.
/// Single ledger row in cached form. Cost is pre-computed at parse time
/// so the hot summary path doesn't redo the model-pricing lookup. Roughly
/// 110 bytes per row including the key; a full retention window of dense use
/// (~100k deduplicated responses observed) sits around 11 MB.
#[derive(Debug, Clone)]
struct LedgerRow {
    ts_ms: u64,
    fresh_input: u64,
    cache_read: u64,
    cache_write: u64,
    output: u64,
    cost_usd: f64,
    msg_key: String,
}

/// Process-local cache of parsed ledger rows. Avoids re-parsing the entire
/// 100MB+ usage_log.csv every time `load_summary` is called (3× per
/// 6-second tick, ~750ms/pass in release-mode Rust at 1.4M rows). With the
/// cache, the only per-tick I/O is reading bytes appended since the last
/// scan — typically zero or a few KB — plus a linear scan of in-memory
/// rows whose ts_ms >= cutoff.
///
/// Rows older than `MAX_RETENTION_MS` are evicted on every refresh so memory
/// doesn't grow unbounded, and `compact_ledger` drops them from the file once
/// it crosses `COMPACT_THRESHOLD_BYTES`.
const MAX_RETENTION_MS: u64 = 31 * 86_400_000;

/// Rows keyed by `msg_key`, which is what makes the ledger idempotent: a
/// rewritten JSONL is re-scanned and re-appended in full (see
/// `scan_and_append_at`), and one API response spans many JSONL lines
/// repeating the same cumulative usage. Both collapse here.
#[derive(Default)]
struct LedgerCache {
    last_scan_size: u64,
    rows: HashMap<String, LedgerRow>,
}

fn cache() -> &'static Mutex<LedgerCache> {
    static CACHE: OnceLock<Mutex<LedgerCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(LedgerCache::default()))
}

/// Read newly-appended bytes since the last refresh and merge them into the
/// in-memory cache. On file truncation (size shrunk) the cache is reset and
/// re-scanned from the start. Errors are swallowed — a missing or
/// unreadable ledger leaves the cache empty, same as the prior file-only
/// implementation.
fn refresh_cache_from(ledger: &Path) {
    let Ok(meta) = fs::metadata(ledger) else {
        return;
    };
    let current_size = meta.len();
    let mut c = cache().lock().expect("ledger cache mutex poisoned");

    let start = if current_size < c.last_scan_size {
        // Truncation / rotation: fall back to a full re-scan.
        c.rows.clear();
        0
    } else if current_size == c.last_scan_size {
        return;
    } else {
        c.last_scan_size
    };

    let Ok(mut file) = File::open(ledger) else {
        return;
    };
    if start > 0 && file.seek(SeekFrom::Start(start)).is_err() {
        return;
    }

    let reader = BufReader::new(&file);
    for (idx, line) in reader.lines().enumerate() {
        let Ok(line) = line else { break };
        // The header line is only present at the start of the file; on
        // tail reads `start > 0` we never encounter it. On full reads
        // `start == 0` it's the first line (idx 0) and the prefix check
        // catches it.
        if start == 0 && idx == 0 && line.starts_with("timestamp_ms") {
            continue;
        }
        let Some(row) = parse_csv_row(&line) else {
            continue;
        };
        insert_row(&mut c.rows, row);
    }
    c.last_scan_size = current_size;

    // Evict rows older than MAX_RETENTION_MS. The CSV is appended by
    // scan_and_append in file-walk order, NOT strict time order — each
    // batch interleaves rows from many JSONLs whose timestamps run in
    // parallel — so the in-memory `rows` are mostly-sorted but contain
    // local out-of-order regions. `Vec::retain` linearly scans every
    // row, which is correct regardless of order; using `partition_point`
    // here was a bug that silently dropped recent rows while keeping
    // ancient ones, undercounting weekly/monthly totals.
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let cutoff = now.saturating_sub(MAX_RETENTION_MS);
    c.rows.retain(|_, r| r.ts_ms >= cutoff);
}

/// Run `body` holding an exclusive advisory lock on the ledger's sidecar lock
/// file, mirroring `sandbox_registry`'s convention.
///
/// Appending and compaction must not interleave: compaction rewrites the file
/// from a snapshot and renames over it, so rows appended in between would be
/// dropped — and the offsets file already records those transcripts as
/// ingested, so nothing would ever re-read them. `SCAN_IN_FLIGHT` doesn't
/// cover this; it only guards the background entry point within one process,
/// leaving a direct `scan_and_append()` call and other agentctl processes
/// free to race.
fn with_ledger_lock<T>(ledger: &Path, body: impl FnOnce() -> T) -> std::io::Result<T> {
    use std::os::fd::AsRawFd;

    let lock_path = ledger.with_extension("lock");
    let lock_file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)?;
    let fd = lock_file.as_raw_fd();
    // SAFETY: `fd` is a valid open descriptor owned by `lock_file` for the
    // duration of the call; `flock` only reads it. LOCK_EX blocks until the
    // lock is acquired.
    if unsafe { libc::flock(fd, libc::LOCK_EX) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let result = body();
    unsafe { libc::flock(fd, libc::LOCK_UN) };
    Ok(result)
}

/// Retire a v1 ledger and rebuild from the transcripts.
///
/// v1 rows carry no `msg_key`, so their duplicates are indistinguishable from
/// genuine rows and cannot be removed after the fact — on a real 257 MB
/// ledger 84% of rows were byte-identical re-appends. Converting would
/// preserve that error, so the file is set aside and the next scan rebuilds
/// from `~/.claude/projects`, which Claude Code retains far longer than the
/// 31-day window anything here can read.
fn migrate_v1_ledger(ledger: &Path, offsets_file: &Path) {
    let Ok(file) = File::open(ledger) else {
        return;
    };
    let mut first = String::new();
    if BufReader::new(file).read_line(&mut first).is_err() {
        return;
    }
    if first.trim_end() != HEADER_V1 {
        return;
    }

    let backup = ledger.with_extension("v1.csv.bak");
    if fs::rename(ledger, &backup).is_err() {
        return;
    }
    // Offsets say "already ingested up to byte N" for every transcript; the
    // rebuild needs to re-read all of them.
    let _ = fs::remove_file(offsets_file);
    reset_cache();
    crate::logger::log(
        "INFO",
        &format!(
            "usage ledger: v1 schema retired to {} — rebuilding from transcripts \
             (v1 rows lacked a message key, so duplicate rows could not be removed)",
            backup.display()
        ),
    );
}

/// Rewrite the ledger with only what it can still answer for: rows inside the
/// retention window, one per `msg_key`.
///
/// The append path never removes anything, and retention is applied only to
/// the in-memory cache, so the file grows without bound — 93% of a real
/// 257 MB ledger was older than the 31-day window and unreadable by any
/// query.
fn compact_ledger(ledger: &Path) -> std::io::Result<u64> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let cutoff = now.saturating_sub(MAX_RETENTION_MS);

    let file = File::open(ledger)?;
    let mut keep: HashMap<String, String> = HashMap::new();
    let mut best: HashMap<String, u64> = HashMap::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if idx == 0 && line.starts_with("timestamp_ms") {
            continue;
        }
        let Some(row) = parse_csv_row(&line) else {
            continue;
        };
        if row.ts_ms < cutoff {
            continue;
        }
        if best.get(&row.msg_key).is_some_and(|out| *out >= row.output) {
            continue;
        }
        best.insert(row.msg_key.clone(), row.output);
        keep.insert(row.msg_key, line);
    }

    let tmp = ledger.with_extension("compact.tmp");
    {
        let mut out = BufWriter::new(File::create(&tmp)?);
        writeln!(out, "{HEADER}")?;
        for line in keep.values() {
            writeln!(out, "{line}")?;
        }
        out.flush()?;
    }
    fs::rename(&tmp, ledger)?;
    reset_cache();
    Ok(keep.len() as u64)
}

/// Drop the parsed-row cache so the next read re-scans from byte 0.
fn reset_cache() {
    if let Ok(mut c) = cache().lock() {
        c.rows.clear();
        c.last_scan_size = 0;
    }
}

/// Keep the highest-output row for a key. Streaming writes the same response
/// several times as it completes, with `output_tokens` growing each time and
/// the input side fixed, so the last/largest observation is the whole
/// response — keeping the first would undercount output badly.
fn insert_row(rows: &mut HashMap<String, LedgerRow>, row: LedgerRow) {
    match rows.get(&row.msg_key) {
        Some(existing) if existing.output >= row.output => {}
        _ => {
            rows.insert(row.msg_key.clone(), row);
        }
    }
}

fn parse_csv_row(line: &str) -> Option<LedgerRow> {
    let fields: Vec<&str> = line.splitn(FIELD_COUNT, ',').collect();
    if fields.len() != FIELD_COUNT {
        return None;
    }
    let ts_ms: u64 = fields[0].parse().ok()?;
    // fields[1] = session_id (unused for summary)
    let model = fields[2];
    let fresh: u64 = fields[3].parse().unwrap_or(0);
    let cache_read: u64 = fields[4].parse().unwrap_or(0);
    let cache_write_5m: u64 = fields[5].parse().unwrap_or(0);
    let cache_write_1h: u64 = fields[6].parse().unwrap_or(0);
    let output: u64 = fields[7].parse().unwrap_or(0);
    let key = fields[8].trim_end_matches('\n');

    let p = models::resolve(model).profile;
    let cost = (fresh as f64 * p.input_per_m
        + cache_read as f64 * p.cache_read_per_m
        + cache_write_5m as f64 * p.cache_write_per_m
        + cache_write_1h as f64 * p.cache_write_1h_per_m
        + output as f64 * p.output_per_m)
        / 1_000_000.0;
    Some(LedgerRow {
        ts_ms,
        fresh_input: fresh,
        cache_read,
        cache_write: cache_write_5m + cache_write_1h,
        output,
        cost_usd: cost,
        msg_key: key.to_string(),
    })
}

pub fn load_summary(since_ms: u64) -> UsageSummary {
    refresh_cache_from(&ledger_path());
    summarize_cached(since_ms)
}

fn summarize_cached(since_ms: u64) -> UsageSummary {
    let c = cache().lock().expect("ledger cache mutex poisoned");
    let mut summary = UsageSummary::default();
    // Linear scan with explicit ts filter. Cannot use partition_point
    // because the row ordering reflects scan_and_append's file-walk
    // order, not strict time ascending — a previous version used
    // `partition_point` here and silently undercounted week/month
    // totals by ~88% on a real ledger (1.4M rows from many concurrent
    // sessions).
    for row in c.rows.values() {
        if row.ts_ms < since_ms {
            continue;
        }
        summary.fresh_input += row.fresh_input;
        summary.cache_read += row.cache_read;
        summary.cache_write += row.cache_write;
        summary.output += row.output;
        summary.cost_usd += row.cost_usd;
        summary.msg_count += 1;
    }
    summary
}

#[cfg(test)]
fn reset_cache_for_tests() {
    let mut c = cache().lock().expect("ledger cache mutex poisoned");
    c.last_scan_size = 0;
    c.rows.clear();
}

/// Serialize tests that mutate the process-global ledger cache. Without
/// this, parallel cargo-test workers race on `cache()` state and one
/// test's `reset_cache_for_tests` can wipe another test's freshly-read
/// rows mid-assertion. Each cache-touching test acquires
/// `cache_test_lock()` at the top and holds it until done.
///
/// Poisoning is recovered from rather than propagated: the lock orders tests,
/// it guards no invariant, and each test installs its own fixture. Failing on
/// it reported one panicking test as four.
#[cfg(test)]
fn cache_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Testable variant: explicit ledger path, NO cache. Used by tests so each
/// test sees only its own fixture; the production path uses the
/// process-wide cache via `load_summary`.
#[cfg(test)]
pub fn load_summary_at(ledger: &Path, since_ms: u64) -> UsageSummary {
    let Ok(file) = File::open(ledger) else {
        return UsageSummary::default();
    };
    let reader = BufReader::new(file);
    let mut rows: HashMap<String, LedgerRow> = HashMap::new();
    for (idx, line) in reader.lines().enumerate() {
        let Ok(line) = line else { break };
        if idx == 0 && line.starts_with("timestamp_ms") {
            continue;
        }
        let Some(row) = parse_csv_row(&line) else {
            continue;
        };
        if row.ts_ms < since_ms {
            continue;
        }
        insert_row(&mut rows, row);
    }

    let mut summary = UsageSummary::default();
    for row in rows.values() {
        summary.fresh_input += row.fresh_input;
        summary.cache_read += row.cache_read;
        summary.cache_write += row.cache_write;
        summary.output += row.output;
        summary.cost_usd += row.cost_usd;
        summary.msg_count += 1;
    }
    summary
}

/// Convenience: current unix time in ms.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Each test gets its own tmp subdirectory; counter ensures uniqueness
    /// even when tests run in parallel.
    static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

    struct TestPaths {
        _root: PathBuf, // kept alive to own the tmp tree
        projects: PathBuf,
        ledger: PathBuf,
        offsets: PathBuf,
        sessions: PathBuf,
    }

    impl TestPaths {
        fn new(label: &str) -> Self {
            let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "claudectl-ledger-{}-{}-{n}",
                std::process::id(),
                label
            ));
            let _ = fs::remove_dir_all(&root);
            let projects = root.join("projects");
            let share = root.join("share");
            let sessions = root.join("sessions");
            fs::create_dir_all(&projects).unwrap();
            fs::create_dir_all(&share).unwrap();
            fs::create_dir_all(&sessions).unwrap();
            Self {
                ledger: share.join("usage_log.csv"),
                offsets: share.join("usage_offsets.json"),
                projects,
                sessions,
                _root: root,
            }
        }

        fn scan(&self) -> ScanReport {
            scan_and_append_at(&self.projects, &self.ledger, &self.offsets, &self.sessions)
        }

        /// Mark the JSONL named `<session_id>.jsonl` as having a live writer
        /// process. Without this, scan_and_append's drained-skip kicks in
        /// after the first scan and subsequent calls return 0 rows
        /// (correct for production — dead-writer files can't grow).
        fn mark_live(&self, session_id: &str) {
            let pid_path = self.sessions.join(format!("{session_id}-test-pid.json"));
            let body = format!(r#"{{"sessionId":"{session_id}"}}"#);
            fs::write(pid_path, body).unwrap();
        }

        fn summary(&self, since_ms: u64) -> UsageSummary {
            load_summary_at(&self.ledger, since_ms)
        }
    }

    impl Drop for TestPaths {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self._root);
        }
    }

    fn write_tmp(path: &Path, contents: &str) {
        let parent = path.parent().unwrap();
        fs::create_dir_all(parent).unwrap();
        let mut f = File::create(path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
    }

    fn fixture_assistant_line(
        ts: &str,
        model: &str,
        inp: u64,
        cr: u64,
        cw: u64,
        out: u64,
    ) -> String {
        format!(
            r#"{{"type":"assistant","timestamp":"{ts}","message":{{"role":"assistant","model":"{model}","usage":{{"input_tokens":{inp},"cache_read_input_tokens":{cr},"cache_creation_input_tokens":{cw},"output_tokens":{out}}},"content":[]}}}}"#
        )
    }

    #[test]
    fn scan_appends_assistant_usage_rows() {
        let p = TestPaths::new("scan-basic");
        let project = p.projects.join("-test/sess-abc.jsonl");
        let content = [
            fixture_assistant_line(
                "2026-04-22T10:00:00.000Z",
                "claude-opus-4-7",
                100,
                50,
                10,
                200,
            ),
            fixture_assistant_line(
                "2026-04-22T10:01:00.000Z",
                "claude-sonnet-4-6",
                80,
                20,
                5,
                120,
            ),
        ]
        .join("\n");
        write_tmp(&project, &content);

        let report = p.scan();
        assert_eq!(report.files_scanned, 1);
        assert_eq!(report.files_updated, 1);
        assert_eq!(report.rows_appended, 2);

        let summary = p.summary(0);
        assert_eq!(summary.msg_count, 2);
        assert_eq!(summary.fresh_input, 180);
        assert_eq!(summary.cache_read, 70);
        assert_eq!(summary.cache_write, 15);
        assert_eq!(summary.output, 320);
        assert!(summary.cost_usd > 0.0);
    }

    #[test]
    fn scan_is_incremental_across_runs() {
        let p = TestPaths::new("incremental");
        let project = p.projects.join("-test/sess-x.jsonl");
        // Mark this session as live so scan_and_append doesn't drain-skip
        // it on the second pass — we want to verify incremental APPENDS,
        // which only happen for live writers.
        p.mark_live("sess-x");
        write_tmp(
            &project,
            &fixture_assistant_line("2026-04-22T10:00:00.000Z", "claude-opus-4-7", 10, 0, 0, 5),
        );
        let r1 = p.scan();
        assert_eq!(r1.rows_appended, 1);

        // Append another message to the same JSONL.
        let mut f = OpenOptions::new().append(true).open(&project).unwrap();
        writeln!(
            f,
            "\n{}",
            fixture_assistant_line("2026-04-22T10:05:00.000Z", "claude-opus-4-7", 30, 0, 0, 7)
        )
        .unwrap();
        drop(f);

        let r2 = p.scan();
        assert_eq!(r2.rows_appended, 1, "only new bytes should be re-parsed");

        let summary = p.summary(0);
        assert_eq!(summary.msg_count, 2);
        assert_eq!(summary.fresh_input, 40);
        assert_eq!(summary.output, 12);
    }

    #[test]
    fn dead_writer_is_drained_once_then_skipped() {
        // First scan: writer is "dead" (no pointer file). Drains the
        // file in one pass and flips `drained=true` in offsets.
        // Second scan: even after appending more bytes, drained-skip
        // means the appended bytes are NOT picked up — correct because
        // a real dead writer cannot append, and we want O(0) work for
        // historical JSONLs.
        let p = TestPaths::new("drain");
        let project = p.projects.join("-test/sess-dead.jsonl");
        write_tmp(
            &project,
            &fixture_assistant_line("2026-04-22T10:00:00.000Z", "claude-opus-4-7", 10, 0, 0, 5),
        );
        let r1 = p.scan();
        assert_eq!(r1.rows_appended, 1);

        // Simulate a "phantom append" that should never happen for a
        // real dead writer but proves drained-skip is short-circuiting.
        let mut f = OpenOptions::new().append(true).open(&project).unwrap();
        writeln!(
            f,
            "\n{}",
            fixture_assistant_line("2026-04-22T10:05:00.000Z", "claude-opus-4-7", 99, 0, 0, 99)
        )
        .unwrap();
        drop(f);

        let r2 = p.scan();
        assert_eq!(
            r2.rows_appended, 0,
            "drained dead-writer files must be skipped without re-stat"
        );

        // Now mark the session as live + scan again. The file is no
        // longer skipped, the appended row is parsed.
        p.mark_live("sess-dead");
        let r3 = p.scan();
        assert_eq!(r3.rows_appended, 1, "live writer un-drains the file");
    }

    #[test]
    fn user_messages_and_zero_usage_are_ignored() {
        let p = TestPaths::new("filter");
        let project = p.projects.join("-test/sess-y.jsonl");
        let content = [
            r#"{"type":"user","message":{"role":"user","content":"hi"}}"#.to_string(),
            fixture_assistant_line("2026-04-22T10:00:00.000Z", "claude-opus-4-7", 0, 0, 0, 0),
            fixture_assistant_line("2026-04-22T10:01:00.000Z", "claude-opus-4-7", 1, 0, 0, 2),
        ]
        .join("\n");
        write_tmp(&project, &content);

        let report = p.scan();
        assert_eq!(report.rows_appended, 1);
        let summary = p.summary(0);
        assert_eq!(summary.msg_count, 1);
        assert_eq!(summary.fresh_input, 1);
        assert_eq!(summary.output, 2);
    }

    #[test]
    fn since_filter_windows_ledger_by_timestamp() {
        let p = TestPaths::new("since");
        let project = p.projects.join("-test/sess-z.jsonl");
        let content = [
            fixture_assistant_line("2026-04-20T10:00:00.000Z", "claude-opus-4-7", 100, 0, 0, 50),
            fixture_assistant_line("2026-04-22T10:00:00.000Z", "claude-opus-4-7", 10, 0, 0, 5),
        ]
        .join("\n");
        write_tmp(&project, &content);
        p.scan();

        // 2026-04-21T00:00:00 UTC ≈ 1776844800000 ms
        let cutoff = 1776844800000u64;
        let recent = p.summary(cutoff);
        assert_eq!(recent.msg_count, 1);
        assert_eq!(recent.fresh_input, 10);

        let all = p.summary(0);
        assert_eq!(all.msg_count, 2);
        assert_eq!(all.fresh_input, 110);
    }

    #[test]
    fn subagent_files_are_scanned_too() {
        let p = TestPaths::new("subagents");
        let sub = p
            .projects
            .join("-test/parent-session/subagents/agent-abc.jsonl");
        write_tmp(
            &sub,
            &fixture_assistant_line("2026-04-22T10:00:00.000Z", "claude-haiku", 100, 0, 0, 50),
        );
        let r = p.scan();
        assert_eq!(r.files_scanned, 1);
        assert_eq!(r.rows_appended, 1);
        let s = p.summary(0);
        assert_eq!(s.fresh_input, 100);
        assert_eq!(s.output, 50);
    }

    // ---- Cache tests --------------------------------------------------
    //
    // The cache is a process-global singleton, so these tests must
    // serialise via the same mutex `cache()` returns. Calling
    // `reset_cache_for_tests()` between scenarios + naming each ledger
    // file uniquely (TestPaths counter) keeps them order-independent.

    /// Render rows in the v2 schema. Each row gets a distinct `msg_key` from
    /// `prefix` + index so dedup treats them as separate responses; tests that
    /// want dedup exercised pass colliding keys explicitly.
    fn csv_body(rows: &[(u64, &str, u64, u64, u64, u64)], prefix: &str) -> String {
        let mut body = String::new();
        for (idx, (ts, model, fresh, cr, cw, out)) in rows.iter().enumerate() {
            body.push_str(&format!(
                "{ts},sess,{model},{fresh},{cr},{cw},0,{out},{prefix}{idx}\n"
            ));
        }
        body
    }

    fn write_csv_rows(path: &Path, rows: &[(u64, &str, u64, u64, u64, u64)]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut body = String::from(HEADER);
        body.push('\n');
        body.push_str(&csv_body(rows, "w"));
        std::fs::write(path, body).unwrap();
    }

    fn append_csv_rows(path: &Path, rows: &[(u64, &str, u64, u64, u64, u64)]) {
        let body = csv_body(rows, "a");
        let mut f = OpenOptions::new().append(true).open(path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    /// All cache tests share the global cache, so each test owns its file
    /// and resets state via `reset_cache_for_tests()`. Keep them in one
    /// `#[test]` so they run sequentially under any test-thread setting.
    #[test]
    fn cache_refresh_is_incremental_truncation_safe_and_filters_by_cutoff() {
        let _g = cache_test_lock();
        let p = TestPaths::new("cache");
        let ledger = p.ledger.clone();
        // Use real-clock-relative timestamps so MAX_RETENTION_MS doesn't
        // evict the test rows: a row from 2 hours ago is well within the
        // 31-day window. Spacing 60s apart keeps the cutoff filter test
        // unambiguous.
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let t1 = now - 7_200_000; // 2h ago
        let t2 = t1 + 60_000;
        let t3 = t2 + 60_000;
        let cutoff_between_t2_t3 = t2 + 30_000;

        // Initial population: cache is empty, refresh reads everything.
        reset_cache_for_tests();
        write_csv_rows(
            &ledger,
            &[
                (t1, "claude-opus-4-7", 100, 0, 0, 50),
                (t2, "claude-opus-4-7", 200, 0, 0, 100),
            ],
        );
        refresh_cache_from(&ledger);
        let s = summarize_cached(0);
        assert_eq!(s.msg_count, 2);
        assert_eq!(s.fresh_input, 300);

        // Append: only the new row should be parsed; cache must show
        // 3 rows total without re-reading the first 2.
        append_csv_rows(&ledger, &[(t3, "claude-opus-4-7", 5, 0, 0, 1)]);
        refresh_cache_from(&ledger);
        let s = summarize_cached(0);
        assert_eq!(s.msg_count, 3);
        assert_eq!(s.fresh_input, 305);

        // Cutoff filter: only rows at-or-after `since_ms` count.
        let s = summarize_cached(cutoff_between_t2_t3);
        assert_eq!(s.msg_count, 1);
        assert_eq!(s.fresh_input, 5);

        // File truncation/rotation: write a smaller file in place. Cache
        // must reset and re-scan from byte 0.
        let t4 = now - 60_000;
        write_csv_rows(&ledger, &[(t4, "claude-opus-4-7", 7, 0, 0, 0)]);
        refresh_cache_from(&ledger);
        let s = summarize_cached(0);
        assert_eq!(s.msg_count, 1);
        assert_eq!(s.fresh_input, 7);

        // Cleanup so a later test starting with the same global cache
        // has no leftover rows from this one.
        reset_cache_for_tests();
    }

    #[test]
    fn cache_correctly_filters_unsorted_rows() {
        let _g = cache_test_lock();
        // Regression: the cache used `partition_point` for both the
        // since-cutoff filter AND the retention eviction. Since the
        // ledger CSV is appended in file-walk order (not strict time
        // ascending), the array can be mostly-sorted with local
        // out-of-order regions — exactly the input that breaks
        // `partition_point`'s binary-search assumption. On the real
        // ledger this undercounted week/month totals by ~88%.
        //
        // Verify the fix by stuffing the cache with deliberately
        // unsorted timestamps and asserting summarize_cached counts
        // EVERY row at-or-after the cutoff regardless of its position.
        let p = TestPaths::new("cache_unsorted");
        let ledger = p.ledger.clone();

        // Use real-clock-relative timestamps to dodge MAX_RETENTION_MS
        // eviction on every row.
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let recent = |secs_ago: u64| now - secs_ago * 1_000;

        // Order chosen to break a binary-search partition: a "newer"
        // timestamp early in the file, then "older" ones, then newer
        // again. partition_point would return a position partway
        // through and miss real rows; the linear scan must find them
        // all.
        reset_cache_for_tests();
        write_csv_rows(
            &ledger,
            &[
                (recent(60), "claude-opus-4-7", 100, 0, 0, 50),
                (recent(120), "claude-opus-4-7", 200, 0, 0, 100),
                (recent(30), "claude-opus-4-7", 5, 0, 0, 1),
                (recent(180), "claude-opus-4-7", 300, 0, 0, 200),
                (recent(15), "claude-opus-4-7", 7, 0, 0, 2),
                (recent(90), "claude-opus-4-7", 10, 0, 0, 4),
            ],
        );
        refresh_cache_from(&ledger);

        let all = summarize_cached(0);
        assert_eq!(
            all.msg_count, 6,
            "all rows must be summed regardless of order"
        );
        assert_eq!(all.fresh_input, 100 + 200 + 5 + 300 + 7 + 10);

        // Cutoff at "75 s ago": should match the 4 rows whose
        // recent_ago <= 75 (i.e. recent(60), recent(30), recent(15),
        // recent(60))? Let me re-check by listing:
        //   recent(60)  → ts > cutoff, included
        //   recent(120) → ts < cutoff, excluded
        //   recent(30)  → included
        //   recent(180) → excluded
        //   recent(15)  → included
        //   recent(90)  → excluded
        // = 3 included. partition_point on this unsorted array would
        // commonly return the wrong start index and miss rows.
        let cutoff = recent(75);
        let recent_summary = summarize_cached(cutoff);
        assert_eq!(
            recent_summary.msg_count, 3,
            "summarize_cached must scan EVERY row, not binary-search a sorted prefix"
        );
        assert_eq!(recent_summary.fresh_input, 100 + 5 + 7);

        reset_cache_for_tests();
    }

    #[test]
    fn cache_eviction_keeps_recent_rows_in_unsorted_array() {
        let _g = cache_test_lock();
        // Same regression class for the retention eviction path. With
        // ages mixed (some > 31d, some <), partition_point used to
        // drain a fragile prefix that didn't correspond to "everything
        // older than 31d"; sometimes recent rows landed in the prefix
        // and got evicted while ancient rows survived. The retain-based
        // fix must drop only the truly-ancient rows.
        let p = TestPaths::new("cache_evict_unsorted");
        let ledger = p.ledger.clone();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let ancient = now.saturating_sub(MAX_RETENTION_MS + 86_400_000);
        let r60 = now - 60_000;
        let r10 = now - 10_000;

        reset_cache_for_tests();
        // Interleave ancient + recent so any partition-based eviction
        // would behave wrong.
        write_csv_rows(
            &ledger,
            &[
                (r60, "claude-opus-4-7", 1, 0, 0, 0),
                (ancient, "claude-opus-4-7", 999, 0, 0, 0),
                (r10, "claude-opus-4-7", 2, 0, 0, 0),
                (ancient + 100, "claude-opus-4-7", 999, 0, 0, 0),
                (r60 + 200, "claude-opus-4-7", 4, 0, 0, 0),
            ],
        );
        refresh_cache_from(&ledger);

        let all = summarize_cached(0);
        assert_eq!(
            all.msg_count, 3,
            "ancient rows must be evicted, recent rows must survive"
        );
        assert_eq!(all.fresh_input, 1 + 2 + 4);

        reset_cache_for_tests();
    }

    #[test]
    fn cache_evicts_rows_older_than_max_retention() {
        let _g = cache_test_lock();
        let p = TestPaths::new("cache_evict");
        let ledger = p.ledger.clone();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        // One row inside the retention window, one well outside it.
        let recent = now - 60_000;
        let ancient = now.saturating_sub(MAX_RETENTION_MS + 86_400_000);

        reset_cache_for_tests();
        write_csv_rows(
            &ledger,
            &[
                (ancient, "claude-opus-4-7", 999, 0, 0, 0),
                (recent, "claude-opus-4-7", 1, 0, 0, 0),
            ],
        );
        refresh_cache_from(&ledger);
        // Asking for the full window must yield only the recent row;
        // the ancient one was evicted on refresh.
        let s = summarize_cached(0);
        assert_eq!(s.msg_count, 1);
        assert_eq!(s.fresh_input, 1);

        reset_cache_for_tests();
    }

    // ── Perf regression smoke checks ───────────────────────────────────
    //
    // These tests prime the in-memory ledger cache with a large number of
    // rows and assert that hot-path operations stay fast. The bounds are
    // intentionally generous — they only catch egregious regressions
    // (e.g. someone reverts a perf optimisation and the path becomes
    // linear or worse). They will NOT flake on slow CI.
    //
    // Calibration on Andre's 2026-04-28 sandbox (Linux, debug build):
    //   - 50_000-row primed cache:
    //       summarize_cached:  ~5 ms (linear scan of all rows)
    //       eviction retain:   ~3 ms
    //   - 100_000-row primed cache:
    //       summarize_cached: ~10 ms
    //       eviction retain:   ~6 ms
    //
    // The asserted bounds (200 ms / 500 ms) leave 20× headroom — they
    // fire only if a path regresses to e.g. O(N²) or accidentally
    // re-parses the CSV on every call.

    /// Regression: `load_summary` must stay fast on a warm cache.
    /// Before commit 0d6e2d3a it linear-scanned the full 105 MB CSV
    /// (~750 ms per call) every time and was called 3× per tick.
    ///
    /// Measured against a parse of the same rows on the same machine instead of
    /// a millisecond ceiling — the ceiling was unreachable through a sandbox
    /// mount and far too generous on a fast disk. Nine cached queries must cost
    /// less than *half* one parse; the regression this guards made each of the
    /// nine cost a whole parse, so the two are never close.
    #[test]
    fn perf_summarize_cached_costs_far_less_than_parsing() {
        let _g = cache_test_lock();
        let paths = TestPaths::new("perf_cached_vs_parse");
        reset_cache_for_tests();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        write_synthetic_ledger(&paths.ledger, 50_000);

        // Both sides are best-of-3. A single wall-clock sample taken while the
        // rest of the suite saturates the CPU can be descheduled mid-run, and
        // one unlucky sample on either side flips the comparison; the minimum
        // is the sample least contaminated by that, and taking it symmetrically
        // keeps the comparison honest.
        let parse = (0..3)
            .map(|_| {
                reset_cache_for_tests();
                let start = std::time::Instant::now();
                refresh_cache_from(&paths.ledger);
                start.elapsed()
            })
            .min()
            .expect("at least one sample");

        let cached = (0..3)
            .map(|_| {
                let start = std::time::Instant::now();
                for _ in 0..3 {
                    // Mimic a tick's day/week/month query trio.
                    let _ = summarize_cached(now.saturating_sub(86_400_000));
                    let _ = summarize_cached(now.saturating_sub(7 * 86_400_000));
                    let _ = summarize_cached(now.saturating_sub(30 * 86_400_000));
                }
                start.elapsed()
            })
            .min()
            .expect("at least one sample");

        assert!(
            cached * 2 < parse,
            "9 cached queries took {} ms against a {} ms parse of the same 50k \
             rows — summarize_cached is re-reading the ledger",
            cached.as_millis(),
            parse.as_millis()
        );
        reset_cache_for_tests();
    }

    /// Regression: `refresh_cache_from` eviction must be linear, not
    /// quadratic. Before the partition_point→retain fix (commit
    /// 0bb91466) it was using binary search on unsorted data — buggy
    /// but fast. The retain path is correct AND must remain fast.
    ///
    /// `#[ignore]` because the assertion compares two wall-clock samples
    /// while the rest of the suite saturates every core, so it measures the
    /// scheduler as much as the code — it failed roughly one run in three,
    /// and best-of-3 sampling only softened that. The eviction *correctness*
    /// this guards is covered deterministically by
    /// `cache_eviction_keeps_recent_rows_in_unsorted_array` and
    /// `cache_correctly_filters_unsorted_rows`; run this one on demand with
    /// `cargo test -- --ignored perf_eviction` when touching the cache path.
    /// Measured 1.91-2.18x for 2x the rows, against a 3x ceiling.
    #[test]
    #[ignore = "wall-clock ratio; flaky under a saturated parallel suite"]
    fn perf_eviction_stays_linear_with_many_rows() {
        let _g = cache_test_lock();
        let p = TestPaths::new("perf_evict");
        let ledger = p.ledger.clone();
        reset_cache_for_tests();

        // Write a CSV with 50k rows where roughly half are within
        // retention and half are ancient — exercises the retain path
        // on a meaningful slice.
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let recent_base = now - 60_000;
        let ancient_base = now.saturating_sub(MAX_RETENTION_MS + 86_400_000);
        let mut body = String::from(HEADER);
        body.push('\n');
        for i in 0..50_000_u64 {
            let ts = if i.is_multiple_of(2) {
                recent_base - i
            } else {
                ancient_base - i
            };
            body.push_str(&format!("{ts},sess,claude-opus-4-7,1,0,0,0,1,perf{i}\n"));
        }
        std::fs::create_dir_all(ledger.parent().unwrap()).unwrap();
        std::fs::write(&ledger, &body).unwrap();

        // Half the work, on the same machine, as the control: linear eviction
        // doubles with the row count, quadratic quadruples. A 3× ceiling
        // separates the two with room for scheduler noise, and unlike a
        // millisecond bound it means the same thing on a laptop, a CI runner and
        // through a sandbox mount.
        let half = p.ledger.with_extension("half.csv");
        let mut half_body = String::from(HEADER);
        half_body.push('\n');
        for line in body.lines().skip(1).take(25_000) {
            half_body.push_str(line);
            half_body.push('\n');
        }
        std::fs::write(&half, half_body).unwrap();

        // Best-of-N per size. A single wall-clock sample taken while the rest
        // of the suite saturates the CPU can be descheduled mid-run, and the
        // ratio of two such samples was flaky enough to fail ~2 runs in 3.
        // The minimum is the sample least contaminated by that noise, so the
        // comparison measures the code rather than the scheduler.
        let best_of = |path: &Path| {
            (0..3)
                .map(|_| {
                    reset_cache_for_tests();
                    let start = std::time::Instant::now();
                    refresh_cache_from(path);
                    start.elapsed()
                })
                .min()
                .expect("at least one sample")
        };
        let half_elapsed = best_of(&half);
        let elapsed = best_of(&ledger);
        assert!(
            elapsed < half_elapsed.max(std::time::Duration::from_millis(1)) * 3,
            "50k rows took {} ms against {} ms for 25k — eviction is scaling \
             worse than linearly",
            elapsed.as_millis(),
            half_elapsed.as_millis()
        );

        // Sanity: roughly half the rows survive eviction (the recent
        // half). Use a generous bound to stay deterministic.
        let s = summarize_cached(0);
        assert!(
            s.msg_count > 20_000 && s.msg_count < 30_000,
            "expected ~25k recent rows after eviction, got {}",
            s.msg_count
        );

        reset_cache_for_tests();
    }

    /// Write `n` synthetic rows into `ledger`, spread across the past 30 days so
    /// the day/week/month cutoffs each match a distinct subset.
    ///
    /// Takes the path rather than returning one: `TestPaths` deletes its tree on
    /// drop, so a helper that owned the guard and handed back a `PathBuf` would
    /// return a path to a directory that no longer exists — and a timing ratio
    /// over a missing ledger passes vacuously. The caller keeps the guard alive.
    /// It also leaves the cache alone, so callers can time `refresh_cache_from`
    /// as their own baseline.
    fn write_synthetic_ledger(ledger: &Path, n: usize) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let mut body = String::from(HEADER);
        body.push('\n');
        for i in 0..n as u64 {
            let ts = now - (i * (30 * 86_400_000) / n.max(1) as u64);
            body.push_str(&format!("{ts},sess,claude-opus-4-7,1,0,0,0,1,perf{i}\n"));
        }
        std::fs::create_dir_all(ledger.parent().unwrap()).unwrap();
        std::fs::write(ledger, body).unwrap();
    }

    /// The usage ledger must live inside the host-shared, bind-mounted root.
    ///
    /// Exercised through the real resolver with `HOME` pinned, not by
    /// re-deriving the path in the test: the regression this guards against was
    /// a call site quietly routed to `~/.local/share/agentctl`, which no
    /// assertion about `product` alone would have caught.
    #[test]
    fn ledger_dir_resolves_inside_the_shared_state_root() {
        let _guard = crate::sandbox_registry::tests::env_guard();
        let home = tempfile::tempdir().unwrap();
        let saved = std::env::var_os("HOME");
        // SAFETY: env access is serialised by the held lock.
        unsafe { std::env::set_var("HOME", home.path()) };
        let resolved = ledger_dir();
        unsafe {
            match saved {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
        }
        assert_eq!(resolved, crate::product::shared_state_root(home.path()));
        assert!(
            resolved.ends_with("claudectl"),
            "the bind mount is named claudectl, got {}",
            resolved.display()
        );
    }

    /// An assistant line carrying the message identity the real transcripts
    /// have. `id`/`requestId` are what make one API response countable once.
    fn fixture_keyed_line(ts: &str, model: &str, id: &str, req: &str, out: u64) -> String {
        format!(
            r#"{{"type":"assistant","timestamp":"{ts}","requestId":"{req}","message":{{"id":"{id}","role":"assistant","model":"{model}","usage":{{"input_tokens":10,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":{out}}},"content":[]}}}}"#
        )
    }

    /// Same, minus the `timestamp` field, so the scanner stamps the row with
    /// the file's mtime — i.e. now. Tests that let compaction run need rows
    /// inside the retention window, or compaction correctly discards them and
    /// the test proves nothing.
    fn fixture_keyed_line_now(id: &str, req: &str, out: u64) -> String {
        format!(
            r#"{{"type":"assistant","requestId":"{req}","message":{{"id":"{id}","role":"assistant","model":"claude-opus-5","usage":{{"input_tokens":10,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":{out}}},"content":[]}}}}"#
        )
    }

    /// Streaming writes one response as several lines that repeat the same
    /// cumulative usage with a growing `output_tokens`. They are one response
    /// and must be billed once, at the final output count.
    #[test]
    fn streaming_partials_of_one_response_are_counted_once() {
        let p = TestPaths::new("dedup-stream");
        let project = p.projects.join("-test/sess-stream.jsonl");
        fs::create_dir_all(project.parent().unwrap()).unwrap();
        let body = [
            fixture_keyed_line(
                "2026-04-22T10:00:00.000Z",
                "claude-opus-5",
                "msg_a",
                "req_a",
                5,
            ),
            fixture_keyed_line(
                "2026-04-22T10:00:01.000Z",
                "claude-opus-5",
                "msg_a",
                "req_a",
                40,
            ),
            fixture_keyed_line(
                "2026-04-22T10:00:02.000Z",
                "claude-opus-5",
                "msg_a",
                "req_a",
                900,
            ),
        ]
        .join("\n");
        fs::write(&project, format!("{body}\n")).unwrap();

        scan_and_append_at(&p.projects, &p.ledger, &p.offsets, &p.sessions);
        let s = load_summary_at(&p.ledger, 0);
        assert_eq!(s.msg_count, 1, "three lines are one response");
        assert_eq!(
            s.output, 900,
            "the completed response, not the first partial"
        );
        assert_eq!(s.fresh_input, 10, "input must not be triple-counted");
    }

    /// Claude Code rewrites transcripts, which shrinks the file and makes the
    /// scanner re-read it from byte 0 and append every row again. That is the
    /// longitudinal leak: it must not change any total.
    #[test]
    fn rescanning_a_rewritten_transcript_does_not_double_count() {
        let p = TestPaths::new("dedup-rewrite");
        let project = p.projects.join("-test/sess-rw.jsonl");
        fs::create_dir_all(project.parent().unwrap()).unwrap();
        let line = fixture_keyed_line(
            "2026-04-22T10:00:00.000Z",
            "claude-opus-5",
            "msg_b",
            "req_b",
            100,
        );
        fs::write(&project, format!("{line}\n{line}\n")).unwrap();

        scan_and_append_at(&p.projects, &p.ledger, &p.offsets, &p.sessions);
        let first = load_summary_at(&p.ledger, 0);

        // Rewrite shorter, forcing the truncation branch to re-append.
        fs::write(&project, format!("{line}\n")).unwrap();
        scan_and_append_at(&p.projects, &p.ledger, &p.offsets, &p.sessions);
        let second = load_summary_at(&p.ledger, 0);

        assert_eq!(first.msg_count, 1);
        assert_eq!(second, first, "a re-scan must not inflate any total");
    }

    /// 1h cache writes bill at 2x base input, not the 5m 1.25x.
    #[test]
    fn hourly_cache_writes_are_billed_at_the_hourly_rate() {
        let p = TestPaths::new("cache-1h");
        let project = p.projects.join("-test/sess-1h.jsonl");
        fs::create_dir_all(project.parent().unwrap()).unwrap();
        // 1M cache-creation tokens, all 1-hour, on a $5/MTok-input model.
        let line = r#"{"type":"assistant","timestamp":"2026-04-22T10:00:00.000Z","requestId":"req_c","message":{"id":"msg_c","role":"assistant","model":"claude-opus-5","usage":{"input_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":1000000,"cache_creation":{"ephemeral_1h_input_tokens":1000000},"output_tokens":0},"content":[]}}"#;
        fs::write(&project, format!("{line}\n")).unwrap();

        scan_and_append_at(&p.projects, &p.ledger, &p.offsets, &p.sessions);
        let s = load_summary_at(&p.ledger, 0);
        assert_eq!(s.cache_write, 1_000_000);
        // 1M tokens x $10/MTok (2x the $5 base), not $6.25 (1.25x).
        assert!(
            (s.cost_usd - 10.0).abs() < 1e-9,
            "expected the 1h rate ($10), got ${:.4}",
            s.cost_usd
        );
    }

    /// Compaction must preserve every total while dropping rows the ledger can
    /// no longer answer for.
    #[test]
    fn compaction_preserves_totals_and_drops_unreadable_rows() {
        let _g = cache_test_lock();
        let p = TestPaths::new("compact");
        let now = now_ms();
        let ancient = now.saturating_sub(MAX_RETENTION_MS + 86_400_000);
        let mut body = String::from(HEADER);
        body.push('\n');
        // Same key three times (a re-append), one in-window unique row, and an
        // ancient row that no query can reach.
        for _ in 0..3 {
            body.push_str(&format!("{now},sess,claude-opus-5,10,0,0,0,100,dup\n"));
        }
        body.push_str(&format!("{now},sess,claude-opus-5,10,0,0,0,100,uniq\n"));
        body.push_str(&format!("{ancient},sess,claude-opus-5,999,0,0,0,999,old\n"));
        fs::create_dir_all(p.ledger.parent().unwrap()).unwrap();
        fs::write(&p.ledger, body).unwrap();

        let before = load_summary_at(&p.ledger, now.saturating_sub(86_400_000));
        let kept = compact_ledger(&p.ledger).unwrap();
        let after = load_summary_at(&p.ledger, now.saturating_sub(86_400_000));

        assert_eq!(kept, 2, "one row per key inside the window");
        assert_eq!(
            before, after,
            "compaction must not change any visible total"
        );
        let raw = fs::read_to_string(&p.ledger).unwrap();
        assert!(!raw.contains(",old\n"), "out-of-window rows are dropped");
    }

    /// Invariant net for concurrent scanning with compaction enabled: no
    /// response may go missing.
    ///
    /// This does NOT reliably reproduce the underlying race — the loss window
    /// (an append through a handle to the inode compaction just renamed away)
    /// is microseconds wide, and this test passed with the lock removed. It is
    /// a regression net for gross breakage, not proof the lock is load-bearing;
    /// `ledger_lock_serialises_holders` covers the lock itself.
    #[test]
    fn concurrent_scans_do_not_lose_rows() {
        let p = TestPaths::new("concurrent");
        // Force compaction on every pass. At the production threshold a test
        // ledger never triggers it, so the race this guards would never be
        // exercised and the test would pass for the wrong reason. Verified:
        // with the ledger lock removed and this set, all 40 responses are lost.
        COMPACT_THRESHOLD_BYTES.store(512, std::sync::atomic::Ordering::Relaxed);
        const TOTAL: usize = 400;
        fs::create_dir_all(&p.projects).unwrap();
        for i in 0..40 {
            let f = p.projects.join(format!("-test/sess-{i}.jsonl"));
            fs::create_dir_all(f.parent().unwrap()).unwrap();
            fs::write(
                &f,
                format!(
                    "{}\n",
                    fixture_keyed_line_now(&format!("msg_{i}"), &format!("req_{i}"), 10)
                ),
            )
            .unwrap();
        }

        // One writer keeps producing transcripts while several scanners ingest
        // and compact. The loss window is an append landing between
        // compaction's snapshot read and its rename: the row vanishes, but the
        // offsets already mark that transcript ingested, so it never returns.
        let done = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for i in 40..TOTAL {
                    let f = p.projects.join(format!("-test/sess-{i}.jsonl"));
                    let _ = fs::create_dir_all(f.parent().unwrap());
                    let _ = fs::write(
                        &f,
                        format!(
                            "{}\n",
                            fixture_keyed_line_now(&format!("msg_{i}"), &format!("req_{i}"), 10)
                        ),
                    );
                }
                done.store(true, std::sync::atomic::Ordering::Release);
            });
            for _ in 0..4 {
                scope.spawn(|| {
                    while !done.load(std::sync::atomic::Ordering::Acquire) {
                        scan_and_append_at(&p.projects, &p.ledger, &p.offsets, &p.sessions);
                    }
                });
            }
        });
        // Final pass so anything written after the last scan is ingested.
        scan_and_append_at(&p.projects, &p.ledger, &p.offsets, &p.sessions);

        let s = load_summary_at(&p.ledger, 0);
        COMPACT_THRESHOLD_BYTES.store(64 * 1024 * 1024, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            s.msg_count, TOTAL as u64,
            "every response must survive concurrent scan+compaction"
        );
        assert_eq!(s.output, TOTAL as u64 * 10);
    }

    /// The lock must actually exclude, since compaction's snapshot-then-rename
    /// is only safe if no appender can run inside it. Threads that overlap
    /// would observe `inside > 1`.
    #[test]
    fn ledger_lock_serialises_holders() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let p = TestPaths::new("lock-excl");
        fs::create_dir_all(p.ledger.parent().unwrap()).unwrap();
        let inside = AtomicUsize::new(0);
        let max_seen = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..50 {
                        with_ledger_lock(&p.ledger, || {
                            let n = inside.fetch_add(1, Ordering::AcqRel) + 1;
                            max_seen.fetch_max(n, Ordering::AcqRel);
                            std::thread::yield_now();
                            inside.fetch_sub(1, Ordering::AcqRel);
                        })
                        .unwrap();
                    }
                });
            }
        });

        assert_eq!(
            max_seen.load(Ordering::Acquire),
            1,
            "two holders were inside the ledger lock at once"
        );
    }
}
