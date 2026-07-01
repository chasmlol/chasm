//! `--replay` parity harness.
//!
//! Two byte-for-byte checks against real Node-produced files, runnable offline
//! before ever loading the game:
//!
//! 1. **Archive round-trip** — parse each archived request in `processed/`, run it
//!    back through [`build_native_archived_request`], and diff against the original
//!    bytes. `processed/` files were produced by the same serializer (which is
//!    deterministic, no timestamp), so a clean parser+archiver reproduces them
//!    exactly. Any diff is a parser or archiver parity bug.
//!
//! 2. **Response framing** — for each leftover Node response in `outbox/`, split it
//!    into lines and re-frame via [`frame_lines`], diffing against the original.
//!    This locks down line sanitization, the `\r\n` joins, and the trailing `\r\n`
//!    against real response bytes. (AI-dependent content — the response text and
//!    the game_master tail — is reproduced verbatim from the golden, so this is a
//!    pure framing check; text/caption stripping parity is a Section 2 concern.)
//!
//! `--replay <path>`: if `<path>/processed` exists, treat `<path>` as a native
//! root and run both checks; otherwise treat `<path>` itself as a directory of
//! request files and run only the archive check.

use std::path::{Path, PathBuf};

use crate::protocol::{
    build_native_archived_request, frame_lines, parse_native_text_request, split_crlf,
};

struct CheckResult {
    checked: usize,
    passed: usize,
    /// Older-format golden files that differ only by trailing empty metadata lines
    /// our current serializer adds. Compatible, not a parity regression.
    legacy: usize,
    diffs: Vec<String>,
}

impl CheckResult {
    fn failed(&self) -> usize {
        self.checked - self.passed - self.legacy
    }
}

pub fn run_replay(target: &Path) -> anyhow::Result<()> {
    if !target.exists() {
        anyhow::bail!("replay target does not exist: {}", target.display());
    }

    let (request_dir, response_dir) = if target.join("processed").is_dir() {
        (target.join("processed"), Some(target.join("outbox")))
    } else {
        (target.to_path_buf(), None)
    };

    println!("== NVBridge Rust bridge --replay parity ==");
    println!("request dir : {}", request_dir.display());

    let archive = check_archive_round_trip(&request_dir)?;
    report("archive round-trip", &archive);

    let mut any_failed = archive.failed() > 0;

    if let Some(dir) = response_dir {
        if dir.is_dir() {
            println!("response dir: {}", dir.display());
            let framing = check_response_framing(&dir)?;
            report("response framing", &framing);
            any_failed |= framing.failed() > 0;
        }
    }

    if archive.checked == 0 {
        anyhow::bail!("no request files found under {}", request_dir.display());
    }
    if any_failed {
        anyhow::bail!("replay parity FAILED — see diffs above");
    }
    println!("\nreplay parity OK — zero diffs.");
    Ok(())
}

fn check_archive_round_trip(dir: &Path) -> anyhow::Result<CheckResult> {
    let mut result = CheckResult {
        checked: 0,
        passed: 0,
        legacy: 0,
        diffs: Vec::new(),
    };
    for path in txt_files(dir)? {
        let original = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => continue, // skip unreadable / non-UTF8 (e.g. stray binaries)
        };
        result.checked += 1;
        let request = parse_native_text_request(&path, &original);
        let rebuilt = build_native_archived_request(&request);
        if rebuilt == original {
            result.passed += 1;
        } else if legacy_compatible(&original, &rebuilt) {
            result.legacy += 1;
        } else if result.diffs.len() < 5 {
            result
                .diffs
                .push(format!("{}: {}", file_name(&path), first_diff(&original, &rebuilt)));
        }
    }
    Ok(result)
}

fn check_response_framing(dir: &Path) -> anyhow::Result<CheckResult> {
    let mut result = CheckResult {
        checked: 0,
        passed: 0,
        legacy: 0,
        diffs: Vec::new(),
    };
    for path in txt_files(dir)? {
        let original = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => continue,
        };
        result.checked += 1;
        let mut lines: Vec<String> = split_crlf(&original).iter().map(|s| s.to_string()).collect();
        // Drop the single trailing empty produced by the file's final `\r\n`.
        if lines.last().map(|s| s.is_empty()).unwrap_or(false) {
            lines.pop();
        }
        let rebuilt = frame_lines(&lines);
        if rebuilt == original {
            result.passed += 1;
        } else if result.diffs.len() < 5 {
            result
                .diffs
                .push(format!("{}: {}", file_name(&path), first_diff(&original, &rebuilt)));
        }
    }
    Ok(result)
}

fn report(label: &str, r: &CheckResult) {
    println!(
        "  {label}: {} checked, {} exact, {} legacy-format, {} failed",
        r.checked,
        r.passed,
        r.legacy,
        r.failed()
    );
    for diff in &r.diffs {
        println!("    DIFF {diff}");
    }
}

/// True when the golden file is an older, shorter archive format that differs
/// from our current output only by trailing empty metadata lines (e.g. the
/// pre-`transcript`/`voice_request` 10-field format). The original must be an
/// exact prefix of our output, with every extra line we add being empty — so a
/// genuine field-level mismatch is never masked.
fn legacy_compatible(original: &str, rebuilt: &str) -> bool {
    let o = content_lines(original);
    let r = content_lines(rebuilt);
    if r.len() <= o.len() {
        return false;
    }
    if o[..] != r[..o.len()] {
        return false;
    }
    r[o.len()..].iter().all(|line| line.is_empty())
}

/// Lines with the single trailing-empty artifact of the final `\r\n` removed.
fn content_lines(s: &str) -> Vec<&str> {
    let mut lines = split_crlf(s);
    if lines.last().map(|l| l.is_empty()).unwrap_or(false) {
        lines.pop();
    }
    lines
}

fn txt_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => anyhow::bail!("reading {}: {e}", dir.display()),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("txt"))
                .unwrap_or(false)
        {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?")
        .to_string()
}

/// Human-readable description of where two strings first diverge.
fn first_diff(a: &str, b: &str) -> String {
    let a_lines: Vec<&str> = split_crlf(a);
    let b_lines: Vec<&str> = split_crlf(b);
    let max = a_lines.len().max(b_lines.len());
    for i in 0..max {
        let av = a_lines.get(i).copied().unwrap_or("<none>");
        let bv = b_lines.get(i).copied().unwrap_or("<none>");
        if av != bv {
            return format!("line {i}: node={av:?} rust={bv:?}");
        }
    }
    if a.len() != b.len() {
        return format!("byte length node={} rust={}", a.len(), b.len());
    }
    "differs (no line-level diff found)".to_string()
}
