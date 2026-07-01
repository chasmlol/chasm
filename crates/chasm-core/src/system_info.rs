//! Boot-time host detection (CPU / RAM / GPU / VRAM) + a model-fit helper that
//! drives the "recommended" badges next to downloadable models (LLM + retrieval).
//!
//! Detection is best-effort and never panics: missing tools just yield `None`.
//! It's meant to be run once at startup and cached in the app state.

use std::process::Command;

use serde::Serialize;

/// A snapshot of the host's relevant hardware, captured once at boot.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SystemInfo {
    /// Logical CPU cores (`available_parallelism`).
    pub cpu_cores: usize,
    /// Total system RAM in GB, when detectable.
    pub ram_gb: Option<f64>,
    /// Primary NVIDIA GPU name, when an `nvidia-smi` is present.
    pub gpu_name: Option<String>,
    /// Total VRAM of that GPU in GB.
    pub vram_total_gb: Option<f64>,
}

/// How a model of a given size fits this host's GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuFit {
    /// Fits with comfortable headroom.
    Comfortable,
    /// Fits, but uses most of the VRAM.
    Tight,
    /// Bigger than this GPU's VRAM — would run on CPU/RAM instead.
    Exceeds,
    /// No usable GPU detected (CPU/RAM only host).
    NoGpu,
}

impl SystemInfo {
    /// Detects the host once. Cheap shell-outs; safe to call at startup.
    pub fn detect() -> Self {
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0);
        let ram_gb = detect_ram_gb();
        let (gpu_name, vram_total_gb) = detect_gpu();
        Self {
            cpu_cores,
            ram_gb,
            gpu_name,
            vram_total_gb,
        }
    }

    /// Whether a model needing `required_gb` of VRAM fits this host's GPU.
    /// `Comfortable` if it leaves ~15% headroom, `Tight` if it just fits.
    pub fn gpu_fit(&self, required_gb: f64) -> GpuFit {
        match self.vram_total_gb {
            None => GpuFit::NoGpu,
            Some(vram) if required_gb <= vram * 0.85 => GpuFit::Comfortable,
            Some(vram) if required_gb <= vram => GpuFit::Tight,
            Some(_) => GpuFit::Exceeds,
        }
    }
}

/// Picks the index of the *recommended* model from `required_gbs`: the largest
/// one that still fits comfortably on the GPU. Falls back to the smallest model
/// when nothing fits comfortably (or there's no GPU), so there is always a
/// sensible pick to badge.
pub fn recommended_index(required_gbs: &[f64], system: &SystemInfo) -> Option<usize> {
    if required_gbs.is_empty() {
        return None;
    }
    // Largest model that's Comfortable on the GPU.
    let mut best: Option<(usize, f64)> = None;
    for (index, &req) in required_gbs.iter().enumerate() {
        if system.gpu_fit(req) == GpuFit::Comfortable {
            match best {
                Some((_, b)) if req <= b => {}
                _ => best = Some((index, req)),
            }
        }
    }
    if let Some((index, _)) = best {
        return Some(index);
    }
    // Nothing comfortable: recommend the smallest (safest) model.
    required_gbs
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
}

fn detect_ram_gb() -> Option<f64> {
    // PowerShell CIM is stable across modern Windows (wmic is deprecated).
    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        .output()
        .ok()?;
    let bytes: f64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    Some(bytes / 1_073_741_824.0)
}

fn detect_gpu() -> (Option<String>, Option<f64>) {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => return (None, None),
    };
    let line = String::from_utf8_lossy(&output.stdout);
    let Some(first) = line.lines().next() else {
        return (None, None);
    };
    let mut parts = first.split(',');
    let name = parts.next().map(|value| value.trim().to_string());
    let vram_gb = parts
        .next()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .map(|mib| mib / 1024.0);
    (name, vram_gb)
}
