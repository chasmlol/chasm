//! Music generation — the "play a song (guitar)" NPC action.
//!
//! When an NPC chooses the play-a-song action during a turn, the bridge hands us a
//! [`SongJob`] and returns immediately (the turn is never blocked). Here we run the
//! job asynchronously:
//!   1. Write the SONG LYRICS as the character — reusing the exact turn-assembly
//!      prompt stack (card, persona, relationships, memories, lore, history) via
//!      [`crate::generate::song_base_messages`], routed through a dedicated song
//!      system prompt that folds in the player's request.
//!   2. Generate the audio with the local ACE-Step engine (DiT mode) on :5004.
//!   3. Store the WAV + a metadata sidecar under the profile (`headless/music/`).
//!   4. Deliver it to the mod via a `control/songs/<id>.json` queue file (a new
//!      queue the plugin polls unconditionally, so the song plays AFTER the turn —
//!      the normal reply/audio path is turn-scoped and won't play unsolicited audio).
//!
//! The guitar idle itself is handled by the shipped action book entry (a trusted
//! GECK `PlayIdle SpecialIdleNVGuitar` script) which fires through the normal action
//! queue during the triggering turn; this module only does the song.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;
use chasm_core::AppSettings;
use chasm_fnv_bridge::chasm::SongJob;

use crate::AppState;

/// Token budget for the lyrics generation. A structured song (a few verses + a
/// chorus) fits comfortably; generous so the model isn't cut mid-chorus.
const SONG_MAX_TOKENS: i64 = 900;

/// The parsed output of the lyrics generation: ACE-Step-ready lyrics (with
/// `[Verse]`/`[Chorus]` section markers) + the style tags for its `caption`.
#[derive(Debug, Clone)]
pub(crate) struct SongLyrics {
    pub lyrics: String,
    pub style_tags: String,
    /// A short human title (best-effort; from the first lyric line) for metadata.
    pub title: String,
}

/// The song system prompt: instruct the model to write lyrics AS this character,
/// grounded in who they are (the surrounding messages carry the full character
/// stack), incorporating the player's request. Structured for ACE-Step: bracketed
/// section markers + a trailing `STYLE:` line of comma-separated tags.
fn song_system_prompt(
    character_name: &str,
    user_message: &str,
    max_seconds: u32,
    is_rap: bool,
) -> String {
    // Rough guide: ~1 short verse + chorus per ~20s. Keep it short enough to fit the
    // duration so ACE-Step doesn't have to pad or drift.
    let sections = if max_seconds <= 45 { "two short verses and a chorus" }
        else if max_seconds <= 90 { "two or three verses and a repeating chorus" }
        else { "three verses, a chorus, and a short bridge" };

    // The performance framing + format differ between a sung song and a rap.
    let (perform, extra_format, style_example) = if is_rap {
        (
            "You are going to RAP, right now — spit bars in your own voice and from your \
own life and knowledge",
            "\n- Write it as a RAP: rhythmic, rhyming, punchy spoken-flow bars (internal \
rhymes welcome), NOT a slow sung melody.",
            "'STYLE: gruff older man, gravelly, hip hop, boom bap, rhythmic flow'",
        )
    } else {
        (
            "You are going to perform a song, right now, with your guitar, in your own \
voice and from your own life and knowledge",
            "",
            "'STYLE: gruff older man, gravelly, weary, acoustic folk, sparse guitar'",
        )
    };

    format!(
        "You are {name}. {perform}. Write the LYRICS of that {kind}.\n\
\n\
The player just said to you: \"{msg}\"\n\
\n\
Write a {kind} that answers or riffs on what they said — make it clearly ABOUT that, \
but told the way {name} would tell it: your outlook, your history, the places and \
people you know, your way of speaking. First person. No meta-talk, no stage \
directions, no explaining yourself — only the {kind}.\n\
\n\
FORMAT (follow exactly):\n\
- Use bracketed section markers on their own lines: [Verse], [Chorus], [Verse 2], [Bridge], [Outro].\n\
- Put the lines under each marker, one line per line.\n\
- Keep it to about {sections} so it fits a ~{secs}-second {kind}.{extra}\n\
- After the lyrics, add ONE final line beginning exactly with 'STYLE:' followed by \
a few comma-separated descriptors. IMPORTANT: describe YOUR OWN VOICE as it truly is \
for who you are — your sex/gender, your age, and its character (for example 'gruff \
older man, gravelly, weathered' or 'bright young woman, warm' or 'raspy weary man'). \
Then add a couple of musical descriptors (genre, mood). Example: {example}.\n\
\n\
Output ONLY the bracketed lyrics and the single STYLE line. Nothing else.",
        name = character_name,
        perform = perform,
        kind = if is_rap { "rap" } else { "song" },
        msg = user_message.replace('"', "'"),
        sections = sections,
        secs = max_seconds,
        extra = extra_format,
        example = style_example,
    )
}

/// Parses the model's song output into lyrics + style tags. The last line starting
/// with `STYLE:` (case-insensitive) is pulled out as the tags; everything else is
/// the lyrics. `base_tags` (from settings) is prepended so the configured base
/// style always leads. Robust to the model omitting the STYLE line (falls back to
/// base tags) or wrapping the output in code fences.
fn parse_song_output(text: &str, base_tags: &str) -> SongLyrics {
    let cleaned = text
        .trim()
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let mut lyric_lines: Vec<&str> = Vec::new();
    let mut model_tags = String::new();
    for line in cleaned.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("style:") {
            // Take the tags from the ORIGINAL-case text after the prefix.
            let start = trimmed.len() - rest.trim_start().len();
            model_tags = trimmed[start..].trim().to_string();
            continue;
        }
        // Drop a stray leading fence or a bracketed [STYLE] header line.
        if trimmed == "```" || lower == "[style]" {
            continue;
        }
        lyric_lines.push(line);
    }

    let lyrics = lyric_lines
        .join("\n")
        .trim()
        .to_string();

    // Merge base + model tags, de-duplicating trivially and dropping empties.
    let mut tags: Vec<String> = Vec::new();
    for tag in base_tags
        .split(',')
        .chain(model_tags.split(','))
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
    {
        let lower = tag.to_ascii_lowercase();
        if !tags.iter().any(|t| t.to_ascii_lowercase() == lower) {
            tags.push(tag.to_string());
        }
    }
    let style_tags = tags.join(", ");

    // Title: first non-marker, non-empty lyric line (trimmed, capped) — best-effort.
    let title = lyric_lines
        .iter()
        .map(|l| l.trim())
        .find(|l| !l.is_empty() && !l.starts_with('['))
        .map(|l| {
            let t: String = l.chars().take(48).collect();
            t
        })
        .unwrap_or_else(|| "Untitled".to_string());

    SongLyrics {
        lyrics,
        style_tags,
        title,
    }
}

/// Generates the song lyrics as the character. Reuses the full turn-assembly prompt
/// stack for `job.character_id` (via [`crate::generate::song_base_messages`]) and
/// appends the [`song_system_prompt`], then runs one completion against the main
/// LLM. Errors are strings (logged by the caller).
pub(crate) async fn generate_song_lyrics(
    state: &Arc<AppState>,
    job: &SongJob,
) -> Result<SongLyrics, String> {
    let settings = AppSettings::load(&state.config.settings_path);
    let max_seconds = chasm_core::normalize_music_max_seconds(settings.music.max_seconds);
    // The rap variant swaps the folk/acoustic base for a hip-hop base (the folk tags
    // would fight a rap) and switches the lyric prompt to spoken-flow bars.
    let is_rap = job.style_hint.trim().eq_ignore_ascii_case("rap");
    let base_tags = if is_rap {
        chasm_core::MUSIC_RAP_STYLE_TAGS.to_string()
    } else {
        settings.music.style_tags.trim().to_string()
    };

    let force_character_id = (!job.character_id.is_empty()).then_some(job.character_id.as_str());
    let (mut messages, speaker_name) = crate::generate::song_base_messages(
        state,
        &job.live_chat_id,
        force_character_id,
        &job.user_message,
    )
    .ok_or_else(|| {
        format!(
            "could not assemble song prompt for live chat '{}' (character '{}')",
            job.live_chat_id, job.character_id
        )
    })?;

    // The speaker name from the assembled turn is the most reliable label; fall back
    // to the job's character name.
    let character_name = if !speaker_name.trim().is_empty() {
        speaker_name
    } else if !job.character_name.trim().is_empty() {
        job.character_name.clone()
    } else {
        job.npc_name.clone()
    };

    // Swap the turn's response instruction for the song instruction: append a strong
    // system message + a user cue. The preceding messages carry the character stack.
    messages.push(json!({
        "role": "system",
        "content": song_system_prompt(&character_name, &job.user_message, max_seconds, is_rap),
    }));
    messages.push(json!({
        "role": "user",
        "content": format!(
            "Now perform your song about: {}. Remember — bracketed sections, then one STYLE: line.",
            job.user_message.trim()
        ),
    }));

    let sampling = crate::llm::Sampling::from_settings(&settings.llm.sampling).with_overrides(
        crate::llm::GenerationOptions {
            temperature: None,
            max_tokens: Some(SONG_MAX_TOKENS),
        },
    );
    let target = crate::llm::LlmTarget::resolve(&settings, &state.config);
    let (text, _metrics) =
        crate::llm::chat_completion_capturing_sampled(&target, &messages, None, sampling).await?;

    let parsed = parse_song_output(&text, &base_tags);
    if parsed.lyrics.trim().is_empty() {
        return Err("lyrics generation returned no usable lyrics".to_string());
    }
    Ok(parsed)
}

/// Resolves the performing NPC's voice reference clip for the ACE-Step reference:
/// `<active voices dir>/<character>/reference.wav` — the same clip TTS clones from.
/// Returns the path string only when the file exists (so a character without a
/// cloned voice just gets no reference, never an error). Tries the character name
/// then the native NPC name.
fn npc_voice_reference(state: &Arc<AppState>, job: &SongJob) -> Option<String> {
    let voices = crate::active_voices_dir(&state.config);
    for name in [job.character_name.as_str(), job.npc_name.as_str()] {
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        let clip = voices.join(name).join("reference.wav");
        if clip.exists() {
            return Some(clip.display().to_string());
        }
    }
    None
}

/// The configured ACE-Step endpoint (`CHASM_ACESTEP_ENDPOINT` or the default).
fn acestep_endpoint() -> String {
    std::env::var("CHASM_ACESTEP_ENDPOINT")
        .unwrap_or_else(|_| chasm_core::DEFAULT_ACESTEP_ENDPOINT.to_string())
}

/// POSTs the lyrics + style tags + duration to the ACE-Step server and returns the
/// WAV bytes. Long timeout — a multi-second song render is expected (the model also
/// loads lazily on the first call). Errors are strings.
async fn request_acestep_wav(
    lyrics: &str,
    style_tags: &str,
    duration_seconds: u32,
    reference_audio: Option<&str>,
) -> Result<Vec<u8>, String> {
    let client = reqwest::Client::builder()
        // Generous: first request pays a lazy model load (tens of seconds) + the
        // render itself. Well above the worst measured cold gen.
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| format!("building music http client: {e}"))?;

    let body = json!({
        "lyrics": lyrics,
        "style_tags": style_tags,
        "duration": duration_seconds,
        // The performing NPC's own voice clip as a style/timbre reference (empty when
        // disabled or the clip is missing — the server ignores an empty value).
        "reference_audio": reference_audio.unwrap_or(""),
    });
    let resp = client
        .post(acestep_endpoint())
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("ACE-Step request failed (is the engine installed + running?): {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        return Err(format!("ACE-Step returned {status}: {}", truncate(&detail, 300)));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("reading ACE-Step audio: {e}"))?;
    if bytes.is_empty() {
        return Err("ACE-Step returned an empty audio body".to_string());
    }
    Ok(bytes.to_vec())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}

/// The PCM duration of a RIFF/WAVE buffer in milliseconds, from its `fmt `
/// byte-rate + `data` chunk size. Returns `None` for a non-WAV / malformed buffer
/// (the caller falls back to the requested duration).
fn wav_duration_ms(bytes: &[u8]) -> Option<u64> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut pos = 12usize;
    let mut byte_rate: Option<u32> = None;
    let mut data_size: Option<u32> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]])
            as usize;
        let body = pos + 8;
        if id == b"fmt " && body + 16 <= bytes.len() {
            byte_rate = Some(u32::from_le_bytes([
                bytes[body + 8],
                bytes[body + 9],
                bytes[body + 10],
                bytes[body + 11],
            ]));
        } else if id == b"data" {
            data_size = Some(size as u32);
        }
        // Chunks are word-aligned (pad byte for odd sizes).
        pos = body + size + (size & 1);
    }
    match (byte_rate, data_size) {
        (Some(br), Some(ds)) if br > 0 => Some((ds as u64) * 1000 / (br as u64)),
        _ => None,
    }
}

/// Milliseconds since the Unix epoch (for ids + metadata timestamps).
fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// A short filesystem-safe id for a stored song.
fn song_id(job: &SongJob) -> String {
    let base = job
        .request_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>();
    let base = base.trim_matches('-');
    let base = if base.is_empty() { "song" } else { base };
    format!("{base}-{}", now_millis())
}

/// A stored song: the on-disk WAV path + its measured duration.
struct StoredSong {
    id: String,
    wav_path: PathBuf,
    duration_ms: u64,
}

/// Writes the song WAV + a metadata sidecar under the active profile's
/// `headless/music/` dir. Returns the stored paths + duration.
fn store_song(
    state: &Arc<AppState>,
    job: &SongJob,
    lyrics: &SongLyrics,
    wav: &[u8],
    requested_seconds: u32,
) -> Result<StoredSong, String> {
    let dir = state.config.active_profile_paths().music_dir();
    store_song_at(&dir, job, lyrics, wav, requested_seconds)
}

/// The pure store: writes the WAV + metadata sidecar into `dir`. Separated from
/// [`store_song`] so it can be unit-tested without a full `AppState`.
fn store_song_at(
    dir: &std::path::Path,
    job: &SongJob,
    lyrics: &SongLyrics,
    wav: &[u8],
    requested_seconds: u32,
) -> Result<StoredSong, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;

    let id = song_id(job);
    let wav_path = dir.join(format!("{id}.wav"));
    std::fs::write(&wav_path, wav).map_err(|e| format!("writing {}: {e}", wav_path.display()))?;

    let duration_ms = wav_duration_ms(wav).unwrap_or((requested_seconds as u64) * 1000);

    let meta = json!({
        "schema": "chasm.song.v1",
        "id": id,
        "requestId": job.request_id,
        "character": job.character_name,
        "characterId": job.character_id,
        "npcKey": job.npc_key,
        "npcName": job.npc_name,
        "prompt": job.user_message,
        "title": lyrics.title,
        "styleTags": lyrics.style_tags,
        "lyrics": lyrics.lyrics,
        "durationMs": duration_ms,
        "requestedSeconds": requested_seconds,
        "createdAtMs": now_millis() as u64,
        "wav": wav_path.file_name().and_then(|n| n.to_str()).unwrap_or_default(),
    });
    let meta_path = dir.join(format!("{id}.json"));
    if let Err(e) = std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta).unwrap_or_default()) {
        // Non-fatal: the WAV is what matters for playback.
        tracing::warn!("music: could not write song metadata {}: {e}", meta_path.display());
    }

    Ok(StoredSong {
        id,
        wav_path,
        duration_ms,
    })
}

/// Writes the mod delivery — a `control/songs/<id>.txt` queue file — into every
/// bridge root. The plugin polls this queue UNCONDITIONALLY (unlike the turn-scoped
/// outbox reply), plays the WAV positionally from the NPC, and runs/stops the guitar
/// idle for the song's duration. Returns how many roots were written.
///
/// LINE-BASED format (the mod reads responses line-by-line; it has no JSON parser):
/// ```text
/// NVBRIDGE_SONG_V1
/// <songId>
/// <npcKey>
/// <npcName>
/// <absolute wav path>
/// <durationMs>
/// <title>
/// ```
fn write_song_delivery(job: &SongJob, song: &StoredSong, lyrics: &SongLyrics) -> usize {
    // Strip newlines from free-text fields so the line-based layout stays intact.
    let sanitize = |s: &str| s.replace(['\r', '\n'], " ");
    let content = format!(
        "NVBRIDGE_SONG_V1\n{id}\n{key}\n{name}\n{wav}\n{dur}\n{title}\n",
        id = song.id,
        key = sanitize(&job.npc_key),
        name = sanitize(&job.npc_name),
        wav = song.wav_path.display(),
        dur = song.duration_ms,
        title = sanitize(&lyrics.title),
    );
    let bytes = content.into_bytes();

    let mut written = 0usize;
    for root in &job.bridge_roots {
        let dir = root.join("control").join("songs");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!("music: could not create {}: {e}", dir.display());
            continue;
        }
        // Write to a temp name then rename, so the plugin never reads a half-written
        // file (the plugin only ever sees the final `<id>.txt`).
        let tmp = dir.join(format!("{}.txt.tmp", song.id));
        let final_path = dir.join(format!("{}.txt", song.id));
        if std::fs::write(&tmp, &bytes).is_ok() && std::fs::rename(&tmp, &final_path).is_ok() {
            written += 1;
        } else {
            let _ = std::fs::remove_file(&tmp);
            tracing::warn!("music: could not write song delivery to {}", final_path.display());
        }
    }
    written
}

/// Spawns the async song job (fire-and-forget). Called by the in-process bridge's
/// `start_song_job`. Never blocks the caller; any failure is logged.
pub(crate) fn spawn_song_job(state: Arc<AppState>, job: SongJob) {
    tokio::spawn(async move {
        if let Err(error) = run_song_job(&state, &job).await {
            tracing::warn!(
                "music: song job for '{}' (req {}) failed: {error}",
                job.character_name, job.request_id
            );
        }
    });
}

/// The full song job: lyrics -> ACE-Step -> store -> deliver. Each stage logs on
/// entry so a failure is traceable to a stage from the log alone.
async fn run_song_job(state: &Arc<AppState>, job: &SongJob) -> Result<(), String> {
    let settings = AppSettings::load(&state.config.settings_path);
    if !settings.music.enabled {
        return Err("music generation is disabled in settings".to_string());
    }
    if chasm_core::normalize_music_engine(&settings.music.engine).is_empty() {
        return Err("no music engine selected".to_string());
    }
    let requested_seconds = chasm_core::normalize_music_max_seconds(settings.music.max_seconds);

    tracing::info!(
        "music: writing lyrics for '{}' about {:?}",
        job.character_name,
        truncate(&job.user_message, 80)
    );
    let lyrics = generate_song_lyrics(state, job).await?;
    tracing::info!(
        "music: lyrics ready ({} chars); style='{}'; generating audio (~{}s) ...",
        lyrics.lyrics.len(),
        lyrics.style_tags,
        requested_seconds
    );

    // Match-voice: pass the performing NPC's own voice clip as a style/timbre
    // reference so the song leans toward how they sound. Falls back to no reference
    // when disabled or the clip is missing.
    let voice_ref = if settings.music.match_npc_voice {
        npc_voice_reference(state, job)
    } else {
        None
    };
    if voice_ref.is_some() {
        tracing::info!("music: using '{}' voice clip as the reference", job.character_name);
    }

    let wav = request_acestep_wav(
        &lyrics.lyrics,
        &lyrics.style_tags,
        requested_seconds,
        voice_ref.as_deref(),
    )
    .await?;
    tracing::info!("music: received {} bytes of audio; storing", wav.len());

    let song = store_song(state, job, &lyrics, &wav, requested_seconds)?;
    let roots = write_song_delivery(job, &song, &lyrics);
    tracing::info!(
        "music: song '{}' stored at {} ({} ms); delivered to {} bridge root(s)",
        song.id,
        song.wav_path.display(),
        song.duration_ms,
        roots
    );
    if roots == 0 {
        return Err("song generated + stored but delivery to the mod failed (no bridge roots written)".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_pulls_style_line_and_merges_base_tags() {
        let out = "[Verse]\nWoke up in Goodsprings, dust on my boots\n[Chorus]\nOld town, old friends\nSTYLE: country, weary, harmonica";
        let parsed = parse_song_output(out, "acoustic guitar, folk");
        assert!(parsed.lyrics.contains("[Verse]"));
        assert!(parsed.lyrics.contains("[Chorus]"));
        assert!(!parsed.lyrics.to_ascii_lowercase().contains("style:"));
        // Base tags lead, model tags appended, de-duped.
        assert!(parsed.style_tags.starts_with("acoustic guitar, folk"));
        assert!(parsed.style_tags.contains("country"));
        assert!(parsed.style_tags.contains("harmonica"));
        assert_eq!(parsed.title, "Woke up in Goodsprings, dust on my boots");
    }

    #[test]
    fn parse_without_style_line_falls_back_to_base_tags() {
        let out = "[Verse]\nJust a tune\n[Chorus]\nLa la la";
        let parsed = parse_song_output(out, "campfire, warm");
        assert_eq!(parsed.style_tags, "campfire, warm");
        assert!(parsed.lyrics.contains("Just a tune"));
    }

    #[test]
    fn parse_strips_code_fences() {
        let out = "```\n[Verse]\nHello\nSTYLE: blues\n```";
        let parsed = parse_song_output(out, "");
        assert_eq!(parsed.lyrics.trim(), "[Verse]\nHello");
        assert_eq!(parsed.style_tags, "blues");
    }

    #[test]
    fn wav_duration_from_header() {
        // Build a minimal 1-second 16-bit mono 24kHz WAV (data only, silence).
        let sample_rate = 24_000u32;
        let byte_rate = sample_rate * 2; // mono, 16-bit
        let data_len = byte_rate; // 1 second
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.extend(std::iter::repeat(0u8).take(data_len as usize));
        assert_eq!(wav_duration_ms(&wav), Some(1000));
    }

    #[test]
    fn wav_duration_none_for_garbage() {
        assert_eq!(wav_duration_ms(b"not a wav"), None);
    }

    #[test]
    fn song_prompt_branches_between_sung_and_rap() {
        // Sung (guitar) variant: mentions the guitar, no rap framing.
        let sung = song_system_prompt("Easy Pete", "a song about Goodsprings", 60, false);
        assert!(sung.contains("guitar"), "sung prompt should mention the guitar");
        assert!(!sung.contains("RAP"), "sung prompt should not ask for a rap");

        // Rap variant: asks for a RAP / bars / spoken-flow and a hip-hop STYLE example.
        let rap = song_system_prompt("MC Test", "spit a rap about the Mojave", 60, true);
        assert!(rap.contains("RAP"), "rap prompt should ask for a RAP");
        let lower = rap.to_ascii_lowercase();
        assert!(
            lower.contains("bars") || lower.contains("spoken-flow"),
            "rap prompt should ask for bars / spoken flow"
        );
        assert!(
            rap.contains("hip hop") || rap.contains("boom bap"),
            "rap STYLE example should be hip-hop"
        );
        assert!(!rap.contains("your guitar"), "rap prompt should not force the guitar");
    }

    #[test]
    fn rap_base_tags_are_hiphop_not_folk() {
        // The rap variant swaps the folk default for a hip-hop base (constant).
        assert!(chasm_core::MUSIC_RAP_STYLE_TAGS.contains("hip hop"));
        assert!(chasm_core::MUSIC_RAP_STYLE_TAGS.contains("rap"));
        assert!(!chasm_core::MUSIC_RAP_STYLE_TAGS.contains("guitar"));
    }

    fn minimal_wav(seconds: u32) -> Vec<u8> {
        let sample_rate = 24_000u32;
        let byte_rate = sample_rate * 2;
        let data_len = byte_rate * seconds;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.extend(std::iter::repeat(0u8).take(data_len as usize));
        wav
    }

    // The store + deliver half of the song job, end-to-end over temp dirs (no
    // AppState / LLM / engine): the WAV + metadata land in the store, and the
    // control/songs delivery is written in the exact line-based format the mod reads.
    #[test]
    fn store_and_deliver_round_trip() {
        let root = std::env::temp_dir().join(format!("chasm-song-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let music_dir = root.join("profile").join("headless").join("music");
        let bridge_root = root.join("bridge");
        std::fs::create_dir_all(&bridge_root).unwrap();

        let job = SongJob {
            request_id: "req_abc123".to_string(),
            live_chat_id: "fnv-goodsprings".to_string(),
            character_id: "Easy Pete".to_string(),
            character_name: "Easy Pete".to_string(),
            npc_key: "GSEasyPete".to_string(),
            npc_name: "Easy Pete".to_string(),
            user_message: "play me a song about Goodsprings".to_string(),
            style_hint: String::new(),
            bridge_roots: vec![bridge_root.clone()],
        };
        let lyrics = SongLyrics {
            lyrics: "[Verse]\nDust and dynamite\n[Chorus]\nGoodsprings, my home".to_string(),
            style_tags: "acoustic guitar, folk, warm".to_string(),
            title: "Dust and dynamite".to_string(),
        };
        let wav = minimal_wav(3);

        let stored = store_song_at(&music_dir, &job, &lyrics, &wav, 75).expect("store");
        // WAV + metadata sidecar landed in the store.
        assert!(stored.wav_path.exists(), "wav should be stored");
        assert_eq!(std::fs::read(&stored.wav_path).unwrap().len(), wav.len());
        assert_eq!(stored.duration_ms, 3000, "duration parsed from the WAV header");
        let meta_path = music_dir.join(format!("{}.json", stored.id));
        assert!(meta_path.exists(), "metadata sidecar should be written");
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
        assert_eq!(meta["character"], "Easy Pete");
        assert_eq!(meta["styleTags"], "acoustic guitar, folk, warm");
        assert!(meta["lyrics"].as_str().unwrap().contains("[Chorus]"));

        // Delivery written to the bridge root in the mod's line-based format.
        let written = write_song_delivery(&job, &stored, &lyrics);
        assert_eq!(written, 1, "one bridge root written");
        let delivery = bridge_root
            .join("control")
            .join("songs")
            .join(format!("{}.txt", stored.id));
        assert!(delivery.exists(), "delivery file should exist");
        let text = std::fs::read_to_string(&delivery).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "NVBRIDGE_SONG_V1");
        assert_eq!(lines[1], stored.id);
        assert_eq!(lines[2], "GSEasyPete");
        assert_eq!(lines[3], "Easy Pete");
        assert_eq!(lines[4], stored.wav_path.display().to_string());
        assert_eq!(lines[5], "3000");
        assert_eq!(lines[6], "Dust and dynamite");
        // No leftover temp files.
        assert!(!bridge_root
            .join("control")
            .join("songs")
            .join(format!("{}.txt.tmp", stored.id))
            .exists());

        let _ = std::fs::remove_dir_all(&root);
    }
}
