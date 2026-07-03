//! Live-chat NPC-turn generation, ported from the Node headless runtime
//! (`src/headless/live-chats.js` + `generation.js`). This is the core pipeline
//! the FNV helper (`tools/fnv/nvbridge-helper.mjs` `generateNpcTurn`) drives:
//!
//! 1. `GET  /live-chats/:id`            — existence probe (404 when missing).
//! 2. `POST /live-chats`                — create a live chat + first segment.
//! 3. `POST /live-chats/:id/presence`   — replace/update participant presence.
//! 4. `POST /live-chats/:id/generate/stream` — stream an NPC turn as NDJSON.
//! 5. `POST /live-chats/:id/generate`   — same, buffered (non-stream).
//!
//! The prompt is assembled by the existing `chasm_prompt::assemble_prompt`
//! API; the local LLM (llama.cpp, OpenAI-compatible) is called for the actual
//! generation; the resulting turn is appended to the live-chat JSONL session so
//! history stays consistent across turns.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, HeaderMap},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use chasm_core::{
    format_message_timestamp, AppSettings, InjectedView, MessageView, ParticipantView,
};
use chasm_prompt::RetrievalCtx;
use chasm_st_compat::{LiveChat, LiveChatParticipant, LiveChatSegment, STJsonlChatMessage};

use crate::{orchestrator, AppState, WebError, WebResult};

/// HTTP header the FNV helper sends carrying the originating game request's trace
/// id, so generation metrics can be correlated to its trace file.
const TRACE_ID_HEADER: &str = "x-chasm-trace-id";

/// Extracts a non-empty `X-Chasm-Trace-Id` from the request headers.
fn trace_id_from_headers(headers: &HeaderMap) -> Option<String> {
    headers
        .get(TRACE_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// Extracts a non-empty `traceId` from a generate request BODY. The in-process
/// FNV bridge calls `generate_stream_core` directly (no HTTP headers), so it
/// carries the game request's trace id in the body instead; without this the
/// LLM usage/timings capture never correlates to in-game turns.
fn trace_id_from_body(body: &Value) -> Option<String> {
    body.get("traceId")
        .or_else(|| body.get("trace_id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

const PLAYER_PARTICIPANT_ID: &str = "player";
const CONTEXT_MESSAGE_LIMIT: usize = 40;

// ---------------------------------------------------------------------------
// GET /live-chats/:id  (existence probe — helper only checks status 404)
// ---------------------------------------------------------------------------

pub async fn get_live_chat(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> WebResult<Json<Value>> {
    let live_chat = state.repository.get_live_chat(&id)?;
    Ok(Json(map_live_chat(&live_chat)))
}

// ---------------------------------------------------------------------------
// POST /live-chats  (create live chat + initial segment + presence)
// ---------------------------------------------------------------------------

pub async fn create_live_chat(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> WebResult<Json<Value>> {
    let id = body
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| web_err("live-chats create requires an 'id'"))?
        .to_string();
    let group_id = body
        .get("groupId")
        .and_then(Value::as_str)
        .unwrap_or(&id)
        .to_string();
    let title = body
        .get("title")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or(&id)
        .to_string();
    let location = body
        .get("location")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let now = now_iso();

    // The segment session id mirrors the Node base64url(JSON) encoding so the
    // existing st-compat reader can resolve the JSONL path.
    let segment_id = if location.is_empty() {
        title.clone()
    } else {
        location.clone()
    };
    let session_id = encode_group_session_id(&group_id, &segment_id);
    let segment = LiveChatSegment {
        id: segment_id.clone(),
        title: segment_id.clone(),
        location: location.clone(),
        chat_id: segment_id.clone(),
        session_id,
        created_at: Some(now.clone()),
        metadata: Value::Null,
    };

    let mut live_chat = LiveChat {
        id: id.clone(),
        title,
        group_id,
        current_segment_id: segment.id.clone(),
        segments: vec![segment],
        created_at: Some(now.clone()),
        updated_at: Some(now.clone()),
        ..Default::default()
    };

    apply_presence(&mut live_chat, &body, /* replace = */ true, &now);

    state.repository.update_store(|store| {
        store.items.insert(id.clone(), live_chat.clone());
    })?;

    Ok(Json(map_live_chat(&live_chat)))
}

// ---------------------------------------------------------------------------
// POST /live-chats/:id/presence
// ---------------------------------------------------------------------------

pub async fn update_presence(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> WebResult<Json<Value>> {
    let now = now_iso();
    let replace = body
        .get("replace")
        .or_else(|| body.get("replacePresence"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let live_chat = state.repository.update_store(|store| {
        let Some(live_chat) = store.items.get_mut(&id) else {
            return None;
        };
        apply_presence(live_chat, &body, replace, &now);
        live_chat.updated_at = Some(now.clone());
        Some(live_chat.clone())
    })?;

    let live_chat = live_chat
        .ok_or_else(|| WebError::from(chasm_st_compat::CompatError::LiveChatNotFound(id)))?;
    Ok(Json(map_live_chat(&live_chat)))
}

// ---------------------------------------------------------------------------
// POST /live-chats/:id/generate/stream  (NDJSON)
// ---------------------------------------------------------------------------

pub async fn generate_stream(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> WebResult<Response> {
    let trace_id = trace_id_from_headers(&headers);
    let stream = generate_stream_core(state, id, body, trace_id).await?;
    Ok((
        [(header::CONTENT_TYPE, "application/x-ndjson; charset=utf-8")],
        Body::from_stream(stream.map(Ok::<String, std::convert::Infallible>)),
    )
        .into_response())
}

/// In-process core of [`generate_stream`]: resolves the turn up front (so setup
/// errors surface before streaming starts) and returns the raw NDJSON event lines.
/// The HTTP handler above streams them as the response body; the in-process bridge
/// client parses each line back into a `Value` — one code path, minus the socket.
/// Appends a generate-side stage marker to the bridge's per-request trace file
/// (best-effort; only when the bridge tracer already opened this request). Gives
/// the trace waterfall visibility INSIDE the generate path — context resolution,
/// speaker selection, prompt assembly — instead of one opaque gap between
/// `live_chat_generate_start` and `live_chat_first_delta`.
fn trace_generate_stage(trace_id: Option<&str>, stage: &str) {
    let Some(id) = trace_id else { return };
    if id.is_empty() || id.contains(['/', '\\', '.']) {
        return; // ids are bridge-generated (req_...); refuse anything path-like
    }
    let Some(local) = std::env::var_os("LOCALAPPDATA") else { return };
    let path = std::path::Path::new(&local)
        .join("chasm")
        .join("bridge")
        .join("traces")
        .join(format!("{id}.jsonl"));
    if !path.exists() {
        return;
    }
    let at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let line = format!(
        "{{\"request_id\":\"{id}\",\"stage\":\"{stage}\",\"at_ms\":{at_ms},\"source\":\"chasm-web\"}}\n"
    );
    if let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(&path) {
        use std::io::Write as _;
        let _ = file.write_all(line.as_bytes());
    }
}

pub async fn generate_stream_core(
    state: Arc<AppState>,
    id: String,
    body: Value,
    trace_id: Option<String>,
) -> WebResult<impl futures_util::Stream<Item = String> + Send> {
    // Header-supplied trace id wins; the in-process bridge (no headers) sends it
    // in the body so the LLM metrics capture still correlates to game requests.
    let trace_id = trace_id.or_else(|| trace_id_from_body(&body));
    // Resolve everything that can fail synchronously up front so a hard error
    // is returned as a non-200 (which the helper's `streamApi` surfaces),
    // rather than mid-stream.
    let ctx = resolve_turn_context(&state, &id, &body)?;
    trace_generate_stage(trace_id.as_deref(), "gen_ctx_resolved");

    // Persist the player message immediately (mirrors the Node append-user step
    // that happens before speaker selection).
    if !ctx.message.is_empty() {
        persist_player_message_ctx(&state, &ctx)?;
    }

    // Run the orchestrator before streaming so selection errors surface as a
    // non-200 (the deterministic fallback never errors here once eligible).
    let (speakers, selector) = orchestrate(&state, &ctx, &body).await?;
    trace_generate_stage(trace_id.as_deref(), "gen_speakers_selected");
    let speaker_summaries: Vec<Value> = speakers.iter().map(speaker_summary).collect();

    // LLM sampling knobs, read fresh per request so UI changes apply on the
    // very next turn. (Speech goes out as ONE delta per line once the model
    // finishes: faster-qwen3-tts streams the AUDIO natively, so no text-side
    // opener-chunk splitting is needed — that legacy path is removed.)
    let live_settings = AppSettings::load(&state.config.settings_path);
    let sampling = crate::llm::Sampling::from_settings(&live_settings.llm.sampling);
    // The active LLM target (managed-local or a hosted API), resolved fresh per
    // request so a provider switch takes effect on the next turn.
    let target = crate::llm::LlmTarget::resolve(&live_settings, &state.config);

    let state = state.clone();
    let live_chat_id = ctx.live_chat.id.clone();
    let segment_id = ctx.segment.id.clone();
    let stream = async_stream::stream! {
        // live.start
        yield ndjson(&json!({ "type": "live.start", "liveChatId": live_chat_id }));
        let trace_id = trace_id;

        let mut turns: Vec<Value> = Vec::new();

        // One streamed turn per selected speaker, in order. Each turn is
        // persisted before the next so later speakers see earlier lines.
        for speaker in &speakers {
            let plan = match prepare_speaker_turn_traced(&state, &ctx, speaker, trace_id.as_deref()) {
                Ok(plan) => plan,
                Err(error) => {
                    yield ndjson(&json!({
                        "type": "live.error",
                        "error": { "message": error.0.to_string() },
                    }));
                    return;
                }
            };
            trace_generate_stage(trace_id.as_deref(), "gen_prompt_assembled");

            // speaker.start
            yield ndjson(&json!({ "type": "speaker.start", "speaker": plan.speaker }));

            // Collect the model output; `collected` keeps the full raw output
            // for finalize_turn, which re-parses it. LLM -> TTS streaming at
            // SENTENCE granularity: each completed sentence is emitted as its
            // own speech.delta the moment it exists, so the bridge synthesizes
            // sentence 1 while the model is still writing sentence 2 — first
            // audio no longer scales with reply length. Sentences (not raw
            // tokens or char counts) are the smallest unit qwen3-tts can speak
            // with natural prosody.
            let mut collected = String::new();
            let mut spoken_len: usize = 0;
            let response_format = plan.structured.then(crate::llm::structured_response_format);
            trace_generate_stage(trace_id.as_deref(), "gen_llm_request_dispatch");
            let mut first_token_seen = false;
            match crate::llm::chat_completion_stream(&target, &plan.chat_messages, response_format.as_ref(), trace_id.as_deref(), sampling)
                .await
            {
                Ok(mut rx) => {
                    while let Some(item) = rx.recv().await {
                        match item {
                            Ok(token) => {
                                if token.is_empty() {
                                    continue;
                                }
                                if !first_token_seen {
                                    first_token_seen = true;
                                    trace_generate_stage(trace_id.as_deref(), "gen_llm_first_token");
                                }
                                collected.push_str(&token);
                                let speech = extracted_speech(plan.structured, &collected);
                                while let Some(end) = next_sentence_end(&speech, spoken_len) {
                                    let segment = speech[spoken_len..end].trim();
                                    if !segment.is_empty() {
                                        yield ndjson(&json!({
                                            "type": "speech.delta",
                                            "text": segment,
                                            "speaker": plan.speaker,
                                        }));
                                    }
                                    spoken_len = end;
                                }
                            }
                            Err(error) => {
                                yield ndjson(&json!({
                                    "type": "live.error",
                                    "error": { "message": error },
                                }));
                                return;
                            }
                        }
                    }
                }
                Err(error) => {
                    yield ndjson(&json!({
                        "type": "live.error",
                        "error": { "message": error },
                    }));
                    return;
                }
            }

            // The remainder (final sentence / anything after the last completed
            // sentence boundary) as the closing delta.
            let full = extracted_speech(plan.structured, &collected);
            let rest = full.get(spoken_len.min(full.len())..).unwrap_or("").trim_start();
            if !rest.is_empty() {
                yield ndjson(&json!({
                    "type": "speech.delta",
                    "text": rest,
                    "speaker": plan.speaker,
                }));
            }

            // Build + persist this speaker's turn.
            match finalize_turn(&state, &plan, &ctx.macros, &collected) {
                Ok(turn) => turns.push(turn),
                Err(error) => {
                    yield ndjson(&json!({
                        "type": "live.error",
                        "error": { "message": error.0.to_string() },
                    }));
                    return;
                }
            }
        }

        // live.completed carries the full multi-turn response (back-compat with
        // the single-turn helper, which reads `turn.turns[]` when present).
        let response = build_live_response(
            &live_chat_id,
            &segment_id,
            &speaker_summaries,
            selector,
            turns,
        );
        yield ndjson(&json!({ "type": "live.completed", "turn": response }));
    };

    Ok(stream)
}

// ---------------------------------------------------------------------------
// POST /live-chats/:id/generate  (buffered)
// ---------------------------------------------------------------------------

pub async fn generate(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> WebResult<Json<Value>> {
    let trace_id = trace_id_from_headers(&headers);
    let ctx = resolve_turn_context(&state, &id, &body)?;
    trace_generate_stage(trace_id.as_deref(), "gen_ctx_resolved");

    // Persist the player message once (before speaker selection, like Node).
    if !ctx.message.is_empty() {
        persist_player_message_ctx(&state, &ctx)?;
    }

    // Run the orchestrator to get the ordered speaker list (empty = silence).
    let (speakers, selector) = orchestrate(&state, &ctx, &body).await?;
    trace_generate_stage(trace_id.as_deref(), "gen_speakers_selected");
    let speaker_summaries: Vec<Value> = speakers.iter().map(speaker_summary).collect();

    // Saved LLM sampling + the active provider target, read fresh per request so
    // UI tweaks / a provider switch apply next turn.
    let live_settings = AppSettings::load(&state.config.settings_path);
    let sampling = crate::llm::Sampling::from_settings(&live_settings.llm.sampling);
    let target = crate::llm::LlmTarget::resolve(&live_settings, &state.config);
    let mut turns: Vec<Value> = Vec::new();
    for speaker in &speakers {
        let plan = prepare_speaker_turn(&state, &ctx, speaker)?;
        let response_format = plan.structured.then(crate::llm::structured_response_format);
        let (text, metrics) = crate::llm::chat_completion_capturing_sampled(
            &target,
            &plan.chat_messages,
            response_format.as_ref(),
            sampling,
        )
        .await
        .map_err(web_err)?;
        // Surface the generation's tokens/sec etc. on the request's trace.
        if let (Some(id), Some(metrics)) = (trace_id.as_deref(), metrics) {
            crate::trace_routes::record_llm_metrics(id, metrics);
        }
        // finalize_turn persists this speaker's message, so the NEXT speaker's
        // history read sees it (matches ST's between-turn writes).
        let turn = finalize_turn(&state, &plan, &ctx.macros, &text)?;
        turns.push(turn);
    }

    Ok(Json(build_live_response(
        &ctx.live_chat.id,
        &ctx.segment.id,
        &speaker_summaries,
        selector,
        turns,
    )))
}

// ---------------------------------------------------------------------------
// Turn preparation
// ---------------------------------------------------------------------------

/// Speaker-agnostic context for one generate call: everything that does NOT
/// depend on which NPC speaks. Built once per request, then reused to build a
/// `TurnPlan` per selected speaker. Mirrors the Node `generateLiveChat`
/// preamble (load store, read message/body fields) before the per-speaker loop.
struct TurnContext {
    live_chat: LiveChat,
    segment: LiveChatSegment,
    message: String,
    player_participant_id: String,
    structured: bool,
    extra_context: String,
    gamestate: Value,
    player_metadata: Value,
    /// The turn's FRESH gamestate macro table (`metadata.macros`, flat
    /// key→value), extracted once so `finalize_turn` can record it verbatim on
    /// the persisted message (`extra.chasm.macros`) — never back-filled from
    /// older turns, so the recorded history stays honest. Prompt-side macro use
    /// goes through `scenario_macros` below; retrieval stays macro-free.
    macros: BTreeMap<String, String>,
    /// The macro table the GLOBAL scenario resolves against this request:
    /// `macros` when the mod sent a table this turn, else the latest recorded
    /// `extra.chasm.macros` (the same source the Gamestate page reads), so
    /// UI/admin-driven turns still see real values. Backend-computed macros
    /// (`{{participants}}`) are merged per speaker in `prepare_speaker_turn`.
    scenario_macros: BTreeMap<String, String>,
    /// The effective global scenario template (Globals store value, else the
    /// built-in default; empty = user disabled the scenario component).
    scenario_template: String,
    /// Action-book scopes the request supplies (`actionBookScopes`). Gates
    /// scope-restricted actions (e.g. admin-only spawn). Empty for regular NPCs
    /// unless the helper sends them.
    requested_scopes: Vec<String>,
    /// Global orchestrator knobs (enabled / max_speakers / temperature / prompt),
    /// loaded from `AppSettings` at request time.
    orchestrator: orchestrator::OrchestratorSettings,
}

/// Parses the request's `actionBookScopes` (array of strings) into the scope list
/// the prompt assembler gates actions on. Accepts camelCase or snake_case.
fn parse_action_book_scopes(body: &Value) -> Vec<String> {
    body.get("actionBookScopes")
        .or_else(|| body.get("action_book_scopes"))
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(|scope| scope.trim().to_string())
                .filter(|scope| !scope.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Everything resolved for one NPC turn before the model is called.
struct TurnPlan {
    live_chat_id: String,
    segment: LiveChatSegment,
    /// The chosen speaker, in the JSON shape the helper consumes
    /// (`{ participantId, characterId, name }`).
    speaker: Value,
    speaker_participant_id: String,
    speaker_character_id: Option<String>,
    speaker_name: String,
    structured: bool,
    /// Player ids this turn is audible to (for the persisted live metadata).
    audible_to: Vec<String>,
    present: Vec<String>,
    location: String,
    /// The OpenAI chat-completion messages to send to the model.
    chat_messages: Vec<Value>,
    /// Lore/quest/action entries injected into THIS turn's prompt, recorded so
    /// `finalize_turn` can persist them onto the produced message's
    /// `extra.chasm.injected` for the per-message panel.
    injected: InjectedView,
}

/// Resolves the request-level (speaker-agnostic) context. Fails synchronously
/// for the conditions the helper surfaces as a non-200 (missing chat / segment).
fn resolve_turn_context(state: &Arc<AppState>, id: &str, body: &Value) -> WebResult<TurnContext> {
    let live_chat = state.repository.get_live_chat(id)?;
    let segment =
        current_segment(&live_chat).ok_or_else(|| web_err("live chat has no current segment"))?;

    let message = body
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let player_participant_id = body
        .get("participantId")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or(PLAYER_PARTICIPANT_ID)
        .to_string();
    let structured = body
        .get("responseFormat")
        .and_then(Value::as_str)
        .map(|value| value == "structured")
        .unwrap_or(false);
    let extra_context = body
        .get("extraContext")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let gamestate = body.get("gamestate").cloned().unwrap_or(Value::Null);

    // Orchestrator knobs come from the GLOBAL LLM settings (persisted JSON),
    // not from the per-chat `live_chat.settings` Value.
    let llm = AppSettings::load(&state.config.settings_path).llm;
    let orchestrator = orchestrator::OrchestratorSettings::new(
        llm.orchestrator_enabled,
        llm.orchestrator_max_speakers,
        llm.orchestrator_temperature,
        &llm.orchestrator_system_prompt,
    );

    let player_metadata = body.get("metadata").cloned().unwrap_or(Value::Null);
    let macros = chasm_prompt::macros_from_metadata(&player_metadata);
    // Scenario resolution uses the freshest table available: this turn's macros
    // when the mod sent them, else the latest recorded table from this chat.
    let scenario_macros = if macros.is_empty() {
        chasm_prompt::macros_from_value(&latest_chat_macros(state, &live_chat).1)
    } else {
        macros.clone()
    };
    let scenario_template = global_scenario_template(state);

    Ok(TurnContext {
        live_chat,
        segment,
        message,
        player_participant_id,
        structured,
        extra_context,
        gamestate,
        player_metadata,
        macros,
        scenario_macros,
        scenario_template,
        requested_scopes: parse_action_book_scopes(body),
        orchestrator,
    })
}

// ---------------------------------------------------------------------------
// Global scenario (the ONLY production surface that resolves gamestate macros)
// ---------------------------------------------------------------------------

/// The effective global scenario template: the Globals store value when saved
/// (empty string = the user explicitly disabled the component), else the
/// built-in `chasm_prompt::DEFAULT_SCENARIO_TEMPLATE`. Read fresh per request
/// so a Globals-page save applies on the very next turn.
pub(crate) fn global_scenario_template(state: &AppState) -> String {
    match state.repository.read_globals() {
        Ok(store) => store
            .scenario_template
            .unwrap_or_else(|| chasm_prompt::DEFAULT_SCENARIO_TEMPLATE.to_string()),
        Err(error) => {
            tracing::warn!(
                "globals store read failed ({error}); using the default scenario template"
            );
            chasm_prompt::DEFAULT_SCENARIO_TEMPLATE.to_string()
        }
    }
}

/// Merges the BACKEND-COMPUTED macros for one prompted character over the
/// turn's table (a computed key deliberately wins over a same-named mod key —
/// the backend knows the conversation better than the game plugin does).
///
/// Today that is `{{participants}}`: the player (named via the table's
/// `player_name`) plus the OTHER NPCs in the conversation. Future computed
/// macros belong here so every caller (live, admin, Globals preview) picks
/// them up together.
fn insert_computed_macros(macros: &mut BTreeMap<String, String>, other_npcs: &[String]) {
    let player_name = macros.get("player_name").cloned().unwrap_or_default();
    macros.insert(
        "participants".to_string(),
        chasm_prompt::participants_macro(&player_name, other_npcs),
    );
}

/// The OTHER present NPCs of a conversation (excluding the character being
/// prompted), by display name — the NPC half of `{{participants}}`. Presence
/// order is the store's participant-id order (BTreeMap), so it is stable.
fn other_npc_names(live_chat: &LiveChat, speaker_participant_id: &str) -> Vec<String> {
    orchestrator::compute_eligible(live_chat)
        .into_iter()
        .filter(|participant| participant.participant_id != speaker_participant_id)
        .map(|participant| participant.name)
        .collect()
}

/// Resolves the GLOBAL scenario text for one prompted character: computed
/// macros merged over `turn_macros`, then `apply_macros` on `template`.
/// Returns "" (component omitted) when the template is blank. This is the
/// scenario-only injection pass — no other prompt component runs macros.
fn resolve_global_scenario(
    template: &str,
    turn_macros: &BTreeMap<String, String>,
    other_npcs: &[String],
) -> String {
    if template.trim().is_empty() {
        return String::new();
    }
    let mut macros = turn_macros.clone();
    insert_computed_macros(&mut macros, other_npcs);
    chasm_prompt::apply_macros(template, &macros)
        .trim()
        .to_string()
}

/// The newest macros-bearing turn of one live chat: `(send_date, macros)` from
/// the persisted `extra.chasm.macros` blobs `finalize_turn` writes.
///
/// The live game path writes each NPC's history to its per-participant
/// PROJECTION session while older / group-mode chats use the shared segment
/// sessions (see `messages_for_participant` in chasm-st-compat) — and the two
/// overlap for turns visible to several NPCs. Scanning both and keeping the
/// newest `send_date` works for every layout: duplicated copies of a turn carry
/// the same macro table, so whichever copy wins the tie is correct.
///
/// Shared by the generation path (scenario fallback when a turn arrives without
/// fresh macros) and the Gamestate/Globals UI pages — one snapshot source.
pub(crate) fn latest_chat_macros(state: &AppState, live_chat: &LiveChat) -> (Option<String>, Value) {
    let mut session_ids: Vec<String> = live_chat
        .segments
        .iter()
        .map(|segment| segment.session_id.clone())
        .collect();
    if let Some(sessions) = live_chat.participant_sessions.as_object() {
        // Projection entries are `{ "sessionId": "…" }` objects (see
        // `participant_session_id`), but tolerate raw strings too.
        session_ids.extend(sessions.values().filter_map(|entry| {
            entry
                .get("sessionId")
                .and_then(Value::as_str)
                .or_else(|| entry.as_str())
                .filter(|id| !id.is_empty())
                .map(str::to_string)
        }));
    }
    session_ids.sort();
    session_ids.dedup();

    let mut best: Option<(String, Value)> = None;
    for session_id in &session_ids {
        // Unreadable / not-yet-created sessions are skipped rather than failing
        // the whole lookup — callers should always get a table (even empty).
        let Ok(messages) = state.repository.read_session_messages(session_id) else {
            continue;
        };
        for message in messages {
            let Some(macros) = message
                .extra
                .get("chasm")
                .and_then(|chasm| chasm.get("macros"))
                .filter(|macros| macros.as_object().is_some_and(|map| !map.is_empty()))
            else {
                continue;
            };
            // ISO-8601 timestamps compare lexicographically; `>=` keeps the
            // later-in-file copy on equal stamps.
            let send_date = message.send_date.clone().unwrap_or_default();
            if best
                .as_ref()
                .map_or(true, |(best_date, _)| send_date.as_str() >= best_date.as_str())
            {
                best = Some((send_date, macros.clone()));
            }
        }
    }

    match best {
        Some((send_date, macros)) => {
            let updated_at = (!send_date.is_empty()).then_some(send_date);
            (updated_at, macros)
        }
        None => (None, json!({})),
    }
}

/// The active live chat: most recently updated first, so a stale chat sitting
/// in the store never shadows the one the game is writing to. Shared by the
/// admin scenario fallback and the Gamestate/Globals UI pages.
pub(crate) fn active_live_chat(state: &AppState) -> WebResult<Option<LiveChat>> {
    let mut chats = state.repository.list_live_chats()?;
    chats.sort_by(|a, b| {
        b.updated_at
            .clone()
            .unwrap_or_default()
            .cmp(&a.updated_at.clone().unwrap_or_default())
    });
    Ok(chats.into_iter().next())
}

/// Runs the orchestrator. The deterministic path picks the forced speaker (when
/// requested) or the first eligible NPC. When the orchestrator is enabled, not
/// forced, and 2+ NPCs are eligible, a single model call decides who speaks and
/// in what order — falling back to the first-eligible speaker on ANY failure.
///
/// Returns `(speakers, selector_meta)`. An empty `speakers` is a VALID outcome
/// (the model chose silence).
async fn orchestrate(
    state: &Arc<AppState>,
    ctx: &TurnContext,
    body: &Value,
) -> WebResult<(Vec<orchestrator::SelectedSpeaker>, Value)> {
    let force_participant_id = body
        .get("forceParticipantId")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let force_character_id = body
        .get("forceCharacterId")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let forced = force_participant_id.is_some() || force_character_id.is_some();

    let input = orchestrator::SelectionInput {
        force_participant_id,
        force_character_id,
    };

    let fallback =
        orchestrator::select_live_speaker_candidates(&ctx.live_chat, &input).map_err(web_err)?;

    // Gate: model selector only when enabled, not forced, and >1 eligible NPC.
    if !orchestrator::should_use_model_speaker_selection(&fallback, &ctx.orchestrator, forced) {
        let selector = json!({ "mode": "fallback", "fallbackReason": fallback.reason });
        return Ok((fallback.speakers, selector));
    }

    // The per-turn LLM director is gone: pick who speaks with an instant weighted
    // score over the game + conversation signals (crosshair, name, proximity,
    // recency, topic). No model call — the whole thing is microseconds.
    let recent = recent_messages_for_selection(state, &ctx.live_chat);
    let selection = orchestrator::select_weighted_speakers(
        &ctx.live_chat,
        &fallback.eligible,
        &recent,
        &ctx.message,
        ctx.orchestrator.max_speakers,
    );
    let selector = json!({
        "mode": "weighted",
        "modelReason": selection
            .speakers
            .first()
            .and_then(|speaker| speaker.model_reason.clone()),
        "fallbackReason": fallback.reason,
    });
    Ok((selection.speakers, selector))
}

/// Reads the recent live messages used to build the selector transcript: read
/// each segment's JSONL, drop the first message + system messages, keep the last
/// `limit`. We read the raw `STJsonlChatMessage`s (not per-participant views) so
/// the selector sees the shared transcript regardless of visibility.
fn recent_messages_for_selection(
    state: &Arc<AppState>,
    live_chat: &LiveChat,
) -> Vec<STJsonlChatMessage> {
    let limit = orchestrator::selector_context_limit();
    let mut messages: Vec<STJsonlChatMessage> = Vec::new();
    for segment in &live_chat.segments {
        if let Ok(segment_messages) = state.repository.read_segment_messages(segment) {
            // `.slice(1)` drops the first message; filter out system messages.
            for message in segment_messages.into_iter().skip(1) {
                if message.is_system {
                    continue;
                }
                messages.push(message);
            }
        }
    }
    let start = messages.len().saturating_sub(limit);
    messages.split_off(start)
}

/// Builds the per-speaker `TurnPlan` from the shared context and one chosen
/// speaker. This is the speaker-dependent half of the old `prepare_turn`:
/// visible history, prompt assembly, and the chat-completion message array.
fn prepare_speaker_turn(
    state: &Arc<AppState>,
    ctx: &TurnContext,
    speaker: &orchestrator::SelectedSpeaker,
) -> WebResult<TurnPlan> {
    prepare_speaker_turn_traced(state, ctx, speaker, None)
}

fn prepare_speaker_turn_traced(
    state: &Arc<AppState>,
    ctx: &TurnContext,
    speaker: &orchestrator::SelectedSpeaker,
    trace_id: Option<&str>,
) -> WebResult<TurnPlan> {
    let speaker_participant_id = speaker.participant.participant_id.clone();
    let speaker_character_id = if speaker.participant.character_id.is_empty() {
        None
    } else {
        Some(speaker.participant.character_id.clone())
    };
    let speaker_name = if speaker.participant.name.is_empty() {
        speaker_participant_id.clone()
    } else {
        speaker.participant.name.clone()
    };

    // Visible history for this speaker (re-read each turn so later speakers see
    // earlier speakers' just-persisted messages — matching ST).
    let history = state
        .repository
        .messages_for_participant(&ctx.live_chat, &speaker_participant_id)?;

    // Resolve the participant view for prompt assembly (character card lookup).
    let view = state.repository.live_chat_view(&ctx.live_chat, None)?;
    let speaker_view = view
        .participants
        .iter()
        .find(|participant| participant.id == speaker_participant_id)
        .cloned()
        .unwrap_or_else(|| fallback_participant_view(speaker, &speaker_name));

    let response_instructions = build_response_instructions(&speaker_name, ctx.structured);

    // Lore / quest activation scans the player MESSAGE ONLY. The gamestate
    // (location + nearby-NPC list) is deliberately NOT in the scan: including it
    // fired location/faction/NPC lorebook entries every turn (e.g. "Caesar's Legion
    // basics" off a nearby Powder Ganger) even when the player never mentioned them.
    // Actions already scanned message-only; lore/quest now match. (Constant entries
    // still always inject — that's their point.)
    let retrieval_settings = AppSettings::load(&state.config.settings_path).retrieval;
    let retriever = state.retriever();
    let cache = state.embed_cache();
    let retrieval_ctx = match (retriever, cache) {
        (Some(retriever), Some(cache)) if retrieval_settings.enabled => Some(RetrievalCtx {
            retriever,
            cache,
            chat_memory_enabled: retrieval_settings.chat_memory_enabled,
            lore_semantic_enabled: retrieval_settings.lore_semantic_enabled,
            action_semantic_enabled: retrieval_settings.action_semantic_enabled,
            quest_semantic_enabled: retrieval_settings.quest_semantic_enabled,
            candidates: retrieval_settings.candidates as usize,
            top_k: retrieval_settings.top_k as usize,
            min_score: retrieval_settings.min_score,
            action_min_score: retrieval_settings.action_min_score,
            chat_memory_limit: retrieval_settings.chat_memory_limit as usize,
            lore_limit: retrieval_settings.lore_limit as usize,
            quest_limit: retrieval_settings.quest_limit as usize,
        }),
        _ => None,
    };
    // GLOBAL scenario for THIS speaker: the Globals template resolved with the
    // turn's macro table + computed macros ({{participants}} excludes the
    // speaker being prompted). NOT given to the assembler (Some("") omits the
    // old card-scenario slot): its per-turn timestamp there busted the LLM
    // prompt cache every turn. build_chat_messages injects it late instead.
    // Deliberately NOT part of the retrieval scan text above.
    let global_scenario = resolve_global_scenario(
        &ctx.scenario_template,
        &ctx.scenario_macros,
        &other_npc_names(&ctx.live_chat, &speaker_participant_id),
    );
    trace_generate_stage(trace_id, "gen_assemble_enter");
    let (assembled, injected) = chasm_prompt::assemble_prompt_with_retrieval_collect(
        &state.repository,
        &speaker_view,
        &history,
        &ctx.message,
        &ctx.message,
        &ctx.requested_scopes,
        retrieval_ctx,
        Some(""),
    );
    trace_generate_stage(trace_id, "gen_assemble_done");

    let scene_roster = build_scene_roster(state, ctx, &speaker_participant_id, &speaker_name);
    let chat_messages = build_chat_messages(
        &assembled,
        &history,
        &ctx.message,
        ctx.structured,
        &response_instructions,
        &ctx.extra_context,
        &ctx.gamestate,
        &speaker_participant_id,
        &scene_roster,
        // The player message is already persisted to the segment before this turn,
        // so it's in `history`; don't re-append it (that would make co-speakers
        // answer the player instead of the prior NPC's just-spoken line).
        false,
        &global_scenario,
    );

    let audible_to = default_audible_to(&ctx.live_chat, &speaker_participant_id);
    let present = present_participant_ids(&ctx.live_chat);
    let location = ctx.segment.location.clone();

    let speaker_json = json!({
        "participantId": speaker_participant_id,
        "characterId": speaker_character_id,
        "name": speaker_name,
        "queueIndex": speaker.queue_index,
        "reason": speaker.reason,
    });

    Ok(TurnPlan {
        live_chat_id: ctx.live_chat.id.clone(),
        segment: ctx.segment.clone(),
        speaker: speaker_json,
        speaker_participant_id,
        speaker_character_id,
        speaker_name,
        structured: ctx.structured,
        audible_to,
        present,
        location,
        chat_messages,
        injected,
    })
}

/// Builds the chat-completion message array a REAL first turn of `live_chat_id`
/// would send — same live chat, same deterministic first-eligible speaker, same
/// structured/text mode, same persisted history — minus the (unknown) new player
/// message. Used by the connect-time warm-up to pre-ingest the static prompt
/// prefix into the LLM server's KV cache: the actual first turn then
/// fast-forwards over everything up to the player's new line.
///
/// Pure read: nothing is persisted and no speaker turn is recorded. Returns
/// `None` when the live chat doesn't exist yet (fresh install before the first
/// ever turn) or has no eligible speaker — callers fall back to a generic
/// warm-up prompt.
pub(crate) fn warmup_chat_messages(
    state: &Arc<AppState>,
    live_chat_id: &str,
    structured: bool,
) -> Option<(Vec<Value>, String)> {
    let body = json!({
        "responseFormat": if structured { "structured" } else { "text" },
    });
    let ctx = resolve_turn_context(state, live_chat_id, &body).ok()?;
    let input = orchestrator::SelectionInput {
        force_participant_id: None,
        force_character_id: None,
    };
    let selection = orchestrator::select_live_speaker_candidates(&ctx.live_chat, &input).ok()?;
    let speaker = selection.speakers.first()?;
    let plan = prepare_speaker_turn(state, &ctx, speaker).ok()?;
    Some((plan.chat_messages, plan.speaker_name))
}

/// Synthesizes a `ParticipantView` when the merged participant list does not
/// contain the selected speaker (defensive; normally the speaker is present).
fn fallback_participant_view(
    speaker: &orchestrator::SelectedSpeaker,
    speaker_name: &str,
) -> ParticipantView {
    ParticipantView {
        id: speaker.participant.participant_id.clone(),
        name: speaker_name.to_string(),
        initial: speaker_name
            .chars()
            .next()
            .map(|ch| ch.to_uppercase().to_string())
            .unwrap_or_else(|| "?".to_string()),
        kind: "npc".to_string(),
        character_id: if speaker.participant.character_id.is_empty() {
            None
        } else {
            Some(speaker.participant.character_id.clone())
        },
        present: true,
        audible: true,
        distance: speaker.participant.distance,
        distance_label: String::new(),
        message_count: 0,
        selected: true,
    }
}

/// Builds the OpenAI chat-completion message array from the assembled prompt
/// components (system parts in send order), the visible history, then the
/// pending player turn. Mirrors `prepareGenerationRun` ordering.
#[allow(clippy::too_many_arguments)]
/// The roster descriptor for a co-present character: their FULL card description
/// (verbatim, line breaks preserved), falling back to personality only when the
/// description is blank — so each NPC gets the same rich context the player-facing
/// card injection carries about the others.
fn roster_descriptor(card: &chasm_st_compat::CharacterCard) -> String {
    let description = card.description.trim();
    if !description.is_empty() {
        return description.to_string();
    }
    card.personality.trim().to_string()
}

/// Builds the group "scene roster" for `speaker`: a block per OTHER present NPC
/// (name + their full description) plus an instruction that they share a
/// conversation and may address/react to one another. Returns "" for a 1-on-1
/// chat (no other NPC present), so single-NPC prompts are unchanged.
fn build_scene_roster(
    state: &Arc<AppState>,
    ctx: &TurnContext,
    speaker_participant_id: &str,
    speaker_name: &str,
) -> String {
    let others: Vec<orchestrator::EligibleParticipant> =
        orchestrator::compute_eligible(&ctx.live_chat)
            .into_iter()
            .filter(|participant| participant.participant_id != speaker_participant_id)
            .collect();
    if others.is_empty() {
        return String::new();
    }
    let mut blocks = Vec::with_capacity(others.len());
    for other in &others {
        let descriptor = state
            .repository
            .read_character_card(&other.character_id)
            .ok()
            .flatten()
            .map(|card| roster_descriptor(&card))
            .unwrap_or_default();
        if descriptor.is_empty() {
            blocks.push(other.name.clone());
        } else {
            blocks.push(format!("{}:\n{descriptor}", other.name));
        }
    }
    format!(
        "You are {speaker_name}, talking with the player in a group. The others present with you:\n\n{}\n\n\
         You may speak directly to them by name and react to what they just said — not only to the \
         player. Voice ONLY {speaker_name}; never write another character's or the player's lines, \
         and don't repeat what someone already said.",
        blocks.join("\n\n")
    )
}

fn build_chat_messages(
    assembled: &chasm_core::PromptAssemblyView,
    history: &[MessageView],
    message: &str,
    structured: bool,
    response_instructions: &str,
    extra_context: &str,
    gamestate: &Value,
    current_speaker_participant_id: &str,
    scene_roster: &str,
    append_player_message: bool,
    global_scenario: &str,
) -> Vec<Value> {
    // Components whose content CHANGES from turn to turn (retrieval picks,
    // book activations, live world state). They must NOT ride in the head
    // system message: any byte changing there invalidates the LLM's cached
    // prefix and forces re-ingestion of the entire history every turn
    // (measured: first-token time grows 1.2s -> 2.2s over a few turns). They
    // are injected at depth 1 instead, like the scenario below, where only the
    // prompt tail reprocesses.
    const VOLATILE_KEYS: [&str; 5] = [
        "lore",
        "chat_vectors",
        "quest_books",
        "action_books",
        "world_state",
    ];
    let included = |component: &&chasm_core::PromptComponentView| {
        component.group == "system"
            && component.status == "included"
            && !component.content.is_empty()
    };
    let mut system_parts: Vec<String> = assembled
        .components
        .iter()
        .filter(included)
        .filter(|component| !VOLATILE_KEYS.contains(&component.key.as_str()))
        .map(|component| component.content.clone())
        .collect();
    let mut volatile_parts: Vec<String> = assembled
        .components
        .iter()
        .filter(included)
        .filter(|component| VOLATILE_KEYS.contains(&component.key.as_str()))
        .map(|component| component.content.clone())
        .collect();

    // Group "scene roster": who else is present + that they may address/react to
    // each other. Empty in 1-on-1 chats, so single-NPC prompts are unchanged.
    if !scene_roster.is_empty() {
        system_parts.push(scene_roster.to_string());
    }

    // Structured-output rules. The quest/action instructions are gated on whether
    // those books actually *activated* this turn (mirrors generation.js, which only
    // appends them `if (questResult.items.length > 0)` / `actionResult.items > 0`),
    // not merely on whether the books are enabled.
    if structured {
        let has_quest_block = assembled.components.iter().any(|component| {
            component
                .content
                .starts_with("Activated Quest Book entries:")
        });
        // Single source of action guidance now lives in STRUCTURED_OUTPUT_INSTRUCTION
        // (lean + positive). The action-book *entries* themselves are still injected
        // as a system component; only the redundant how-to blocks were removed.
        system_parts.push(chasm_prompt::STRUCTURED_OUTPUT_INSTRUCTION.to_string());
        if has_quest_block {
            // Rides with the quest entries in the volatile block: it appears
            // only on turns where a quest book activated, so in the head
            // system message it would churn the cached prefix.
            volatile_parts
                .push(chasm_prompt::QUEST_BOOK_STRUCTURED_OUTPUT_INSTRUCTION.to_string());
        }
    }

    // Gamestate injection into the prompt is DISABLED per request. The game state
    // (player location + nearby NPCs) is STILL built, passed in, and used for the
    // retrieval keyword-scan + NPC resolution — it's just no longer shown to the
    // model in the prompt. Re-enable by uncommenting the push below.
    let _ = gamestate;
    // if !gamestate.is_null() {
    //     if let Ok(text) = serde_json::to_string(gamestate) {
    //         system_parts.push(format!("Gamestate:\n{text}"));
    //     }
    // }
    if !extra_context.is_empty() {
        system_parts.push(format!("Additional external context:\n{extra_context}"));
    }
    if !response_instructions.is_empty() {
        system_parts.push(response_instructions.to_string());
    }

    let mut messages: Vec<Value> = Vec::new();
    if !system_parts.is_empty() {
        messages.push(json!({
            "role": "system",
            "content": system_parts.join("\n\n"),
        }));
    }

    // In a GROUP chat the visible history holds lines from other NPCs too. Mapped
    // to bare `assistant` turns with no attribution, the model treats those other
    // characters' lines as its OWN prior turns — so personas blend and the NPCs
    // start echoing/repeating each other. Prefix each line spoken by a DIFFERENT
    // participant with their name so the model can tell who said what. Only when
    // the history actually contains 2+ distinct NPC speakers (a real group): 1-on-1
    // live chats and admin sessions are left byte-for-byte unchanged.
    // The window start advances in BLOCKS, not message-by-message. A start that
    // slides 2 forward every turn changes the first history message every turn,
    // which kills the LLM's prefix cache right after the head (measured live:
    // prompt_ms ~1.2s, ~4900 of 5748 tokens reprocessed per turn — llama.cpp's
    // --cache-reuse chunk shifting is disabled when --mmproj is loaded, so only
    // the exact prefix survives). Quantized, the window holds byte-stable for
    // WINDOW_DROP_BLOCK/2 turns (only appends), then pays one full reprocess.
    // The window is never SMALLER than CONTEXT_MESSAGE_LIMIT — it runs up to
    // BLOCK-1 messages larger until the next quantized drop.
    const WINDOW_DROP_BLOCK: usize = 16;
    let overflow = history.len().saturating_sub(CONTEXT_MESSAGE_LIMIT);
    let start = (overflow / WINDOW_DROP_BLOCK) * WINDOW_DROP_BLOCK;
    let window = &history[start..];
    let distinct_npc_speakers = window
        .iter()
        .filter(|m| m.role != "player" && m.role != "system")
        .filter_map(|m| m.speaker_participant_id.as_deref())
        .filter(|id| !id.is_empty())
        .collect::<std::collections::HashSet<_>>()
        .len();
    // A turn counts as a group either when the history already holds 2+ distinct NPC
    // speakers OR when 2+ NPCs are present this turn (`scene_roster` non-empty). The
    // roster check fixes the FIRST group turn: speakers generate sequentially within
    // one turn, so when the 2nd speaker (e.g. Sunny) generates, history holds only the
    // 1st speaker's (Pete's) just-persisted line — distinct_npc_speakers == 1 — and
    // without the roster signal she'd see his line as her OWN prior `assistant` turn
    // and echo it verbatim (it "self-heals" on turn 2 once history has 2 speakers).
    // `scene_roster` is "" for 1-on-1 live chats and admin, so both stay unchanged.
    let is_group = distinct_npc_speakers > 1 || !scene_roster.is_empty();
    for message_view in window {
        let base_role = match message_view.role.as_str() {
            "player" => "user",
            "system" => "system",
            _ => "assistant",
        };
        // In a group, a line spoken by a DIFFERENT NPC is INPUT to the current
        // speaker, not the speaker's own prior output. Present it as a `user` turn
        // labeled with the speaker's name. This does two things: (a) the model can
        // tell the personas apart, and (b) the transcript keeps alternating, so the
        // model writes a NEW reply for the current speaker instead of echoing the
        // previous NPC's `assistant` turn verbatim (it was repeating it 1:1 when the
        // prompt ended on another NPC's assistant line). Only the current speaker's
        // OWN past lines stay `assistant`.
        let other_speaker = is_group
            && base_role == "assistant"
            && !message_view.speaker_name.is_empty()
            && message_view
                .speaker_participant_id
                .as_deref()
                .is_some_and(|id| !id.is_empty() && id != current_speaker_participant_id);
        let (role, content) = if other_speaker {
            (
                "user",
                format!("{}: {}", message_view.speaker_name, message_view.content),
            )
        } else {
            (base_role, message_view.content.clone())
        };
        messages.push(json!({
            "role": role,
            "content": content,
        }));
    }

    // Append the current player message as the final user turn ONLY when it isn't
    // already in `history`. The live path persists it before generating (so it's
    // in history, in its correct position before any co-speaker's reply) — appending
    // again would put the player's line AFTER the prior NPC's reply, making each
    // later speaker answer the player instead of bouncing off what was just said.
    // The admin path doesn't persist it, so it still needs the append.
    if append_player_message && !message.is_empty() {
        messages.push(json!({ "role": "user", "content": message }));
    }

    // Volatile retrieved context (lore, past-chat memory, quest/action book
    // activations), injected LATE at depth 1 for the same cache-preserving
    // reason as the scenario below.
    if !volatile_parts.is_empty() {
        let volatile_message =
            json!({ "role": "system", "content": volatile_parts.join("\n\n") });
        let insert_at = if messages.len() > 1 {
            messages.len() - 1
        } else {
            messages.len()
        };
        messages.insert(insert_at, volatile_message);
    }

    // GLOBAL scenario, injected LATE (ST-style in-chat injection at depth 1 -
    // just before the newest line). The scenario carries a per-turn timestamp;
    // sitting in the early system prompt it invalidated the LLM's cached prefix
    // every turn, forcing re-ingestion of everything after it (chat history
    // included) - measured at 1-2s per turn. Down here, the static system
    // prompt and the history stay cached and only ~a hundred tokens reprocess.
    if !global_scenario.is_empty() {
        let scenario_message = json!({ "role": "system", "content": global_scenario });
        let insert_at = if messages.len() > 1 {
            messages.len() - 1
        } else {
            messages.len()
        };
        messages.insert(insert_at, scenario_message);
    }

    messages
}

/// Finds the byte index just past the next COMPLETE sentence in `speech`,
/// starting the scan at `from`. A sentence is complete when a terminator
/// (`.` `!` `?` `…`, plus any closing quotes/parens) is followed by whitespace
/// and at least one more non-whitespace character — i.e. the next sentence has
/// visibly started. The final sentence therefore never matches (the remainder
/// path emits it), and mid-number periods ("2.5") never match. Common honorific
/// abbreviations ("Mr. House") and single-letter initials are guarded.
fn next_sentence_end(speech: &str, from: usize) -> Option<usize> {
    const CLOSERS: [char; 5] = ['"', '\'', ')', '\u{201d}', '\u{2019}'];
    const ABBREVIATIONS: [&str; 14] = [
        "mr", "mrs", "ms", "dr", "st", "lt", "sgt", "col", "gen", "capt", "prof", "jr", "sr", "vs",
    ];
    let tail = speech.get(from..)?;
    let mut chars = tail.char_indices().peekable();
    while let Some((offset, ch)) = chars.next() {
        if !matches!(ch, '.' | '!' | '?' | '\u{2026}') {
            continue;
        }
        // Absorb runs of terminators ("?!", "...") and trailing closers.
        let mut end = offset + ch.len_utf8();
        while let Some(&(next_offset, next_ch)) = chars.peek() {
            if matches!(next_ch, '.' | '!' | '?' | '\u{2026}') || CLOSERS.contains(&next_ch) {
                end = next_offset + next_ch.len_utf8();
                chars.next();
            } else {
                break;
            }
        }
        // Confirmed complete only when whitespace + a following word exist.
        let after = &tail[end..];
        let mut after_chars = after.chars();
        let Some(first_after) = after_chars.next() else {
            return None; // end of text: leave it for the remainder delta
        };
        if !first_after.is_whitespace() {
            continue; // "2.5", "e.g", mid-token punctuation
        }
        match after.trim_start().chars().next() {
            None => return None,
            // A real sentence start is capitalized (or a digit/quote); a
            // lowercase continuation means the terminator was mid-sentence
            // (quoted speech like «"Well..." she said»).
            Some(next_start) if next_start.is_lowercase() => continue,
            Some(_) => {}
        }
        if ch == '.' {
            // Abbreviation guard: the word directly before the period.
            let before = &tail[..offset];
            let word: String = before
                .chars()
                .rev()
                .take_while(|c| c.is_alphanumeric())
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            let lower = word.to_lowercase();
            if ABBREVIATIONS.contains(&lower.as_str()) || (word.len() == 1 && word.chars().all(|c| c.is_uppercase())) {
                continue;
            }
        }
        return Some(from + end);
    }
    None
}

/// Per-speaker response instruction, mirroring the Node `responseInstructions`.
fn build_response_instructions(speaker_name: &str, structured: bool) -> String {
    if structured {
        // The JSON shape + action guidance live in STRUCTURED_OUTPUT_INSTRUCTION; this
        // only adds the speaker-specific speech rule so it isn't duplicated.
        format!(
            "In \"speech\", write only {speaker_name}'s spoken words — do not start with \
\"{speaker_name}:\" or repeat any speaker label."
        )
    } else {
        format!(
            "Output only {speaker_name}'s spoken words. Do not start with \"{speaker_name}:\" \
and do not repeat any speaker label."
        )
    }
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

/// Appends the player's message to the segment JSONL with headless live
/// metadata, mirroring the Node `appendLiveMessage({ role: 'user' })`. Player
/// audibility = the present participants (the player is heard by everyone
/// present). Called once per generate request, before speaker selection.
fn persist_player_message_ctx(state: &Arc<AppState>, ctx: &TurnContext) -> WebResult<()> {
    let present = present_participant_ids(&ctx.live_chat);
    let live = json!({
        "liveChatId": ctx.live_chat.id,
        "segmentId": ctx.segment.id,
        "speakerParticipantId": ctx.player_participant_id,
        "present": present,
        "audibleTo": present,
        "location": ctx.segment.location,
        "strictVisibility": true,
    });
    let mut headless = serde_json::Map::new();
    headless.insert("characterId".to_string(), Value::Null);
    let mut metadata = serde_json::Map::new();
    metadata.insert("live".to_string(), live);
    if let Value::Object(map) = &ctx.player_metadata {
        for (key, value) in map {
            metadata.insert(key.clone(), value.clone());
        }
    }
    headless.insert("metadata".to_string(), Value::Object(metadata));

    let message = STJsonlChatMessage {
        name: "Player".to_string(),
        is_user: true,
        is_system: false,
        send_date: Some(now_iso()),
        mes: ctx.message.clone(),
        extra: json!({ "headless": Value::Object(headless) }),
        original_avatar: None,
    };
    state
        .repository
        .append_segment_message(&ctx.segment, &message)?;
    Ok(())
}

/// Builds the final turn JSON (in the shape the helper consumes), appends the
/// assistant message to the segment JSONL, and bumps the live-chat updatedAt.
/// `macros` is the request's gamestate macro table (`ctx.macros`), recorded
/// verbatim onto the persisted message's `extra.chasm.macros`.
fn finalize_turn(
    state: &Arc<AppState>,
    plan: &TurnPlan,
    macros: &BTreeMap<String, String>,
    raw: &str,
) -> WebResult<Value> {
    let raw_trimmed = strip_reasoning_channel(raw);
    // Structured output: try to parse a JSON object with `speech`/`message`, then
    // resolve emitted action aliases to canonical action ids so the helper can
    // match them (mirrors ST `normalizeStructuredActionAliases`).
    // Resolve the action alias map once so it can both normalize the emitted
    // actions and recover each action's alias for the persisted `turn_actions`.
    let aliases = structured_action_aliases(state);
    let (content, structured) = if plan.structured {
        match parse_structured(&raw_trimmed) {
            Some((speech, mut structured)) => {
                if !aliases.is_empty() {
                    normalize_structured_action_aliases(&mut structured, &aliases);
                }
                (speech, Some(structured))
            }
            None => (raw_trimmed.clone(), None),
        }
    } else {
        (raw_trimmed.clone(), None)
    };
    // Final guard: strip any speaker label prefix the model echoed.
    let content = strip_speaker_prefix(&content, &plan.speaker_name);

    // Per-message panel blob: what was injected into this turn's prompt + the
    // actions the NPC chose + the turn's gamestate macros. Stored under
    // `extra.chasm`, ADDITIVE to the existing `headless` block (the
    // `/api/headless/v1/*` response shapes the mod sees are unchanged — this
    // only enriches the persisted message on disk).
    let chasm_extra = build_chasm_extra(&plan.injected, structured.as_ref(), &aliases, macros);

    let live = json!({
        "liveChatId": plan.live_chat_id,
        "segmentId": plan.segment.id,
        "speakerParticipantId": plan.speaker_participant_id,
        "present": plan.present,
        "audibleTo": plan.audible_to,
        "location": plan.location,
        "strictVisibility": true,
    });
    let mut metadata = serde_json::Map::new();
    metadata.insert("live".to_string(), live);
    if let Some(structured) = &structured {
        metadata.insert("structured".to_string(), structured.clone());
    }
    let mut headless = serde_json::Map::new();
    headless.insert(
        "characterId".to_string(),
        plan.speaker_character_id
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    headless.insert("metadata".to_string(), Value::Object(metadata.clone()));

    // Persist the assistant message (only when it has content, like Node).
    if !content.is_empty() {
        let assistant = STJsonlChatMessage {
            name: plan.speaker_name.clone(),
            is_user: false,
            is_system: false,
            send_date: Some(now_iso()),
            mes: content.clone(),
            extra: json!({
                "headless": Value::Object(headless),
                "chasm": chasm_extra,
            }),
            original_avatar: None,
        };
        state
            .repository
            .append_segment_message(&plan.segment, &assistant)?;
    }

    // Bump updatedAt on the live chat.
    let id = plan.live_chat_id.clone();
    let now = now_iso();
    state.repository.update_store(|store| {
        if let Some(live_chat) = store.items.get_mut(&id) {
            live_chat.updated_at = Some(now);
        }
    })?;

    // Per-turn payload — matches the per-turn fields the helper reads via
    // `turn.turns[]`: `speaker`, `message.content`, optional `structured`. This
    // is one element; the top-level response builder assembles the `turns` array
    // and back-compat single-turn fields. Mirrors a `generateLiveChat` turn
    // object (`{ ...result, speaker, message, metadata }`).
    let message_obj = json!({
        "role": "assistant",
        "content": content,
        "name": plan.speaker_name,
    });
    let mut turn = serde_json::Map::new();
    turn.insert("liveChatId".to_string(), json!(plan.live_chat_id));
    turn.insert("segmentId".to_string(), json!(plan.segment.id));
    turn.insert("speaker".to_string(), plan.speaker.clone());
    turn.insert("message".to_string(), message_obj);
    // Relay the activated actions' trusted execution/binding ON THE TURN ONLY (the
    // headless message persisted above was cloned before this), so the helper's
    // `collectActivatedActions` can build native commands for non-native actions
    // (gestures, spawn) without bloating the saved chat history with GECK scripts.
    if !plan.injected.activated_actions.is_empty() {
        metadata.insert(
            "activatedActions".to_string(),
            serde_json::to_value(&plan.injected.activated_actions).unwrap_or_else(|_| json!([])),
        );
    }
    turn.insert("metadata".to_string(), Value::Object(metadata));
    if let Some(structured) = structured {
        turn.insert("structured".to_string(), structured);
    }
    Ok(Value::Object(turn))
}

/// Builds the `extra.chasm` blob persisted onto an NPC turn:
/// `{ "injected": { "lore"|"quests"|"actions": [...] }, "turn_actions": [...],
/// "macros": { key: value, ... } }`.
/// `injected` is the set of entries the assembler folded into this turn's prompt;
/// `turn_actions` is the NPC's chosen actions (flattened from the parsed
/// structured output, with each action's alias recovered from `aliases`). When
/// the turn was plain-text (no structured output), `turn_actions` is empty.
/// `macros` is the turn's gamestate macro table (`metadata.macros` from the
/// request) — recorded for the Gamestate page / test harness, NOT injected into
/// any prompt. This is read back by the st-compat message-view mapper for the
/// per-message panel and by the `/api/ui/v1/gamestate` view.
fn build_chasm_extra(
    injected: &InjectedView,
    structured: Option<&Value>,
    aliases: &[(String, String)],
    macros: &BTreeMap<String, String>,
) -> Value {
    let turn_actions = structured
        .map(|structured| chasm_prompt::turn_actions_from_structured(structured, aliases))
        .unwrap_or_default();
    json!({
        "injected": serde_json::to_value(injected).unwrap_or_else(|_| json!({})),
        "turn_actions": serde_json::to_value(&turn_actions).unwrap_or_else(|_| json!([])),
        "macros": serde_json::to_value(macros).unwrap_or_else(|_| json!({})),
    })
}

/// Assembles the multi-turn response payload (mirrors `generateLiveChat`'s
/// response object, lines 1505-1518). Keeps the top-level single-turn fields
/// populated from the FIRST turn for back-compat with the FNV helper, which
/// reads `turn.turns[]` when present and otherwise `turn.speaker`/`turn.message`.
fn build_live_response(
    live_chat_id: &str,
    segment_id: &str,
    speakers: &[Value],
    selector: Value,
    turns: Vec<Value>,
) -> Value {
    let first = turns.first().cloned().unwrap_or(Value::Null);
    let first_speaker = first.get("speaker").cloned().unwrap_or(Value::Null);
    let first_message = first.get("message").cloned().unwrap_or(Value::Null);
    let first_metadata = first.get("metadata").cloned().unwrap_or_else(|| json!({}));
    let messages: Vec<Value> = turns
        .iter()
        .filter_map(|turn| turn.get("message").cloned())
        .collect();

    let mut response = serde_json::Map::new();
    response.insert("liveChatId".to_string(), json!(live_chat_id));
    response.insert("segmentId".to_string(), json!(segment_id));
    response.insert("speaker".to_string(), first_speaker);
    response.insert("speakers".to_string(), json!(speakers));
    response.insert("speakerSelection".to_string(), selector);
    response.insert("message".to_string(), first_message);
    response.insert("messages".to_string(), json!(messages));
    response.insert("turns".to_string(), json!(turns));
    response.insert("metadata".to_string(), first_metadata);
    if let Some(structured) = first.get("structured").cloned() {
        response.insert("structured".to_string(), structured);
    }
    Value::Object(response)
}

/// The JSON shape the helper consumes for each entry in `speakers`.
fn speaker_summary(speaker: &orchestrator::SelectedSpeaker) -> Value {
    let character_id = if speaker.participant.character_id.is_empty() {
        Value::Null
    } else {
        Value::String(speaker.participant.character_id.clone())
    };
    let mut value = json!({
        "participantId": speaker.participant.participant_id,
        "characterId": character_id,
        "name": speaker.participant.name,
        "queueIndex": speaker.queue_index,
        "reason": speaker.reason,
    });
    if let Some(model_reason) = &speaker.model_reason {
        value["modelReason"] = json!(model_reason);
    }
    if let Some(confidence) = speaker.confidence {
        value["confidence"] = json!(confidence);
    }
    value
}

/// Strips a model reasoning/thinking preamble before the real answer. Handles
/// the harmony-style `<|channel>thought ... <channel|>` (and `<|...|>`) markers
/// some local models emit, plus `<think>...</think>` blocks. Best-effort: when
/// no recognizable marker is present the input is returned trimmed.
fn strip_reasoning_channel(raw: &str) -> String {
    let mut text = raw.trim().to_string();

    // <think>...</think> (DeepSeek-style).
    if let Some(end) = text.find("</think>") {
        text = text[end + "</think>".len()..].trim().to_string();
    }

    // Harmony channel markers: drop everything up to and including the last
    // closing channel marker, then strip any stray `<|...|>` / `<...|>` tokens.
    for closer in ["<channel|>", "<|channel|>", "<|message|>", "<|end|>"] {
        if let Some(pos) = text.rfind(closer) {
            text = text[pos + closer.len()..].to_string();
        }
    }
    // Remove any remaining angle-bracket control tokens like `<|channel>` /
    // `<|assistant|>` that may bookend the content.
    let cleaned: String = strip_control_tokens(&text);
    cleaned.trim().to_string()
}

/// Removes short `<|...|>` style control tokens from `text`. A token starts at
/// `<|` and ends at the next `|>` within a small window (so we don't eat real
/// prose that merely contains a `<`).
fn strip_control_tokens(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<|") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        match after.find("|>") {
            Some(rel) if rel <= 24 => {
                rest = &after[rel + 2..];
            }
            _ => {
                // Not a control token; keep the `<|` literally and move on.
                out.push_str("<|");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Drops a leading `Name:` speaker label the model may have echoed, mirroring
/// the helper's `stripSpeakerPrefix`.
fn strip_speaker_prefix(content: &str, speaker_name: &str) -> String {
    let trimmed = content.trim_start();
    let prefix = format!("{speaker_name}:");
    if let Some(rest) = trimmed.strip_prefix(&prefix) {
        return rest.trim_start().to_string();
    }
    trimmed.to_string()
}

/// Parses a structured-output object, returning `(speech, structuredValue)`.
/// Tolerant of a reasoning/thinking preamble (e.g. harmony `<|channel>thought`
/// markers) and ```json fences: it extracts the first balanced top-level JSON
/// object found in the raw text and parses that.
/// Loads the profile's action books and returns their `(action_id, alias)` map,
/// used to resolve a model's emitted action alias/id to the canonical action id.
fn structured_action_aliases(state: &Arc<AppState>) -> Vec<(String, String)> {
    let actions: Vec<chasm_st_compat::ActionEntry> =
        match state.repository.list_action_books() {
            Ok(books) => books
                .into_iter()
                .flat_map(|book| book.entries.into_iter())
                .collect(),
            Err(_) => Vec::new(),
        };
    chasm_prompt::action_alias_pairs(&actions)
}

/// Rewrites `structured["actions"]` so each emitted action carries the canonical
/// `id`, resolving aliases — string entries (`"follow"`) or objects keyed by
/// `id`/`actionId`/`action_id`/`name`/`alias`. Drops actions that match no known
/// alias/id. Mirrors ST `normalizeStructuredActionAliases`; the `alias` key is
/// added to the object lookup since small models often emit `{"alias": "..."}`.
fn normalize_structured_action_aliases(structured: &mut Value, aliases: &[(String, String)]) {
    let Some(raw_actions) = structured.get("actions").and_then(Value::as_array).cloned() else {
        return;
    };
    let mut by_alias: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (action_id, alias) in aliases {
        by_alias.insert(
            chasm_prompt::slug_action_alias(alias),
            action_id.clone(),
        );
        ids.insert(action_id.clone());
    }

    let mut out: Vec<Value> = Vec::new();
    for raw in &raw_actions {
        match raw {
            Value::String(text) => {
                if let Some(id) = by_alias.get(&chasm_prompt::slug_action_alias(text)) {
                    out.push(json!({
                        "id": id,
                        "target": "",
                        "parameters": {},
                        "reason": format!("Selected action alias \"{text}\"."),
                    }));
                }
            }
            Value::Object(map) => {
                let raw_id = ["id", "actionId", "action_id", "name", "alias"]
                    .iter()
                    .find_map(|key| map.get(*key).and_then(Value::as_str))
                    .map(str::trim)
                    .unwrap_or("");
                let canonical = by_alias
                    .get(&chasm_prompt::slug_action_alias(raw_id))
                    .cloned()
                    .or_else(|| ids.contains(raw_id).then(|| raw_id.to_string()));
                if let Some(id) = canonical {
                    let mut obj = map.clone();
                    obj.insert("id".to_string(), json!(id));
                    if !obj.get("parameters").map(Value::is_object).unwrap_or(false) {
                        obj.insert("parameters".to_string(), json!({}));
                    }
                    let target = obj
                        .get("target")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    obj.insert("target".to_string(), json!(target));
                    out.push(Value::Object(obj));
                }
            }
            _ => {}
        }
    }

    if let Some(obj) = structured.as_object_mut() {
        obj.insert("actions".to_string(), Value::Array(out));
    }
}

fn parse_structured(raw: &str) -> Option<(String, Value)> {
    let candidate = first_json_object(strip_code_fence(raw))?;
    let value: Value = serde_json::from_str(candidate).ok()?;
    let obj = value.as_object()?;
    let speech = obj
        .get("speech")
        .or_else(|| obj.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    Some((speech, value))
}


/// Speech text extracted from the (partial or complete) raw model output: the
/// structured `speech` field via the incremental extractor, or plain text with
/// the reasoning channel stripped. Grows monotonically as more output arrives, so
/// a byte offset taken from an earlier call stays valid against a later one.
fn extracted_speech(structured: bool, collected: &str) -> String {
    if structured {
        extract_structured_speech_prefix(collected)
    } else {
        strip_reasoning_channel(collected)
    }
}


/// Incrementally extracts the `speech` string value from a partial structured
/// JSON buffer (mirrors ST `extractStructuredSpeechPrefix`). Finds the first
/// `"speech": "` — which naturally skips any reasoning/preamble before the JSON
/// — then reads the string contents up to the next unescaped quote, decoding
/// JSON escapes. Lets the stream emit only spoken text, never the reasoning
/// channel or JSON syntax.
fn extract_structured_speech_prefix(raw: &str) -> String {
    let Some(key) = raw.find("\"speech\"") else {
        return String::new();
    };
    let bytes = raw.as_bytes();
    let mut i = key + "\"speech\"".len();
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b':' {
        return String::new();
    }
    i += 1;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'"' {
        return String::new();
    }
    i += 1; // first content byte
    let start = i;
    let mut escaped = false;
    let mut end = bytes.len();
    while i < bytes.len() {
        let c = bytes[i];
        if !escaped && c == b'"' {
            end = i;
            break;
        }
        escaped = !escaped && c == b'\\';
        i += 1;
    }
    // `start`/`end` sit on ASCII (`"`) boundaries, so the slice is valid UTF-8.
    decode_json_string_prefix(&raw[start..end])
}

/// Decodes JSON string escapes in a (possibly truncated) string body, mirroring
/// ST `decodeJsonStringPrefix`. A trailing incomplete escape (a lone `\` or a
/// partial `\uXXXX`) is dropped, so no half-decoded garbage is emitted before
/// the rest of the token arrives.
fn decode_json_string_prefix(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    let mut out = String::with_capacity(value.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c != '\\' {
            out.push(c);
            i += 1;
            continue;
        }
        i += 1;
        if i >= chars.len() {
            break; // lone trailing backslash
        }
        match chars[i] {
            '"' => out.push('"'),
            '\\' => out.push('\\'),
            '/' => out.push('/'),
            'b' => out.push('\u{0008}'),
            'f' => out.push('\u{000C}'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            'u' => {
                if i + 4 < chars.len() {
                    let hex: String = chars[i + 1..i + 5].iter().collect();
                    match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                        Some(ch) => {
                            out.push(ch);
                            i += 4;
                        }
                        None => return out,
                    }
                } else {
                    return out; // incomplete \uXXXX — wait for more
                }
            }
            other => out.push(other),
        }
        i += 1;
    }
    out
}

/// Returns the substring spanning the first balanced `{ ... }` JSON object in
/// `text`, honoring quoted strings and escapes so braces inside strings don't
/// break the match. Returns `None` when no balanced object is present.
fn first_json_object(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let start = text.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, &byte) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..=index]);
                }
            }
            _ => {}
        }
    }
    None
}

fn strip_code_fence(raw: &str) -> &str {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        // Drop an optional language tag on the first line, then the trailing ```.
        let rest = rest.split_once('\n').map(|(_, body)| body).unwrap_or(rest);
        return rest.trim_end().trim_end_matches("```").trim();
    }
    trimmed
}

// ---------------------------------------------------------------------------
// Live-chat helpers (presence, mapping, visibility)
// ---------------------------------------------------------------------------

/// Applies a presence body to the live chat (mirrors `updateLivePresence`):
/// upserts each incoming participant into `participants` + `presence`, marks
/// absent NPCs when `replace` is set, and refreshes the active id list.
fn apply_presence(live_chat: &mut LiveChat, body: &Value, replace: bool, now: &str) {
    let incoming: Vec<LiveChatParticipant> = body
        .get("participants")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| normalize_participant(item, now))
                .collect()
        })
        .unwrap_or_default();
    let incoming_ids: std::collections::BTreeSet<String> = incoming
        .iter()
        .map(|participant| participant.participant_id.clone())
        .collect();

    if replace {
        for participant in live_chat.presence.values_mut() {
            if participant.kind == "npc" && !incoming_ids.contains(&participant.participant_id) {
                participant.present = Some(false);
                participant.audible = Some(false);
                participant.updated_at = Some(now.to_string());
            }
        }
    }

    for participant in incoming {
        live_chat
            .participants
            .insert(participant.participant_id.clone(), participant.clone());
        live_chat
            .presence
            .insert(participant.participant_id.clone(), participant);
    }

    live_chat.active_participant_ids = live_chat
        .presence
        .values()
        .filter(|participant| participant.present.unwrap_or(false))
        .map(|participant| participant.participant_id.clone())
        .collect();
    live_chat.updated_at = Some(now.to_string());
}

/// Normalizes one incoming presence participant object.
fn normalize_participant(item: &Value, now: &str) -> Option<LiveChatParticipant> {
    let participant_id = item
        .get("participantId")
        .and_then(Value::as_str)?
        .to_string();
    if participant_id.is_empty() {
        return None;
    }
    Some(LiveChatParticipant {
        participant_id,
        kind: item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("npc")
            .to_string(),
        character_id: item
            .get("characterId")
            .and_then(Value::as_str)
            .map(str::to_string),
        name: item
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        present: item.get("present").and_then(Value::as_bool),
        audible: item.get("audible").and_then(Value::as_bool),
        distance: item.get("distance").and_then(Value::as_f64),
        metadata: item.get("metadata").cloned().unwrap_or(Value::Null),
        updated_at: Some(now.to_string()),
        segment_id: None,
    })
}

fn current_segment(live_chat: &LiveChat) -> Option<LiveChatSegment> {
    live_chat
        .segments
        .iter()
        .find(|segment| segment.id == live_chat.current_segment_id)
        .or_else(|| live_chat.segments.last())
        .cloned()
}

/// Default audibility: the speaker plus every present participant (so the
/// player and co-present NPCs hear the line). Mirrors `getDefaultAudibleTo`'s
/// present-set behaviour for the single-speaker case.
fn default_audible_to(live_chat: &LiveChat, speaker_id: &str) -> Vec<String> {
    let mut ids: Vec<String> = present_participant_ids(live_chat);
    if !ids.iter().any(|id| id == speaker_id) {
        ids.push(speaker_id.to_string());
    }
    ids
}

fn present_participant_ids(live_chat: &LiveChat) -> Vec<String> {
    live_chat
        .presence
        .values()
        .filter(|participant| participant.present.unwrap_or(false))
        .map(|participant| participant.participant_id.clone())
        .collect()
}

/// Maps a `LiveChat` to the JSON the helper's `GET`/create/presence calls
/// expect. The helper only inspects status codes here, so a faithful-but-light
/// projection is sufficient.
fn map_live_chat(live_chat: &LiveChat) -> Value {
    json!({
        "id": live_chat.id,
        "title": live_chat.title,
        "groupId": live_chat.group_id,
        "currentSegmentId": live_chat.current_segment_id,
        "activeParticipantIds": live_chat.active_participant_ids,
        "segments": live_chat.segments.iter().map(|segment| json!({
            "id": segment.id,
            "title": segment.title,
            "location": segment.location,
            "sessionId": segment.session_id,
        })).collect::<Vec<_>>(),
        "updatedAt": live_chat.updated_at,
        "createdAt": live_chat.created_at,
    })
}

/// Base64url(no-pad) JSON session id for a group segment, matching the Node
/// encoding the st-compat reader decodes.
fn encode_group_session_id(group_id: &str, chat_id: &str) -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    let payload = json!({ "mode": "group", "groupId": group_id, "chatId": chat_id });
    URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload).unwrap_or_default())
}

fn now_iso() -> String {
    // Minimal RFC3339-ish timestamp without pulling in chrono. Uses the system
    // clock; format mirrors the Node `new Date().toISOString()` shape closely
    // enough for the metadata fields (which are informational here).
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", iso8601(now.as_secs(), now.subsec_millis()))
}

/// Tiny epoch-seconds -> ISO 8601 UTC formatter (no external deps).
fn iso8601(secs: u64, millis: u32) -> String {
    // Days since epoch and time of day.
    let days = secs / 86_400;
    let rem = secs % 86_400;
    let (hour, minute, second) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Civil date from days (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

fn web_err(message: impl Into<String>) -> WebError {
    WebError::from(anyhow::anyhow!(message.into()))
}

fn ndjson(value: &Value) -> String {
    let mut line = serde_json::to_string(value).unwrap_or_default();
    line.push('\n');
    line
}

// ===========================================================================
// Admin / "Todd" single-character generation  (POST /generate, /generate/stream)
// ===========================================================================
//
// Faithful port of SillyTavern's standalone headless generation endpoint
// (`src/headless/generation.js`):
//
//   * `generateHeadless`         (generation.js:1101) -> `generate_headless`
//   * `streamGenerateHeadless`   (generation.js:1287) -> `generate_headless_stream`
//
// Both delegate to `prepareGenerationRun` (generation.js:668), the shared prompt
// builder, then call the provider and run `finalizeGenerationResult`
// (generation.js:434). Unlike the live-chat path there is NO speaker
// selection/orchestrator: it is ONE character (`characterId`) with optional
// `sessionId` history. The FNV helper `generateAdminTurn`
// (`tools/fnv/nvbridge-helper.mjs:2864`) builds the request body and consumes the
// response via `getGeneratedLineItems`/`getSelectedSpeakerInfo`, which read
// `turn.turns[]||[turn]` (each `{ speaker, message.content, metadata, structured }`)
// plus `turn.message.content || turn.structured.message` and `turn.sessionId`.
//
// We reuse the live-chat machinery wherever possible: the `chasm-prompt`
// assembler (`assemble_prompt_with_retrieval`), `build_chat_messages`,
// `build_response_instructions`, `llm::chat_completion[_stream]` +
// `structured_response_format`, and the structured parse/cleanup helpers
// (`strip_reasoning_channel`, `parse_structured`, `strip_speaker_prefix`).

/// Default context window for admin session history (mirrors the live path's
/// `CONTEXT_MESSAGE_LIMIT` and `prepareGenerationRun`'s `.slice(-40)`).
const ADMIN_HISTORY_LIMIT: usize = CONTEXT_MESSAGE_LIMIT;

/// Speaker-agnostic resolution of an admin generate request: the character card
/// (by `characterId`), the visible history (from `sessionId`, when present), and
/// the request fields that shape the prompt. Mirrors the non-live resolution
/// `prepareGenerationRun` does from the request body.
struct AdminRun {
    /// `characterId` from the body or the decoded session — `None` is allowed
    /// (a card-less prompt), matching `prepareGenerationRun` where `character`
    /// may be null.
    character_id: Option<String>,
    /// Resolved card name (for speaker label + `stripSpeakerLabel`).
    character_name: String,
    /// `sessionId` echoed back in the response (empty when none supplied).
    session_id: String,
    structured: bool,
    gamestate: Value,
    /// Prior turns for this session, already mapped to `MessageView` (empty when
    /// no readable `sessionId`).
    history: Vec<MessageView>,
    /// The OpenAI chat-completion messages to send to the model.
    chat_messages: Vec<Value>,
    /// Per-request temperature / max_tokens (honors `generationOptions`, with the
    /// structured-output minimum-token budget applied for structured runs).
    options: crate::llm::GenerationOptions,
    /// `assistantName` override for the speaker label, else the card name.
    assistant_name: String,
    /// `stripSpeakerLabel` flag (the FNV admin helper sets this true).
    strip_speaker_label: bool,
    /// `metadata` echoed into the structured/non-structured turn metadata.
    request_metadata: Value,
    /// Activated actions' trusted execution/binding + scoped-catalog candidates,
    /// relayed via the turn's `metadata.activatedActions` so the helper can build
    /// native commands for Todd's ACTION_BOOK actions (gestures, spawn).
    activated_actions: Vec<chasm_core::ActivatedActionView>,
    /// (action_id, alias) pairs used to normalize the model's emitted action ids
    /// (which are aliases like `spawn_entity`) back to canonical ids
    /// (`world.spawn_entity`) so the helper can match them to `activatedActions`.
    aliases: Vec<(String, String)>,
}

/// Structured-output minimum token budget, matching
/// `STRUCTURED_OUTPUT_MIN_TOKENS` in generation.js.
const STRUCTURED_OUTPUT_MIN_TOKENS: i64 = 768;

/// Resolves an admin generate request body into an [`AdminRun`]. Card + session
/// reads are best-effort: a missing card yields `character_found = false` (not an
/// error), and an unreadable/absent `sessionId` falls back to empty history with
/// a logged note, matching the "best effort" spirit of `prepareGenerationRun`.
fn resolve_admin_run(state: &Arc<AppState>, body: &Value) -> WebResult<AdminRun> {
    let message = body
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if message.trim().is_empty() {
        // `getString(body, 'message', { required: true })` in prepareGenerationRun.
        return Err(web_err("message is required."));
    }

    let session_id = body
        .get("sessionId")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or("")
        .to_string();

    // characterId from the body, else from the decoded session id (single mode).
    let session_character_id = if session_id.is_empty() {
        None
    } else {
        chasm_st_compat::decode_session_id(&session_id)
            .ok()
            .and_then(|payload| {
                payload
                    .get("characterId")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
    };
    let character_id = body
        .get("characterId")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(|value| value.trim_end_matches(".png").to_string())
        .or(session_character_id);

    let structured = body
        .get("responseFormat")
        .and_then(Value::as_str)
        .map(|value| value == "structured")
        .unwrap_or(false);
    let extra_context = body
        .get("extraContext")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let gamestate = body.get("gamestate").cloned().unwrap_or(Value::Null);

    // Resolve the character card (best-effort: None => card-less prompt).
    let card = character_id
        .as_deref()
        .and_then(|id| state.repository.read_character_card(id).ok().flatten());
    let character_name = card
        .as_ref()
        .map(|card| card.name.clone())
        .unwrap_or_default();

    // Session history: read the single-session JSONL when a sessionId is present.
    // A `single`-mode session resolves to `chats/<characterId>/<chatId>.jsonl`
    // (the same reader the ST `/generate` path loads via `getChatData`). When no
    // sessionId is supplied — or it cannot be read — we fall back to empty
    // history. TODO(admin-session-history): the FNV admin helper does not persist
    // a sessionId across turns (it sends a fresh body each turn and only adopts a
    // returned sessionId), so admin history is typically empty here; ST's
    // `appendMessage`/session-create write path is not replicated for the admin
    // route, so multi-turn admin memory relies on the caller supplying a sessionId
    // that already exists on disk.
    let history = if session_id.is_empty() {
        Vec::new()
    } else {
        read_session_history(state, &session_id)
    };

    // Generation options (temperature / max_tokens), honoring `generationOptions`
    // and bumping max_tokens to the structured minimum for structured runs.
    let mut options = parse_generation_options(body.get("generationOptions"));
    if structured {
        options.max_tokens = Some(
            options
                .max_tokens
                .map(|tokens| tokens.max(STRUCTURED_OUTPUT_MIN_TOKENS))
                .unwrap_or(STRUCTURED_OUTPUT_MIN_TOKENS),
        );
    }

    let assistant_name = body
        .get("assistantName")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| character_name.clone());
    let strip_speaker_label = body
        .get("stripSpeakerLabel")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let request_metadata = body.get("metadata").cloned().unwrap_or(Value::Null);

    // Build a synthetic participant view so the prompt assembler reads the card.
    let participant = admin_participant_view(character_id.as_deref(), &character_name);

    // Lore / quest activation scans the player MESSAGE ONLY (the gamestate is not
    // in the scan — see the live path above; it otherwise fired location/NPC/faction
    // lore every turn). Constant entries still always inject.
    let retrieval_settings = AppSettings::load(&state.config.settings_path).retrieval;
    let retriever = state.retriever();
    let cache = state.embed_cache();
    let retrieval_ctx = match (retriever, cache) {
        (Some(retriever), Some(cache)) if retrieval_settings.enabled => Some(RetrievalCtx {
            retriever,
            cache,
            chat_memory_enabled: retrieval_settings.chat_memory_enabled,
            lore_semantic_enabled: retrieval_settings.lore_semantic_enabled,
            action_semantic_enabled: retrieval_settings.action_semantic_enabled,
            quest_semantic_enabled: retrieval_settings.quest_semantic_enabled,
            candidates: retrieval_settings.candidates as usize,
            top_k: retrieval_settings.top_k as usize,
            min_score: retrieval_settings.min_score,
            action_min_score: retrieval_settings.action_min_score,
            chat_memory_limit: retrieval_settings.chat_memory_limit as usize,
            lore_limit: retrieval_settings.lore_limit as usize,
            quest_limit: retrieval_settings.quest_limit as usize,
        }),
        _ => None,
    };
    // GLOBAL scenario for the admin turn: same template + macro-table rules as
    // the live path (fresh request `metadata.macros`, else the latest recorded
    // table of the active live chat). Admin sessions are 1-on-1, so
    // {{participants}} names the player alone.
    let global_scenario = {
        let mut macros = chasm_prompt::macros_from_metadata(&request_metadata);
        if macros.is_empty() {
            if let Ok(Some(live_chat)) = active_live_chat(state) {
                macros =
                    chasm_prompt::macros_from_value(&latest_chat_macros(state, &live_chat).1);
            }
        }
        resolve_global_scenario(&global_scenario_template(state), &macros, &[])
    };
    // Admin (Todd) path uses the _collect variant so it ALSO gets the activated
    // actions' trusted execution/scoped-catalogs — relayed below via
    // metadata.activatedActions so Todd can fire ACTION_BOOK actions (gestures,
    // spawn), not just the 3 hardcoded natives. Scopes (incl. `admin`) gate which
    // actions Todd is offered (e.g. admin-only spawn).
    let (assembled, injected) = chasm_prompt::assemble_prompt_with_retrieval_collect(
        &state.repository,
        &participant,
        &history,
        &message,
        &message,
        &parse_action_book_scopes(body),
        retrieval_ctx,
        Some(""),
    );

    // Admin = Todd, who must stay terse. Append a hard one-sentence rule to the
    // response instructions (the LAST thing before generation, so it overrides the
    // verbose persona/examples that otherwise win).
    let response_instructions = format!(
        "{} Reply with EXACTLY ONE short sentence — a single sentence only, never two, no \
         second clause after the first period.",
        build_response_instructions(&assistant_name, structured)
    );
    // Admin sessions are single-character — no group attribution or scene roster
    // needed (empty current-speaker id; the in-fn group guard keeps it unchanged).
    let chat_messages = build_chat_messages(
        &assembled,
        &history,
        &message,
        structured,
        &response_instructions,
        &extra_context,
        &gamestate,
        "",
        "",
        // Admin sessions don't persist the player message to history, so it must be
        // appended as the final user turn here.
        true,
        &global_scenario,
    );

    Ok(AdminRun {
        character_id,
        character_name,
        session_id,
        structured,
        gamestate,
        history,
        chat_messages,
        options,
        assistant_name,
        strip_speaker_label,
        request_metadata,
        activated_actions: injected.activated_actions,
        aliases: structured_action_aliases(state),
    })
}

/// Reads the prior turns of a `single`-mode session and maps them to
/// `MessageView`s for prompt assembly. Best-effort: any read/parse failure (or a
/// non-single session) yields an empty history. Drops the leading "first message"
/// (greeting) row to mirror ST's `chatData.slice(1)`, keeps the last
/// [`ADMIN_HISTORY_LIMIT`] turns.
fn read_session_history(state: &Arc<AppState>, session_id: &str) -> Vec<MessageView> {
    let Ok(payload) = chasm_st_compat::decode_session_id(session_id) else {
        return Vec::new();
    };
    // Only single-character sessions live under `chats/<characterId>/...`; the
    // admin route is single-character by construction.
    if payload.get("mode").and_then(Value::as_str) != Some("single") {
        return Vec::new();
    }
    // A minimal single-session reader: build a one-segment LiveChat whose segment
    // session_id is this id, then reuse the repository's JSONL reader + view
    // mapping. We do NOT apply live-visibility filtering (single sessions have no
    // `headless.metadata.live` block); the fallback mapping keeps every row.
    let segment = LiveChatSegment {
        id: String::from("admin"),
        title: String::new(),
        location: String::new(),
        chat_id: String::new(),
        session_id: session_id.to_string(),
        created_at: None,
        metadata: Value::Null,
    };
    let raw = match state.repository.read_segment_messages(&segment) {
        Ok(messages) => messages,
        Err(_) => return Vec::new(),
    };
    let mut views: Vec<MessageView> = raw
        .into_iter()
        .enumerate()
        .skip(1) // drop the greeting/first message (ST `chatData.slice(1)`)
        .map(|(index, message)| admin_message_view(index, &message))
        .filter(|view| !view.content.is_empty())
        .collect();
    let start = views.len().saturating_sub(ADMIN_HISTORY_LIMIT);
    views.split_off(start)
}

/// Maps a raw single-session JSONL message to a `MessageView`. Role mapping
/// mirrors the live path's `role_for_message`: user messages -> "player",
/// system -> "system", else "assistant" (and `build_chat_messages` maps "player"
/// back to the "user" chat role).
fn admin_message_view(index: usize, message: &STJsonlChatMessage) -> MessageView {
    let role = if message.is_user {
        "player"
    } else if message.is_system {
        "system"
    } else {
        "assistant"
    };
    let name = if message.name.is_empty() {
        "Unknown".to_string()
    } else {
        message.name.clone()
    };
    MessageView {
        id: format!("m_{index}"),
        role: role.to_string(),
        speaker_participant_id: None,
        speaker_name: name.clone(),
        speaker_initial: name
            .chars()
            .next()
            .map(|ch| ch.to_uppercase().to_string())
            .unwrap_or_else(|| "?".to_string()),
        content: message.mes.clone(),
        created_at: message.send_date.clone(),
        created_at_label: message
            .send_date
            .as_deref()
            .map(format_message_timestamp)
            .unwrap_or_default(),
        segment_id: None,
        location: None,
        audible_to: Vec::new(),
        visible_reason: "admin".to_string(),
        // Admin history is only used to assemble the next prompt, never rendered.
        injected: None,
        turn_actions: Vec::new(),
    }
}

/// Builds the synthetic `ParticipantView` the assembler uses to resolve the
/// admin character's card (only `character_id` + `name` drive card lookup +
/// the character-name lore filter).
fn admin_participant_view(character_id: Option<&str>, character_name: &str) -> ParticipantView {
    let name = if character_name.is_empty() {
        "Assistant".to_string()
    } else {
        character_name.to_string()
    };
    ParticipantView {
        id: "admin".to_string(),
        initial: name
            .chars()
            .next()
            .map(|ch| ch.to_uppercase().to_string())
            .unwrap_or_else(|| "?".to_string()),
        name,
        kind: "npc".to_string(),
        character_id: character_id.map(str::to_string),
        present: true,
        audible: true,
        distance: None,
        distance_label: String::new(),
        message_count: 0,
        selected: true,
    }
}

/// Parses `generationOptions.{temperature,max_tokens|maxTokens}` into the LLM
/// `GenerationOptions`, mirroring `getGenerationOptions` (clamps max_tokens to
/// (0, 32000]). `None`/absent fields leave the LLM client defaults in place.
fn parse_generation_options(options: Option<&Value>) -> crate::llm::GenerationOptions {
    let temperature = options
        .and_then(|value| value.get("temperature"))
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite());
    let max_tokens = options
        .and_then(|value| value.get("max_tokens").or_else(|| value.get("maxTokens")))
        .and_then(Value::as_i64)
        .filter(|value| *value > 0)
        .map(|value| value.min(32_000));
    crate::llm::GenerationOptions {
        temperature,
        max_tokens,
    }
}

/// Builds the response metadata block, mirroring the shape
/// `finalizeGenerationResult` returns (the FNV admin helper only reads
/// `metadata.structured`-adjacent fields opportunistically, but we keep the
/// informative fields so the contract matches ST).
fn admin_metadata(run: &AdminRun, streamed: bool, structured_ok: bool) -> Value {
    json!({
        "gamestateInjected": !run.gamestate.is_null(),
        "structuredOutputEnforced": run.structured,
        "structuredOk": structured_ok,
        "streamed": streamed,
        "historyMessages": run.history.len(),
        "responseFormat": if run.structured { "structured" } else { "text" },
        "limitations": [
            "Admin generation reuses the live-chat prompt assembler; ST's browser prompt builder is not reusable here.",
            "Lorebook / Action Book / Quest Book activation is best-effort server-side matching plus optional vector retrieval.",
            "Session history is read from an existing single-character session file when a sessionId is supplied; the admin route does not itself persist new session messages."
        ],
    })
}

/// Finalizes an admin run into the response JSON the FNV helper consumes. Mirrors
/// `finalizeGenerationResult`: structured parse (when requested), speaker-label
/// stripping, then the `{ sessionId, characterId, message, structured?, raw?,
/// metadata }` shape — plus a `speaker` object and a single-element `turns[]`
/// array so the helper's `turn.turns[]||[turn]` consumers work unchanged.
fn finalize_admin_result(run: &AdminRun, raw: &str, streamed: bool) -> WebResult<Value> {
    let raw_trimmed = strip_reasoning_channel(raw);

    // Structured output: parse a JSON object with `speech`/`message`.
    let (mut content, structured) = if run.structured {
        match parse_structured(&raw_trimmed) {
            Some((speech, mut value)) => {
                // Normalize emitted action aliases (e.g. `spawn_entity`) to canonical
                // ids (`world.spawn_entity`) so the helper can match them to the
                // relayed activatedActions — the live path does this too. Without it
                // Todd's ACTION_BOOK actions (gestures, spawn) never resolve.
                normalize_structured_action_aliases(&mut value, &run.aliases);
                (speech, Some(value))
            }
            None => {
                // Graceful salvage instead of a hard 500 (the live path is
                // graceful too): on a big admin prompt the model sometimes emits
                // structured output that doesn't parse. Pull the speech out of the
                // partial/loose text so Todd still talks; drop actions this turn.
                // Log the raw so the malformation can be diagnosed for actions.
                tracing::warn!(
                    "admin structured parse failed; salvaging speech. raw (first 500): {}",
                    raw_trimmed.chars().take(500).collect::<String>()
                );
                let salvaged = extract_structured_speech_prefix(&raw_trimmed);
                let content = if salvaged.trim().is_empty() {
                    raw_trimmed.clone()
                } else {
                    salvaged
                };
                (content, None)
            }
        }
    } else {
        (raw_trimmed.clone(), None)
    };

    // stripSpeakerLabel: drop a leading "Name:" the model echoed.
    if run.strip_speaker_label {
        let label = if run.assistant_name.is_empty() {
            run.character_name.as_str()
        } else {
            run.assistant_name.as_str()
        };
        if !label.is_empty() {
            content = strip_speaker_prefix(&content, label);
        }
    }

    let structured_ok = structured.is_some();
    let mut metadata = admin_metadata(run, streamed, structured_ok);
    // Relay the activated actions' trusted execution/binding + scoped-catalog
    // candidates onto the turn metadata (mirrors the live `finalize_turn`), so the
    // helper can build native commands for Todd's ACTION_BOOK actions.
    if !run.activated_actions.is_empty() {
        if let Value::Object(map) = &mut metadata {
            map.insert(
                "activatedActions".to_string(),
                serde_json::to_value(&run.activated_actions).unwrap_or_else(|_| json!([])),
            );
        }
    }

    let character_id = run
        .character_id
        .clone()
        .map(Value::String)
        .unwrap_or(Value::Null);
    let speaker_name = if run.assistant_name.is_empty() {
        run.character_name.clone()
    } else {
        run.assistant_name.clone()
    };
    let speaker = json!({
        "participantId": "system:admin",
        "characterId": character_id,
        "name": speaker_name,
    });
    let message_obj = json!({
        "role": "assistant",
        "content": content,
        "name": speaker_name,
    });

    // One-element turn matching the per-turn fields the helper reads via
    // `turn.turns[]` (`speaker`, `message.content`, optional `structured`).
    let mut turn = serde_json::Map::new();
    turn.insert("speaker".to_string(), speaker.clone());
    turn.insert("message".to_string(), message_obj.clone());
    turn.insert("metadata".to_string(), metadata.clone());
    if let Some(structured) = &structured {
        turn.insert("structured".to_string(), structured.clone());
    }

    let mut response = serde_json::Map::new();
    response.insert("sessionId".to_string(), json!(run.session_id));
    response.insert("characterId".to_string(), character_id);
    response.insert("speaker".to_string(), speaker);
    response.insert("message".to_string(), message_obj);
    response.insert("turns".to_string(), json!([Value::Object(turn)]));
    response.insert("metadata".to_string(), metadata);
    if let Some(structured) = structured {
        response.insert("structured".to_string(), structured.clone());
        response.insert("raw".to_string(), json!(raw));
    }
    // Echo the request metadata under a stable key for debugging parity (ST keeps
    // request metadata on the persisted message, which we don't write here).
    if !run.request_metadata.is_null() {
        response.insert("requestMetadata".to_string(), run.request_metadata.clone());
    }
    Ok(Value::Object(response))
}

/// POST /generate — buffered admin / "Todd" single-character generation.
/// Mirrors `generateHeadless` (generation.js:1101).
pub async fn generate_headless(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> WebResult<Json<Value>> {
    let trace_id = trace_id_from_headers(&headers);
    let run = resolve_admin_run(&state, &body)?;
    let response_format = run.structured.then(crate::llm::structured_response_format);
    // Saved sampling overlaid with the request's explicit generationOptions, plus
    // the active provider target.
    let admin_settings = AppSettings::load(&state.config.settings_path);
    let sampling =
        crate::llm::Sampling::from_settings(&admin_settings.llm.sampling).with_overrides(run.options);
    let target = crate::llm::LlmTarget::resolve(&admin_settings, &state.config);
    let (raw, metrics) = crate::llm::chat_completion_capturing_sampled(
        &target,
        &run.chat_messages,
        response_format.as_ref(),
        sampling,
    )
    .await
    .map_err(web_err)?;
    if let (Some(id), Some(metrics)) = (trace_id.as_deref(), metrics) {
        crate::trace_routes::record_llm_metrics(id, metrics);
    }
    Ok(Json(finalize_admin_result(&run, &raw, false)?))
}

/// POST /generate/stream — admin generation streamed over SSE. Mirrors
/// `streamGenerateHeadless` (generation.js:1287): emits `run.started`, a `token`
/// event per content delta, then a final `run.completed` carrying the finalized
/// turn (so a streaming admin caller gets the same payload as the buffered path),
/// or an `error` event.
pub async fn generate_headless_stream(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> WebResult<Response> {
    let trace_id = trace_id_from_headers(&headers);
    // Resolve synchronously so hard errors surface as a non-200 (matching the
    // helper's `streamApi`, which checks the response status before reading).
    let run = resolve_admin_run(&state, &body)?;
    // Saved sampling overlaid with the request's explicit generationOptions, plus
    // the active provider target, computed before the stream takes ownership of `run`.
    let admin_settings = AppSettings::load(&state.config.settings_path);
    let sampling =
        crate::llm::Sampling::from_settings(&admin_settings.llm.sampling).with_overrides(run.options);
    let target = crate::llm::LlmTarget::resolve(&admin_settings, &state.config);

    let session_id = run.session_id.clone();
    let character_id = run
        .character_id
        .clone()
        .map(Value::String)
        .unwrap_or(Value::Null);

    let stream = async_stream::stream! {
        yield sse_event("run.started", &json!({
            "sessionId": session_id,
            "characterId": character_id,
        }));
        let trace_id = trace_id;

        let response_format = run.structured.then(crate::llm::structured_response_format);
        let mut raw = String::new();
        match crate::llm::chat_completion_stream(&target, &run.chat_messages, response_format.as_ref(), trace_id.as_deref(), sampling)
            .await
        {
            Ok(mut rx) => {
                while let Some(item) = rx.recv().await {
                    match item {
                        Ok(delta) => {
                            if delta.is_empty() {
                                continue;
                            }
                            raw.push_str(&delta);
                            yield sse_event("token", &json!({ "text": delta }));
                        }
                        Err(error) => {
                            yield sse_event("error", &json!({ "error": { "message": error } }));
                            return;
                        }
                    }
                }
            }
            Err(error) => {
                yield sse_event("error", &json!({ "error": { "message": error } }));
                return;
            }
        }

        match finalize_admin_result(&run, &raw, true) {
            Ok(turn) => yield sse_event("run.completed", &turn),
            Err(error) => {
                yield sse_event("error", &json!({ "error": { "message": error.0.to_string() } }));
            }
        }
    };

    Ok((
        [(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")],
        Body::from_stream(stream.map(Ok::<String, std::convert::Infallible>)),
    )
        .into_response())
}

/// Formats one server-sent event (`event:`/`data:` lines + blank separator),
/// mirroring the Node `writeSse` helper.
fn sse_event(event: &str, data: &Value) -> String {
    format!(
        "event: {event}\ndata: {}\n\n",
        serde_json::to_string(data).unwrap_or_default()
    )
}

use futures_util::StreamExt as _;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_first_balanced_json_object() {
        let raw =
            "<|channel>thought\n<channel|>{\n  \"speech\": \"Hi {there}\",\n  \"actions\": []\n}";
        let cleaned = strip_reasoning_channel(raw);
        let (speech, value) = parse_structured(&cleaned).expect("structured");
        assert_eq!(speech, "Hi {there}");
        assert!(value.get("actions").is_some());
    }

    #[test]
    fn strips_harmony_channel_preamble() {
        let raw = "<|channel>thought\n<channel|>Hello there.";
        assert_eq!(strip_reasoning_channel(raw), "Hello there.");
    }

    #[test]
    fn streaming_speech_prefix_skips_reasoning_and_json() {
        // Full structured output with a reasoning preamble: only the speech text.
        let raw = "<|channel|>analysis<|message|>thinking...<|channel|>final<|message|>\
                   {\"speech\":\"Same as it ever was.\",\"stateUpdates\":[],\"actions\":[]}";
        assert_eq!(
            extract_structured_speech_prefix(raw),
            "Same as it ever was."
        );
    }

    #[test]
    fn streaming_speech_prefix_grows_monotonically_on_partial_json() {
        // Before "speech" appears, nothing is spoken.
        assert_eq!(
            extract_structured_speech_prefix("<|channel|>final<|message|>{\"stateUpdates\":[]"),
            ""
        );
        // Partial speech string (still streaming, no closing quote) extracts the
        // text so far, and each later prefix extends the earlier one.
        let p1 = extract_structured_speech_prefix("{\"speech\":\"Howdy");
        let p2 = extract_structured_speech_prefix("{\"speech\":\"Howdy, stranger");
        assert_eq!(p1, "Howdy");
        assert!(p2.starts_with(&p1));
        assert_eq!(p2, "Howdy, stranger");
    }

    #[test]
    fn streaming_speech_prefix_decodes_escapes_and_drops_partial() {
        assert_eq!(
            extract_structured_speech_prefix("{\"speech\":\"line1\\nline2\""),
            "line1\nline2"
        );
        // A trailing incomplete escape is dropped until it completes.
        assert_eq!(
            extract_structured_speech_prefix("{\"speech\":\"done\\"),
            "done"
        );
    }


    #[test]
    fn extracted_speech_pulls_structured_field_and_plain_text() {
        assert_eq!(
            extracted_speech(true, "{\"speech\":\"Howdy, stranger.\",\"actions\":[]}"),
            "Howdy, stranger."
        );
        assert_eq!(extracted_speech(false, "Just talking."), "Just talking.");
        // Partial structured output: extracts the prefix that's arrived so far.
        assert_eq!(
            extracted_speech(true, "{\"speech\":\"Howdy, str"),
            "Howdy, str"
        );
    }

    #[test]
    fn normalizes_emitted_alias_object_to_canonical_id() {
        let aliases = vec![("movement.follow_target".to_string(), "follow".to_string())];
        // The exact shape the small model emitted in-game.
        let mut s = json!({
            "speech": "ok",
            "actions": [{ "alias": "follow", "parameters": { "confidence": 0.8, "target": "player" } }],
        });
        normalize_structured_action_aliases(&mut s, &aliases);
        let actions = s["actions"].as_array().unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0]["id"], "movement.follow_target");
        assert_eq!(actions[0]["parameters"]["target"], "player");
    }

    #[test]
    fn normalizes_string_and_id_actions_and_drops_unknown() {
        let aliases = vec![
            ("movement.follow_target".to_string(), "follow".to_string()),
            ("combat.start".to_string(), "attack".to_string()),
        ];
        let mut s = json!({
            "actions": ["follow", { "id": "combat.start" }, "teleport", { "alias": "nonsense" }],
        });
        normalize_structured_action_aliases(&mut s, &aliases);
        let ids: Vec<&str> = s["actions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["movement.follow_target", "combat.start"]);
    }

    #[test]
    fn strips_think_block() {
        let raw = "<think>reasoning here</think>\nActual answer.";
        assert_eq!(strip_reasoning_channel(raw), "Actual answer.");
    }

    #[test]
    fn strips_speaker_prefix() {
        assert_eq!(
            strip_speaker_prefix("Easy Pete: Howdy.", "Easy Pete"),
            "Howdy."
        );
        assert_eq!(strip_speaker_prefix("Howdy.", "Easy Pete"), "Howdy.");
    }

    #[test]
    fn keeps_plain_prose_unchanged() {
        let raw = "Just a normal line with no markers.";
        assert_eq!(strip_reasoning_channel(raw), raw);
    }

    #[test]
    fn encodes_group_session_id_roundtrip() {
        let id = encode_group_session_id("fnv-goodsprings", "Goodsprings");
        let decoded = chasm_st_compat::decode_session_id(&id).expect("decode");
        assert_eq!(decoded["mode"], "group");
        assert_eq!(decoded["chatId"], "Goodsprings");
        assert_eq!(decoded["groupId"], "fnv-goodsprings");
    }

    // --- Global scenario resolution -----------------------------------------

    /// A LiveChat with the given present+audible NPCs (id, name).
    fn group_chat(npcs: &[(&str, &str)]) -> LiveChat {
        let mut chat = LiveChat::default();
        for (id, name) in npcs {
            chat.presence.insert(
                id.to_string(),
                LiveChatParticipant {
                    participant_id: id.to_string(),
                    kind: "npc".to_string(),
                    character_id: Some(format!("char-{id}")),
                    name: name.to_string(),
                    present: Some(true),
                    audible: Some(true),
                    ..Default::default()
                },
            );
        }
        chat
    }

    #[test]
    fn other_npc_names_excludes_the_prompted_speaker() {
        let chat = group_chat(&[
            ("npc-pete", "Easy Pete"),
            ("npc-sunny", "Sunny Smiles"),
            ("npc-trudy", "Trudy"),
        ]);
        assert_eq!(
            other_npc_names(&chat, "npc-pete"),
            vec!["Sunny Smiles".to_string(), "Trudy".to_string()]
        );
        // 1-on-1: no other NPCs.
        let solo = group_chat(&[("npc-pete", "Easy Pete")]);
        assert!(other_npc_names(&solo, "npc-pete").is_empty());
    }

    #[test]
    fn sentence_boundaries_stream_correctly() {
        // Basic: two complete sentences, third unfinished.
        let s = "Howdy there. Watch the geckos! And also";
        let first = next_sentence_end(s, 0).unwrap();
        assert_eq!(s[..first].trim(), "Howdy there.");
        let second = next_sentence_end(s, first).unwrap();
        assert_eq!(s[first..second].trim(), "Watch the geckos!");
        assert_eq!(next_sentence_end(s, second), None, "unfinished tail waits");

        // The final sentence never fires mid-stream (remainder path owns it).
        assert_eq!(next_sentence_end("One sentence only.", 0), None);

        // Abbreviations and initials do not split.
        let s = "Mr. House runs the Strip. Ask J. Smith later.";
        let first = next_sentence_end(s, 0).unwrap();
        assert_eq!(s[..first].trim(), "Mr. House runs the Strip.");
        assert_eq!(next_sentence_end(s, first), None);

        // Decimals do not split; ellipses + quotes absorb into the boundary.
        assert_eq!(next_sentence_end("It costs 2.5 caps total", 0), None);
        let s = "\"Well...\" she said. More text";
        let first = next_sentence_end(s, 0).unwrap();
        assert!(s[..first].ends_with("said."));
    }

    #[test]
    fn late_scenario_injects_as_system_message_at_depth_one() {
        let assembled = chasm_core::PromptAssemblyView::default();
        let history = vec![
            MessageView {
                role: "player".to_string(),
                content: "hello".to_string(),
                ..Default::default()
            },
            MessageView {
                role: "assistant".to_string(),
                content: "hi there".to_string(),
                ..Default::default()
            },
        ];
        let messages = build_chat_messages(
            &assembled, &history, "", false, "", "", &Value::Null, "", "", false,
            "It is 1:22PM. You are in the saloon.",
        );
        // Scenario rides as a SYSTEM message at depth 1: after the history bulk,
        // directly before the newest line - so the cached prompt prefix survives
        // the per-turn timestamp. Never inside the leading system prompt.
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], "system");
        assert_eq!(messages[1]["content"], "It is 1:22PM. You are in the saloon.");
        assert_eq!(messages[2]["content"], "hi there");

        // Empty scenario -> no injection at all.
        let plain = build_chat_messages(
            &assembled, &history, "", false, "", "", &Value::Null, "", "", false, "",
        );
        assert_eq!(plain.len(), 2);
    }

    #[test]
    fn volatile_retrieval_components_inject_at_depth_one_not_in_head() {
        let mut assembled = chasm_core::PromptAssemblyView::default();
        let component = |key: &str, content: &str| chasm_core::PromptComponentView {
            order: 0,
            group: "system".to_string(),
            key: key.to_string(),
            label: String::new(),
            role: "system".to_string(),
            status: "included".to_string(),
            note: String::new(),
            content: content.to_string(),
            char_count: content.chars().count(),
        };
        assembled.components = vec![
            component("character", "You are Sunny Smiles."),
            component("lore", "Activated lore:\nGeckos roam the hills."),
            component("chat_vectors", "Relevant past chat context:\nDoc patched you up."),
            // Relationships change only on game save, never per turn — they
            // must ride in the STABLE HEAD, not the depth-1 volatile block.
            component(
                "relationships",
                "Sunny Smiles's relationships:\n- Toward Courier: Sunny trusts her after the gecko hunt.",
            ),
        ];
        let history = vec![
            MessageView {
                role: "player".to_string(),
                content: "hello".to_string(),
                ..Default::default()
            },
            MessageView {
                role: "assistant".to_string(),
                content: "hi there".to_string(),
                ..Default::default()
            },
        ];
        let messages = build_chat_messages(
            &assembled, &history, "", false, "", "", &Value::Null, "", "", false, "",
        );
        // Head system message: stable card only — retrieval picks must NOT be
        // there or every turn's differing picks would invalidate the LLM's
        // cached prefix and force full-history reprocessing.
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0]["role"], "system");
        let head = messages[0]["content"].as_str().unwrap();
        assert!(head.contains("Sunny Smiles"));
        assert!(!head.contains("Geckos"));
        assert!(!head.contains("Doc patched"));
        // Relationships are HEAD content (save-cadence, not per-turn).
        assert!(head.contains("Sunny trusts her after the gecko hunt."));
        // Volatile block rides at depth 1: after history, before the newest line.
        assert_eq!(messages[2]["role"], "system");
        let volatile = messages[2]["content"].as_str().unwrap();
        assert!(volatile.contains("Geckos roam the hills."));
        assert!(volatile.contains("Doc patched you up."));
        assert!(!volatile.contains("gecko hunt"));
        assert_eq!(messages[3]["content"], "hi there");
    }

    #[test]
    fn resolves_global_scenario_with_computed_participants() {
        let macros: BTreeMap<String, String> = [
            ("player_name".to_string(), "Courier".to_string()),
            ("time_of_day".to_string(), "11:10PM".to_string()),
        ]
        .into_iter()
        .collect();
        let resolved = resolve_global_scenario(
            "It is {{time_of_day}}. You speak with {{participants}}.",
            &macros,
            &["Sunny Smiles".to_string(), "Trudy".to_string()],
        );
        assert_eq!(
            resolved,
            "It is 11:10PM. You speak with Courier, Sunny Smiles, and Trudy."
        );

        // 1-on-1 turn: participants is the player alone.
        let solo = resolve_global_scenario("With {{participants}}.", &macros, &[]);
        assert_eq!(solo, "With Courier.");

        // Blank template = user disabled the component: resolves to "" (the
        // assembler then omits the scenario entirely).
        assert_eq!(resolve_global_scenario("   ", &macros, &[]), "");

        // Empty macro table degrades but never leaks placeholders; the
        // computed participants still names the player.
        let empty = resolve_global_scenario(
            chasm_prompt::DEFAULT_SCENARIO_TEMPLATE,
            &BTreeMap::new(),
            &[],
        );
        assert!(!empty.contains("{{"), "no unresolved macros: {empty}");
        assert!(empty.contains("You are in a conversation with the player."));
    }

    // --- Admin / "Todd" single-character generation ------------------------

    /// Builds an `AdminRun` with empty `chat_messages`/history for finalize-shape
    /// tests (the finalizer never inspects the prompt — only the request flags).
    fn admin_run_fixture(structured: bool) -> AdminRun {
        AdminRun {
            character_id: Some("Todd".to_string()),
            character_name: "Todd".to_string(),
            session_id: "sess-1".to_string(),
            structured,
            gamestate: json!({ "location": "Goodsprings" }),
            history: Vec::new(),
            chat_messages: Vec::new(),
            options: crate::llm::GenerationOptions::default(),
            assistant_name: "Todd".to_string(),
            strip_speaker_label: true,
            request_metadata: json!({ "adminMode": true }),
            activated_actions: Vec::new(),
            aliases: Vec::new(),
        }
    }

    #[test]
    fn finalizes_text_admin_turn_with_helper_fields() {
        let run = admin_run_fixture(false);
        // stripSpeakerLabel must drop the echoed "Todd:" prefix.
        let result =
            finalize_admin_result(&run, "Todd: My child, listen.", false).expect("text finalize");

        // Top-level helper contract: sessionId, characterId, message.content.
        assert_eq!(result["sessionId"], "sess-1");
        assert_eq!(result["characterId"], "Todd");
        assert_eq!(result["message"]["content"], "My child, listen.");
        assert_eq!(result["message"]["role"], "assistant");
        // No structured payload on the text path.
        assert!(result.get("structured").is_none());
        assert!(result.get("raw").is_none());

        // `turn.turns[]||[turn]` consumers: a single-element turns array, each
        // with `speaker` + `message.content`.
        let turns = result["turns"].as_array().expect("turns array");
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0]["message"]["content"], "My child, listen.");
        assert_eq!(turns[0]["speaker"]["name"], "Todd");
        assert_eq!(turns[0]["speaker"]["characterId"], "Todd");
        assert_eq!(result["metadata"]["responseFormat"], "text");
    }

    #[test]
    fn finalizes_structured_admin_turn_with_speech_and_actions() {
        let run = admin_run_fixture(true);
        let raw = "<|channel>thought\n<channel|>{\n  \"speech\": \"Be still.\",\n  \"stateUpdates\": {},\n  \"actions\": []\n}";
        let result = finalize_admin_result(&run, raw, false).expect("structured finalize");

        // content == structured.speech; structured + raw echoed back.
        assert_eq!(result["message"]["content"], "Be still.");
        assert_eq!(result["structured"]["speech"], "Be still.");
        assert!(result["structured"]["actions"].is_array());
        assert!(result.get("raw").is_some());
        assert_eq!(result["metadata"]["responseFormat"], "structured");
        assert_eq!(result["metadata"]["structuredOk"], true);

        // The turn mirrors the structured payload for `turn.turns[]` consumers.
        let turns = result["turns"].as_array().expect("turns array");
        assert_eq!(turns[0]["structured"]["speech"], "Be still.");
        assert_eq!(turns[0]["message"]["content"], "Be still.");
    }

    #[test]
    fn chasm_extra_blob_carries_injected_and_actions() {
        // The injected set the assembler would have produced for this turn.
        let injected = InjectedView {
            lore: vec![chasm_core::InjectedEntryView {
                source: "lore".to_string(),
                id: "Goodsprings".to_string(),
                title: "Goodsprings".to_string(),
                reason: "keyword".to_string(),
            }],
            quests: Vec::new(),
            actions: vec![chasm_core::InjectedEntryView {
                source: "action".to_string(),
                id: "movement.follow_target".to_string(),
                title: "Follow target".to_string(),
                reason: "vector".to_string(),
            }],
            activated_actions: Vec::new(),
        };
        // A normalized structured payload (post alias-resolution): one action obj.
        let structured = json!({
            "speech": "Right behind you.",
            "actions": [
                { "id": "movement.follow_target", "target": "player", "parameters": {}, "reason": "Asked to follow." }
            ],
        });
        let aliases = vec![("movement.follow_target".to_string(), "follow".to_string())];
        let macros: BTreeMap<String, String> = [
            ("player_name".to_string(), "Courier".to_string()),
            ("major_location".to_string(), "Goodsprings".to_string()),
        ]
        .into_iter()
        .collect();

        let extra = build_chasm_extra(&injected, Some(&structured), &aliases, &macros);

        // injected groups round-trip under the documented keys.
        assert_eq!(extra["injected"]["lore"][0]["id"], "Goodsprings");
        assert_eq!(extra["injected"]["lore"][0]["reason"], "keyword");
        assert_eq!(
            extra["injected"]["actions"][0]["id"],
            "movement.follow_target"
        );
        assert_eq!(extra["injected"]["actions"][0]["reason"], "vector");
        // turn_actions flattens the chosen actions with the recovered alias.
        let turn_actions = extra["turn_actions"]
            .as_array()
            .expect("turn_actions array");
        assert_eq!(turn_actions.len(), 1);
        assert_eq!(turn_actions[0]["id"], "movement.follow_target");
        assert_eq!(turn_actions[0]["alias"], "follow");
        assert_eq!(turn_actions[0]["target"], "player");
        assert_eq!(turn_actions[0]["reason"], "Asked to follow.");
        // The turn's gamestate macro table is recorded verbatim.
        assert_eq!(extra["macros"]["player_name"], "Courier");
        assert_eq!(extra["macros"]["major_location"], "Goodsprings");

        // A plain-text turn (no structured output) -> empty turn_actions, but the
        // injected set is still recorded; no macros that turn -> empty object.
        let text_extra = build_chasm_extra(&injected, None, &aliases, &BTreeMap::new());
        assert!(text_extra["turn_actions"].as_array().unwrap().is_empty());
        assert_eq!(text_extra["injected"]["lore"][0]["id"], "Goodsprings");
        assert!(text_extra["macros"].as_object().unwrap().is_empty());
    }

    #[test]
    fn structured_malformed_json_salvages_speech_no_error() {
        let run = admin_run_fixture(true);
        // Malformed structured output no longer 500s (Todd would go silent):
        // a partial `"speech":"…"` is salvaged so the character still talks,
        // and the turn is marked structured-not-ok (no actions that turn).
        let result =
            finalize_admin_result(&run, "{\"speech\":\"Be still", false).expect("graceful");
        assert_eq!(result["metadata"]["structuredOk"], false);
        assert_eq!(result["message"]["content"], "Be still");

        // No JSON / no speech field at all -> falls back to the raw text, still Ok.
        let result2 = finalize_admin_result(&run, "not json at all", false).expect("graceful");
        assert_eq!(result2["metadata"]["structuredOk"], false);
        assert_eq!(result2["message"]["content"], "not json at all");
    }

    #[test]
    fn parses_generation_options_temperature_and_max_tokens() {
        let opts = parse_generation_options(Some(&json!({
            "temperature": 0.3,
            "maxTokens": 1024
        })));
        assert_eq!(opts.temperature, Some(0.3));
        assert_eq!(opts.max_tokens, Some(1024));

        // max_tokens clamps to (0, 32000]; non-positive is ignored.
        let clamped = parse_generation_options(Some(&json!({ "max_tokens": 99_999 })));
        assert_eq!(clamped.max_tokens, Some(32_000));
        let ignored = parse_generation_options(Some(&json!({ "max_tokens": 0 })));
        assert_eq!(ignored.max_tokens, None);
        // Absent options leave the client defaults (None) in place.
        let empty = parse_generation_options(None);
        assert_eq!(empty.temperature, None);
        assert_eq!(empty.max_tokens, None);
    }

    #[test]
    fn admin_message_view_maps_roles() {
        let user = STJsonlChatMessage {
            name: "Player".to_string(),
            is_user: true,
            is_system: false,
            send_date: None,
            mes: "Hello".to_string(),
            extra: Value::Null,
            original_avatar: None,
        };
        assert_eq!(admin_message_view(1, &user).role, "player");

        let assistant = STJsonlChatMessage {
            name: "Todd".to_string(),
            is_user: false,
            is_system: false,
            send_date: None,
            mes: "Greetings".to_string(),
            extra: Value::Null,
            original_avatar: None,
        };
        let view = admin_message_view(2, &assistant);
        assert_eq!(view.role, "assistant");
        assert_eq!(view.speaker_name, "Todd");
    }

    /// A real [`AppState`] over throwaway temp dirs (retrieval disabled so the
    /// prompt path never tries to load ONNX models in tests).
    fn fixture_state(tag: &str) -> Arc<AppState> {
        let root = std::env::temp_dir().join(format!(
            "sb-generate-fixture-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let settings_path = root.join("settings.json");
        let mut settings = AppSettings::load(&settings_path); // defaults (file absent)
        settings.retrieval.enabled = false;
        settings.save(&settings_path).unwrap();
        let config = chasm_core::AppConfig {
            bind_addr: "127.0.0.1:0".into(),
            data_root: root.join("data"),
            workspace_root: root.clone(),
            settings_path,
            engines_dir: root.join("engines"),
            profiles_dir: root.join("profiles"),
            voices_dir: root.join("voices"),
            llm_models_dir: root.join("models-llm"),
            stt_endpoint: "http://127.0.0.1:9/v1/audio/transcriptions".into(),
            parakeet_stt_endpoint: "http://127.0.0.1:9/v1/audio/transcriptions".into(),
            llm_endpoint: "http://127.0.0.1:9".into(),
            tts_endpoint: "http://127.0.0.1:9".into(),
        };
        Arc::new(AppState::new(config))
    }

    /// Every file under `root` as sorted `(relative path, bytes)` pairs — a full
    /// content snapshot, so "persists nothing" is byte-for-byte provable.
    fn dir_snapshot(root: &std::path::Path) -> Vec<(String, Vec<u8>)> {
        fn walk(dir: &std::path::Path, base: &std::path::Path, out: &mut Vec<(String, Vec<u8>)>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, base, out);
                } else if let Ok(bytes) = std::fs::read(&path) {
                    let rel = path
                        .strip_prefix(base)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .to_string();
                    out.push((rel, bytes));
                }
            }
        }
        let mut out = Vec::new();
        walk(root, root, &mut out);
        out.sort();
        out
    }

    /// The connect-time warm-up's LLM priming is only worth anything if it
    /// ingests EXACTLY the prefix the real first turn sends. Prove it on a
    /// disk-backed live chat: `warmup_chat_messages` (a) is a pure read — the
    /// data root is byte-for-byte unchanged — and (b) equals the real first
    /// turn's chat-completion array minus the trailing player message, so
    /// the LLM runtime's `cache_prompt` fast-forwards over everything the warm-up
    /// pre-ingested and turn 1 only pays for the player's new line.
    #[tokio::test]
    async fn warmup_prompt_is_the_real_first_turn_prefix_and_persists_nothing() {
        let state = fixture_state("warmup-prefix");
        create_live_chat(
            State(state.clone()),
            Json(json!({
                "id": "fnv-goodsprings",
                "title": "Fallout New Vegas - Goodsprings",
                "location": "Goodsprings",
                "participants": [
                    { "participantId": "player", "type": "player", "name": "Player",
                      "present": true, "audible": true },
                    { "participantId": "npc:easy_pete", "type": "npc", "characterId": "Easy Pete",
                      "name": "Easy Pete", "present": true, "audible": true },
                ],
            })),
        )
        .await
        .expect("create live chat");

        // (a) Pure read: same result twice, and NOTHING under the data root moved.
        let before = dir_snapshot(&state.config.data_root);
        let (warm_messages, speaker) =
            warmup_chat_messages(&state, "fnv-goodsprings", true).expect("warmup plan resolves");
        assert_eq!(speaker, "Easy Pete", "deterministic first-eligible speaker");
        assert_eq!(
            dir_snapshot(&state.config.data_root),
            before,
            "warmup_chat_messages must persist nothing"
        );

        // (b) The real first turn: player line persisted first (as the live path
        // does), then the same deterministic speaker plan.
        let body = json!({ "message": "Hi there, Pete.", "responseFormat": "structured" });
        let ctx = resolve_turn_context(&state, "fnv-goodsprings", &body).expect("turn context");
        persist_player_message_ctx(&state, &ctx).expect("persist player line");
        let selection = orchestrator::select_live_speaker_candidates(
            &ctx.live_chat,
            &orchestrator::SelectionInput {
                force_participant_id: None,
                force_character_id: None,
            },
        )
        .expect("speaker selection");
        let plan = prepare_speaker_turn(&state, &ctx, selection.speakers.first().expect("speaker"))
            .expect("turn plan");

        // KV-priming property: real array == warm-up array + the player's line.
        assert_eq!(plan.chat_messages.len(), warm_messages.len() + 1);
        assert_eq!(
            plan.chat_messages[..warm_messages.len()],
            warm_messages[..],
            "the warm-up prompt must be an exact prefix of the real first turn"
        );
        let last = plan.chat_messages.last().expect("player turn present");
        assert_eq!(last["role"], json!("user"));
        assert_eq!(last["content"], json!("Hi there, Pete."));
    }
}
