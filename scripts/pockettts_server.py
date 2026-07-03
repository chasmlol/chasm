"""Chasm's PocketTTS streaming TTS server (OpenAI-compatible).

The PocketTTS twin of `qwen3_tts_server.py`: llama.cpp handles the LLM and the
Parakeet server handles STT, while this handles TTS. Streams raw int16 PCM (or a
streaming WAV) as PocketTTS generates,
so the Rust backend can slice it into gapless mini-chunks for the game. Both
servers expose the SAME contract on :5002, so the picker can swap between
faster-qwen3-tts and PocketTTS with no change to the Rust routing layer.

  GET  /health             -> {"status":"ok","model_loaded":bool}
  POST /v1/audio/speech     {model,input,voice,response_format}  (pcm|wav)

Run inside the PocketTTS engine venv (engines/pockettts/.venv):
  python pockettts_server.py --voices-dir <profile voices dir> --host 127.0.0.1 --port 5002

Voices are resolved by directory convention (no voices.json): for a requested
voice `<name>` we use the cloned trimmed prompt at
`<voices_dir>/<name>/pockettts/prompt.wav` if present, else the shared reference
clip `<voices_dir>/<name>/reference.wav`. The per-voice model state (the speaker
prompt) is computed once and cached. PocketTTS streams Mimi at 24 kHz, matching
the Rust router's TTS_SAMPLE_RATE, so no resampling is needed.
"""
import argparse
import asyncio
import inspect
import io
import logging
import os
import queue
import re
import struct
import sys
import threading
from typing import AsyncGenerator, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pockettts_server")

app = FastAPI(title="chasm pockettts")

tts_model = None
voices_dir: Optional[str] = None
SAMPLE_RATE = 24000  # PocketTTS (Mimi) is 24 kHz; updated once the model loads
ROUTER_SAMPLE_RATE = 24000  # what the Rust router assumes when slicing PCM
_states: dict = {}            # character -> cached model state (speaker prompt)
_states_lock = threading.Lock()
_model_lock = threading.Lock()  # serialize GPU inference (one turn at a time)
_default_noise_clamp = None       # captured at load; used when noise_clamp <= 0 ("off")
_gen_kwargs_accepted: set = set()  # which generate_audio_stream kwargs this build accepts


class SpeechRequest(BaseModel):
    model: str = "pockettts"
    input: str
    voice: str = ""
    response_format: str = "pcm"  # pcm | wav
    speed: float = 1.0  # accepted, not yet applied
    # Silence padding (ms), sent live per request by the Rust router from the TTS
    # tuning settings. lead_in protects the speech onset from playback-startup
    # clipping; sentence_gap separates PocketTTS's sentence-chunked output so it
    # doesn't run together; trailing pads the end.
    lead_in_ms: int = 150
    sentence_gap_ms: int = 180
    trailing_ms: int = 70
    # PocketTTS sampling knobs, also sent live per request. temp/lsd/eos/noise_clamp
    # are read off the model instance at every generation step; max_tokens (chunk
    # size) + frames_after_eos (post-EOS tail; 0 = auto) go to generate_audio_stream.
    temperature: float = 0.7
    lsd_decode_steps: int = 1
    eos_threshold: float = -4.0
    noise_clamp: float = 0.0
    max_tokens: int = 50
    frames_after_eos: int = 0
    # Prepend padding spaces to short (<5 word) inputs. PocketTTS "does not perform
    # well when there are very few tokens" — our per-sentence split makes many short
    # inputs, so this gives the model a runway and preserves the word's onset.
    pad_short_inputs: bool = True


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """float32 [-1,1] -> raw 16-bit little-endian PCM bytes."""
    return np.clip(pcm * 32768.0, -32768, 32767).astype("<i2").tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """RIFF/WAVE header. data_len=0xFFFFFFFF for streaming (unknown length)."""
    n_channels, bits = 1, 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate, byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_numpy(audio) -> np.ndarray:
    """Bring a torch tensor (CPU or CUDA) back to a 1-D float32 numpy array."""
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    arr = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _prompt_path(character: str) -> Optional[str]:
    """The reference clip for a voice: the cloned trimmed prompt if present, else
    the shared reference. Mirrors the clone layout (voices/<name>/<engine>/...)."""
    if not voices_dir:
        return None
    trimmed = os.path.join(voices_dir, character, "pockettts", "prompt.wav")
    if os.path.exists(trimmed):
        return trimmed
    reference = os.path.join(voices_dir, character, "reference.wav")
    if os.path.exists(reference):
        return reference
    return None


def _first_available_voice() -> Optional[str]:
    """The first character under voices_dir that has a usable clip (the default)."""
    if not voices_dir or not os.path.isdir(voices_dir):
        return None
    for name in sorted(os.listdir(voices_dir)):
        if _prompt_path(name):
            return name
    return None


def resolve_state(voice_name: str):
    """The cached PocketTTS speaker state for a voice, falling back to the first
    available voice when the requested one has no clip (matches the qwen server's
    default-voice behaviour)."""
    name = voice_name.strip()
    prompt = _prompt_path(name) if name else None
    if prompt is None:
        fallback = _first_available_voice()
        if fallback is None:
            raise HTTPException(
                status_code=400,
                detail=f"voice {voice_name!r} has no clip under {voices_dir!r} and no fallback voice exists",
            )
        if name and name != fallback:
            logger.warning("voice %r has no clip; using default %r", voice_name, fallback)
        name = fallback
        prompt = _prompt_path(name)

    with _states_lock:
        state = _states.get(name)
        if state is None:
            logger.info("building speaker state for %r from %s", name, prompt)
            state = tts_model.get_state_for_audio_prompt(prompt)
            _states[name] = state
        return state


# Sentence boundaries: split AFTER ., !, ?, … followed by whitespace. We split the
# line ourselves (rather than letting generate_audio_stream's internal splitter own
# it) so we can insert a clean silence gap between sentences.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _split_sentences(text: str) -> list:
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    return parts or [text.strip()]


def _silence_pcm(ms: int) -> bytes:
    """`ms` of int16 mono silence at the model sample rate (clamped to a sane max)."""
    ms = max(0, min(int(ms), 5000))
    n = int(round(ms * SAMPLE_RATE / 1000.0))
    return np.zeros(n, dtype="<i2").tobytes() if n > 0 else b""


def _apply_model_knobs(req: "SpeechRequest") -> None:
    """Set PocketTTS's per-step sampling knobs on the shared model from the request.
    Caller MUST hold _model_lock. These attrs are read at every generation step, so
    setting them here applies the Tuning sliders live, per request. A knob absent on
    this build is skipped."""
    if hasattr(tts_model, "temp"):
        tts_model.temp = float(req.temperature)
    if hasattr(tts_model, "lsd_decode_steps"):
        tts_model.lsd_decode_steps = max(1, int(req.lsd_decode_steps))
    if hasattr(tts_model, "eos_threshold"):
        tts_model.eos_threshold = float(req.eos_threshold)
    if hasattr(tts_model, "noise_clamp"):
        nc = float(req.noise_clamp)
        # <= 0 means "off" → restore the library default captured at load.
        tts_model.noise_clamp = nc if nc > 0.0 else _default_noise_clamp
    if hasattr(tts_model, "pad_with_spaces_for_short_inputs"):
        tts_model.pad_with_spaces_for_short_inputs = bool(req.pad_short_inputs)


def _stream_gen_kwargs(req: "SpeechRequest") -> dict:
    """generate_audio_stream kwargs from the request, filtered to what this build
    accepts. frames_after_eos=0 means "auto" (omit so the library picks 1-3)."""
    kwargs = {}
    if "max_tokens" in _gen_kwargs_accepted and int(req.max_tokens) > 0:
        kwargs["max_tokens"] = int(req.max_tokens)
    if "frames_after_eos" in _gen_kwargs_accepted and int(req.frames_after_eos) > 0:
        kwargs["frames_after_eos"] = int(req.frames_after_eos)
    return kwargs


async def _stream_sentence(state, text: str, req: "SpeechRequest") -> AsyncGenerator[bytes, None]:
    """Run PocketTTS's streaming generator for ONE sentence in a worker thread,
    yielding int16 PCM per chunk as it is decoded (PocketTTS parallelises
    generation + Mimi decode). Applies the request's sampling knobs under the lock."""
    q: "queue.Queue" = queue.Queue()
    done = object()
    gen_kwargs = _stream_gen_kwargs(req)

    def producer():
        try:
            with _model_lock, torch.no_grad():
                _apply_model_knobs(req)
                # copy_state=True so the cached speaker state is preserved for reuse.
                for chunk in tts_model.generate_audio_stream(
                    state, text, copy_state=True, **gen_kwargs
                ):
                    q.put(_to_numpy(chunk))
        except Exception as exc:  # noqa: BLE001 — surfaced to the request
            q.put(exc)
        finally:
            q.put(done)

    threading.Thread(target=producer, daemon=True).start()
    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is done:
            break
        if isinstance(item, Exception):
            raise item
        yield _to_pcm16(item)


async def _stream_pcm(state, req: "SpeechRequest") -> AsyncGenerator[bytes, None]:
    """Stream the WHOLE line in ONE continuous generate_audio_stream pass (chunks
    flow as Mimi decodes), matching the faster-qwen3-tts delivery pattern that plays
    cleanly in-game.

    We deliberately do NOT pre-split into sentences + insert silence gaps anymore:
    that made the byte stream BURSTY (an instant silence burst, then a per-sentence
    generation stall), and the FNV plugin's real-time DirectSound buffer mishandled
    the bursts (onset cuts / glitches in-game) even though the rendered audio was
    clean. PocketTTS's own split_into_best_sentences still handles multi-sentence
    text internally, so prosody is unaffected. Only the line-edge pads remain."""
    lead = _silence_pcm(req.lead_in_ms)
    if lead:
        yield lead
    async for raw in _stream_sentence(state, req.input, req):
        yield raw
    trail = _silence_pcm(req.trailing_ms)
    if trail:
        yield trail


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tts_model is not None}


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")

    state = resolve_state(req.voice)
    fmt = req.response_format.lower()
    content_types = {"pcm": "audio/pcm", "wav": "audio/wav"}
    if fmt not in content_types:
        raise HTTPException(status_code=400, detail=f"response_format {fmt!r} not supported (pcm|wav)")

    async def audio_stream():
        if fmt == "wav":
            yield _wav_header(SAMPLE_RATE)  # streaming, unknown length
        async for raw in _stream_pcm(state, req):
            yield raw

    return StreamingResponse(audio_stream(), media_type=content_types[fmt])


def _parse_args():
    p = argparse.ArgumentParser(description="chasm pockettts streaming server")
    p.add_argument("--voices-dir", default=os.environ.get("POCKETTTS_VOICES_DIR"),
                   help="profile voices dir; voices resolve to <dir>/<name>/{pockettts/prompt.wav|reference.wav}")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5002)
    return p.parse_args()


def main():
    global tts_model, voices_dir, SAMPLE_RATE, _default_noise_clamp, _gen_kwargs_accepted
    args = _parse_args()

    if not args.voices_dir:
        print("ERROR: pass --voices-dir <dir>", file=sys.stderr)
        sys.exit(1)
    voices_dir = args.voices_dir

    from pocket_tts import TTSModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # TF32 matmuls are a free speedup on Ampere+ with no audible quality loss.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info("loading PocketTTS on %s …", device)
    tts_model = TTSModel.load_model()
    tts_model.to(device)
    # Capture the library's default noise_clamp (used when a request says "off") and
    # which generate_audio_stream kwargs this build accepts, so per-request knobs
    # degrade gracefully on a build with a different signature.
    _default_noise_clamp = getattr(tts_model, "noise_clamp", None)
    try:
        _gen_kwargs_accepted = set(inspect.signature(tts_model.generate_audio_stream).parameters)
    except (TypeError, ValueError):
        _gen_kwargs_accepted = set()
    SAMPLE_RATE = int(getattr(tts_model, "sample_rate", ROUTER_SAMPLE_RATE))
    if SAMPLE_RATE != ROUTER_SAMPLE_RATE:
        logger.warning(
            "PocketTTS sample_rate=%d != router's %d; in-game audio may sound wrong "
            "(the Rust slicer assumes %d Hz)",
            SAMPLE_RATE, ROUTER_SAMPLE_RATE, ROUTER_SAMPLE_RATE,
        )
    logger.info("model ready; sample_rate=%d; listening on http://%s:%d", SAMPLE_RATE, args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
