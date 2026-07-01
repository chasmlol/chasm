"""Chasm's faster-qwen3-tts streaming TTS server (OpenAI-compatible).

Serves NPC voice-cloned speech for the FNV bridge: koboldcpp handles LLM+STT,
this handles TTS. Streams raw int16 PCM (or a streaming WAV) as the model
generates, so the Rust backend can slice it into gapless mini-chunks for the game.

Why chasm ships its own server instead of faster-qwen3-tts's `examples/openai_server.py`:
the stock server doesn't expose `xvec_only`, so with our no-transcript NPC clips it
would run ICL mode (needs an accurate transcript) and clone poorly. Our clips have no
transcript, so we force `xvec_only=True` (ref_text ignored) — matching the old koboldcpp
`x_vector_only_mode=True` baseline.

  GET  /health             -> {"status":"ok","model_loaded":bool}
  POST /v1/audio/speech     {model,input,voice,response_format}  (pcm|wav|mp3)

Run inside the faster-qwen3-tts venv:
  python qwen3_tts_server.py --voices voices.json --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
      --host 127.0.0.1 --port 5002

voices.json maps a voice name to a reference-audio config:
  {
    "Easy Pete": {"ref_audio": "C:/.../Easy Pete.wav", "ref_text": "",
                  "language": "English", "chunk_size": 4, "xvec_only": true}
  }
`ref_text` is ignored when `xvec_only` is true. `chunk_size` is codec steps per
streamed chunk (~chunk_size/12 s of audio; 2≈160ms, 4≈320ms).
"""
import argparse
import asyncio
import io
import json
import logging
import os
import queue
import struct
import sys
import threading
from typing import AsyncGenerator, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen3_tts_server")

app = FastAPI(title="chasm faster-qwen3-tts")

tts_model = None
voices: dict = {}
default_voice: Optional[str] = None
voices_path: Optional[str] = None  # path to the --voices file, for hot-reload
voices_mtime: float = 0.0          # last-seen mtime, to detect on-disk edits
voices_dir: Optional[str] = None   # a profile voices dir (--voices-dir) to auto-discover
SAMPLE_RATE = 24000  # updated once the model loads
_model_lock = threading.Lock()  # serialize GPU inference (one turn at a time)


def _dir_ref_audio(character: str) -> Optional[str]:
    """The reference clip for a voice discovered under --voices-dir: the cloned
    trimmed prompt if present, else the shared reference. Mirrors the clone layout
    (voices/<name>/<engine>/...) used by the PocketTTS server."""
    if not voices_dir:
        return None
    trimmed = os.path.join(voices_dir, character, "faster-qwen3-tts", "prompt.wav")
    if os.path.exists(trimmed):
        return trimmed
    reference = os.path.join(voices_dir, character, "reference.wav")
    if os.path.exists(reference):
        return reference
    return None


def _dir_voice_cfg(character: str) -> Optional[dict]:
    """Build a qwen3 voice config for a character discovered under --voices-dir.
    Our cloned clips have no transcript, so xvec_only=True (ref_text ignored)."""
    ref = _dir_ref_audio(character)
    if ref is None:
        return None
    return {"ref_audio": ref, "ref_text": "", "language": "English",
            "chunk_size": 4, "xvec_only": True}


def _first_dir_voice() -> Optional[str]:
    """The first character under --voices-dir that has a usable clip (the default)."""
    if not voices_dir or not os.path.isdir(voices_dir):
        return None
    for name in sorted(os.listdir(voices_dir)):
        if _dir_ref_audio(name):
            return name
    return None


class SpeechRequest(BaseModel):
    model: str = "qwen3-tts"
    input: str
    voice: str = ""
    response_format: str = "pcm"  # pcm | wav | mp3
    speed: float = 1.0  # accepted, not yet applied


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


def _maybe_reload_voices() -> None:
    """Re-read the voices file when it changes on disk, so voices added to
    voices.json take effect without restarting the server."""
    global voices, default_voice, voices_mtime
    if not voices_path:
        return
    try:
        mtime = os.path.getmtime(voices_path)
    except OSError:
        return
    if mtime == voices_mtime:
        return
    try:
        with open(voices_path, encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("voices reload failed (keeping current): %s", exc)
        voices_mtime = mtime  # don't retry a broken file every request
        return
    if loaded:
        voices = loaded
        if default_voice not in voices:
            default_voice = next(iter(voices))
        voices_mtime = mtime
        logger.info("reloaded %d voice(s) from %s", len(voices), voices_path)


def resolve_voice(voice_name: str) -> dict:
    _maybe_reload_voices()
    name = voice_name.strip()
    if name in voices:
        return voices[name]
    # Directory mode (--voices-dir): auto-discover the clip from the clone layout.
    if voices_dir is not None:
        cfg = _dir_voice_cfg(name) if name else None
        if cfg is not None:
            return cfg
        fallback = _first_dir_voice()
        if fallback is not None:
            if name and name != fallback:
                logger.warning("voice %r has no clip; using default %r", voice_name, fallback)
            cfg = _dir_voice_cfg(fallback)
            if cfg is not None:
                return cfg
        raise HTTPException(
            status_code=400,
            detail=f"voice {voice_name!r} has no clip under {voices_dir!r} and no fallback voice exists",
        )
    if default_voice and default_voice in voices:
        logger.warning("voice %r not configured; using default %r", voice_name, default_voice)
        return voices[default_voice]
    raise HTTPException(status_code=400, detail=f"voice {voice_name!r} not configured; have {list(voices)}")


async def _stream_pcm(voice_cfg: dict, text: str) -> AsyncGenerator[bytes, None]:
    """Run the streaming generator in a worker thread, yield int16 PCM per chunk."""
    q: "queue.Queue" = queue.Queue()
    done = object()

    def producer():
        try:
            with _model_lock:
                for chunk, _sr, _timing in tts_model.generate_voice_clone_streaming(
                    text=text,
                    language=voice_cfg.get("language", "English"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                    xvec_only=voice_cfg.get("xvec_only", True),
                    chunk_size=int(voice_cfg.get("chunk_size", 4)),
                    non_streaming_mode=False,
                ):
                    q.put(chunk)
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


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tts_model is not None}


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")

    voice_cfg = resolve_voice(req.voice)
    fmt = req.response_format.lower()
    content_types = {"pcm": "audio/pcm", "wav": "audio/wav", "mp3": "audio/mpeg"}
    if fmt not in content_types:
        raise HTTPException(status_code=400, detail=f"response_format {fmt!r} not supported (pcm|wav|mp3)")

    if fmt == "mp3":
        try:
            from pydub import AudioSegment
        except ImportError:
            raise HTTPException(status_code=400, detail="response_format='mp3' requires pydub")
        loop = asyncio.get_event_loop()

        def _gen():
            with _model_lock:
                return tts_model.generate_voice_clone(
                    text=req.input,
                    language=voice_cfg.get("language", "English"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                    xvec_only=voice_cfg.get("xvec_only", True),
                )

        arrs, sr = await loop.run_in_executor(None, _gen)
        audio = arrs[0] if arrs else np.zeros(1, dtype=np.float32)
        seg = AudioSegment(_to_pcm16(audio), frame_rate=sr, sample_width=2, channels=1)
        buf = io.BytesIO()
        seg.export(buf, format="mp3")
        return Response(content=buf.getvalue(), media_type="audio/mpeg")

    async def audio_stream():
        if fmt == "wav":
            yield _wav_header(SAMPLE_RATE)  # streaming, unknown length
        async for raw in _stream_pcm(voice_cfg, req.input):
            yield raw

    return StreamingResponse(audio_stream(), media_type=content_types[fmt])


def _parse_args():
    p = argparse.ArgumentParser(description="chasm faster-qwen3-tts streaming server")
    p.add_argument("--model", default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
    p.add_argument("--voices", default=os.environ.get("QWEN_TTS_VOICES"), metavar="FILE",
                   help="JSON mapping voice names to {ref_audio, ref_text, language, chunk_size, xvec_only}")
    p.add_argument("--voices-dir", default=os.environ.get("QWEN_TTS_VOICES_DIR"), metavar="DIR",
                   help="profile voices dir (<name>/reference.wav); voices auto-discovered + hot-reloaded")
    p.add_argument("--ref-audio", default=os.environ.get("QWEN_TTS_REF_AUDIO"),
                   help="single reference audio when --voices is not used")
    p.add_argument("--ref-text", default=os.environ.get("QWEN_TTS_REF_TEXT", ""))
    p.add_argument("--language", default=os.environ.get("QWEN_TTS_LANGUAGE", "English"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5002)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    global tts_model, voices, default_voice, SAMPLE_RATE, voices_path, voices_mtime, voices_dir
    args = _parse_args()

    if args.voices_dir:
        # Directory mode: voices auto-discovered per request from the clone layout
        # (<name>/reference.wav). Matches the PocketTTS server's --voices-dir, so the
        # Rust launcher spawns both engines the same way.
        voices_dir = args.voices_dir
        n = sum(1 for name in (os.listdir(voices_dir) if os.path.isdir(voices_dir) else [])
                if _dir_ref_audio(name))
        logger.info("voices dir %s (%d voice(s) discovered)", voices_dir, n)
    elif args.voices:
        voices_path = args.voices
        with open(args.voices, encoding="utf-8") as f:
            voices = json.load(f)
        try:
            voices_mtime = os.path.getmtime(args.voices)
        except OSError:
            voices_mtime = 0.0
        default_voice = next(iter(voices))
        logger.info("loaded %d voice(s) from %s", len(voices), args.voices)
    elif args.ref_audio:
        voices = {"default": {"ref_audio": args.ref_audio, "ref_text": args.ref_text,
                              "language": args.language, "xvec_only": not args.ref_text}}
        default_voice = "default"
    else:
        print("ERROR: pass --voices-dir <dir>, --voices <config.json>, or --ref-audio <file>",
              file=sys.stderr)
        sys.exit(1)

    from faster_qwen3_tts import FasterQwen3TTS

    logger.info("loading %s on %s …", args.model, args.device)
    tts_model = FasterQwen3TTS.from_pretrained(args.model, device=args.device, dtype=torch.bfloat16)
    SAMPLE_RATE = tts_model.sample_rate
    logger.info("model ready; sample_rate=%d; listening on http://%s:%d", SAMPLE_RATE, args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
