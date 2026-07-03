"""Chasm's Parakeet STT server (OpenAI-compatible).

Serves push-to-talk transcription for the FNV bridge on its OWN port, so voice
input never queues behind an LLM generation. The Rust backend POSTs a standard
OpenAI-style multipart transcription form to this server's endpoint.

Model: NVIDIA Parakeet TDT 0.6B v3 via nano-parakeet (pure-PyTorch TDT
inference, CUDA). nano-parakeet downloads the `.nemo` from HuggingFace into the
HF cache on first load (the engine install prefetches it); `--model` overrides
the repo id.

  GET  /health                    -> {"status":"ok","model_loaded":bool}
  POST /v1/audio/transcriptions      multipart: file (wav), model, language,
                                     prompt (both accepted + ignored: Parakeet
                                     v3 auto-detects language and takes no
                                     biasing prompt) -> {"text": "..."}

Run inside the engines/parakeet venv (see scripts/install-engine.ps1):
  python parakeet_stt_server.py --host 127.0.0.1 --port 5003
"""
import argparse
import asyncio
import io
import json
import logging
import os
import sys
import threading
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parakeet_stt_server")

app = FastAPI(title="chasm parakeet stt")

asr_model = None
_model_lock = threading.Lock()  # serialize GPU inference (one clip at a time)

TARGET_SR = 16_000  # Parakeet's expected input rate

# --- Word boosting / custom vocabulary --------------------------------------
# Parakeet's greedy TDT decoder cannot be biased mid-decode (no beam search), so
# we snap near-miss proper nouns back to the caller's vocabulary AFTER decoding.
# All of this is OPTIONAL and additive: with no vocabulary supplied the server
# behaves exactly as before. The corrector lives in a sibling module so it can
# be unit-tested without importing torch/fastapi.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from stt_vocab_boost import VocabCorrector, scorer_backend

    _VOCAB_OK = True
    logger.info("vocab boosting available (scorer backend: %s)", scorer_backend())
except Exception as _vocab_import_error:  # missing file or optional dep
    VocabCorrector = None  # type: ignore
    _VOCAB_OK = False
    logger.warning(
        "vocab boosting unavailable (%s) — transcription runs unbiased",
        _vocab_import_error,
    )

_vocab_lock = threading.Lock()
# Server-side default vocabulary set via POST /v1/vocabulary (chasm pushes the
# current character+lore names here). Applied when a request omits its own.
_server_corrector = None            # VocabCorrector | None
_server_vocab_count = 0
# Tiny memo so an unchanged per-request vocab list isn't rebuilt every clip.
_request_corrector_cache: dict = {}  # {hash(words_json): (VocabCorrector, count)}


def _build_corrector(words):
    """Build a corrector from a list of strings, or None if unusable/empty."""
    if not _VOCAB_OK or not words:
        return None, 0
    try:
        corrector = VocabCorrector(words)
    except Exception as error:
        logger.warning("failed to build vocab corrector: %s", error)
        return None, 0
    if corrector.is_empty():
        return None, 0
    return corrector, corrector.size


def _corrector_for_request(vocab_field: str):
    """Resolve the corrector for one transcription: the per-request `vocab`
    field if present, else the server-side default. Cached by content hash."""
    vocab_field = (vocab_field or "").strip()
    if vocab_field:
        key = hash(vocab_field)
        cached = _request_corrector_cache.get(key)
        if cached is not None:
            return cached[0]
        try:
            words = json.loads(vocab_field)
            if not isinstance(words, list):
                words = []
        except Exception as error:
            logger.warning("ignoring malformed vocab field: %s", error)
            words = []
        corrector, count = _build_corrector([str(w) for w in words])
        # Keep the memo tiny — this endpoint sees one vocab list at a time.
        if len(_request_corrector_cache) > 4:
            _request_corrector_cache.clear()
        _request_corrector_cache[key] = (corrector, count)
        return corrector
    with _vocab_lock:
        return _server_corrector


def _decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode an audio payload to float32 mono samples + sample rate.

    soundfile handles WAV/OGG/FLAC (the bridge always sends WAV). No ffmpeg
    dependency: unsupported containers raise -> a clean 400 for the caller.
    """
    import soundfile as sf

    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return wav.mean(axis=1), int(sr)


def _resample(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz with torchaudio when needed (game audio is 16 kHz
    already; browser/test clips may not be)."""
    if sr == TARGET_SR:
        return wav
    import torchaudio

    tensor = torch.from_numpy(wav).unsqueeze(0)
    out = torchaudio.functional.resample(tensor, sr, TARGET_SR)
    return out.squeeze(0).numpy()


def _transcribe_blocking(wav: np.ndarray) -> str:
    """Run Parakeet on float32 16 kHz samples (GPU, lock-serialized)."""
    with _model_lock:
        tensor = torch.from_numpy(wav)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        result = asr_model.transcribe(tensor)
    if isinstance(result, (list, tuple)):
        result = result[0] if result else ""
    return str(result).strip()


@app.get("/health")
async def health():
    with _vocab_lock:
        vocab_count = _server_vocab_count
    return {
        "status": "ok",
        "model_loaded": asr_model is not None,
        "vocab_boost": _VOCAB_OK,
        "vocab_count": vocab_count,
    }


@app.post("/v1/vocabulary")
async def set_vocabulary(request: Request):
    """Set the server-side boost vocabulary (chasm pushes character + lore names).

    Body: {"words": ["Sunny Smiles", "Novac", ...]}. An empty list clears it.
    Backward-safe: callers that never hit this endpoint get today's behaviour.
    """
    global _server_corrector, _server_vocab_count
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    words = body.get("words") if isinstance(body, dict) else None
    if words is None or not isinstance(words, list):
        raise HTTPException(status_code=400, detail="expected {\"words\": [...]}")
    corrector, count = _build_corrector([str(w) for w in words])
    with _vocab_lock:
        _server_corrector = corrector
        _server_vocab_count = count
    _request_corrector_cache.clear()
    logger.info("vocabulary set: %d boost entries", count)
    return {"ok": True, "vocab_count": count, "vocab_boost": _VOCAB_OK}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(""),        # accepted for OpenAI-compat; ignored
    language: str = Form(""),     # Parakeet v3 auto-detects; ignored
    prompt: str = Form(""),       # no biasing-prompt support; ignored
    vocab: str = Form(""),        # optional JSON array of boost words (chasm)
    response_format: str = Form("json"),
):
    if asr_model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio payload")
    try:
        wav, sr = _decode_wav(data)
    except Exception as error:  # malformed/unsupported audio
        raise HTTPException(status_code=400, detail=f"could not decode audio: {error}")
    if wav.size == 0:
        raise HTTPException(status_code=400, detail="audio contained no samples")
    wav = _resample(wav, sr)

    started = time.perf_counter()
    try:
        text = await asyncio.to_thread(_transcribe_blocking, wav)
    except Exception as error:
        logger.exception("transcription failed")
        raise HTTPException(status_code=500, detail=f"transcription failed: {error}")
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # Word boosting: snap near-miss proper nouns to the boost vocabulary. No-op
    # (identity) when no vocabulary is configured, so absent-vocab is unchanged.
    corrector = _corrector_for_request(vocab)
    if corrector is not None and text:
        try:
            corrected = corrector.correct(text)
        except Exception as error:  # never let correction break a transcription
            logger.warning("vocab correction failed (returning raw text): %s", error)
            corrected = text
        if corrected != text:
            logger.info("vocab boost: %r -> %r", text[:120], corrected[:120])
            text = corrected

    logger.info(
        "transcribed %.1fs of audio in %.0f ms: %r",
        wav.size / TARGET_SR, elapsed_ms, text[:120],
    )
    # OpenAI json shape; `verbose_json` callers still find `.text`.
    return {"text": text}


def main() -> None:
    parser = argparse.ArgumentParser(description="chasm Parakeet STT server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5003)
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-tdt-0.6b-v3",
        help="HuggingFace repo id nano-parakeet loads (the .nemo build)",
    )
    args = parser.parse_args()

    global asr_model
    from nano_parakeet import from_pretrained

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading %s on %s ...", args.model, device)
    started = time.perf_counter()
    asr_model = from_pretrained(args.model, device=device)
    logger.info("model loaded in %.1fs", time.perf_counter() - started)

    # Absorb the first-decode warm-up so the first real push-to-talk line is fast.
    try:
        _transcribe_blocking(np.zeros(TARGET_SR, dtype=np.float32))
        logger.info("warmup decode done")
    except Exception as error:
        logger.warning("warmup decode failed (continuing): %s", error)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
