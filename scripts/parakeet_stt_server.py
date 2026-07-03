"""Chasm's Parakeet STT server (OpenAI-compatible).

Serves push-to-talk transcription for the FNV bridge on its OWN port, so voice
input never queues behind an LLM generation (koboldcpp's Whisper shares the
LLM's single slot). The Rust backend POSTs the exact same multipart form it
sends to koboldcpp's Whisper, so selecting Parakeet changes only the endpoint.

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
import logging
import threading
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parakeet_stt_server")

app = FastAPI(title="chasm parakeet stt")

asr_model = None
_model_lock = threading.Lock()  # serialize GPU inference (one clip at a time)

TARGET_SR = 16_000  # Parakeet's expected input rate


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
    return {"status": "ok", "model_loaded": asr_model is not None}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(""),        # accepted for OpenAI-compat; ignored
    language: str = Form(""),     # Parakeet v3 auto-detects; ignored
    prompt: str = Form(""),       # no biasing-prompt support; ignored
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
