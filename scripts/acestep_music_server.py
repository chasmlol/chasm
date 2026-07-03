"""Chasm's ACE-Step music-generation server (DiT mode only).

Generates a full song (lyrics + style tags -> WAV) for the "play a song" NPC
action. The Rust backend POSTs lyrics + style tags + duration; this returns a
WAV. ACE-Step is a heavy, GPU-shared model (it competes with the LLM, TTS and
the game for VRAM), so the model is loaded LAZILY on the first request and
RELEASED after an idle timeout — chasm only pays the VRAM while a song is
actually being made.

DiT mode only: we pass `thinking=False` and `llm_handler=None` to
`generate_music`, so the 5Hz LM planner / vllm path is never initialised (that
also sidesteps the Blackwell/sm_120 vllm build). The DiT synthesises directly
from the caption (style tags) + lyrics + duration we provide.

  GET  /health          -> {"status":"ok","model_loaded":bool,"generating":bool}
  POST /v1/music         {lyrics, style_tags|caption, duration, seed?, steps?}
                         -> audio/wav  (headers: X-Gen-Seconds, X-Peak-VRAM-MB)

Run inside the engines/acestep venv (see scripts/install-engine.ps1), with the
ACE-Step checkout as the project root and the weights dir exported:
  set ACESTEP_PROJECT_ROOT=<engines>/acestep/ACE-Step-1.5
  set ACESTEP_CHECKPOINTS_DIR=<engines>/acestep/checkpoints
  python acestep_music_server.py --host 127.0.0.1 --port 5004
"""
import argparse
import io
import logging
import os
import sys
import threading
import time
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("acestep_music_server")

app = FastAPI(title="chasm acestep music")

# --- lazily-loaded model state (guarded by _model_lock for GPU work) ----------
_handler = None                       # AceStepHandler (None until first request)
_model_lock = threading.Lock()        # serialize GPU generation (one song at a time)
_state_lock = threading.Lock()        # guards _handler swap + _last_used
_last_used = 0.0                      # monotonic time of the last generation
_generating = False                   # a request is mid-generation (for /health)

# Configured once in main() from CLI args / env.
_cfg = {
    "project_root": "",
    "config_path": "acestep-v15-turbo",  # the DiT turbo checkpoint (DiT mode)
    "device": "cuda",
    "offload_to_cpu": False,
    "idle_timeout": 300.0,               # seconds idle before the model is freed
}


class MusicRequest(BaseModel):
    lyrics: str = ""
    # `style_tags` (chasm's name) or `caption` (ACE-Step's name) — accept both.
    style_tags: str = ""
    caption: str = ""
    duration: float = 60.0
    seed: int = -1
    steps: int = 8            # turbo DiT default (8-step distilled)
    guidance_scale: float = 7.0
    # Optional style/timbre reference (a path to an audio clip). ACE-Step samples a
    # 30s reference tensor and guides the DiT toward its style/timbre. Empty = none.
    reference_audio: str = ""


def _now() -> float:
    return time.monotonic()


def _peak_vram_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _ensure_model():
    """Load the ACE-Step DiT handler if it isn't resident. Caller holds _model_lock."""
    global _handler
    if _handler is not None:
        return _handler

    root = _cfg["project_root"]
    if root and root not in sys.path:
        sys.path.insert(0, root)

    from acestep.handler import AceStepHandler

    device = _cfg["device"]
    logger.info(
        "loading ACE-Step DiT (%s) on %s (offload_to_cpu=%s) ...",
        _cfg["config_path"], device, _cfg["offload_to_cpu"],
    )
    started = _now()
    handler = AceStepHandler()
    handler.initialize_service(
        project_root=root,
        config_path=_cfg["config_path"],
        device=device,
        offload_to_cpu=bool(_cfg["offload_to_cpu"]),
    )
    logger.info("ACE-Step DiT loaded in %.1fs", _now() - started)
    with _state_lock:
        _handler = handler
    return handler


def _unload_model(reason: str):
    """Drop the handler + free VRAM. Caller must hold _model_lock (no generation
    in flight)."""
    global _handler
    with _state_lock:
        if _handler is None:
            return
        _handler = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass
    logger.info("ACE-Step model unloaded (%s); VRAM released", reason)


def _idle_reaper():
    """Background: free the model after `idle_timeout` seconds with no generation,
    so an idle music engine doesn't sit on VRAM the game/LLM/TTS need."""
    while True:
        time.sleep(15.0)
        timeout = float(_cfg["idle_timeout"])
        if timeout <= 0:
            continue
        with _state_lock:
            resident = _handler is not None
            idle_for = _now() - _last_used if _last_used else 0.0
            busy = _generating
        if resident and not busy and idle_for >= timeout:
            # Only unload when we can take the GPU lock without blocking a request.
            if _model_lock.acquire(blocking=False):
                try:
                    if not _generating:
                        _unload_model(f"idle {idle_for:.0f}s")
                finally:
                    _model_lock.release()


def _generate_wav(req: MusicRequest) -> tuple[bytes, float, float]:
    """Run one DiT-mode generation; return (wav_bytes, gen_seconds, peak_vram_mb)."""
    global _last_used, _generating
    caption = (req.caption or req.style_tags or "").strip()
    lyrics = (req.lyrics or "").strip()
    duration = float(req.duration) if req.duration and req.duration > 0 else 60.0

    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    with _model_lock:
        with _state_lock:
            _generating = True
        try:
            handler = _ensure_model()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:  # noqa: BLE001
                pass

            ref_audio = (req.reference_audio or "").strip() or None
            if ref_audio and not os.path.exists(ref_audio):
                logger.warning("reference_audio %r not found; ignoring", ref_audio)
                ref_audio = None
            params = GenerationParams(
                caption=caption,
                lyrics=lyrics,
                duration=duration,
                thinking=False,          # DiT mode only — no LM planner / vllm
                inference_steps=int(req.steps) if req.steps else 8,
                guidance_scale=float(req.guidance_scale),
                seed=int(req.seed),
                reference_audio=ref_audio,
            )
            config = GenerationConfig(
                batch_size=1,
                audio_format="wav",
                use_random_seed=(int(req.seed) < 0),
                seeds=None if int(req.seed) < 0 else int(req.seed),
            )

            out_dir = os.path.join(
                _cfg["project_root"] or ".", "outputs", uuid.uuid4().hex
            )
            os.makedirs(out_dir, exist_ok=True)

            started = _now()
            # DiT mode: llm_handler=None so the LM/vllm path is never touched.
            result = generate_music(handler, None, params, config, save_dir=out_dir)
            gen_seconds = _now() - started

            if not getattr(result, "success", False):
                raise RuntimeError(getattr(result, "error", "generation failed"))
            audios = getattr(result, "audios", []) or []
            if not audios:
                raise RuntimeError("generation returned no audio")
            path = audios[0].get("path")
            if not path or not os.path.exists(path):
                raise RuntimeError(f"generated audio missing on disk: {path!r}")
            with open(path, "rb") as f:
                wav_bytes = f.read()

            peak = _peak_vram_mb()
            logger.info(
                "generated %.0fs song in %.1fs (%.2fx realtime), peak VRAM %.0f MB, %d bytes",
                duration, gen_seconds, (duration / gen_seconds) if gen_seconds else 0.0,
                peak, len(wav_bytes),
            )
            # Best-effort cleanup of the on-disk render (we return the bytes).
            try:
                os.remove(path)
                os.rmdir(out_dir)
            except OSError:
                pass
            return wav_bytes, gen_seconds, peak
        finally:
            with _state_lock:
                _generating = False
                _last_used = _now()


@app.get("/health")
async def health():
    with _state_lock:
        return {
            "status": "ok",
            "model_loaded": _handler is not None,
            "generating": _generating,
        }


@app.post("/v1/music")
async def create_music(req: MusicRequest):
    if not (req.lyrics.strip() or req.caption.strip() or req.style_tags.strip()):
        raise HTTPException(status_code=400, detail="need lyrics or style tags")
    import asyncio

    try:
        wav_bytes, gen_seconds, peak = await asyncio.to_thread(_generate_wav, req)
    except Exception as error:  # noqa: BLE001
        logger.exception("music generation failed")
        raise HTTPException(status_code=500, detail=f"generation failed: {error}")
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Gen-Seconds": f"{gen_seconds:.1f}",
            "X-Peak-VRAM-MB": f"{peak:.0f}",
        },
    )


def main():
    parser = argparse.ArgumentParser(description="chasm ACE-Step music server (DiT mode)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5004)
    parser.add_argument(
        "--project-root",
        default=os.environ.get("ACESTEP_PROJECT_ROOT", ""),
        help="the ACE-Step 1.5 checkout dir (its code + default checkpoints root)",
    )
    parser.add_argument(
        "--config-path",
        default=os.environ.get("ACESTEP_DIT_CONFIG", "acestep-v15-turbo"),
        help="DiT checkpoint config (the only pipeline mode exposed)",
    )
    parser.add_argument("--device", default=os.environ.get("ACESTEP_DEVICE", "cuda"))
    parser.add_argument(
        "--offload-to-cpu",
        action="store_true",
        default=os.environ.get("ACESTEP_OFFLOAD_TO_CPU", "") not in ("", "0", "false"),
        help="offload idle weights to CPU RAM to shrink resident VRAM",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=float(os.environ.get("ACESTEP_IDLE_TIMEOUT", "300")),
        help="seconds idle before the model is unloaded to free VRAM (0 = never)",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=os.environ.get("ACESTEP_WARMUP", "") not in ("", "0", "false"),
        help="load the model in the background at startup so the FIRST song is fast",
    )
    args = parser.parse_args()

    _cfg["project_root"] = args.project_root
    _cfg["config_path"] = args.config_path
    _cfg["device"] = args.device
    _cfg["offload_to_cpu"] = bool(args.offload_to_cpu)
    _cfg["idle_timeout"] = float(args.idle_timeout)

    if not _cfg["project_root"] or not os.path.isdir(_cfg["project_root"]):
        logger.warning(
            "ACESTEP project root %r not found; set --project-root / ACESTEP_PROJECT_ROOT",
            _cfg["project_root"],
        )

    # The model loads on the first /v1/music (lazy) — the server comes up instantly.
    # With --warmup we ALSO kick a background load now, so the first in-game song
    # doesn't pay the ~12s model-load cost. The idle reaper still frees VRAM later.
    threading.Thread(target=_idle_reaper, daemon=True).start()
    if args.warmup:
        def _warm():
            try:
                with _model_lock:
                    _ensure_model()
                with _state_lock:
                    global _last_used
                    _last_used = _now()
                logger.info("ACE-Step warmup complete; first song will be fast")
            except Exception as error:  # noqa: BLE001
                logger.warning("ACE-Step warmup failed (will lazy-load on first song): %s", error)
        threading.Thread(target=_warm, daemon=True).start()
    logger.info(
        "ACE-Step music server listening on http://%s:%d (%s; idle unload %.0fs)",
        args.host, args.port, "warming up" if args.warmup else "lazy load", _cfg["idle_timeout"],
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
