"""Clone each character's reference voice with a chosen TTS engine.

Run inside that engine's venv. Loads the engine's model once, then for every
voices/<name>/reference.wav synthesizes a short line in the cloned voice to
voices/<name>/<engine>/sample.wav (proof the engine cloned that character).

Usage: python tts_clone.py --engine pockettts --voices-dir <dir>
"""
import argparse
import os
import sys
import traceback

import soundfile as sf

SAMPLE_TEXT = (
    "Hello there, this is a test of my cloned voice. "
    "If you can hear me speaking clearly, then the voice cloning is working as intended. "
    "Thanks for listening."
)


def trimmed_prompt(ref, out_dir, seconds=18):
    """A short, mono prompt clip for cloning (long refs slow some engines)."""
    audio, sr = sf.read(ref)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if len(audio) > int(seconds * sr):
        audio = audio[: int(seconds * sr)]
    path = os.path.join(out_dir, "prompt.wav")
    sf.write(path, audio, sr)
    return path


def make_synth(engine, text):
    """Return synth(prompt_path) -> (audio, sample_rate) for the engine."""
    if engine == "pockettts":
        from pocket_tts import TTSModel
        model = TTSModel.load_model()

        def synth(prompt):
            state = model.get_state_for_audio_prompt(prompt)
            audio = model.generate_audio(state, text)
            return (audio.numpy() if hasattr(audio, "numpy") else audio), model.sample_rate
        return synth

    if engine == "faster-qwen3-tts":
        import torch
        from faster_qwen3_tts import FasterQwen3TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = FasterQwen3TTS.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device=device, dtype=torch.bfloat16,
        )

        def synth(prompt):
            # xvec_only: our extracted clips have no transcript, so clone from the
            # speaker embedding only (ref_text ignored). Matches qwen3_tts_server.
            arrs, sr = model.generate_voice_clone(
                text=text, language="English", ref_audio=prompt, ref_text="", xvec_only=True,
            )
            audio = arrs[0] if arrs else None
            return (audio.numpy() if hasattr(audio, "numpy") else audio), sr
        return synth

    print(f"ERROR unknown engine {engine}", flush=True)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--voices-dir", required=True)
    ap.add_argument("--text", default=SAMPLE_TEXT)
    args = ap.parse_args()

    chars = [n for n in sorted(os.listdir(args.voices_dir))
             if os.path.exists(os.path.join(args.voices_dir, n, "reference.wav"))]
    print(f"INFO engine={args.engine} characters={len(chars)}", flush=True)

    synth = make_synth(args.engine, args.text)

    for name in chars:
        ref = os.path.join(args.voices_dir, name, "reference.wav")
        out_dir = os.path.join(args.voices_dir, name, args.engine)
        os.makedirs(out_dir, exist_ok=True)
        try:
            prompt = trimmed_prompt(ref, out_dir)
            audio, sr = synth(prompt)
            sf.write(os.path.join(out_dir, "sample.wav"), audio, sr)
            print(f"PROGRESS {name} ok", flush=True)
        except Exception:
            print(f"PROGRESS {name} failed: {traceback.format_exc().splitlines()[-1]}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
