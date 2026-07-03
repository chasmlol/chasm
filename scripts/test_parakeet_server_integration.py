"""End-to-end HTTP test of the Parakeet server's word-boosting plumbing.

Runs against the real FastAPI app via TestClient, with the ASR model stubbed
(so no GPU / model download): we replace `_transcribe_blocking` with a canned
transcript and verify the vocabulary field + /v1/vocabulary endpoint correct it,
and — crucially — that WITHOUT a vocabulary the text is returned unchanged.

    python scripts/test_parakeet_server_integration.py
"""
import io
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parakeet_stt_server as server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

CANNED = "let's go to novak and see sunny smells today"
VOCAB = ["Novac", "Sunny Smiles", "Boone", "Goodsprings"]

_fails = []


def check(cond, msg):
    print(("  ok  : " if cond else "  FAIL: ") + msg)
    if not cond:
        _fails.append(msg)


def tiny_wav() -> bytes:
    """A short, valid 16 kHz mono WAV (content irrelevant — ASR is stubbed)."""
    buf = io.BytesIO()
    samples = np.zeros(8000, dtype=np.float32)  # 0.5 s of silence
    sf.write(buf, samples, 16_000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def post_transcribe(client, wav, vocab=None):
    data = {"model": "parakeet"}
    if vocab is not None:
        data["vocab"] = json.dumps(vocab)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", wav, "audio/wav")},
        data=data,
    )
    assert r.status_code == 200, (r.status_code, r.text)
    return r.json()["text"]


def main():
    # Stub the model so the endpoint returns our canned transcript.
    server.asr_model = object()
    server._transcribe_blocking = lambda wav: CANNED

    wav = tiny_wav()
    with TestClient(server.app) as client:
        # Health advertises boosting availability + an empty vocab to start.
        h = client.get("/health").json()
        check(h["model_loaded"] is True, "health: model_loaded")
        check(h.get("vocab_boost") is True, "health: vocab boosting available")
        check(h.get("vocab_count") == 0, "health: vocab_count starts at 0")

        # No vocab supplied anywhere -> byte-for-byte unchanged (regression gate).
        check(post_transcribe(client, wav) == CANNED,
              "no vocab -> transcript unchanged")
        check(post_transcribe(client, wav, vocab=[]) == CANNED,
              "empty vocab list -> transcript unchanged")

        # Per-request vocab -> proper nouns snapped.
        got = post_transcribe(client, wav, vocab=VOCAB)
        check(got == "let's go to Novac and see Sunny Smiles today",
              f"per-request vocab corrects proper nouns (got {got!r})")

        # Push a server-side vocabulary; subsequent requests need no vocab field.
        r = client.post("/v1/vocabulary", json={"words": VOCAB}).json()
        check(r["ok"] and r["vocab_count"] > 0, "POST /v1/vocabulary accepted")
        check(client.get("/health").json()["vocab_count"] == r["vocab_count"],
              "health reflects pushed vocab_count")
        got = post_transcribe(client, wav)  # no vocab field this time
        check(got == "let's go to Novac and see Sunny Smiles today",
              f"server-side vocab corrects without a per-request field (got {got!r})")

        # Per-request empty list overrides the server-side vocab (opt-out path).
        check(post_transcribe(client, wav, vocab=[]) == CANNED,
              "per-request empty list overrides server vocab")

        # Clearing the server vocab restores raw behaviour.
        r = client.post("/v1/vocabulary", json={"words": []}).json()
        check(r["vocab_count"] == 0, "clearing vocab -> count 0")
        check(post_transcribe(client, wav) == CANNED,
              "after clear -> transcript unchanged again")

        # Malformed vocab field must not break transcription.
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", wav, "audio/wav")},
            data={"vocab": "{not valid json"},
        )
        check(r.status_code == 200 and r.json()["text"] == CANNED,
              "malformed vocab field -> raw text, no error")

    print()
    if _fails:
        print(f"FAILED {len(_fails)} check(s)")
        sys.exit(1)
    print("ALL INTEGRATION TESTS PASSED")


if __name__ == "__main__":
    main()
