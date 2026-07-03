"""Unit tests for the STT vocabulary corrector (scripts/stt_vocab_boost.py).

Plain-assert tests so they run with any Python:
    python scripts/test_stt_vocab_boost.py

They pass under BOTH orthographic backends (rapidfuzz and the difflib
fallback): every positive case is anchored on either a phonetic-key match or a
difflib ratio that clears the threshold regardless of backend.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stt_vocab_boost import (  # noqa: E402
    VocabCorrector,
    phonetic_key,
    scorer_backend,
)

VOCAB = [
    "Sunny Smiles",
    "Caesar",
    "Novac",
    "Boone",
    "Doc Mitchell",
    "Goodsprings",
    "Mojave",
    "Manny Vargas",
]

_fails = []


def check(cond, msg):
    if cond:
        print(f"  ok  : {msg}")
    else:
        print(f"  FAIL: {msg}")
        _fails.append(msg)


def eq(corrector, text, expected):
    got = corrector.correct(text)
    check(got == expected, f"{text!r} -> {got!r} (expected {expected!r})")


def main():
    print(f"orthographic backend: {scorer_backend()}")

    # --- phonetic key: homophone-ish substitutions collide -----------------
    print("\n[phonetic_key]")
    check(phonetic_key("smiles") == phonetic_key("smells"), "smiles ~ smells")
    check(phonetic_key("sunny") == phonetic_key("sonny"), "sunny ~ sonny")
    check(phonetic_key("novac") == phonetic_key("novak"), "novac ~ novak")
    check(phonetic_key("goodsprings") == phonetic_key("godsprings"),
          "goodsprings ~ godsprings")
    check(phonetic_key("dad") != phonetic_key("ada"), "dad != ada")
    check(phonetic_key("") == "", "empty -> empty")

    c = VocabCorrector(VOCAB)
    print(f"\ncorrector size (phrases + split words): {c.size}")

    # --- POSITIVE: near-misses snap to the canonical proper noun -----------
    print("\n[positive corrections]")
    eq(c, "sunny smells", "Sunny Smiles")            # the headline case
    eq(c, "i talked to sunny smells today", "i talked to Sunny Smiles today")
    eq(c, "let's head to novak", "let's head to Novac")
    eq(c, "go see godsprings", "go see Goodsprings")
    eq(c, "boon is at the dinosaur", "Boone is at the dinosaur")  # boon->Boone
    eq(c, "hey, novak!", "hey, Novac!")               # punctuation preserved
    eq(c, "sunny   smells", "Sunny Smiles")           # collapses extra spaces
    eq(c, "manny vargus guards novak", "Manny Vargas guards Novac")  # two fixes
    # exact-but-lowercase spellings are left as-is (never re-cased), which is
    # what stops a real common word from being clobbered by a same-spelled name.
    eq(c, "manny vargas guards novac", "manny vargas guards novac")

    # --- NEGATIVE: ordinary speech and out-of-vocab left untouched ---------
    print("\n[negative / no over-correction]")
    eq(c, "the cave smells bad", "the cave smells bad")   # 'smells' guarded
    eq(c, "it smells in here", "it smells in here")
    eq(c, "i went to the store", "i went to the store")
    eq(c, "funny stories by the fire", "funny stories by the fire")
    eq(c, "on sunday we rest", "on sunday we rest")       # sunday !~ Sunny
    eq(c, "helios one power plant", "helios one power plant")  # OOV proper noun
    eq(c, "mojave desert is hot", "mojave desert is hot")  # exact-lower: no recase
    eq(c, "", "")                                          # empty text

    # --- Empty vocab is a strict identity ----------------------------------
    print("\n[empty vocab = identity]")
    empty = VocabCorrector([])
    check(empty.is_empty(), "empty corrector reports is_empty")
    eq(empty, "sunny smells anything at all", "sunny smells anything at all")

    # --- Split-word behaviour ---------------------------------------------
    print("\n[vocabulary construction]")
    long = VocabCorrector(["Doc Mitchell"])
    # "Doc" is < min_len (4) so only "Mitchell" + the phrase are registered.
    check(long.correct("mitchel patched me up") == "Mitchell patched me up",
          "single split word 'Mitchell' corrects mitchel")

    print()
    if _fails:
        print(f"FAILED {len(_fails)} check(s):")
        for f in _fails:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
