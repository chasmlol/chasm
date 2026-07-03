"""Conservative fuzzy vocabulary correction for the Parakeet STT server.

Why this exists
---------------
`nano-parakeet` decodes with a *pure greedy* TDT loop (argmax per step, no beam
search, GPU path captured in a CUDA graph). Real context-biasing / shallow
fusion needs a beam search + a context graph that nano-parakeet simply does not
ship, so proper decoder biasing is not feasible here without rewriting the
decoder. Instead we snap near-miss proper nouns back to the caller's vocabulary
*after* transcription — cheap, decoder-agnostic, and very effective for exactly
the failure mode we care about ("sunny smells" -> "Sunny Smiles").

Design goals
------------
* CONSERVATIVE: whole-word / whole-phrase replacement only, behind high
  similarity thresholds, with a common-English-word guard so ordinary speech is
  never corrupted. When in doubt, leave the text alone.
* SELF-CONTAINED: no torch / fastapi imports, so it unit-tests standalone.
* DEGRADES GRACEFULLY: `rapidfuzz` is used when present for orthographic
  scoring, else stdlib `difflib`. The phonetic key is our own deterministic
  function (no external phonetic lib), so behaviour is identical everywhere.
* BACKWARD-SAFE: an empty vocabulary yields a corrector whose `correct()` is the
  identity function.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

# --- Orthographic scorer: rapidfuzz if available, else difflib --------------
try:  # pragma: no cover - import guard
    from rapidfuzz import fuzz as _rf_fuzz

    def _ratio(a: str, b: str) -> float:
        """Levenshtein-based similarity in 0..1.

        Deliberately NOT Jaro-Winkler: JW's shared-character / prefix bonus rates
        "funny"~"sunny" and "sunday"~"sunny" at ~0.87, which would clobber
        ordinary speech. Edit-distance ratio keeps those safely below threshold
        while still scoring genuine suffix/vowel typos highly.
        """
        return _rf_fuzz.ratio(a, b) / 100.0

    _BACKEND = "rapidfuzz"
except Exception:  # pragma: no cover - fallback path
    import difflib

    def _ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    _BACKEND = "difflib"


def scorer_backend() -> str:
    """Which orthographic backend is active ('rapidfuzz' or 'difflib')."""
    return _BACKEND


# --- Phonetic key -----------------------------------------------------------
# A compact, deterministic metaphone-ish reduction. It is intentionally lossy:
# it collapses vowels and near-homophone consonant classes so that ASR
# substitutions that *sound* alike ("smiles"/"smells", "sunny"/"sonny") produce
# the same key. It is a booster, not the sole signal — orthographic similarity
# still gates every replacement.

_VOWELS = set("aeiouy")
_CONS_CLASS = {
    "b": "B", "p": "B",
    "f": "F", "v": "F",
    "c": "K", "k": "K", "q": "K", "g": "K", "j": "K", "x": "K",
    "s": "S", "z": "S",
    "t": "T", "d": "T",
    "m": "M", "n": "M",
    "l": "L",
    "r": "R",
}


def phonetic_key(word: str) -> str:
    """Reduce a single word to a coarse phonetic skeleton.

    Empty string for anything with no usable letters. A leading vowel is marked
    'A' so "Ada" and "Ida" collide but "Ada"/"Dad" do not.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    out: list[str] = []
    i = 0
    n = len(w)
    first = True
    while i < n:
        c = w[i]
        nxt = w[i + 1] if i + 1 < n else ""
        code = ""
        if c in _VOWELS:
            if first:
                code = "A"  # only a leading vowel survives
        elif c == "p" and nxt == "h":
            code, i = "F", i + 1          # ph -> f
        elif c == "c" and nxt in ("e", "i", "y"):
            code = "S"                    # soft c
        elif c == "c" and nxt == "h":
            code, i = "J", i + 1          # ch
        elif c == "s" and nxt == "h":
            code, i = "J", i + 1          # sh
        elif c == "c" and nxt == "k":
            code, i = "K", i + 1          # ck -> k
        elif c == "g" and nxt == "h":
            code, i = "", i + 1           # silent gh
        elif c in ("w", "h"):
            code = ""                     # w/h drop (even leading: low value)
        else:
            code = _CONS_CLASS.get(c, "")
        if code and (not out or out[-1] != code):
            out.append(code)
        first = False
        i += 1
    return "".join(out)


def _phrase_phonetic(text: str) -> str:
    """Phonetic key of a (possibly multi-word) phrase: per-word keys joined."""
    return " ".join(phonetic_key(tok) for tok in text.split())


# --- Common-word guard ------------------------------------------------------
# Single recognized words that are ordinary English are NEVER replaced, so a
# stray "smells" only snaps to "Smiles" when it is genuinely a near-miss AND not
# itself a common word. (Multi-word phrases are inherently safer and skip this.)
_COMMON_WORDS = {
    "the", "and", "are", "you", "your", "for", "was", "with", "his", "her",
    "him", "she", "they", "them", "that", "this", "then", "than", "there",
    "here", "have", "has", "had", "not", "but", "all", "any", "one", "two",
    "who", "why", "how", "what", "when", "where", "which", "will", "would",
    "can", "could", "should", "did", "does", "done", "get", "got", "out",
    "off", "over", "into", "just", "like", "know", "now", "new", "old",
    "good", "bad", "yes", "man", "men", "way", "day", "say", "see", "let",
    "come", "want", "need", "take", "make", "give", "tell", "talk", "look",
    "well", "back", "down", "from", "some", "more", "most", "much", "many",
    "very", "also", "about", "still", "even", "only", "such", "these", "those",
    "been", "being", "were", "our", "its", "him", "hey", "yeah", "okay", "ok",
    "sir", "please", "hello", "thanks", "sorry", "sure", "fine", "name",
    "people", "friend", "help", "thing", "things", "time", "place", "world",
    "wait", "stop", "here", "there", "yellow", "smell", "smells",
}


class VocabCorrector:
    """Snaps near-miss spans of a transcript onto a caller-supplied vocabulary.

    Parameters
    ----------
    words:
        Proper nouns to bias toward — character names, lore titles, lore keys.
        Multi-word entries are registered whole *and* split into their words.
    ortho_threshold / phon_threshold:
        A span is replaced when orthographic similarity >= ortho_threshold, OR
        (phonetic keys match AND orthographic >= phon_threshold). Higher =
        more conservative.
    min_len:
        Shortest single word eligible to match (guards against 2-3 char noise).
    """

    def __init__(
        self,
        words: Iterable[str],
        *,
        ortho_threshold: float = 0.86,
        phon_threshold: float = 0.66,
        min_len: int = 4,
    ) -> None:
        self.ortho_threshold = ortho_threshold
        self.phon_threshold = phon_threshold
        self.min_len = min_len
        # entries grouped by word-count -> list of (canonical, norm, phon)
        self._by_len: dict[int, list[tuple[str, str, str]]] = {}
        self._max_words = 1
        seen: set[str] = set()

        def add(canonical: str) -> None:
            canonical = canonical.strip()
            if not canonical:
                return
            toks = canonical.split()
            norm = " ".join(toks).lower()
            if not norm or norm in seen:
                return
            seen.add(norm)
            nwords = len(toks)
            entry = (canonical, norm, _phrase_phonetic(canonical))
            self._by_len.setdefault(nwords, []).append(entry)
            self._max_words = max(self._max_words, nwords)

        for raw in words:
            if not raw:
                continue
            phrase = " ".join(str(raw).split())
            add(phrase)
            toks = phrase.split()
            if len(toks) > 1:
                for tok in toks:
                    cleaned = tok.strip("'\".,!?;:()[]{}")
                    if len(re.sub(r"[^A-Za-z0-9]", "", cleaned)) >= self.min_len:
                        add(cleaned)

    @property
    def size(self) -> int:
        """Number of distinct vocabulary entries (phrases + split words)."""
        return sum(len(v) for v in self._by_len.values())

    def is_empty(self) -> bool:
        return self._max_words == 1 and not self._by_len

    def _best_match(self, candidate: str, nwords: int) -> Optional[str]:
        """Return the canonical form to replace `candidate` with, or None."""
        entries = self._by_len.get(nwords)
        if not entries:
            return None
        cand_norm = candidate.lower()
        # Already exactly a vocab entry (ignoring case) -> never touch it. This
        # is what keeps a genuine common word like "sunny" (in "a sunny day")
        # from being re-cased when "Sunny" is a character.
        cand_phon = _phrase_phonetic(candidate)
        best_canon: Optional[str] = None
        best_score = 0.0
        for canonical, norm, phon in entries:
            if cand_norm == norm:
                return None
            score = _ratio(cand_norm, norm)
            # Phonetic equality is a strong signal but still gated by `score`
            # below, so short 2-char keys ("BM" for boon/Boone) are safe: a
            # genuinely different word ("bun") fails the orthographic gate.
            phon_ok = bool(phon) and phon == cand_phon and len(phon.replace(" ", "")) >= 2
            accept = score >= self.ortho_threshold or (phon_ok and score >= self.phon_threshold)
            if not accept:
                continue
            # Prefer the highest orthographic score; phonetic ties break upward.
            ranked = score + (0.05 if phon_ok else 0.0)
            if ranked > best_score:
                best_score = ranked
                best_canon = canonical
        return best_canon

    def correct(self, text: str) -> str:
        """Return `text` with near-miss vocabulary spans snapped to canonical.

        Longest phrases win (left-to-right, greedy), so "sunny smiles" is fixed
        as a unit before its individual words are considered.
        """
        if not text or self.is_empty():
            return text
        # Split keeping separators: odd indices are word tokens.
        parts = re.split(r"([A-Za-z0-9']+)", text)
        word_idx = [i for i in range(1, len(parts), 2)]
        words = [parts[i] for i in word_idx]
        n = len(words)
        result = list(parts)
        i = 0
        while i < n:
            replaced = False
            upper = min(self._max_words, n - i)
            for wlen in range(upper, 0, -1):
                span = words[i:i + wlen]
                candidate = " ".join(span)
                if wlen == 1:
                    bare = re.sub(r"[^A-Za-z0-9]", "", candidate)
                    if len(bare) < self.min_len or bare.lower() in _COMMON_WORDS:
                        continue
                canonical = self._best_match(candidate, wlen)
                if canonical is None:
                    continue
                # Write the canonical into the first word slot, blank the rest
                # (and the separators between them); canonical carries its own
                # internal spacing.
                first_part = word_idx[i]
                last_part = word_idx[i + wlen - 1]
                result[first_part] = canonical
                for p in range(first_part + 1, last_part + 1):
                    result[p] = ""
                i += wlen
                replaced = True
                break
            if not replaced:
                i += 1
        return "".join(result)


def build_corrector(words: Iterable[str]) -> VocabCorrector:
    """Convenience factory (kept separate so callers can catch build errors)."""
    return VocabCorrector(words)
