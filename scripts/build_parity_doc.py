"""Generates docs/generation-parity-reference.md from the captured llama.cpp
requests/responses, so the Rust generation can be diffed against ST ground truth.
"""
import json
import os
import glob

RS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = os.path.join(RS, "scripts", "captures")
OUT = os.path.join(RS, "docs", "generation-parity-reference.md")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def load(n, kind):
    p = os.path.join(CAP, f"{n:03d}-{kind}.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


req1 = load(1, "request")
req2 = load(2, "request")
msgs1 = req1["messages"]
sys1 = msgs1[0]["content"]

# Role sequence + a couple of representative history turns (redact long bodies).
roles = [m["role"] for m in msgs1]
history = msgs1[1:]


def trim(s, n=400):
    s = s.replace("\r\n", "\n")
    return s if len(s) <= n else s[:n] + f"\n…[+{len(s)-n} chars]"


lines = []
w = lines.append
w("# Generation parity reference (SillyTavern ground truth)")
w("")
w("> Captured live from the running SillyTavern stack on 2026-06-25 via a logging")
w("> proxy (`scripts/llm_capture_proxy.py`) inserted on the LLM hop with")
w("> `providerOptions.custom_url`. Non-persisting (`saveUserMessage/saveAssistantMessage=false`),")
w("> so the real `fnv-goodsprings` chat was untouched. This is the exact request ST")
w("> sends to llama.cpp; the Rust `/live-chats/:id/generate/stream` must reproduce it.")
w("")
w("## How to re-capture")
w("```")
w("# 1. start the proxy (forwards 8099 -> llama.cpp 8080, logs to scripts/captures/)")
w("python scripts/llm_capture_proxy.py 8099 http://127.0.0.1:8080")
w("# 2. POST to ST with providerOptions.custom_url=http://127.0.0.1:8099/v1 and")
w("#    saveUserMessage=false, saveAssistantMessage=false (see scenarios below)")
w("# 3. python scripts/build_parity_doc.py   # regenerates this file")
w("```")
w("")
w("## 1. Request envelope to llama.cpp (`POST /v1/chat/completions`)")
w("")
w("Exact top-level body fields ST sends (scenario 1):")
w("")
w("| field | value |")
w("|---|---|")
for k in ("model", "temperature", "max_tokens", "seed", "stream"):
    w(f"| `{k}` | `{json.dumps(req1.get(k))}` |")
w(f"| `messages` | {len(msgs1)} items (1 system + {len(history)} history/user) |")
w(f"| `response_format` | present: {'response_format' in req1} (structured — schema below) |")
w("")
w("Field key order as serialized: `" + "`, `".join(req1.keys()) + "`")
w("")
w("## 2. System prompt — section order")
w("")
w("ST joins these `systemParts` with `\\n\\n` (from `src/headless/generation.js`),")
w("in this exact order; each present only when it has content:")
w("")
w("1. `Character: <name>`")
w("2. `System prompt:\\n<card.system_prompt>`")
w("3. `Description:\\n<card.description>`")
w("4. `Personality:\\n<card.personality>`")
w("5. `Scenario:\\n<card.scenario>`")
w("6. `Example dialogue:\\n<card.mes_example>`")
w("7. `Activated lore:\\n<entries joined by blank lines>`")
w("8. `Relevant past chat context:\\n<chat-vector hits>`  (vector; not in Rust yet)")
w("9. `Activated Quest Book entries:\\n<formatted>`")
w("10. `Activated Action Book entries:\\n<formatted>`")
w("11. `External world state:\\n<worldState>`")
w("12. `Gamestate:\\n<gamestate>`")
w("13. `Additional external context:\\n<extraContext>`")
w("14. `Response instructions:\\n<responseInstructions>`")
w("15. `STRUCTURED_OUTPUT_INSTRUCTION` (when responseFormat=structured)")
w("16. `QUEST_BOOK_STRUCTURED_OUTPUT_INSTRUCTION` (structured + quests present)")
w("17. `ACTION_BOOK_STRUCTURED_OUTPUT_INSTRUCTION` (structured + actions present)")
w("18. audio-tags instruction (when the TTS audio-tags prompt is enabled)")
w("")
w(f"Scenario 1 system prompt = **{len(sys1)} chars**. Full literal text:")
w("")
w("````text")
w(sys1)
w("````")
w("")
w("## 3. Message array shape")
w("")
w(f"`messages` = 1 system + {len(history)} chat messages (ST's last-N history window,")
w("then the trailing player line). Role sequence (scenario 1):")
w("")
w("```")
w(" ".join(roles))
w("```")
w("")
w("First two history turns and the trailing user turn (bodies trimmed):")
w("")
for m in history[:2] + history[-1:]:
    w(f"- **{m['role']}**: {trim(m['content'])}")
    w("")
w("## 4. Structured-output `response_format` (verbatim)")
w("")
w("````json")
w(json.dumps(req1.get("response_format", {}), indent=2, ensure_ascii=False))
w("````")
w("")
w("## 5. Scenario 2 deltas (action-triggering prompt)")
w("")
w(f"Same envelope; system prompt = {len(req2['messages'][0]['content'])} chars,")
w(f"{len(req2['messages'])} messages. Player line: "
  f"\"{trim(req2['messages'][-1]['content'], 200)}\"")
w("Model produced structured action `movement.follow_target` (alias \"follow\").")
w("")
w("## 6. Parity checklist for the Rust port")
w("")
w("- [ ] System prompt byte-identical: section set, order, labels, `\\n\\n` joins.")
w("- [ ] `temperature`, `max_tokens`, `seed`, `model`, `stream` match ST's connection.")
w("- [ ] `response_format` JSON schema identical (structured mode).")
w("- [ ] History window = same last-N selection + role mapping.")
w("- [ ] Action/quest/lore activation selects the same entries (note: ST also uses")
w("      vector retrieval — Rust covers keyword/constant only for now).")
w("- [ ] Audio-tags instruction present/absent matches the TTS setting.")
w("")
w("Raw captures live in `scripts/captures/00N-request.json` / `-response.json`.")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", OUT, f"({os.path.getsize(OUT)} bytes)")
