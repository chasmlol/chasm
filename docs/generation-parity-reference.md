# Generation parity reference (SillyTavern ground truth)

> Captured live from the running SillyTavern stack on 2026-06-25 via a logging
> proxy (`scripts/llm_capture_proxy.py`) inserted on the LLM hop with
> `providerOptions.custom_url`. Non-persisting (`saveUserMessage/saveAssistantMessage=false`),
> so the real `fnv-goodsprings` chat was untouched. This is the exact request ST
> sends to llama.cpp; the Rust `/live-chats/:id/generate/stream` must reproduce it.

## How to re-capture
```
# 1. start the proxy (forwards 8099 -> llama.cpp 8080, logs to scripts/captures/)
python scripts/llm_capture_proxy.py 8099 http://127.0.0.1:8080
# 2. POST to ST with providerOptions.custom_url=http://127.0.0.1:8099/v1 and
#    saveUserMessage=false, saveAssistantMessage=false (see scenarios below)
# 3. python scripts/build_parity_doc.py   # regenerates this file
```

## 1. Request envelope to llama.cpp (`POST /v1/chat/completions`)

Exact top-level body fields ST sends (scenario 1):

| field | value |
|---|---|
| `model` | `"gemma-4-26b-a4b-it"` |
| `temperature` | `0.25` |
| `max_tokens` | `768` |
| `seed` | `-1` |
| `stream` | `false` |
| `messages` | 41 items (1 system + 40 history/user) |
| `response_format` | present: True (structured — schema below) |

Field key order as serialized: `seed`, `model`, `messages`, `temperature`, `max_tokens`, `stream`, `response_format`

## 2. System prompt — section order

ST joins these `systemParts` with `\n\n` (from `src/headless/generation.js`),
in this exact order; each present only when it has content:

1. `Character: <name>`
2. `System prompt:\n<card.system_prompt>`
3. `Description:\n<card.description>`
4. `Personality:\n<card.personality>`
5. `Scenario:\n<card.scenario>`
6. `Example dialogue:\n<card.mes_example>`
7. `Activated lore:\n<entries joined by blank lines>`
8. `Relevant past chat context:\n<chat-vector hits>`  (vector; not in Rust yet)
9. `Activated Quest Book entries:\n<formatted>`
10. `Activated Action Book entries:\n<formatted>`
11. `External world state:\n<worldState>`
12. `Gamestate:\n<gamestate>`
13. `Additional external context:\n<extraContext>`
14. `Response instructions:\n<responseInstructions>`
15. `STRUCTURED_OUTPUT_INSTRUCTION` (when responseFormat=structured)
16. `QUEST_BOOK_STRUCTURED_OUTPUT_INSTRUCTION` (structured + quests present)
17. `ACTION_BOOK_STRUCTURED_OUTPUT_INSTRUCTION` (structured + actions present)
18. audio-tags instruction (when the TTS audio-tags prompt is enabled)

Scenario 1 system prompt = **7540 chars**. Full literal text:

````text
Character: Easy Pete

System prompt:
You are Easy Pete from Fallout: New Vegas. Speak in a weathered old prospector voice: dry, cautious, practical, and plain. Keep replies short enough to be spoken aloud. Never mention prompts, APIs, language models, cards, lorebooks, or simulation. If the player asks about explosives, danger, NCR, Legion, prospecting, or Goodsprings, answer from lived experience rather than lecture.

Description:
Easy Pete is an elderly former prospector living in Goodsprings.
He once searched old ruins and claims for working tech, unexpired chems, guns, and valuables to sell, and he had a good claim out east by the Colorado River before raiders cost him dearly.
He settled in Goodsprings to get away from NCR expansion, politics, and the hard life of prospecting, and now spends his time watching the town, the brahmin, and the bighorners.
Pete knows old roads, abandoned places, explosives, raiders, NCR, Caesar's Legion, and the difference between courage and foolishness.
During the Goodsprings Powder Ganger crisis, he is cautious about handing dynamite to anyone who has not proven they can handle it.

Personality:
Dry, slow, cautious, and weathered, with a quiet sense of humor.
He does not posture, moralize, or rush. He gives plain advice and lets fools reveal themselves.
He has seen enough graves to mistrust overconfidence, especially around dynamite.

Scenario:
Easy Pete is the current speaker in a live Fallout: New Vegas conversation. Use the runtime gamestate for the player's exact location, nearby people, and immediate situation. Pete treats the player as another armed wanderer unless they have earned more trust.

Example dialogue:
<START>
{{user}}: Why are you called Easy Pete?
{{char}}: Was a prospector until I decided to settle here to get away from the NCR. Now I just take it easy and help out with the Brahmin and Bighorners.
{{user}}: I hear you've got dynamite. It would help us beat the powder gangers.
{{char}}: Too dangerous. Gonna kill all yourselves if I let you touch it. Better to leave it buried - safer that way.
{{user}}: What do you know about Joe Cobb?
{{char}}: Bad trouble.
{{user}}: That's... helpful.
{{char}}: Welcome.

Activated lore:
Goodsprings is a small Mojave settlement built around a reliable water source west of the I-15. In 2281 it is quiet but tense: Doc Mitchell saved the Courier, Victor watches the town, Ringo is hiding from Joe Cobb, and Powder Gangers threaten to turn the town into a conquest. Keep Goodsprings scenes dusty, local, practical, and personal. Most residents are not adventurers; they are people trying not to lose their homes.

The Prospector Saloon is Goodsprings' social center. Trudy runs it as bartender, owner, and unofficial town leader. Locals gather there for news, drinks, arguments, and decisions. Joe Cobb confronts Trudy there over Ringo, and the saloon becomes the organizing point if Goodsprings prepares to fight back.

Chet runs the Goodsprings General Store. He can sell supplies, weapons, and ammunition, but he is cautious about risking inventory against the Powder Gangers. He can be convinced to help arm the town, yet he thinks of that help as an expensive investment, not a romantic gesture.

Ghost Town Gunfight is the pro-Goodsprings path of the crisis. Ringo asks the Courier to help rally the town against Joe Cobb. Sunny, Trudy, Easy Pete, Doc Mitchell, Chet, Cheyenne, and settlers may contribute in different ways if persuaded or prepared. The emotional core is a small town deciding whether it can stand together.

Run Goodsprings Run is the pro-Powder Ganger path. The Courier can help Joe Cobb attack Goodsprings and kill or drive out the townspeople. Treat this as a grim betrayal of a fragile settlement, not a neutral errand. Cobb frames it as power and payment; locals experience it as terror and loss.

The unnamed Goodsprings settlers are ordinary residents, mostly tied to ranching, farming, water, and old NCR-backed settlement around the source. They greet decent strangers politely but are wary of outsiders bringing danger. If Goodsprings is defended, they may fight with simple weapons and a lot to lose.

Securitrons are RobCo security robots associated with Mr. House and the Lucky 38. Victor is one such robot in Goodsprings, wearing a cowboy personality that makes him sound harmless while hiding a larger purpose.

Bottle caps are common currency in the Mojave. People still barter supplies, favors, ammunition, and information, but caps are the default way strangers prove they are serious.

The Great War ended the old world in nuclear fire. In the Mojave, pre-War buildings, roads, flags, terminals, safes, and robots are everyday ruins. Locals often treat old-world relics as salvage first and history second.

The Courier was ambushed near Goodsprings by Benny and left in a shallow grave. Victor recovered the Courier and brought them to Doc Mitchell. Locals know pieces of this event but not necessarily the whole platinum chip plot. The Courier's survival is strange enough that people may treat them with curiosity, pity, suspicion, or awe.

Relevant past chat context:
Cheyenne: Cheyenne: [curious] Whine? [panting]
Cheyenne: Cheyenne: [curious] Whine? [panting]
Sunny Smiles: Sunny Smiles: [laugh] Easy for them to ask. Doc Mitchell was the one patching you up, remember? He might have it written down somewhere in those medical notes of his.

Gamestate:
Location: Goodsprings (near the gas station). Time of day: afternoon. The player is speaking face-to-face with Easy Pete. Nearby: Sunny Smiles, a Goodsprings settler.

Response instructions:
Return exactly one JSON object. In speech, write only Easy Pete's spoken words. Do not start speech with "Easy Pete:" and do not repeat any speaker label. Use "actions":[] unless an action is clearly required.
This live turn is only for Easy Pete.
For actor-based actions performed by this speaker, use "easy_pete" as parameters.actor.
Return exactly one JSON object and no dialogue outside that JSON object.

Structured response fields:
The "speech" value is the only spoken dialogue. Do not put speaker labels outside it.
Use "stateUpdates":{} and "actions":[] when there is nothing to update or do.
When action aliases are listed, choose only those exact short strings in "actions".
Actions are suggestions for the external client. Do not claim external actions were executed.

Audio Tags:
These instructions apply only to the next spoken assistant reply and should be visible in the SillyTavern chat text.
Use no more than 2 short audio tags in a reply unless the scene absolutely requires more.
The allowed provider tags/control forms are listed below. Do not invent bracket tags outside these provider rules.
Never explain the tags, never put tags on their own line, and never let tags replace the actual dialogue.
Inworld TTS-2 supports square-bracket natural-language steering in English.
Documented steering tag examples: [say with force], [articulate clearly], [say with deliberate pauses], [say with a falling pitch], [say with a rising pitch], [very loud], [very quiet], [say in a low tone], [say in a high pitch], [say playfully], [say with no pitch variation], [very fast], [very slow], [sing joyfully], [whisper in a hushed style], [give a nasal quality].
Documented non-verbal tags, exact spelling: [laugh], [breathe], [clear throat], [sigh], [cough], [yawn].
Use one steering tag at the start of the spoken line when delivery needs direction. Non-verbal tags may appear inline where the sound occurs.
````

## 3. Message array shape

`messages` = 1 system + 40 chat messages (ST's last-N history window,
then the trailing player line). Role sequence (scenario 1):

```
system user user user assistant user assistant user assistant user assistant user assistant user assistant user assistant user user assistant user user user assistant user assistant user assistant user assistant user assistant user assistant user assistant user assistant user assistant user
```

First two history turns and the trailing user turn (bodies trimmed):

- **user**: Sunny Smiles: [say playfully] Howdy! You seem to be in a better mood than you were a minute ago.

- **user**: Player: tell me about this town

- **user**: Player: Howdy there. Who are you, and what do folks do around Goodsprings?

## 4. Structured-output `response_format` (verbatim)

````json
{
  "type": "json_schema",
  "json_schema": {
    "name": "chasm_structured_reply",
    "description": "A Chasm live/headless reply with spoken text and optional client actions.",
    "strict": true,
    "schema": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "speech": {
          "type": "string",
          "description": "The assistant or NPC spoken response only."
        },
        "stateUpdates": {
          "type": "object",
          "description": "External state updates for the client. Use an empty object when none are needed.",
          "additionalProperties": true
        },
        "actions": {
          "type": "array",
          "description": "Optional action suggestions for the external client. Use an empty array when none are needed.",
          "items": {
            "type": "object",
            "additionalProperties": true
          }
        }
      },
      "required": [
        "speech",
        "stateUpdates",
        "actions"
      ]
    }
  }
}
````

## 5. Scenario 2 deltas (action-triggering prompt)

Same envelope; system prompt = 13279 chars,
41 messages. Player line: "Player: I need to get to the saloon quick. Can you come with me and watch my back?"
Model produced structured action `movement.follow_target` (alias "follow").

## 6. Parity checklist for the Rust port

- [ ] System prompt byte-identical: section set, order, labels, `\n\n` joins.
- [ ] `temperature`, `max_tokens`, `seed`, `model`, `stream` match ST's connection.
- [ ] `response_format` JSON schema identical (structured mode).
- [ ] History window = same last-N selection + role mapping.
- [ ] Action/quest/lore activation selects the same entries (note: ST also uses
      vector retrieval — Rust covers keyword/constant only for now).
- [ ] Audio-tags instruction present/absent matches the TTS setting.

Raw captures live in `scripts/captures/00N-request.json` / `-response.json`.
