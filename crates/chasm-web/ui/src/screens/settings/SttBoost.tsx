import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { configApi, type SttConfig } from "@/lib/api";
import { SettingsPage, ToggleRow } from "@/components/ui/settings-page";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { FormRow, StatusPill } from "@/components/ui/page";

// STT word boosting — a dedicated page for the Parakeet custom-vocabulary
// feature. The vocabulary (character names + lore entry names/keys) is gathered
// live from the active profile and shipped with each transcription so the
// server snaps near-miss proper nouns to the real name ("sunny smells" ->
// "Sunny Smiles"). It ALWAYS tracks the current books — add a character or lore
// entry and it is picked up automatically, no refresh. Persists via the shared
// /api/ui/v1/config/stt endpoint (same SttConfig the STT page uses).

export function SttBoost() {
  const qc = useQueryClient();
  const config = useQuery({
    queryKey: ["config", "stt"],
    queryFn: () => configApi.get("stt"),
  });

  const initial = config.data?.stt;
  const [form, setForm] = useState<SttConfig | null>(initial ?? null);
  const [justSaved, setJustSaved] = useState(false);
  useEffect(() => setForm(initial ?? null), [initial]);

  const dirty = useMemo(
    () => !!form && !!initial && JSON.stringify(form) !== JSON.stringify(initial),
    [form, initial],
  );

  const save = useMutation({
    mutationFn: (body: SttConfig) => configApi.saveStt(body),
    onSuccess: (fresh) => {
      qc.setQueryData(["config", "stt"], fresh);
      setJustSaved(true);
      window.setTimeout(() => setJustSaved(false), 2200);
    },
  });

  const set = <K extends keyof SttConfig>(key: K, value: SttConfig[K]) =>
    setForm((f) => (f ? { ...f, [key]: value } : f));

  const total = form?.boosted_word_count ?? 0;
  const chars = form?.boosted_character_count ?? 0;
  const lore = form?.boosted_lore_count ?? 0;
  const sample = form?.boost_sample ?? [];

  return (
    <SettingsPage
      eyebrow="AI"
      title="Word boosting"
      description="Bias Parakeet speech-to-text toward your world's proper nouns so names come through right."
      save={
        form
          ? {
              dirty,
              saving: save.isPending,
              error: save.isError,
              justSaved,
              onReset: () => initial && setForm(initial),
              onSave: () => form && save.mutate(form),
              saveLabel: "Save boosting",
            }
          : undefined
      }
    >
      {form && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Custom vocabulary</CardTitle>
              <CardDescription>
                Character names and lore entry names are sent with each
                transcription so near-misses snap to the real name (e.g. "sunny
                smells" → "Sunny Smiles"). The list is rebuilt automatically
                whenever you edit the books — new characters and lore are always
                included. Parakeet only; no effect on Whisper.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <ToggleRow
                label="Enable word boosting"
                help="Master switch. When off, transcription is the raw Parakeet output."
                checked={form.boost_vocab}
                onChange={(v) => set("boost_vocab", v)}
              />
              <ToggleRow
                label="Boost character names"
                help={`Names from the Characters book — ${chars.toLocaleString()} available.`}
                checked={form.boost_characters}
                disabled={!form.boost_vocab}
                onChange={(v) => set("boost_characters", v)}
              />
              <ToggleRow
                label="Boost lore names"
                help={`Entry names + trigger keys from the Lore book — ${lore.toLocaleString()} available.`}
                checked={form.boost_lore}
                disabled={!form.boost_vocab}
                onChange={(v) => set("boost_lore", v)}
              />
              <FormRow
                label="Words boosted"
                help="Distinct names + sub-words currently sent (updates on save)."
                control={
                  <StatusPill tone={form.boost_vocab && total > 0 ? "ok" : "idle"}>
                    {total.toLocaleString()} words
                  </StatusPill>
                }
              />
            </CardContent>
          </Card>

          {sample.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Preview</CardTitle>
                <CardDescription>
                  A sample of the proper nouns being boosted right now.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1.5">
                  {sample.map((word) => (
                    <span
                      key={word}
                      className="inline-flex items-center rounded-full border border-[var(--border)] bg-[var(--color-ink-850)] px-2.5 py-1 text-xs text-[var(--muted-foreground)]"
                    >
                      {word}
                    </span>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </SettingsPage>
  );
}
