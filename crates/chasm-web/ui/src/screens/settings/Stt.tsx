import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { configApi, modelsApi, type ModelDto, type SttConfig } from "@/lib/api";
import { SettingsPage } from "@/components/ui/settings-page";
import { ModelPicker, type ModelCard } from "@/components/ModelPicker";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, FormRow, TextArea } from "@/components/ui/page";

// STT settings — picks the whisper model (ModelPicker) AND exposes the
// language / biasing prompt / request timeout the legacy settings page saved.
// The config form hits /api/ui/v1/config/stt, which reuses the legacy
// apply_stt_form path (trim + timeout clamp); the prompt biases koboldcpp's
// whisper decoding, the timeout bounds the transcription POST.

function toCard(dto: ModelDto): ModelCard {
  return {
    id: dto.id,
    name: dto.name,
    description: dto.description,
    installed: dto.installed,
    recommended: dto.recommended,
    meta: dto.meta,
    status: dto.status,
  };
}

export function Stt() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["models", "stt"],
    queryFn: () => modelsApi.get("stt"),
  });
  const config = useQuery({
    queryKey: ["config", "stt"],
    queryFn: () => configApi.get("stt"),
  });

  const select = useMutation({
    mutationFn: (id: string) => modelsApi.select("stt", id),
    onSuccess: (fresh) => qc.setQueryData(["models", "stt"], fresh),
  });
  const download = useMutation({
    mutationFn: (id: string) => modelsApi.download("stt", id),
    onSuccess: (fresh) => qc.setQueryData(["models", "stt"], fresh),
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

  return (
    <SettingsPage
      eyebrow="AI"
      title="Speech-to-text"
      description="The whisper model that transcribes your microphone for the player's turn."
      save={
        form
          ? {
              dirty,
              saving: save.isPending,
              error: save.isError,
              justSaved,
              onReset: () => initial && setForm(initial),
              onSave: () => form && save.mutate(form),
              saveLabel: "Save transcription",
            }
          : undefined
      }
    >
      <ModelPicker
        models={(query.data?.models ?? []).map(toCard)}
        selectedId={query.data?.selected_id}
        folder={query.data?.folder ? { path: query.data.folder } : undefined}
        isLoading={query.isLoading}
        isError={query.isError}
        downloadingId={download.isPending ? download.variables : null}
        onSelect={(id) => select.mutate(id)}
        onDownload={(id) => download.mutate(id)}
      />

      {form && (
        <Card>
          <CardHeader>
            <CardTitle>Transcription</CardTitle>
            <CardDescription>
              Defaults forwarded to koboldcpp's Whisper when a request doesn't
              supply its own. Read fresh per request — no restart.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <FormRow
              label="Language hint"
              help="An ISO code like en, or blank for auto-detect."
              control={
                <Field
                  className="w-40"
                  placeholder="auto"
                  value={form.language}
                  onChange={(e) => set("language", e.target.value)}
                />
              }
            />
            <FormRow
              stacked
              label="Biasing prompt"
              help="A hint that biases decoding toward expected vocabulary (forwarded as the OpenAI prompt field). Blank for none."
              control={
                <TextArea
                  rows={3}
                  placeholder="e.g. names, jargon, or proper nouns the model should expect"
                  value={form.prompt}
                  onChange={(e) => set("prompt", e.target.value)}
                />
              }
            />
            <FormRow
              label="Timeout (ms)"
              help="Per-request transcription timeout (1000–300000)."
              control={
                <Field
                  type="number"
                  className="w-32 text-right"
                  value={Number.isFinite(form.timeout_ms) ? form.timeout_ms : 0}
                  step={1000}
                  min={1000}
                  max={300000}
                  onChange={(e) => set("timeout_ms", Number(e.target.value))}
                />
              }
            />
          </CardContent>
        </Card>
      )}
    </SettingsPage>
  );
}
