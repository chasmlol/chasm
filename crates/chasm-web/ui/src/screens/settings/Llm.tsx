import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { configApi, modelsApi, type LlmConfig, type ModelDto } from "@/lib/api";
import { SettingsPage } from "@/components/ui/settings-page";
import { ModelPicker, type ModelCard } from "@/components/ModelPicker";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, FormRow } from "@/components/ui/page";

// LLM settings — picks the language model (ModelPicker) AND exposes the
// per-request generation sampling the legacy settings page saved (temperature,
// top-p, top-k, min-p, repeat penalty, max tokens, n_ctx, seed). The config form
// hits /api/ui/v1/config/llm, which reuses the legacy apply_llm_form path so the
// values normalize + take effect on the next turn exactly as before.

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

/** A labeled number input row, shared by the config fields below. */
function NumberRow({
  label,
  help,
  value,
  onChange,
  step,
  min,
  max,
}: {
  label: string;
  help?: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <FormRow
      label={label}
      help={help}
      control={
        <Field
          type="number"
          className="w-28 text-right"
          value={Number.isFinite(value) ? value : 0}
          step={step}
          min={min}
          max={max}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      }
    />
  );
}

export function Llm() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["models", "llm"],
    queryFn: () => modelsApi.get("llm"),
  });
  const config = useQuery({
    queryKey: ["config", "llm"],
    queryFn: () => configApi.get("llm"),
  });

  const select = useMutation({
    mutationFn: (id: string) => modelsApi.select("llm", id),
    onSuccess: (fresh) => qc.setQueryData(["models", "llm"], fresh),
  });
  const download = useMutation({
    mutationFn: (id: string) => modelsApi.download("llm", id),
    onSuccess: (fresh) => qc.setQueryData(["models", "llm"], fresh),
  });

  const initial = config.data?.llm;
  const [form, setForm] = useState<LlmConfig | null>(initial ?? null);
  const [justSaved, setJustSaved] = useState(false);
  useEffect(() => setForm(initial ?? null), [initial]);

  const dirty = useMemo(
    () => !!form && !!initial && JSON.stringify(form) !== JSON.stringify(initial),
    [form, initial],
  );

  const save = useMutation({
    mutationFn: (body: LlmConfig) => configApi.saveLlm(body),
    onSuccess: (fresh) => {
      qc.setQueryData(["config", "llm"], fresh);
      setJustSaved(true);
      window.setTimeout(() => setJustSaved(false), 2200);
    },
  });

  const set = <K extends keyof LlmConfig>(key: K, value: LlmConfig[K]) =>
    setForm((f) => (f ? { ...f, [key]: value } : f));

  return (
    <SettingsPage
      eyebrow="AI"
      title="Language model"
      description="The model that drives NPC dialogue and decisions. Served locally by koboldcpp."
      save={
        form
          ? {
              dirty,
              saving: save.isPending,
              error: save.isError,
              justSaved,
              onReset: () => initial && setForm(initial),
              onSave: () => form && save.mutate(form),
              saveLabel: "Save sampling",
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
            <CardTitle>Generation sampling</CardTitle>
            <CardDescription>
              Forwarded to koboldcpp on every NPC / admin turn. Read fresh per
              request, so a change applies on the next line — no restart.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <NumberRow
              label="Temperature"
              help="Higher = more varied, lower = more deterministic (0–2)."
              value={form.temperature}
              onChange={(v) => set("temperature", v)}
              step={0.05}
              min={0}
              max={2}
            />
            <NumberRow
              label="Top-p"
              help="Nucleus sampling cutoff (0–1). 1 disables it."
              value={form.top_p}
              onChange={(v) => set("top_p", v)}
              step={0.01}
              min={0}
              max={1}
            />
            <NumberRow
              label="Top-k"
              help="Top-k cutoff. 0 disables it."
              value={form.top_k}
              onChange={(v) => set("top_k", v)}
              step={1}
              min={0}
              max={200}
            />
            <NumberRow
              label="Min-p"
              help="Min-p cutoff (0–1). 0 disables it."
              value={form.min_p}
              onChange={(v) => set("min_p", v)}
              step={0.01}
              min={0}
              max={1}
            />
            <NumberRow
              label="Repeat penalty"
              help="Penalizes repeated tokens. 1 is off (0–2)."
              value={form.repeat_penalty}
              onChange={(v) => set("repeat_penalty", v)}
              step={0.01}
              min={0}
              max={2}
            />
            <NumberRow
              label="Max tokens"
              help="Cap on generated tokens. 0 = no limit."
              value={form.max_tokens}
              onChange={(v) => set("max_tokens", v)}
              step={1}
              min={0}
              max={8192}
            />
            <NumberRow
              label="Context size (n_ctx)"
              help="Context-window hint. 0 = the loaded model's default."
              value={form.n_ctx}
              onChange={(v) => set("n_ctx", v)}
              step={512}
              min={0}
              max={131072}
            />
            <NumberRow
              label="Seed"
              help="RNG seed. -1 = random each call."
              value={form.seed}
              onChange={(v) => set("seed", v)}
              step={1}
              min={-1}
            />
          </CardContent>
        </Card>
      )}
    </SettingsPage>
  );
}
