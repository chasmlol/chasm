import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { configApi, modelsApi, type LlmConfig } from "@/lib/api";
import { useProvider } from "@/lib/useProvider";
import { SettingsPage } from "@/components/ui/settings-page";
import { ProviderPicker } from "@/components/ProviderPicker";
import { ApiProviderConfig } from "@/components/ApiProviderConfig";
import { RuntimeStatus } from "@/components/RuntimeStatus";
import { ModelPlacement } from "@/components/ModelPlacement";
import { InstalledModelSelector } from "@/components/InstalledModelSelector";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, FormRow, Stack } from "@/components/ui/page";

// LLM settings — picks the provider (local llama.cpp or a hosted API), guides
// manual placement of a recommended .gguf when local, or shows the hosted API
// config, and keeps the per-request generation sampling below (all providers).
// The sampling form hits /api/ui/v1/config/llm.

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
  const provider = useProvider("llm");

  // The recommended .gguf catalog + target folder for manual placement.
  const models = useQuery({
    queryKey: ["models", "llm"],
    queryFn: () => modelsApi.get("llm"),
  });

  const config = useQuery({
    queryKey: ["config", "llm"],
    queryFn: () => configApi.get("llm"),
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
      description="The model that drives NPC dialogue and decisions. Run it locally with llama.cpp, or connect a hosted API."
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
      <ProviderPicker
        providers={provider.view?.providers ?? []}
        selectedId={provider.selectedId}
        onSelect={(id) => provider.select.mutate(id)}
        selectingId={provider.select.isPending ? provider.select.variables : null}
        isLoading={provider.query.isLoading}
      />

      {provider.isLocal ? (
        provider.view && (
          <Stack>
            <RuntimeStatus runtime={provider.view.local_runtime} />
            <InstalledModelSelector
              domain="llm"
              models={models.data?.models ?? []}
              selectedId={models.data?.selected_id}
              isLoading={models.isLoading}
              isError={models.isError}
              emptyTitle="No model installed yet"
              emptyDescription="Place a recommended .gguf below to make it available, then it appears here to activate."
            />
            <ModelPlacement
              domain="llm"
              folderKind="llm"
              models={models.data?.models ?? []}
              folder={models.data?.folder}
              isLoading={models.isLoading}
              isError={models.isError}
            />
          </Stack>
        )
      ) : (
        provider.selectedProvider && (
          <ApiProviderConfig
            capability="llm"
            provider={provider.selectedProvider}
            onSave={(f) => provider.saveConfig.mutate(f)}
            saving={provider.saveConfig.isPending}
            error={provider.saveError}
            justSaved={provider.justSaved}
          />
        )
      )}

      {form && (
        <Card>
          <CardHeader>
            <CardTitle>Generation sampling</CardTitle>
            <CardDescription>
              Sent on every NPC / admin turn. Read fresh per request, so a change
              applies on the next line — no restart.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <FormRow
              label="Constrain action verbs (experiment)"
              help="Locks each NPC action verb to the action book's aliases + verb lexicon at sampling time (grammar enum). Off = free verb, corrected by the resolver."
              control={
                <input
                  type="checkbox"
                  className="h-4 w-4 accent-primary"
                  checked={form.npc_action_enum}
                  onChange={(e) => set("npc_action_enum", e.target.checked)}
                />
              }
            />
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
