import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  configApi,
  modelsApi,
  type ModelDto,
  type RetrievalConfig,
} from "@/lib/api";
import {
  SettingsPage,
  SegmentedControl,
  ToggleRow,
} from "@/components/ui/settings-page";
import { ModelPicker, type ModelCard } from "@/components/ModelPicker";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, FormRow, Section } from "@/components/ui/page";

// Retrieval settings — picks the embedder (ModelPicker) AND exposes the RAG
// tuning the legacy settings page saved: master + per-source toggles, reranker
// tier/toggle, execution provider, and the recall/score knobs. The config form
// hits /api/ui/v1/config/retrieval, which reuses the legacy apply_retrieval_form
// path (the embedder tier stays the picker's job and is left untouched here).

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

function SliderRow({
  label,
  help,
  value,
  onChange,
}: {
  label: string;
  help?: string;
  value: number;
  onChange: (v: number) => void;
}) {
  const v = Number.isFinite(value) ? value : 0;
  return (
    <FormRow
      label={label}
      help={help}
      control={
        <span className="flex w-56 items-center gap-3">
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={v}
            onChange={(e) => onChange(Number(e.target.value))}
            className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-[var(--color-ink-700)] accent-[var(--color-accent)]"
            aria-label={label}
          />
          <Field
            type="number"
            className="w-[4.5rem] text-right"
            value={v}
            step={0.01}
            min={0}
            max={1}
            onChange={(e) => onChange(Number(e.target.value))}
          />
        </span>
      }
    />
  );
}

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

export function Retrieval() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["models", "retrieval"],
    queryFn: () => modelsApi.get("retrieval"),
  });
  const config = useQuery({
    queryKey: ["config", "retrieval"],
    queryFn: () => configApi.get("retrieval"),
  });

  const select = useMutation({
    mutationFn: (id: string) => modelsApi.select("retrieval", id),
    onSuccess: (fresh) => qc.setQueryData(["models", "retrieval"], fresh),
  });
  const download = useMutation({
    mutationFn: (id: string) => modelsApi.download("retrieval", id),
    onSuccess: (fresh) => qc.setQueryData(["models", "retrieval"], fresh),
  });

  const initial = config.data?.retrieval;
  const [form, setForm] = useState<RetrievalConfig | null>(initial ?? null);
  const [justSaved, setJustSaved] = useState(false);
  useEffect(() => setForm(initial ?? null), [initial]);

  const dirty = useMemo(
    () => !!form && !!initial && JSON.stringify(form) !== JSON.stringify(initial),
    [form, initial],
  );

  const save = useMutation({
    mutationFn: (body: RetrievalConfig) => configApi.saveRetrieval(body),
    onSuccess: (fresh) => {
      qc.setQueryData(["config", "retrieval"], fresh);
      setJustSaved(true);
      window.setTimeout(() => setJustSaved(false), 2200);
    },
  });

  const set = <K extends keyof RetrievalConfig>(
    key: K,
    value: RetrievalConfig[K],
  ) => setForm((f) => (f ? { ...f, [key]: value } : f));

  // Split the retrieval catalog into its two kinds (the backend tags each card's
  // meta with Kind: Embedder / Reranker). You need one of each.
  const allModels = query.data?.models ?? [];
  const kindOf = (m: ModelDto) =>
    m.meta?.find((x) => x.label === "Kind")?.value;
  const embedders = allModels.filter((m) => kindOf(m) === "Embedder").map(toCard);
  const rerankers = allModels.filter((m) => kindOf(m) === "Reranker").map(toCard);

  return (
    <SettingsPage
      eyebrow="AI"
      title="Retrieval"
      description="The embedding model that powers semantic lookup of lore, quests and the spawn catalog."
      save={
        form
          ? {
              dirty,
              saving: save.isPending,
              error: save.isError,
              justSaved,
              onReset: () => initial && setForm(initial),
              onSave: () => form && save.mutate(form),
              saveLabel: "Save retrieval",
            }
          : undefined
      }
    >
      <ModelPicker
        title="Embedder"
        description="Turns text into vectors for semantic search. Download one and select it."
        models={embedders}
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
            <CardTitle>Cutoff</CardTitle>
            <CardDescription>
              How closely an entry must match the player&apos;s words before
              it&apos;s injected into the prompt. Higher = stricter (less junk,
              but loose phrasings may stop matching). Applies on the next turn —
              no restart needed.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-5">
            <SliderRow
              label="Lore, quest & memory cutoff"
              help="Semantic hits below this score are dropped."
              value={form.min_score}
              onChange={(v) => set("min_score", v)}
            />
            <SliderRow
              label="Action cutoff"
              help="Separate floor for vector-matched actions (terse commands score lower)."
              value={form.action_min_score}
              onChange={(v) => set("action_min_score", v)}
            />
          </CardContent>
        </Card>
      )}

      <ModelPicker
        title="Reranker"
        description="Re-scores the top candidates for better precision. Download one, then enable it and pick its tier below."
        models={rerankers}
        isLoading={query.isLoading}
        isError={query.isError}
        downloadingId={download.isPending ? download.variables : null}
        onDownload={(id) => download.mutate(id)}
      />

      {form && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Sources</CardTitle>
              <CardDescription>
                The master switch and which content types are searched
                semantically.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <ToggleRow
                label="Enable semantic retrieval"
                help="When off, no retriever loads and consumers skip retrieval entirely."
                checked={form.enabled}
                onChange={(v) => set("enabled", v)}
              />
              <ToggleRow
                label="Chat memory"
                checked={form.chat_memory_enabled}
                onChange={(v) => set("chat_memory_enabled", v)}
              />
              <ToggleRow
                label="Lore"
                checked={form.lore_semantic_enabled}
                onChange={(v) => set("lore_semantic_enabled", v)}
              />
              <ToggleRow
                label="Actions"
                checked={form.action_semantic_enabled}
                onChange={(v) => set("action_semantic_enabled", v)}
              />
              <ToggleRow
                label="Quests"
                checked={form.quest_semantic_enabled}
                onChange={(v) => set("quest_semantic_enabled", v)}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Reranker &amp; execution</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-5">
              <ToggleRow
                label="Enable reranker"
                help="A cross-encoder re-scores recall candidates. Off by default (overkill for small corpora)."
                checked={form.reranker_enabled}
                onChange={(v) => set("reranker_enabled", v)}
              />
              <Section title="Reranker tier">
                <SegmentedControl
                  layoutId="retrieval-reranker-tier"
                  value={form.reranker_tier}
                  onChange={(v) => set("reranker_tier", v)}
                  options={[
                    { value: "small", label: "Small" },
                    { value: "large", label: "Large" },
                  ]}
                />
              </Section>
              <Section
                title="Execution"
                description="GPU falls back to CPU if CUDA is unavailable."
              >
                <SegmentedControl
                  layoutId="retrieval-execution"
                  value={form.execution}
                  onChange={(v) => set("execution", v)}
                  options={[
                    { value: "cpu", label: "CPU" },
                    { value: "gpu", label: "GPU (CUDA)" },
                  ]}
                />
              </Section>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recall &amp; scoring</CardTitle>
              <CardDescription>
                How many candidates are considered, how many survive, and the
                score floors a hit must clear.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <NumberRow
                label="Top-k (final)"
                help="Results returned to the prompt after reranking."
                value={form.top_k}
                onChange={(v) => set("top_k", v)}
                step={1}
                min={1}
                max={50}
              />
              <NumberRow
                label="Candidates"
                help="Recall candidates considered before reranking."
                value={form.candidates}
                onChange={(v) => set("candidates", v)}
                step={1}
                min={1}
                max={500}
              />
              <NumberRow
                label="Chat-memory limit"
                help="Max hits chat memory may contribute."
                value={form.chat_memory_limit}
                onChange={(v) => set("chat_memory_limit", v)}
                step={1}
                min={0}
                max={50}
              />
              <NumberRow
                label="Lore limit"
                help="Max hits lore may contribute."
                value={form.lore_limit}
                onChange={(v) => set("lore_limit", v)}
                step={1}
                min={0}
                max={50}
              />
              <NumberRow
                label="Quest limit"
                help="Max hits quests may contribute."
                value={form.quest_limit}
                onChange={(v) => set("quest_limit", v)}
                step={1}
                min={0}
                max={50}
              />
            </CardContent>
          </Card>
        </>
      )}
    </SettingsPage>
  );
}
