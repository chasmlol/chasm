import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";

import { configApi, modelsApi, type SttConfig } from "@/lib/api";
import { useProvider } from "@/lib/useProvider";
import { SettingsPage } from "@/components/ui/settings-page";
import { ProviderPicker } from "@/components/ProviderPicker";
import { ApiProviderConfig } from "@/components/ApiProviderConfig";
import { RuntimeStatus } from "@/components/RuntimeStatus";
import { InstalledModelSelector } from "@/components/InstalledModelSelector";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Field,
  FormRow,
  Section,
  Stack,
  TextArea,
} from "@/components/ui/page";

// STT settings — picks the provider (local Parakeet engine or a hosted API) and
// keeps the language / biasing prompt / request timeout config. The config form
// hits /api/ui/v1/config/stt.

// The local STT model selector — mirrors the LLM active-model list. The managed
// Parakeet engine is the only local STT today, so this shows one entry marked
// Active, but it's built as a general selector so multiple would work. Selecting
// a card POSTs /models/stt/select {id}. Below it, an Install-on-Runtimes link
// for when the engine isn't installed yet.
function SttModelSelector() {
  const stt = useQuery({
    queryKey: ["models", "stt"],
    queryFn: () => modelsApi.get("stt"),
  });
  const models = stt.data?.models ?? [];
  const anyInstalled = models.some((m) => m.installed);

  return (
    <Section
      title="Voice input"
      description="Voice input is transcribed locally by the Parakeet engine, which runs on its own port. Pick which installed model is active."
    >
      <InstalledModelSelector
        domain="stt"
        title=""
        description=""
        models={models}
        selectedId={stt.data?.selected_id}
        isLoading={stt.isLoading}
        isError={stt.isError}
        emptyTitle="Parakeet engine not installed"
        emptyDescription="Install the Parakeet STT engine on the Runtimes page, then it appears here to activate."
      />
      {!anyInstalled && !stt.isLoading && (
        <div className="mt-2 flex justify-end">
          <Link
            to="/settings/runtimes"
            className="inline-flex items-center gap-1 text-[13px] font-medium text-[var(--color-accent)] hover:underline"
          >
            Install on Runtimes <ArrowRight className="size-3.5" />
          </Link>
        </div>
      )}
    </Section>
  );
}

export function Stt() {
  const qc = useQueryClient();
  const provider = useProvider("stt");

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

  return (
    <SettingsPage
      eyebrow="AI"
      title="Speech-to-text"
      description="Transcribes your microphone for the player's turn. Run it locally with Parakeet, or connect a hosted API."
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
            <SttModelSelector />
          </Stack>
        )
      ) : (
        provider.selectedProvider && (
          <ApiProviderConfig
            capability="stt"
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
            <CardTitle>Transcription</CardTitle>
            <CardDescription>
              Defaults used when a request doesn't supply its own. Read fresh per
              request — no restart.
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
              help="A hint that biases decoding toward expected vocabulary. Blank for none."
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
