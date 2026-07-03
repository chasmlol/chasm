import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { Check, Loader2 } from "lucide-react";

import {
  configApi,
  modelsApi,
  systemApi,
  ttsApi,
  type ModelDto,
  type ProviderDto,
  type TtsConfig,
} from "@/lib/api";
import { useProvider } from "@/lib/useProvider";
import { SettingsPage } from "@/components/ui/settings-page";
import { ModelPicker, type ModelCard } from "@/components/ModelPicker";
import { ProviderPicker } from "@/components/ProviderPicker";
import { ApiProviderConfig } from "@/components/ApiProviderConfig";
import { RuntimeStatus } from "@/components/RuntimeStatus";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  EmptyState,
  Field,
  FormRow,
  Section,
  SectionLabel,
  Stack,
  StatusPill,
  type StatusTone,
} from "@/components/ui/page";

// TTS settings — picks the provider (local qwen3-tts / engine picker, or a
// hosted API), keeps the voice-cloning panel and the volumes + synthesis tuning
// config. The config form hits /api/ui/v1/config/tts.

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

/** A percent volume slider (0–200%, 100% = unity). */
function VolumeSlider({
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
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <SectionLabel>{label}</SectionLabel>
        <span className="font-mono text-[13px] text-[var(--color-accent)]">
          {Math.round(value)}%
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={200}
        step={5}
        value={Number.isFinite(value) ? value : 100}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-3 w-full accent-[var(--color-accent)]"
      />
      {help && (
        <p className="mt-1.5 text-[13px] text-[var(--muted-foreground)]">
          {help}
        </p>
      )}
    </div>
  );
}

// Voice cloning — clones each NPC's real in-game voice for the SELECTED engine.
// The active profile carries the character list + its own extractor; cloning runs
// the extractor (pull references from the game) then clones each with the engine,
// writing voices/<name>/<engine>/sample.wav which we play back here.
function VoiceCloning() {
  const qc = useQueryClient();
  const status = useQuery({
    queryKey: ["voice-clone"],
    queryFn: systemApi.voiceCloneStatus,
    refetchInterval: (q) => (q.state.data?.any_cloning ? 2000 : false),
  });
  const clone = useMutation({
    mutationFn: systemApi.voiceCloneStart,
    onSuccess: (v) => qc.setQueryData(["voice-clone"], v),
  });
  const v = status.data;

  const toneFor = (s: string): StatusTone =>
    s === "cloned"
      ? "ok"
      : s === "cloning"
        ? "busy"
        : s === "failed"
          ? "error"
          : "idle";

  return (
    <Card>
      <CardHeader>
        <CardTitle>Voice cloning</CardTitle>
        <CardDescription>
          Clone each NPC's real in-game voice for the selected engine
          {v?.engine_label ? ` (${v.engine_label})` : ""}. Runs the profile's own
          extractor to pull references from the game, then clones per character.
          Each engine clones separately.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {!v?.has_profile ? (
          <EmptyState
            title="No game profile"
            description="Connect the game so its profile + character list import, then clone."
          />
        ) : (
          <>
            <div className="flex items-center justify-between gap-3">
              <div className="text-[13px] text-[var(--muted-foreground)]">
                Profile:{" "}
                <span className="text-[var(--foreground)]">{v.profile_name}</span>{" "}
                · {v.cloned_count}/{v.characters.length} cloned
              </div>
              <Button
                size="sm"
                disabled={v.any_cloning || clone.isPending}
                onClick={() => clone.mutate()}
              >
                {v.any_cloning || clone.isPending ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Cloning…
                  </>
                ) : (
                  `Clone voices (${v.engine_label})`
                )}
              </Button>
            </div>
            <div className="flex flex-col gap-2">
              {v.characters.map((c) => (
                <div
                  key={c.name}
                  className="flex items-center justify-between gap-3 rounded-lg border border-[var(--border)] bg-[var(--color-ink-850)] px-3 py-2"
                >
                  <div className="flex min-w-0 items-center gap-2">
                    <StatusPill
                      tone={toneFor(c.status)}
                      pulse={c.status === "cloning"}
                    >
                      {c.status_label}
                    </StatusPill>
                    <span className="truncate text-sm">{c.name}</span>
                  </div>
                  {c.status === "cloned" && (
                    <audio
                      controls
                      preload="none"
                      className="h-8 max-w-[240px]"
                      src={`/voices/${encodeURIComponent(c.name)}/${v.engine_id}/sample.wav`}
                    />
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

// API voice cloning — the hosted-provider equivalent of the LOCAL VoiceCloning
// panel above. Shown when an API TTS provider is selected. It clones each
// character's recorded reference clip (the same one the local panel records)
// into the ACTIVE hosted provider's cloning API; the character then speaks in
// the returned cloned voice. Reuses the voice-clone status for the character
// list, and GET /tts/api-voices for which characters already have a cloned id.
function ApiVoiceCloning({ provider }: { provider: ProviderDto }) {
  const qc = useQueryClient();

  // Reuse the same source the local VoiceCloning panel uses for the character
  // list (the active profile's characters + reference-clip state).
  const status = useQuery({
    queryKey: ["voice-clone"],
    queryFn: systemApi.voiceCloneStatus,
  });

  // Which characters already have a cloned voice id for the active provider.
  const apiVoices = useQuery({
    queryKey: ["tts", "api-voices"],
    queryFn: ttsApi.listApiVoices,
  });

  // Per-character clone error (keyed by character name), surfaced readably.
  const [errors, setErrors] = useState<Record<string, string>>({});

  const clone = useMutation({
    mutationFn: (character: string) => ttsApi.cloneApiVoice(character),
    onSuccess: (res, character) => {
      if (res.ok) {
        setErrors((e) => {
          const { [character]: _drop, ...rest } = e;
          return rest;
        });
        // Refetch which characters have a cloned voice id now.
        qc.invalidateQueries({ queryKey: ["tts", "api-voices"] });
      } else {
        setErrors((e) => ({
          ...e,
          [character]: res.error ?? "Cloning failed.",
        }));
      }
    },
    onError: (err, character) =>
      setErrors((e) => ({
        ...e,
        [character]: (err as Error).message || "Cloning failed.",
      })),
  });

  const v = status.data;
  const clonedVoices = apiVoices.data?.voices ?? {};

  return (
    <Card>
      <CardHeader>
        <CardTitle>Clone character voices via {provider.name}</CardTitle>
        <CardDescription>
          Cloning sends each character's recorded reference clip (the same one
          the local voice panel records) to {provider.name}'s cloning API. The
          character then speaks in the cloned voice. Requires an API key set
          above and a recorded reference for the character.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {status.isLoading ? (
          <div className="grid place-items-center py-8 text-[var(--muted-foreground)]">
            <Loader2 className="size-5 animate-spin" />
          </div>
        ) : !v?.has_profile ? (
          <EmptyState
            title="No game profile"
            description="Connect the game so its profile + character list import, then clone."
          />
        ) : v.characters.length === 0 ? (
          <EmptyState
            title="No characters"
            description="This profile has no characters to clone yet."
          />
        ) : (
          <div className="flex flex-col gap-2">
            {v.characters.map((c) => {
              const voiceId = clonedVoices[c.name];
              const cloned = Boolean(voiceId);
              const busy = clone.isPending && clone.variables === c.name;
              const err = errors[c.name];
              return (
                <div
                  key={c.name}
                  className="flex flex-col gap-1.5 rounded-lg border border-[var(--border)] bg-[var(--color-ink-850)] px-3 py-2"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex min-w-0 items-center gap-2">
                      {cloned ? (
                        <StatusPill tone="ok">
                          <Check className="size-3.5" /> Cloned
                        </StatusPill>
                      ) : (
                        <StatusPill tone="idle">Not cloned</StatusPill>
                      )}
                      <span className="truncate text-sm">{c.name}</span>
                    </div>
                    <Button
                      size="sm"
                      variant={cloned ? "secondary" : "default"}
                      disabled={busy}
                      onClick={() => clone.mutate(c.name)}
                    >
                      {busy ? (
                        <>
                          <Loader2 className="size-4 animate-spin" /> Cloning…
                        </>
                      ) : cloned ? (
                        "Re-clone voice"
                      ) : (
                        "Clone voice"
                      )}
                    </Button>
                  </div>
                  {cloned && (
                    <p className="truncate font-mono text-[11px] text-[var(--muted-foreground)]">
                      voice id: {voiceId}
                    </p>
                  )}
                  {err && (
                    <p className="text-[13px] text-[var(--color-danger)]">
                      {err}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function Tts() {
  const qc = useQueryClient();
  const provider = useProvider("tts");

  const query = useQuery({
    queryKey: ["models", "tts"],
    queryFn: () => modelsApi.get("tts"),
  });
  const config = useQuery({
    queryKey: ["config", "tts"],
    queryFn: () => configApi.get("tts"),
  });

  const select = useMutation({
    mutationFn: (id: string) => modelsApi.select("tts", id),
    onSuccess: (fresh) => qc.setQueryData(["models", "tts"], fresh),
  });

  const initial = config.data?.tts;
  const [form, setForm] = useState<TtsConfig | null>(initial ?? null);
  const [justSaved, setJustSaved] = useState(false);
  useEffect(() => setForm(initial ?? null), [initial]);

  const dirty = useMemo(
    () => !!form && !!initial && JSON.stringify(form) !== JSON.stringify(initial),
    [form, initial],
  );

  const save = useMutation({
    mutationFn: (body: TtsConfig) => configApi.saveTts(body),
    onSuccess: (fresh) => {
      qc.setQueryData(["config", "tts"], fresh);
      setJustSaved(true);
      window.setTimeout(() => setJustSaved(false), 2200);
    },
  });

  const set = <K extends keyof TtsConfig>(key: K, value: TtsConfig[K]) =>
    setForm((f) => (f ? { ...f, [key]: value } : f));

  return (
    <SettingsPage
      eyebrow="AI"
      title="Text-to-speech"
      description="The voice that speaks NPC lines. Run it locally with the managed engine, or connect a hosted API."
      save={
        form
          ? {
              dirty,
              saving: save.isPending,
              error: save.isError,
              justSaved,
              onReset: () => initial && setForm(initial),
              onSave: () => form && save.mutate(form),
              saveLabel: "Save tuning",
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
            <ModelPicker
              title="Engine"
              models={(query.data?.models ?? []).map(toCard)}
              selectedId={query.data?.selected_id}
              folder={
                query.data?.folder ? { path: query.data.folder } : undefined
              }
              isLoading={query.isLoading}
              isError={query.isError}
              onSelect={(id) => select.mutate(id)}
            />
            <VoiceCloning />
          </Stack>
        )
      ) : (
        provider.selectedProvider && (
          <Stack>
            <ApiProviderConfig
              capability="tts"
              provider={provider.selectedProvider}
              onSave={(f) => provider.saveConfig.mutate(f)}
              saving={provider.saveConfig.isPending}
              error={provider.saveError}
              justSaved={provider.justSaved}
            />
            <ApiVoiceCloning provider={provider.selectedProvider} />
          </Stack>
        )
      )}

      {form && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Volume</CardTitle>
              <CardDescription>
                Playback gain, applied to the synthesized samples and read fresh
                per request (100% = unchanged). Boosting above 100% is genuine, not
                just attenuation.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-5">
              <VolumeSlider
                label="NPC voices"
                help="Directional, in-world NPC voices."
                value={form.npc_volume_pct}
                onChange={(v) => set("npc_volume_pct", v)}
              />
              <VolumeSlider
                label="Admin voice"
                help='The non-positional "admin" voice spoken straight into your ear (e.g. Todd).'
                value={form.admin_volume_pct}
                onChange={(v) => set("admin_volume_pct", v)}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Synthesis tuning</CardTitle>
              <CardDescription>
                Live per-request knobs applied by the warm worker — silence pads,
                output gain, and the PocketTTS generation parameters. No restart.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <Section title="Pacing">
                <div className="flex flex-col gap-4">
                  <NumberRow
                    label="Lead-in silence (ms)"
                    help="Leading pad to protect speech onset from startup clipping."
                    value={form.lead_in_ms}
                    onChange={(v) => set("lead_in_ms", v)}
                    step={10}
                    min={0}
                    max={2000}
                  />
                  <NumberRow
                    label="Trailing silence (ms)"
                    help="Trailing pad so the end of a line isn't clipped."
                    value={form.trailing_ms}
                    onChange={(v) => set("trailing_ms", v)}
                    step={10}
                    min={0}
                    max={2000}
                  />
                  <NumberRow
                    label="Sentence gap (ms)"
                    help="Silence inserted between sentences of a line."
                    value={form.sentence_gap_ms}
                    onChange={(v) => set("sentence_gap_ms", v)}
                    step={10}
                    min={0}
                    max={2000}
                  />
                  <NumberRow
                    label="Output gain (dB)"
                    help="Gain applied to the rendered samples. 0 = unchanged."
                    value={form.gain_db}
                    onChange={(v) => set("gain_db", v)}
                    step={0.5}
                    min={-24}
                    max={12}
                  />
                </div>
              </Section>

              <Section
                title="PocketTTS model"
                description="Generation parameters read off the live TTSModel per request."
              >
                <div className="flex flex-col gap-4">
                  <NumberRow
                    label="Temperature"
                    help="Sampling temperature. Higher = more varied (0–2)."
                    value={form.temperature}
                    onChange={(v) => set("temperature", v)}
                    step={0.05}
                    min={0}
                    max={2}
                  />
                  <NumberRow
                    label="LSD decode steps"
                    help="More steps can raise quality at more compute (≥1)."
                    value={form.lsd_decode_steps}
                    onChange={(v) => set("lsd_decode_steps", v)}
                    step={1}
                    min={1}
                    max={16}
                  />
                  <NumberRow
                    label="EOS threshold"
                    help="Higher keeps generating longer tails (-12 to 0)."
                    value={form.eos_threshold}
                    onChange={(v) => set("eos_threshold", v)}
                    step={0.25}
                    min={-12}
                    max={0}
                  />
                  <NumberRow
                    label="Noise clamp"
                    help="Bounds sampling noise. 0 = off."
                    value={form.noise_clamp}
                    onChange={(v) => set("noise_clamp", v)}
                    step={0.1}
                    min={0}
                    max={4}
                  />
                  <NumberRow
                    label="Max tokens / chunk"
                    help="Size of the sentence chunks long text is split into."
                    value={form.max_tokens}
                    onChange={(v) => set("max_tokens", v)}
                    step={1}
                    min={8}
                    max={200}
                  />
                  <NumberRow
                    label="Frames after EOS"
                    help="Fixed tail length. 0 = let the library auto-pick."
                    value={form.frames_after_eos}
                    onChange={(v) => set("frames_after_eos", v)}
                    step={1}
                    min={0}
                    max={50}
                  />
                </div>
              </Section>
            </CardContent>
          </Card>
        </>
      )}
    </SettingsPage>
  );
}
