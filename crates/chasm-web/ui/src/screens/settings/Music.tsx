import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Download, Loader2 } from "lucide-react";

import { modelsApi, systemApi, type ModelDto, type MusicForm } from "@/lib/api";
import { SettingsPage } from "@/components/ui/settings-page";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
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
  StatusPill,
  type StatusTone,
  TextArea,
} from "@/components/ui/page";

// Music settings — the play-a-song (guitar) action. One managed engine, ACE-Step
// (DiT mode). Reads GET /api/ui/v1/settings/music (the `music` panel) + GET
// /models/music (the engine card), and saves the non-picker fields (enable, style
// tags, max length) to POST /settings/music/save. The engine installs from here
// (or the Runtimes page) via POST /models/music/download.

function engineStatusPill(status: string, running: boolean): {
  tone: StatusTone;
  label: string;
} {
  if (running) return { tone: "ok", label: "Running" };
  switch (status) {
    case "installed":
      return { tone: "ok", label: "Installed" };
    case "installing":
      return { tone: "busy", label: "Installing…" };
    case "failed":
      return { tone: "error", label: "Install failed" };
    default:
      return { tone: "idle", label: "Not installed" };
  }
}

/** The ACE-Step engine card with a one-click install (mirrors the Runtimes card),
 * polling while an install is in flight. */
function AceStepCard({ running }: { running: boolean }) {
  const qc = useQueryClient();
  const key = ["models", "music"] as const;

  const install = useMutation({
    mutationFn: () => modelsApi.download("music", "acestep"),
    onSuccess: (fresh) => qc.setQueryData(key, fresh),
  });

  const query = useQuery({
    queryKey: key,
    queryFn: () => modelsApi.get("music"),
    refetchInterval: () => (install.isPending ? 2000 : false),
  });

  const card: ModelDto | undefined =
    query.data?.models?.find((m) => m.id === "acestep") ??
    query.data?.models?.[0];
  const installed = Boolean(card?.installed);
  const installing = install.isPending;
  const status = installing
    ? { tone: "busy" as StatusTone, label: "Installing…" }
    : engineStatusPill(installed ? "installed" : "not_installed", running);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <CardTitle>ACE-Step (DiT)</CardTitle>
            <CardDescription className="mt-1.5">
              Local music generation on its own server (:5004). Writes an
              in-character song and performs it in-game. Large install (~10 GB);
              loads lazily and frees VRAM when idle.
            </CardDescription>
          </div>
          <StatusPill tone={status.tone} pulse={status.tone === "busy"}>
            {status.label}
          </StatusPill>
        </div>
      </CardHeader>
      <CardContent>
        {installed ? (
          <span className="inline-flex items-center gap-1.5 text-[13px] font-medium text-[var(--color-player)]">
            <Check className="size-4" /> Installed
          </span>
        ) : (
          <Button
            size="sm"
            disabled={installing || query.isLoading}
            onClick={() => install.mutate()}
          >
            {installing ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Download className="size-4" />
            )}
            {installing ? "Installing" : "Install"}
          </Button>
        )}
        {install.isError && (
          <p className="mt-2 text-[13px] text-[var(--color-danger)]">
            Install failed. Check the logs and try again.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

export function Music() {
  const qc = useQueryClient();
  const settings = useQuery({
    queryKey: ["settings", "music"],
    queryFn: () => systemApi.settings("music"),
  });

  const panel = settings.data?.music;
  const initial: MusicForm | null = useMemo(
    () =>
      panel
        ? {
            enabled: panel.enabled,
            style_tags: panel.style_tags,
            max_seconds: panel.max_seconds,
            match_npc_voice: panel.match_npc_voice,
          }
        : null,
    [panel],
  );

  const [form, setForm] = useState<MusicForm | null>(initial);
  const [justSaved, setJustSaved] = useState(false);
  useEffect(() => setForm(initial), [initial]);

  const dirty = useMemo(
    () =>
      !!form && !!initial && JSON.stringify(form) !== JSON.stringify(initial),
    [form, initial],
  );

  const save = useMutation({
    mutationFn: (body: MusicForm) => systemApi.saveMusic(body),
    onSuccess: (fresh) => {
      qc.setQueryData(["settings", "music"], fresh);
      setJustSaved(true);
      window.setTimeout(() => setJustSaved(false), 2200);
    },
  });

  const set = <K extends keyof MusicForm>(key: K, value: MusicForm[K]) =>
    setForm((f) => (f ? { ...f, [key]: value } : f));

  const minSec = panel?.max_seconds_min ?? 20;
  const maxSec = panel?.max_seconds_max ?? 180;

  return (
    <SettingsPage
      eyebrow="AI"
      title="Music"
      description="An NPC can write an in-character song about what you ask and perform it in the game with a guitar. Powered locally by ACE-Step (DiT mode)."
      save={
        form
          ? {
              dirty,
              saving: save.isPending,
              error: save.isError,
              justSaved,
              onReset: () => initial && setForm(initial),
              onSave: () => form && save.mutate(form),
              saveLabel: "Save music",
            }
          : undefined
      }
    >
      <AceStepCard running={Boolean(panel?.engine_running)} />

      {form && (
        <Card>
          <CardHeader>
            <CardTitle>Song generation</CardTitle>
            <CardDescription>
              The NPC's lyrics come from their own character prompt; these settings
              shape the sound and length.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <FormRow
              label="Enable music generation"
              help="When on, the play-a-song action is available and the engine starts with the stack."
              control={
                <Switch
                  checked={form.enabled}
                  onCheckedChange={(v) => set("enabled", v)}
                />
              }
            />
            <FormRow
              label="Match the NPC's voice"
              help="Use the performing NPC's own voice clip as a style reference, so the song leans toward how they sound. Falls back gracefully when a character has no voice clip."
              control={
                <Switch
                  checked={form.match_npc_voice}
                  onCheckedChange={(v) => set("match_npc_voice", v)}
                />
              }
            />
            <FormRow
              stacked
              label="Default style tags"
              help="The base musical style for every song (comma-separated: genre, instrument, mood). Leave the singing voice out — the character describes their own voice from who they are. The model adds a few tags of its own on top."
              control={
                <TextArea
                  rows={2}
                  placeholder="acoustic guitar, folk, campfire, warm"
                  value={form.style_tags}
                  onChange={(e) => set("style_tags", e.target.value)}
                />
              }
            />
            <FormRow
              label="Max song length (seconds)"
              help={`How long a generated song can be (${minSec}–${maxSec}). Longer songs take longer to generate and share the GPU with the game.`}
              control={
                <Field
                  type="number"
                  className="w-28 text-right"
                  value={
                    Number.isFinite(form.max_seconds) ? form.max_seconds : 0
                  }
                  step={5}
                  min={minSec}
                  max={maxSec}
                  onChange={(e) => set("max_seconds", Number(e.target.value))}
                />
              }
            />
          </CardContent>
        </Card>
      )}
    </SettingsPage>
  );
}
