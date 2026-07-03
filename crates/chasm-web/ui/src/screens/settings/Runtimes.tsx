import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Download, Loader2 } from "lucide-react";

import { modelsApi, type ModelDomain, type ModelDto } from "@/lib/api";
import { SettingsPage } from "@/components/ui/settings-page";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { StatusPill, type StatusTone } from "@/components/ui/page";

// Runtimes settings — the one-click ENGINE installers that STAY (the model FILES
// moved to guided manual placement, and there's no LLM runtime choice anymore).
// Three managed engines, each with live status + an Install button:
//   * llama.cpp       — GET /models/runtime, install POST /models/runtime/download {id:"llamacpp"}
//   * Parakeet STT    — GET /models/stt (parakeet card), install .../stt/download {id:"parakeet"}
//   * qwen3-tts       — GET /models/tts, install .../tts/download {id:"faster-qwen3-tts"}
// While an install is in flight the query polls every ~2s so the pill flips to
// installed on its own.

interface EngineSpec {
  key: string;
  title: string;
  domain: ModelDomain;
  /** The card id within GET /models/:domain to read status from + install. */
  id: string;
  blurb: string;
}

const ENGINES: EngineSpec[] = [
  {
    key: "llamacpp",
    title: "llama.cpp",
    domain: "runtime",
    id: "llamacpp",
    blurb:
      "The managed local LLM server (llama-server on :5001). Serves the placed .gguf with prompt-cache slots for fast speaker swaps.",
  },
  {
    key: "parakeet",
    title: "Parakeet STT engine",
    domain: "stt",
    id: "parakeet",
    blurb:
      "The managed local speech-to-text engine. Runs on its own port and transcribes your microphone for the player's turn.",
  },
  {
    key: "qwen3-tts",
    title: "qwen3-tts engine",
    domain: "tts",
    id: "faster-qwen3-tts",
    blurb:
      "The managed local text-to-speech engine. Synthesizes NPC voices and streams the audio to the game.",
  },
  {
    key: "acestep",
    title: "ACE-Step music engine",
    domain: "music",
    id: "acestep",
    blurb:
      "The managed local music-generation engine (ACE-Step, DiT mode) on its own port (:5004). Powers the play-a-song action — an NPC writes and performs a song. Large install (~10 GB); loads lazily and frees VRAM when idle.",
  },
];

function statusFor(installed: boolean, installing: boolean): {
  tone: StatusTone;
  label: string;
} {
  if (installing) return { tone: "busy", label: "Installing…" };
  if (installed) return { tone: "ok", label: "Installed" };
  return { tone: "idle", label: "Not installed" };
}

function EngineCard({ spec }: { spec: EngineSpec }) {
  const qc = useQueryClient();
  const key = ["models", spec.domain];

  const install = useMutation({
    mutationFn: () => modelsApi.download(spec.domain, spec.id),
    onSuccess: (fresh) => qc.setQueryData(key, fresh),
  });

  const query = useQuery({
    queryKey: key,
    queryFn: () => modelsApi.get(spec.domain),
    // Poll while an install is running so the status flips on its own.
    refetchInterval: () => (install.isPending ? 2000 : false),
  });

  const card: ModelDto | undefined =
    query.data?.models?.find((m) => m.id === spec.id) ?? query.data?.models?.[0];
  const installed = Boolean(card?.installed);
  const installing = install.isPending;
  const status = statusFor(installed, installing);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <CardTitle>{spec.title}</CardTitle>
            <CardDescription className="mt-1.5">{spec.blurb}</CardDescription>
          </div>
          <StatusPill tone={status.tone} pulse={status.tone === "busy"}>
            {status.label}
          </StatusPill>
        </div>
      </CardHeader>
      <CardContent>
        {installed ? (
          <span className="inline-flex items-center gap-1.5 text-[13px] font-medium text-[var(--color-player)]">
            <Check className="size-4" /> Ready
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

export function Runtimes() {
  return (
    <SettingsPage
      eyebrow="AI"
      title="Runtimes"
      description="One-click installers for the managed local engines. Model files are placed manually on their own pages (LLM, Retrieval); the engines below install automatically."
    >
      {ENGINES.map((spec) => (
        <EngineCard key={spec.key} spec={spec} />
      ))}
    </SettingsPage>
  );
}
