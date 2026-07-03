import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "motion/react";
import { Loader2, Play } from "lucide-react";

import { systemApi, type StackStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

// "Start models" control + a per-service status light for each model, sitting
// directly under the connection pill in the sidebar. Polls `/api/stack/status`
// every 2s (like the connection pill) and reflects each service: a green pulsing
// dot when it's up/loaded, a dim dot when it's not. The button POSTs
// `/api/stack/start`, which spawns the configured LLM, STT and TTS providers
// (the managed local engines when selected) and warms the retriever; the lights
// flip to green as each becomes reachable.

// Core services that drive the "Start models" button + "All models running" state.
const MODELS: { key: keyof StackStatus; label: string }[] = [
  { key: "llm", label: "LLM" },
  { key: "stt", label: "STT" },
  { key: "tts", label: "TTS" },
  { key: "embedder", label: "Embedder" },
  { key: "reranker", label: "Reranker" },
];

// Every light shown in the grid. Music is optional (off by default), so it's
// displayed but kept OUT of the button's allOn/anyBusy logic below — a user who
// doesn't use music generation still sees "All models running".
const LIGHTS: { key: keyof StackStatus; label: string }[] = [
  ...MODELS,
  { key: "music", label: "Music" },
];

// "ok" = up (green) · "busy" = coming up / runtime downloading (amber) · idle (gray).
const TONE_COLOR: Record<string, string> = {
  ok: "var(--color-player)",
  busy: "var(--color-npc)",
  idle: "#3a4150",
};

function ModelLight({ label, tone }: { label: string; tone: string }) {
  const color = TONE_COLOR[tone] ?? TONE_COLOR.idle;
  const lit = tone === "ok" || tone === "busy";
  return (
    <div className="flex items-center gap-2">
      <span className="relative flex size-2">
        {lit && (
          <motion.span
            className="absolute inline-flex size-full rounded-full"
            style={{ background: color }}
            animate={{ opacity: [0.6, 0, 0.6], scale: [1, 2.4, 1] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeOut" }}
          />
        )}
        <span
          className="relative inline-flex size-2 rounded-full"
          style={{ background: color }}
        />
      </span>
      <span
        className={cn(
          "text-[11px] font-medium tracking-wide",
          lit ? "text-[var(--foreground)]" : "text-[var(--muted-foreground)]",
        )}
      >
        {label}
      </span>
    </div>
  );
}

export function StackControls() {
  const qc = useQueryClient();
  const { data } = useQuery({
    queryKey: ["stack-status"],
    queryFn: systemApi.stackStatus,
    refetchInterval: 2000,
  });
  const start = useMutation({
    mutationFn: systemApi.startStack,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["stack-status"] }),
  });

  const toneOf = (key: keyof StackStatus) => data?.[key] ?? "idle";
  const allOn = data ? MODELS.every((m) => data[m.key] === "ok") : false;
  const anyBusy = data ? MODELS.some((m) => data[m.key] === "busy") : false;
  const starting = start.isPending || anyBusy;

  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={() => start.mutate()}
        disabled={start.isPending}
        className={cn(
          "flex w-full items-center justify-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors disabled:opacity-70",
          allOn
            ? "border-[var(--color-player)]/40 text-[var(--color-player)]"
            : "border-[var(--border)] bg-[var(--color-ink-800)] text-[var(--foreground)] hover:border-[var(--color-ink-600)]",
        )}
      >
        {starting ? (
          <>
            <Loader2 className="size-3.5 animate-spin" /> Starting…
          </>
        ) : allOn ? (
          "All models running"
        ) : (
          <>
            <Play className="size-3.5" /> Start models
          </>
        )}
      </button>

      <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 rounded-lg border border-[var(--border)] bg-[var(--color-ink-850)] px-3 py-2.5">
        {LIGHTS.map((m) => (
          <ModelLight key={m.key} label={m.label} tone={toneOf(m.key)} />
        ))}
      </div>
    </div>
  );
}
