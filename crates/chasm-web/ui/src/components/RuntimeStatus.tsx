import { Link } from "react-router-dom";
import { ArrowRight, Check } from "lucide-react";

import type { LocalRuntimeDto } from "@/lib/api";
import { StatusPill } from "@/components/ui/page";

// ===========================================================================
// RuntimeStatus — a small line for a capability's LOCAL managed runtime (shown
// on the LLM/STT/TTS pages when the "local" provider is selected). Surfaces
// whether the required engine (llama.cpp / Parakeet / qwen3-tts) is installed,
// and links to the Runtimes screen to install it when it isn't.
// ===========================================================================

export function RuntimeStatus({ runtime }: { runtime: LocalRuntimeDto }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-[var(--border)] bg-[var(--color-ink-850)] px-3 py-2.5">
      <div className="flex min-w-0 items-center gap-2.5">
        <StatusPill tone={runtime.installed ? "ok" : "warn"}>
          {runtime.installed ? (
            <>
              <Check className="size-3.5" /> Installed
            </>
          ) : (
            "Not installed"
          )}
        </StatusPill>
        <div className="min-w-0">
          <p className="truncate text-[13px] text-[var(--foreground)]">
            Needs the {runtime.name}
          </p>
          {runtime.hint && (
            <p className="truncate text-[12px] text-[var(--muted-foreground)]">
              {runtime.hint}
            </p>
          )}
        </div>
      </div>
      {!runtime.installed && (
        <Link
          to="/settings/runtimes"
          className="inline-flex shrink-0 items-center gap-1 text-[13px] font-medium text-[var(--color-accent)] hover:underline"
        >
          Go to Runtimes <ArrowRight className="size-3.5" />
        </Link>
      )}
    </div>
  );
}
