import { Check, Loader2, Server, Cloud } from "lucide-react";

import { cn } from "@/lib/utils";
import type { ProviderDto } from "@/lib/api";
import { Section } from "@/components/ui/page";

// ===========================================================================
// ProviderPicker — a radio list of the providers for a capability (LLM / STT /
// TTS), from GET /providers/:capability. The first card is always the managed
// "local" runtime; hosted API providers follow. Selecting a card calls
// .../select. The selected card is highlighted; each shows name + blurb.
// ===========================================================================

export interface ProviderPickerProps {
  providers: ProviderDto[];
  /** The currently-selected provider id. */
  selectedId: string;
  /** Select a provider (persists via .../select). */
  onSelect: (id: string) => void;
  /** Id whose select is in flight (shows a spinner on that card). */
  selectingId?: string | null;
  isLoading?: boolean;
}

export function ProviderPicker({
  providers,
  selectedId,
  onSelect,
  selectingId,
  isLoading,
}: ProviderPickerProps) {
  return (
    <Section
      title="Provider"
      description="Run this locally with the managed engine, or point it at a hosted API."
    >
      {isLoading ? (
        <div className="grid place-items-center py-8 text-[var(--muted-foreground)]">
          <Loader2 className="size-5 animate-spin" />
        </div>
      ) : (
        <div className="flex flex-col gap-2.5">
          {providers.map((provider) => {
            const selected = provider.id === selectedId;
            const busy = selectingId === provider.id;
            const Icon = provider.kind === "local" ? Server : Cloud;
            return (
              <button
                key={provider.id}
                type="button"
                onClick={() => !selected && onSelect(provider.id)}
                className={cn(
                  "flex items-start gap-3 rounded-xl border bg-[var(--card)] p-[var(--card-pad,15px)] text-left transition-colors",
                  selected
                    ? "border-[var(--color-accent)] ring-1 ring-[var(--color-accent)]/30"
                    : "border-[var(--border)] hover:border-[var(--color-ink-600)]",
                )}
              >
                <span
                  className={cn(
                    "mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg border",
                    selected
                      ? "border-[var(--color-accent)]/40 text-[var(--color-accent)]"
                      : "border-[var(--border)] text-[var(--muted-foreground)]",
                  )}
                >
                  <Icon className="size-4" strokeWidth={2} />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-2">
                    <span className="text-sm font-medium">{provider.name}</span>
                    {provider.kind === "local" && (
                      <span className="rounded-full border border-[var(--border)] px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-[var(--muted-foreground)]">
                        Managed
                      </span>
                    )}
                  </span>
                  {provider.blurb && (
                    <span className="mt-0.5 block text-[13px] leading-relaxed text-[var(--muted-foreground)]">
                      {provider.blurb}
                    </span>
                  )}
                </span>
                <span className="shrink-0 pt-0.5">
                  {busy ? (
                    <Loader2 className="size-4 animate-spin text-[var(--color-accent)]" />
                  ) : selected ? (
                    <span className="inline-flex items-center gap-1 text-[12px] font-medium text-[var(--color-accent)]">
                      <Check className="size-3.5" /> Selected
                    </span>
                  ) : null}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </Section>
  );
}
