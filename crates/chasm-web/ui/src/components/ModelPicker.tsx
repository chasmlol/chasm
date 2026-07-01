import type { ReactNode } from "react";
import { Check, Download, FolderOpen, Loader2, Star } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Section,
  EmptyState,
  StatusPill,
  type StatusTone,
} from "@/components/ui/page";

// ===========================================================================
// ModelPicker — the SHARED model-card list for the AI settings screens (LLM /
// TTS / STT / Retrieval). One card per model: name + meta, a "recommended"
// badge, a status pill (ready / downloading / not installed / error), a
// select/active affordance, and a download button. Below the list, an optional
// "drop files here" folder path with an Open button.
//
// ---------------------------------------------------------------------------
// PROP CONTRACT (read this if you're filling in an AI settings screen)
// ---------------------------------------------------------------------------
// Map your backend's model list to `ModelCard[]`, pass the selected id, and
// wire the callbacks. The chrome (cards, badges, status, layout) is owned here
// so LLM/TTS/STT/Retrieval all look identical.
//
//   <ModelPicker
//     title="Language model"
//     models={cards}                 // ModelCard[]
//     selectedId={selected}
//     onSelect={(id) => mutate(id)}  // optional: pick the active model
//     onDownload={(id) => mutate(id)}// optional: download a model
//     downloadingId={pendingId}      // optional: id mid-download (shows spinner)
//     folder={{ path, onOpen }}      // optional: the "drop files here" dir
//   />
// ===========================================================================

/** One model row in the picker. */
export interface ModelCard {
  /** Stable id (used for select/download/key). */
  id: string;
  /** Display name. */
  name: string;
  /** Optional one-line description / repo / quant. */
  description?: ReactNode;
  /** Small meta chips (e.g. size, VRAM, params). */
  meta?: { label: string; value: ReactNode }[];
  /** Is this model installed / present on disk? */
  installed?: boolean;
  /** Flagged as recommended for this host. */
  recommended?: boolean;
  /** Status pill content + tone; when omitted, derived from installed/selected. */
  status?: { tone: StatusTone; label: ReactNode };
}

export interface ModelPickerProps {
  title?: ReactNode;
  description?: ReactNode;
  models: ModelCard[];
  /** The currently selected/active model id. */
  selectedId?: string;
  /** Pick the active model. When omitted, cards aren't selectable. */
  onSelect?: (id: string) => void;
  /** Download a model. When omitted, no download button shows. */
  onDownload?: (id: string) => void;
  /** Id currently downloading (shows a spinner on that card). */
  downloadingId?: string | null;
  /** The "drop files here" folder with an Open-in-Explorer action. */
  folder?: { path: string; onOpen?: () => void };
  isLoading?: boolean;
  isError?: boolean;
}

export function ModelPicker({
  title,
  description,
  models,
  selectedId,
  onSelect,
  onDownload,
  downloadingId,
  folder,
  isLoading,
  isError,
}: ModelPickerProps) {
  return (
    <Section title={title} description={description}>
      {isLoading ? (
        <div className="grid place-items-center py-12 text-[var(--muted-foreground)]">
          <Loader2 className="size-5 animate-spin" />
        </div>
      ) : isError ? (
        <EmptyState
          title="Couldn't load models."
          description="The backend returned an error."
        />
      ) : models.length === 0 ? (
        <EmptyState
          title="No models available."
          description="Drop a model file into the folder below to get started."
        />
      ) : (
        <div className="flex flex-col gap-2.5">
          {models.map((model) => (
            <ModelRow
              key={model.id}
              model={model}
              selected={model.id === selectedId}
              selectable={Boolean(onSelect)}
              onSelect={onSelect}
              onDownload={onDownload}
              downloading={downloadingId === model.id}
            />
          ))}
        </div>
      )}

      {folder && (
        <div className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-[var(--border)] bg-[var(--color-ink-850)] px-3 py-2.5">
          <div className="min-w-0">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
              Models folder
            </p>
            <p
              className="mt-0.5 truncate font-mono text-[11px] text-[var(--foreground)]"
              title={folder.path}
            >
              {folder.path}
            </p>
          </div>
          {folder.onOpen && (
            <Button variant="secondary" size="sm" onClick={folder.onOpen}>
              <FolderOpen className="size-4" /> Open
            </Button>
          )}
        </div>
      )}
    </Section>
  );
}

/** Default status tone/label when a card doesn't supply its own. */
function defaultStatus(
  model: ModelCard,
  selected: boolean,
): { tone: StatusTone; label: ReactNode } {
  if (!model.installed) return { tone: "idle", label: "Not installed" };
  if (selected) return { tone: "ok", label: "Active" };
  return { tone: "ok", label: "Ready" };
}

function ModelRow({
  model,
  selected,
  selectable,
  onSelect,
  onDownload,
  downloading,
}: {
  model: ModelCard;
  selected: boolean;
  selectable: boolean;
  onSelect?: (id: string) => void;
  onDownload?: (id: string) => void;
  downloading: boolean;
}) {
  const status = model.status ?? defaultStatus(model, selected);
  const canSelect = selectable && model.installed && !selected;

  return (
    <div
      className={cn(
        "rounded-xl border bg-[var(--card)] p-[var(--card-pad,15px)] transition-colors",
        selected
          ? "border-[var(--color-accent)] ring-1 ring-[var(--color-accent)]/30"
          : "border-[var(--border)]",
        canSelect && "cursor-pointer hover:border-[var(--color-ink-600)]",
      )}
      onClick={canSelect ? () => onSelect?.(model.id) : undefined}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-medium">{model.name}</span>
            {model.recommended && (
              <span className="inline-flex items-center gap-1 rounded-full border border-[var(--color-npc)]/40 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-npc)]">
                <Star className="size-3" /> Recommended
              </span>
            )}
          </div>
          {model.description && (
            <p className="mt-0.5 truncate text-[13px] text-[var(--muted-foreground)]">
              {model.description}
            </p>
          )}
          {model.meta && model.meta.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
              {model.meta.map((m) => (
                <span key={m.label} className="text-[12px] text-[var(--muted-foreground)]">
                  <span className="text-[var(--muted-foreground)]/70">
                    {m.label}:
                  </span>{" "}
                  <span className="text-[var(--foreground)]">{m.value}</span>
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="flex shrink-0 flex-col items-end gap-2">
          <StatusPill tone={status.tone} pulse={status.tone === "busy"}>
            {status.label}
          </StatusPill>
          <div className="flex items-center gap-2">
            {selected && (
              <span className="inline-flex items-center gap-1 text-[12px] font-medium text-[var(--color-accent)]">
                <Check className="size-3.5" /> Selected
              </span>
            )}
            {!model.installed && onDownload && (
              <Button
                variant="secondary"
                size="sm"
                disabled={downloading}
                onClick={(e) => {
                  e.stopPropagation();
                  onDownload(model.id);
                }}
              >
                {downloading ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Download className="size-4" />
                )}
                {downloading ? "Downloading" : "Download"}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
