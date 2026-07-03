import { useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  Copy,
  ExternalLink,
  FolderOpen,
  Loader2,
  Star,
  UploadCloud,
} from "lucide-react";

import { cn } from "@/lib/utils";
import {
  systemApi,
  type ModelDto,
  type OpenFolderKind,
  type PlaceDomain,
  type PlaceModelResult,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Section, StatusPill, EmptyState } from "@/components/ui/page";

// ===========================================================================
// ModelPlacement — the GUIDED MANUAL PLACEMENT surface for a model file that
// can't be auto-downloaded in-app (the LLM .gguf, the embedder .onnx). It:
//   1. Lists the recommended models (from GET /models/:domain). Each row opens
//      the model's Hugging Face page in the REAL browser (openUrl).
//   2. Shows the exact target folder (copyable + open-in-Explorer).
//   3. Offers a drag-drop zone AND a choose-file button — BOTH upload the File's
//      raw bytes via placeModel(domain, file). Shows progress + the result.
// After a successful placement it invalidates ["models", domain] so the matching
// row flips to installed.
// ===========================================================================

const EXT: Record<PlaceDomain, string> = { llm: ".gguf", retrieval: ".onnx" };

export interface ModelPlacementProps {
  /** Which placement domain (drives the accepted extension + upload endpoint). */
  domain: PlaceDomain;
  /** Which OS folder to reveal in Explorer. */
  folderKind: OpenFolderKind;
  /** The recommended model rows to list (already filtered by the caller). */
  models: ModelDto[];
  /** The exact target folder path (monospace, copy + open). */
  folder?: string;
  isLoading?: boolean;
  isError?: boolean;
  /** Section title / description overrides. */
  title?: string;
  description?: string;
}

export function ModelPlacement({
  domain,
  folderKind,
  models,
  folder,
  isLoading,
  isError,
  title = "Recommended models",
  description,
}: ModelPlacementProps) {
  const qc = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [copied, setCopied] = useState(false);
  const [result, setResult] = useState<PlaceModelResult | null>(null);

  const openPage = useMutation({
    mutationFn: (repo: string) =>
      systemApi.openUrl(`https://huggingface.co/${repo}`),
  });
  const openFolder = useMutation({
    mutationFn: () => systemApi.openFolder(folderKind),
  });
  const place = useMutation({
    mutationFn: (file: File) => systemApi.placeModel(domain, file),
    onSuccess: (res) => {
      setResult(res);
      if (res.ok) {
        // Flip the matching card to installed/ready.
        qc.invalidateQueries({ queryKey: ["models", domain] });
      }
    },
    onError: (err) =>
      setResult({ ok: false, error: (err as Error).message || "Upload failed" }),
  });

  const accept = EXT[domain];

  const upload = (file: File) => {
    setResult(null);
    place.mutate(file);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) upload(file);
  };

  const copyFolder = async () => {
    if (!folder) return;
    try {
      await navigator.clipboard.writeText(folder);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard may be unavailable; the Open button still works */
    }
  };

  return (
    <Section
      title={title}
      description={
        description ??
        `Download one of these, then place its ${accept} file into the folder below.`
      }
    >
      {isLoading ? (
        <div className="grid place-items-center py-8 text-[var(--muted-foreground)]">
          <Loader2 className="size-5 animate-spin" />
        </div>
      ) : isError ? (
        <EmptyState
          title="Couldn't load models."
          description="The backend returned an error."
        />
      ) : models.length === 0 ? (
        <EmptyState
          title="No recommended models."
          description="Place a compatible model file into the folder below to get started."
        />
      ) : (
        <div className="flex flex-col gap-2.5">
          {models.map((model) => (
            <div
              key={model.id}
              className={cn(
                "rounded-xl border bg-[var(--card)] p-[var(--card-pad,15px)]",
                model.installed
                  ? "border-[var(--color-player)]/40"
                  : "border-[var(--border)]",
              )}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="truncate text-sm font-medium">
                      {model.name}
                    </span>
                    {model.recommended && (
                      <span className="inline-flex items-center gap-1 rounded-full border border-[var(--color-npc)]/40 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-npc)]">
                        <Star className="size-3" /> Recommended
                      </span>
                    )}
                  </div>
                  {model.description && (
                    <p
                      className="mt-0.5 truncate font-mono text-[12px] text-[var(--muted-foreground)]"
                      title={model.description}
                    >
                      {model.description}
                    </p>
                  )}
                  {model.meta && model.meta.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
                      {model.meta.map((m) => (
                        <span
                          key={m.label}
                          className="text-[12px] text-[var(--muted-foreground)]"
                        >
                          <span className="text-[var(--muted-foreground)]/70">
                            {m.label}:
                          </span>{" "}
                          <span className="text-[var(--foreground)]">
                            {m.value}
                          </span>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                <div className="flex shrink-0 flex-col items-end gap-2">
                  <StatusPill tone={model.installed ? "ok" : "idle"}>
                    {model.installed ? (
                      <>
                        <Check className="size-3.5" /> Installed
                      </>
                    ) : (
                      "Not placed"
                    )}
                  </StatusPill>
                  {model.description && (
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={openPage.isPending}
                      onClick={() => openPage.mutate(model.description as string)}
                    >
                      <ExternalLink className="size-4" /> Open download page
                    </Button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Target folder — copyable + open in Explorer. */}
      {folder && (
        <div className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-[var(--border)] bg-[var(--color-ink-850)] px-3 py-2.5">
          <div className="min-w-0">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
              Place the file here
            </p>
            <p
              className="mt-0.5 truncate font-mono text-[11px] text-[var(--foreground)]"
              title={folder}
            >
              {folder}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Button variant="secondary" size="sm" onClick={copyFolder}>
              {copied ? (
                <Check className="size-4" />
              ) : (
                <Copy className="size-4" />
              )}
              {copied ? "Copied" : "Copy"}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              disabled={openFolder.isPending}
              onClick={() => openFolder.mutate()}
            >
              <FolderOpen className="size-4" /> Open folder
            </Button>
          </div>
        </div>
      )}

      {/* Drop zone + choose-file. Both paths upload the raw file bytes. */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
        className={cn(
          "mt-3 grid place-items-center rounded-xl border border-dashed px-6 py-8 text-center transition-colors",
          dragActive
            ? "border-[var(--color-accent)] bg-[var(--color-accent)]/5"
            : "border-[var(--border)]",
        )}
      >
        <div className="max-w-sm">
          <div className="mx-auto mb-3 grid size-11 place-items-center rounded-2xl border border-[var(--border)] bg-[var(--color-ink-800)] text-[var(--color-accent)]">
            {place.isPending ? (
              <Loader2 className="size-5 animate-spin" />
            ) : (
              <UploadCloud className="size-5" />
            )}
          </div>
          <p className="text-sm font-medium text-[var(--foreground)]">
            {place.isPending
              ? "Placing model…"
              : `Drop a ${accept} file here`}
          </p>
          <p className="mt-1 text-[13px] leading-relaxed text-[var(--muted-foreground)]">
            The file is copied into the folder above. It never leaves your
            machine.
          </p>
          <div className="mt-4 flex justify-center">
            <input
              ref={fileInput}
              type="file"
              accept={accept}
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) upload(file);
                // allow re-picking the same file
                e.target.value = "";
              }}
            />
            <Button
              variant="secondary"
              size="sm"
              disabled={place.isPending}
              onClick={() => fileInput.current?.click()}
            >
              Choose file
            </Button>
          </div>

          {result && (
            <p
              className={cn(
                "mt-3 break-words text-[13px]",
                result.ok
                  ? "text-[var(--color-player)]"
                  : "text-[var(--color-danger)]",
              )}
            >
              {result.ok
                ? `Placed${result.path ? ` → ${result.path}` : ""}. Ready.`
                : `Couldn't place the file: ${result.error ?? "unknown error"}`}
            </p>
          )}
        </div>
      </div>
    </Section>
  );
}
