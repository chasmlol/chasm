import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Braces, Globe, RotateCcw, Save } from "lucide-react";

import { globalsApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  EmptyState,
  PageBody,
  PageHeader,
  Section,
  Stack,
  StatusPill,
  TextArea,
} from "@/components/ui/page";
import { cn } from "@/lib/utils";

// ===========================================================================
// Globals → Scenario — the GLOBAL scenario template.
//
// One app-wide template replaces the per-character card "Scenario" field.
// Every NPC turn the backend resolves its {{macro}} placeholders against the
// gamestate the mod sent for that turn (falling back to the latest recorded
// table) plus backend-computed macros ({{participants}}), and injects the
// result into the prompt exactly where the card scenario used to sit.
//
// TOP: the template editor + Save / Reset to default.
// MIDDLE: a LIVE resolved preview — the current draft rendered through the
// latest recorded macros, so you see what the NPC will actually read.
// BOTTOM: the macro vocabulary (documented set + anything the mod recorded).
// ===========================================================================

/** The documented macro vocabulary (mod-source/docs/gamestate-macros.md) +
 *  the backend-computed additions. Keys the mod records that aren't listed
 *  here still show up under "Recorded by the mod". */
const MACRO_VOCABULARY: { key: string; hint: string; computed?: boolean }[] = [
  {
    key: "participants",
    hint: "Computed each turn: the player plus the other NPCs in the conversation (excluding the speaker).",
    computed: true,
  },
  { key: "player_name", hint: "The player's chosen name, e.g. \"Courier\"." },
  { key: "level", hint: "Player level, e.g. \"12\"." },
  {
    key: "major_location",
    hint: "Nearest world-map marker, e.g. \"Goodsprings\".",
  },
  {
    key: "minor_location",
    hint: "Nearest local landmark / cell, e.g. \"Prospector Saloon\".",
  },
  {
    key: "time_of_day",
    hint: "12-hour game clock, e.g. \"6:30PM\".",
  },
  { key: "health", hint: "Current/max HP, e.g. \"185/220\"." },
  { key: "health_percent", hint: "Derived percentage, e.g. \"84%\"." },
  { key: "radiation", hint: "e.g. \"23 rads\"." },
  { key: "condition", hint: "Limb condition, e.g. \"Left Arm crippled; rest OK\"." },
  { key: "effects", hint: "Active effects, e.g. \"Well Rested, Med-X\"." },
  { key: "special", hint: "S.P.E.C.I.A.L. values." },
  { key: "skills", hint: "All visible skill values." },
  { key: "perks", hint: "Perks and traits." },
  { key: "equipped_weapon", hint: "e.g. \"9mm Pistol\"." },
  { key: "equipped_apparel", hint: "Worn armor/apparel names." },
  { key: "inventory", hint: "Curated aid/ammo/weapons/apparel list." },
  { key: "quests", hint: "Named quests with a live objective." },
  { key: "misc_quests", hint: "Unnamed (Misc) quest objectives." },
];

/** "Jun 20, 2026, 9:28 PM"-style label for the recorded-at timestamp. */
function formatUpdatedAt(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

/** Debounces a value so the preview doesn't fire per keystroke. */
function useDebounced<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const handle = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(handle);
  }, [value, delayMs]);
  return debounced;
}

export function Globals() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["globals", "scenario"],
    queryFn: () => globalsApi.scenario(),
  });

  // The editor draft. `null` until the saved template loads (so we never
  // clobber user edits with a refetch — only the FIRST load seeds it).
  const [draft, setDraft] = useState<string | null>(null);
  useEffect(() => {
    if (draft === null && query.data) setDraft(query.data.template);
  }, [draft, query.data]);

  const template = draft ?? query.data?.template ?? "";
  const savedTemplate = query.data?.template ?? "";
  const dirty = draft !== null && query.data !== undefined && draft !== savedTemplate;

  const save = useMutation({
    mutationFn: (nextTemplate: string) =>
      globalsApi.saveScenario({ template: nextTemplate }),
    onSuccess: (data) => {
      qc.setQueryData(["globals", "scenario"], data);
      setDraft(data.template);
    },
  });

  // Live resolved preview of the CURRENT DRAFT through the latest recorded
  // macros (debounced; also refetched periodically so an in-game turn shows
  // up while the page is open).
  const debouncedTemplate = useDebounced(template, 400);
  const preview = useQuery({
    queryKey: ["globals", "scenario-preview", debouncedTemplate],
    queryFn: () => globalsApi.previewScenario({ template: debouncedTemplate }),
    enabled: query.data !== undefined,
    refetchInterval: 4000,
  });

  /** Clicking a macro chip appends its placeholder to the template. */
  const appendMacro = (key: string) =>
    setDraft((current) => {
      const base = current ?? savedTemplate;
      return base.length === 0 || base.endsWith(" ")
        ? `${base}{{${key}}}`
        : `${base} {{${key}}}`;
    });

  // Recorded-but-undocumented keys still get chips so nothing is hidden.
  const documentedKeys = useMemo(
    () => new Set(MACRO_VOCABULARY.map((entry) => entry.key)),
    [],
  );
  const extraRecordedKeys = Object.keys(preview.data?.macros ?? {})
    .filter((key) => !documentedKeys.has(key))
    .sort();

  const statusPill = query.data ? (
    template.trim().length === 0 ? (
      <StatusPill tone="warn">Scenario disabled (empty template)</StatusPill>
    ) : query.data.is_default && !dirty ? (
      <StatusPill tone="idle">Built-in default</StatusPill>
    ) : (
      <StatusPill tone="ok">Custom template</StatusPill>
    )
  ) : null;

  return (
    <PageBody width="wide" className="overflow-y-auto">
      <PageHeader
        eyebrow="Globals"
        title="Scenario"
        description={
          <>
            One global scenario for every NPC, replacing the per-character
            card field. Its{" "}
            <code className="font-mono">{"{{macro}}"}</code> placeholders are
            resolved with the live gamestate on every turn and the result is
            injected into each NPC's prompt.
          </>
        }
        actions={statusPill}
      />

      <Stack className="py-[var(--gap,14px)]">
        <Section
          title="Template"
          description="Edit and save the global scenario. Missing macros resolve to empty, so prefer short, separate sentences. Clear the template entirely to omit the scenario from prompts."
        >
          {query.isLoading ? (
            <EmptyState icon={<Globe />} title="Loading template…" />
          ) : query.isError ? (
            <EmptyState
              icon={<Globe />}
              title="Failed to load the template"
              description="Is the backend running? Reload to retry."
            />
          ) : (
            <div className="flex flex-col gap-3 rounded-xl border border-[var(--border)] bg-[var(--color-ink-900,transparent)] p-3.5">
              <TextArea
                rows={5}
                value={template}
                onChange={(event) => setDraft(event.target.value)}
                placeholder={query.data?.default_template}
                spellCheck={false}
                className="font-mono text-[13px]"
              />
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  onClick={() => save.mutate(template)}
                  disabled={save.isPending || !dirty}
                >
                  <Save />
                  {save.isPending ? "Saving…" : dirty ? "Save" : "Saved"}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setDraft(query.data?.default_template ?? "")}
                  disabled={
                    save.isPending ||
                    template === (query.data?.default_template ?? "")
                  }
                  title="Restore the built-in default template into the editor (save to apply)"
                >
                  <RotateCcw />
                  Reset to default
                </Button>
                {save.isError && (
                  <p className="text-[13px] text-[var(--color-danger)]">
                    Save failed: {(save.error as Error).message}
                  </p>
                )}
              </div>
            </div>
          )}
        </Section>

        <Section
          title="Resolved preview"
          description="The draft above rendered through the most recently recorded gamestate — what the NPC will actually read. Updates as you type and as new in-game turns arrive."
        >
          <div className="flex flex-col gap-3 rounded-xl border border-[var(--border)] bg-[var(--color-ink-900,transparent)] p-3.5">
            {preview.data?.note && (
              <p className="text-[13px] text-[var(--color-npc,orange)]">
                {preview.data.note}
              </p>
            )}
            {preview.isError ? (
              <p className="text-[13px] text-[var(--color-danger)]">
                Preview failed: {(preview.error as Error).message}
              </p>
            ) : preview.data ? (
              <div className="min-w-0">
                <p className="mb-1.5 flex items-center justify-between text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted-foreground)]">
                  <span>Resolved scenario</span>
                  {preview.data.updated_at && (
                    <span className="normal-case tracking-normal">
                      Macros from {formatUpdatedAt(preview.data.updated_at)}
                    </span>
                  )}
                </p>
                <pre
                  className={cn(
                    "max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-xl border border-[var(--border)] bg-[var(--color-ink-850)] px-3.5 py-3 font-mono text-[13px] leading-relaxed",
                    preview.data.resolved.trim().length > 0
                      ? "text-[var(--foreground)]"
                      : "text-[var(--muted-foreground)]",
                  )}
                >
                  {preview.data.resolved.trim().length > 0
                    ? preview.data.resolved
                    : "(empty — the scenario component would be omitted)"}
                </pre>
              </div>
            ) : (
              <p className="flex items-center gap-1.5 text-[13px] text-[var(--muted-foreground)]">
                <Braces className="size-3.5" />
                Resolving preview…
              </p>
            )}
          </div>
        </Section>

        <Section
          title="Available macros"
          description="Click a macro to append it to the template. The mod sends these every turn; whatever it cannot read resolves to empty. {{participants}} is computed by the backend."
        >
          <div className="flex flex-col gap-1.5">
            {MACRO_VOCABULARY.map((entry) => (
              <div key={entry.key} className="flex items-baseline gap-2.5">
                <button
                  type="button"
                  onClick={() => appendMacro(entry.key)}
                  title={`Append {{${entry.key}}} to the template`}
                  className="shrink-0 rounded-md border border-[var(--border)] bg-[var(--color-ink-850)] px-1.5 py-0.5 font-mono text-[12px] text-[var(--color-accent)] transition-colors hover:border-[var(--color-accent)]"
                >
                  {`{{${entry.key}}}`}
                </button>
                <span className="text-[13px] text-[var(--muted-foreground)]">
                  {entry.computed ? "(computed) " : ""}
                  {entry.hint}
                </span>
              </div>
            ))}
            {extraRecordedKeys.length > 0 && (
              <>
                <p className="mt-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted-foreground)]">
                  Also recorded by the mod
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {extraRecordedKeys.map((key) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => appendMacro(key)}
                      title={`Append {{${key}}} to the template`}
                      className="rounded-md border border-[var(--border)] bg-[var(--color-ink-850)] px-1.5 py-0.5 font-mono text-[12px] text-[var(--color-accent)] transition-colors hover:border-[var(--color-accent)]"
                    >
                      {`{{${key}}}`}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        </Section>
      </Stack>
    </PageBody>
  );
}
