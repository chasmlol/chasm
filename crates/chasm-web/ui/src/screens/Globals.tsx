import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowDown,
  ArrowUp,
  Braces,
  Globe,
  RotateCcw,
  Save,
} from "lucide-react";

import { globalsApi } from "@/lib/api";
import type {
  GlobalsScenarioPreviewState,
  GlobalsScenarioVariantConfig,
  GlobalsScenarioVariantDto,
} from "@/lib/api";
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
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

// ===========================================================================
// Globals → Scenario — the GLOBAL scenario template + DYNAMIC variants.
//
// One app-wide template replaces the per-character card "Scenario" field.
// Every NPC turn the backend resolves its {{macro}} placeholders against the
// gamestate the mod sent for that turn (falling back to the latest recorded
// table) plus backend-computed macros ({{participants}}), and injects the
// result into the prompt exactly where the card scenario used to sit.
//
// DYNAMIC SCENARIOS: the wording now varies with the situation. A fixed
// catalog of variants (companion / following / sneaking-together / traveling /
// waiting / …) each carries its own editable template; per turn the backend
// picks the highest-priority enabled variant whose gamestate condition holds,
// else the default template below. Conditions are read-only — they bind to
// real engine flags, never to chat content.
//
// TOP: the default template editor + Save / Reset to default.
// THEN: the variants list (enable, reword, reprioritise).
// THEN: a LIVE resolved preview — the current drafts rendered through the
// latest recorded macros, with a state-picker to preview any situation.
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
  {
    key: "travel_destination",
    hint: "Computed: where the NPC's active journey is headed (empty when not traveling).",
    computed: true,
  },
  {
    key: "travel_arrival_time",
    hint: "Computed: the journey's scheduled arrival, e.g. \"3:00PM\" (empty when not traveling).",
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

/** The state-picker flags, in display order, with human labels. */
const STATE_FLAGS: { key: keyof GlobalsScenarioPreviewState; label: string }[] = [
  { key: "teammate", label: "Companion" },
  { key: "following", label: "Following" },
  { key: "waiting", label: "Waiting" },
  { key: "sneaking", label: "NPC sneaking" },
  { key: "player_sneaking", label: "Player sneaking" },
  { key: "weapon_drawn", label: "NPC weapon drawn" },
  { key: "player_weapon_drawn", label: "Player weapon drawn" },
  { key: "sitting", label: "Sitting" },
  { key: "player_swimming", label: "Player swimming" },
  { key: "traveling", label: "Traveling" },
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

/** The editable slice of a variant (what PUT sends). */
function toConfig(variant: GlobalsScenarioVariantDto): GlobalsScenarioVariantConfig {
  return {
    id: variant.id,
    enabled: variant.enabled,
    priority: variant.priority,
    template: variant.template,
  };
}

export function Globals() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["globals", "scenario"],
    queryFn: () => globalsApi.scenario(),
  });

  // The editor drafts. `null` until the saved data loads (so we never clobber
  // user edits with a refetch — only the FIRST load seeds them).
  const [draft, setDraft] = useState<string | null>(null);
  const [variantsDraft, setVariantsDraft] = useState<
    GlobalsScenarioVariantConfig[] | null
  >(null);
  useEffect(() => {
    if (draft === null && query.data) setDraft(query.data.template);
    if (variantsDraft === null && query.data)
      setVariantsDraft(query.data.variants.map(toConfig));
  }, [draft, variantsDraft, query.data]);

  const template = draft ?? query.data?.template ?? "";
  const savedTemplate = query.data?.template ?? "";
  const variants = useMemo(
    () => variantsDraft ?? (query.data?.variants ?? []).map(toConfig),
    [variantsDraft, query.data],
  );
  /** Catalog facts (label, condition, defaults) by variant id. */
  const catalog = useMemo(() => {
    const map = new Map<string, GlobalsScenarioVariantDto>();
    for (const variant of query.data?.variants ?? []) map.set(variant.id, variant);
    return map;
  }, [query.data]);

  const templateDirty =
    draft !== null && query.data !== undefined && draft !== savedTemplate;
  const variantsDirty =
    variantsDraft !== null &&
    query.data !== undefined &&
    JSON.stringify(variantsDraft) !==
      JSON.stringify(query.data.variants.map(toConfig));
  const dirty = templateDirty || variantsDirty;

  const save = useMutation({
    mutationFn: () =>
      globalsApi.saveScenario({ template, variants }),
    onSuccess: (data) => {
      qc.setQueryData(["globals", "scenario"], data);
      setDraft(data.template);
      setVariantsDraft(data.variants.map(toConfig));
    },
  });

  // --- variant editing helpers --------------------------------------------
  const updateVariant = (
    id: string,
    change: Partial<GlobalsScenarioVariantConfig>,
  ) =>
    setVariantsDraft(
      variants.map((variant) =>
        variant.id === id ? { ...variant, ...change } : variant,
      ),
    );

  /** Variants in SELECTION order (priority desc, id for stable ties). */
  const orderedVariants = useMemo(
    () =>
      [...variants].sort(
        (a, b) => b.priority - a.priority || a.id.localeCompare(b.id),
      ),
    [variants],
  );

  /** Moves a variant one slot up/down the selection order by swapping the
   *  priority numbers with its neighbour. */
  const moveVariant = (id: string, direction: -1 | 1) => {
    const index = orderedVariants.findIndex((variant) => variant.id === id);
    const neighbour = orderedVariants[index + direction];
    if (index < 0 || !neighbour) return;
    const self = orderedVariants[index];
    // Equal priorities would make the swap a no-op; nudge instead.
    const selfPriority =
      self.priority === neighbour.priority
        ? neighbour.priority + (direction === -1 ? 1 : -1)
        : neighbour.priority;
    setVariantsDraft(
      variants.map((variant) =>
        variant.id === self.id
          ? { ...variant, priority: selfPriority }
          : variant.id === neighbour.id
            ? { ...variant, priority: self.priority }
            : variant,
      ),
    );
  };

  // --- preview (with optional state-picker) --------------------------------
  const [pickState, setPickState] = useState(false);
  const [pickedFlags, setPickedFlags] = useState<GlobalsScenarioPreviewState>({});
  const debouncedTemplate = useDebounced(template, 400);
  const debouncedVariants = useDebounced(variants, 400);
  const preview = useQuery({
    queryKey: [
      "globals",
      "scenario-preview",
      debouncedTemplate,
      pickState ? JSON.stringify(debouncedVariants) : "",
      pickState ? JSON.stringify(pickedFlags) : "",
    ],
    queryFn: () =>
      globalsApi.previewScenario(
        pickState
          ? {
              template: debouncedTemplate,
              variants: debouncedVariants,
              state: pickedFlags,
            }
          : { template: debouncedTemplate },
      ),
    enabled: query.data !== undefined,
    refetchInterval: 4000,
  });

  /** Clicking a macro chip appends its placeholder to the default template. */
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
            resolved with the live gamestate on every turn, and the wording
            adapts to the situation via the variants below — companion,
            sneaking, traveling, and more.
          </>
        }
        actions={statusPill}
      />

      <Stack className="py-[var(--gap,14px)]">
        <Section
          title="Default template"
          description="Used when no situation variant below matches. Missing macros resolve to empty, so prefer short, separate sentences. Clear the template entirely to omit the scenario from prompts."
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
                  onClick={() => save.mutate()}
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
          title="Situation variants"
          description="When an NPC's game state matches a condition, that variant's wording replaces the default template (highest priority wins; exactly one scenario is ever injected). Conditions are fixed engine states — edit the wording, order, and enablement. A blank template falls through to the next match. Combat is separate: the combat directive still rides on top of whichever scenario is active."
        >
          <div className="flex flex-col gap-2.5">
            {orderedVariants.map((variant, index) => {
              const facts = catalog.get(variant.id);
              const isDefaultText =
                facts !== undefined && variant.template === facts.default_template;
              return (
                <div
                  key={variant.id}
                  className={cn(
                    "flex flex-col gap-2 rounded-xl border border-[var(--border)] bg-[var(--color-ink-900,transparent)] p-3",
                    !variant.enabled && "opacity-60",
                  )}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <Switch
                      checked={variant.enabled}
                      onCheckedChange={(checked) =>
                        updateVariant(variant.id, { enabled: checked })
                      }
                      title={variant.enabled ? "Enabled" : "Disabled"}
                    />
                    <span className="text-[13.5px] font-semibold">
                      {facts?.label || variant.id}
                    </span>
                    <span className="rounded-md border border-[var(--border)] bg-[var(--color-ink-850)] px-1.5 py-0.5 font-mono text-[11px] text-[var(--muted-foreground)]">
                      priority {variant.priority}
                    </span>
                    <span className="ml-auto flex items-center gap-1">
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => moveVariant(variant.id, -1)}
                        disabled={index === 0}
                        title="Match earlier (higher priority)"
                      >
                        <ArrowUp />
                      </Button>
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => moveVariant(variant.id, 1)}
                        disabled={index === orderedVariants.length - 1}
                        title="Match later (lower priority)"
                      >
                        <ArrowDown />
                      </Button>
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() =>
                          facts &&
                          updateVariant(variant.id, {
                            template: facts.default_template,
                            priority: facts.default_priority,
                            enabled: true,
                          })
                        }
                        disabled={
                          !facts ||
                          (isDefaultText &&
                            variant.priority === facts.default_priority &&
                            variant.enabled)
                        }
                        title="Restore this variant's shipped wording, priority, and enablement"
                      >
                        <RotateCcw />
                      </Button>
                    </span>
                  </div>
                  <p className="text-[12.5px] text-[var(--muted-foreground)]">
                    <span className="font-semibold uppercase tracking-[0.08em] text-[11px]">
                      When:{" "}
                    </span>
                    {facts?.condition_hint ||
                      "Unknown condition (saved by a newer build) — never matches."}
                  </p>
                  <TextArea
                    rows={3}
                    value={variant.template}
                    onChange={(event) =>
                      updateVariant(variant.id, { template: event.target.value })
                    }
                    placeholder="(blank — falls through to the next matching variant)"
                    spellCheck={false}
                    className="font-mono text-[13px]"
                  />
                </div>
              );
            })}
            {orderedVariants.length === 0 && !query.isLoading && (
              <EmptyState icon={<Globe />} title="No variants" />
            )}
          </div>
        </Section>

        <Section
          title="Resolved preview"
          description="The drafts above rendered through the most recently recorded gamestate — what the NPC will actually read. Pick a situation to see which variant wins without being in game."
        >
          <div className="flex flex-col gap-3 rounded-xl border border-[var(--border)] bg-[var(--color-ink-900,transparent)] p-3.5">
            <div className="flex flex-wrap items-center gap-2">
              <Switch checked={pickState} onCheckedChange={setPickState} />
              <span className="text-[13px]">
                Preview a situation (state-picker)
              </span>
              {pickState && (
                <Button
                  variant="outline"
                  size="sm"
                  className="ml-auto"
                  onClick={() => setPickedFlags({})}
                  disabled={Object.values(pickedFlags).every((flag) => !flag)}
                >
                  Clear flags
                </Button>
              )}
            </div>
            {pickState && (
              <div className="flex flex-wrap gap-1.5">
                {STATE_FLAGS.map((flag) => {
                  const active = pickedFlags[flag.key] === true;
                  return (
                    <button
                      key={flag.key}
                      type="button"
                      onClick={() =>
                        setPickedFlags((current) => ({
                          ...current,
                          [flag.key]: !active,
                        }))
                      }
                      className={cn(
                        "rounded-md border px-2 py-1 text-[12.5px] transition-colors",
                        active
                          ? "border-[var(--color-accent)] bg-[var(--color-ink-850)] text-[var(--color-accent)]"
                          : "border-[var(--border)] bg-[var(--color-ink-850)] text-[var(--muted-foreground)] hover:border-[var(--color-accent)]",
                      )}
                    >
                      {flag.label}
                    </button>
                  );
                })}
              </div>
            )}
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
                  <span>
                    Resolved scenario
                    {pickState && preview.data.variant_label && (
                      <span className="ml-2 rounded-md border border-[var(--border)] bg-[var(--color-ink-850)] px-1.5 py-0.5 font-mono normal-case tracking-normal text-[var(--color-accent)]">
                        variant: {preview.data.variant_label}
                      </span>
                    )}
                  </span>
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
          description="Click a macro to append it to the default template (paste into any variant). The mod sends these every turn; whatever it cannot read resolves to empty. Computed macros come from the backend."
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
