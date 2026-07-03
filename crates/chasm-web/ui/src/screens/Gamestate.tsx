import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Braces, Play, Variable } from "lucide-react";

import { gamestateApi, type GamestateTestDto } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  EmptyState,
  Field,
  PageBody,
  PageHeader,
  Section,
  Stack,
  StatusPill,
  Table,
  Td,
  TextArea,
  Th,
} from "@/components/ui/page";
import { cn } from "@/lib/utils";

// ===========================================================================
// Gamestate — the macros page.
//
// TOP: the LATEST recorded gamestate macro table — the flat `metadata.macros`
// map the FNV mod extracted on the most recent in-game turn (player name,
// location, time of day, health, SPECIAL, skills, perks, equipment, inventory,
// quests, …), grouped for eyeballing. This is how you confirm extraction is
// right: `major_location` should say `Goodsprings` while you stand in
// Goodsprings.
//
// BOTTOM: the tester — paste a system-prompt template containing `{{macro}}`
// placeholders and Run test. The backend resolves it against the latest
// recorded table (unknown macros → empty), runs ONE minimal system+user
// generation, and returns the RESOLVED prompt + the model's reply. The tester
// itself never touches real NPC prompts; production macro use is scoped to
// the GLOBAL scenario template (Globals → Scenario).
// ===========================================================================

const DEFAULT_TEMPLATE =
  "You are speaking to {{player_name}} in {{major_location}}. It is {{time_of_day}}.";
const DEFAULT_USER_MESSAGE = "Greet me and mention where we are.";

/** Display grouping for the known macro vocabulary; anything the mod sends
 *  that isn't listed lands in "Other" so new keys are never hidden. */
const MACRO_GROUPS: { label: string; keys: string[] }[] = [
  { label: "Player", keys: ["player_name", "level"] },
  { label: "Location", keys: ["major_location", "minor_location"] },
  { label: "Time", keys: ["time_of_day"] },
  {
    label: "Status",
    keys: ["health", "health_percent", "radiation", "condition", "effects"],
  },
  { label: "S.P.E.C.I.A.L.", keys: ["special"] },
  { label: "Skills", keys: ["skills"] },
  { label: "Perks", keys: ["perks"] },
  { label: "Equipment", keys: ["equipped_weapon", "equipped_apparel"] },
  { label: "Inventory", keys: ["inventory"] },
  { label: "Quests", keys: ["quests", "misc_quests"] },
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

/** One rendered group of the macro table. */
interface MacroGroupRows {
  label: string;
  rows: [string, string][];
}

/** Buckets the flat macro map into the display groups (+ "Other"). */
function groupMacros(macros: Record<string, string>): MacroGroupRows[] {
  const remaining = new Map(Object.entries(macros));
  const groups: MacroGroupRows[] = [];
  for (const group of MACRO_GROUPS) {
    const rows: [string, string][] = [];
    for (const key of group.keys) {
      const value = remaining.get(key);
      if (value !== undefined) {
        rows.push([key, value]);
        remaining.delete(key);
      }
    }
    if (rows.length > 0) groups.push({ label: group.label, rows });
  }
  if (remaining.size > 0) {
    groups.push({
      label: "Other",
      rows: [...remaining.entries()].sort(([a], [b]) => a.localeCompare(b)),
    });
  }
  return groups;
}

/** A monospace output panel of the tester (resolved prompt / model reply). */
function OutputPanel({
  label,
  text,
  accent = false,
}: {
  label: string;
  text: string;
  accent?: boolean;
}) {
  return (
    <div className="min-w-0">
      <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted-foreground)]">
        {label}
      </p>
      <pre
        className={cn(
          "max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-xl border border-[var(--border)] bg-[var(--color-ink-850)] px-3.5 py-3 font-mono text-[13px] leading-relaxed",
          accent ? "text-[var(--foreground)]" : "text-[var(--muted-foreground)]",
        )}
      >
        {text}
      </pre>
    </div>
  );
}

export function Gamestate() {
  const query = useQuery({
    queryKey: ["gamestate", "view"],
    queryFn: () => gamestateApi.view(),
    // The user is live in-game; a talk-to-NPC turn should show up unprompted.
    refetchInterval: 4000,
  });

  const macros = query.data?.macros ?? {};
  const groups = useMemo(() => groupMacros(macros), [macros]);
  const hasMacros = groups.length > 0;

  const [template, setTemplate] = useState(DEFAULT_TEMPLATE);
  const [userMessage, setUserMessage] = useState(DEFAULT_USER_MESSAGE);
  const [result, setResult] = useState<GamestateTestDto | null>(null);

  const test = useMutation({
    mutationFn: () =>
      gamestateApi.test({ template, user_message: userMessage }),
    onSuccess: (data) => setResult(data),
  });

  /** Clicking a key chip appends its placeholder to the template. */
  const appendMacro = (key: string) =>
    setTemplate((current) =>
      current.length === 0 || current.endsWith(" ")
        ? `${current}{{${key}}}`
        : `${current} {{${key}}}`,
    );

  return (
    <PageBody width="wide" className="overflow-y-auto">
      <PageHeader
        eyebrow="Main"
        title="Gamestate"
        description={
          <>
            The live macro table the mod extracts each turn, and a tester that
            resolves <code className="font-mono">{"{{macro}}"}</code>{" "}
            placeholders against it and runs a generation. In production these
            macros drive the global scenario (Globals → Scenario); this page is
            the raw proof surface.
          </>
        }
        actions={
          query.data?.updated_at ? (
            <StatusPill tone="ok">
              Updated {formatUpdatedAt(query.data.updated_at)}
            </StatusPill>
          ) : (
            <StatusPill tone="idle">No turn recorded yet</StatusPill>
          )
        }
      />

      <Stack className="py-[var(--gap,14px)]">
        <Section
          title="Latest macros"
          description="Recorded from the most recent in-game turn's metadata.macros. Click a key to append its placeholder to the test template."
        >
          {query.isLoading ? (
            <EmptyState icon={<Variable />} title="Loading gamestate…" />
          ) : hasMacros ? (
            <Table
              head={
                <tr>
                  <Th className="w-40">Group</Th>
                  <Th className="w-56">Macro</Th>
                  <Th>Value</Th>
                </tr>
              }
            >
              {groups.flatMap((group) =>
                group.rows.map(([key, value], index) => (
                  <tr
                    key={key}
                    className="transition-colors hover:bg-[var(--color-ink-850)]/60"
                  >
                    {index === 0 && (
                      // One label cell spanning the group's rows.
                      <td
                        rowSpan={group.rows.length}
                        className="border-b border-[var(--line-soft)] px-3 py-2.5 align-top text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]"
                      >
                        {group.label}
                      </td>
                    )}
                    <Td className="align-top">
                      <button
                        type="button"
                        onClick={() => appendMacro(key)}
                        title={`Append {{${key}}} to the test template`}
                        className="rounded-md border border-[var(--border)] bg-[var(--color-ink-850)] px-1.5 py-0.5 font-mono text-[12px] text-[var(--color-accent)] transition-colors hover:border-[var(--color-accent)]"
                      >
                        {`{{${key}}}`}
                      </button>
                    </Td>
                    <Td className="whitespace-pre-wrap break-words text-[13px] leading-relaxed">
                      {value}
                    </Td>
                  </tr>
                )),
              )}
            </Table>
          ) : (
            <EmptyState
              icon={<Variable />}
              title="No macros recorded yet"
              description="Talk to an NPC in-game (with the bridge running) and the turn's extracted gamestate will appear here."
            />
          )}
        </Section>

        <Section
          title="Test a template"
          description="Write a system prompt with {{macro}} placeholders, then run a single test generation. The resolved prompt shows each macro filled in from the table above; unknown macros resolve to empty."
        >
          <div className="flex flex-col gap-3 rounded-xl border border-[var(--border)] bg-[var(--color-ink-900,transparent)] p-3.5">
            <TextArea
              rows={3}
              value={template}
              onChange={(event) => setTemplate(event.target.value)}
              placeholder={DEFAULT_TEMPLATE}
              spellCheck={false}
              className="font-mono text-[13px]"
            />
            <div className="flex items-center gap-2">
              <Field
                value={userMessage}
                onChange={(event) => setUserMessage(event.target.value)}
                placeholder={DEFAULT_USER_MESSAGE}
                aria-label="User message"
              />
              <Button
                onClick={() => test.mutate()}
                disabled={test.isPending || template.trim().length === 0}
                className="shrink-0"
              >
                <Play />
                {test.isPending ? "Running…" : "Run test"}
              </Button>
            </div>

            {test.isError && (
              <p className="text-[13px] text-[var(--color-danger)]">
                Test failed: {(test.error as Error).message}. Is the LLM
                running? (Models → LLM)
              </p>
            )}

            {result && !test.isError && (
              <div className="flex flex-col gap-3">
                {result.note && (
                  <p className="text-[13px] text-[var(--color-npc,orange)]">
                    {result.note}
                  </p>
                )}
                <OutputPanel label="Resolved prompt" text={result.resolved_prompt} />
                {result.reply.length > 0 && (
                  <OutputPanel label="Model reply" text={result.reply} accent />
                )}
              </div>
            )}

            {!result && !test.isError && !test.isPending && (
              <p className="flex items-center gap-1.5 text-[13px] text-[var(--muted-foreground)]">
                <Braces className="size-3.5" />
                The resolved prompt and the model's reply will appear here.
              </p>
            )}
          </div>
        </Section>
      </Stack>
    </PageBody>
  );
}
