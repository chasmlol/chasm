import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { RefreshCw, UserRound } from "lucide-react";

import { personaApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  EmptyState,
  PageBody,
  PageHeader,
  Section,
  Stack,
  StatusPill,
  Table,
  Td,
  Th,
} from "@/components/ui/page";

// ===========================================================================
// Persona — the player-persona page.
//
// The FNV mod captures the player's character data (stats + appearance: sex,
// race, hair, eyes, facial hair, outfit) every time the game is saved; the
// backend turns it into a two-paragraph third-person description with the
// main LLM and injects it into NPC prompts at SillyTavern's persona slot.
// This page shows the generated description, when it was generated, the
// character-data snapshot it used, the exact prompt sent to the LLM, and a
// Regenerate button (the manual test hook that re-runs generation from the
// last capture).
// ===========================================================================

/** Display order + labels for the character-data snapshot table. */
const STAT_ROWS: { key: string; label: string }[] = [
  { key: "player_name", label: "Name" },
  { key: "level", label: "Level" },
  { key: "sex", label: "Sex" },
  { key: "race", label: "Race" },
  { key: "age_years", label: "Age (FaceGen)" },
  { key: "hair_style", label: "Hair style" },
  { key: "hair_color", label: "Hair color" },
  { key: "hair_length", label: "Hair length" },
  { key: "eye_color", label: "Eye color" },
  { key: "facial_hair", label: "Facial hair" },
  { key: "special", label: "S.P.E.C.I.A.L." },
  { key: "skills", label: "Skills" },
  { key: "perks", label: "Perks" },
  { key: "equipped_weapon", label: "Equipped weapon" },
  { key: "equipped_apparel", label: "Outfit" },
  { key: "location", label: "Location" },
];

/** "Jun 20, 2026, 9:28 PM"-style label for ISO timestamps. */
function formatTimestamp(iso: string): string {
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

export function Persona() {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["persona", "view"],
    queryFn: () => personaApi.view(),
    // Captures land unprompted while the user plays; keep the page live.
    refetchInterval: 5000,
  });

  const regenerate = useMutation({
    mutationFn: () => personaApi.regenerate(),
    onSuccess: (data) => queryClient.setQueryData(["persona", "view"], data),
  });

  const view = query.data;
  const hasAnything = Boolean(view && (view.has_capture || view.description));
  const statRows = STAT_ROWS.map(({ key, label }) => ({
    key,
    label,
    value: view?.stats?.[key],
  })).filter((row) => row.value !== undefined && `${row.value}`.length > 0);

  return (
    <PageBody width="wide" className="overflow-y-auto">
      <PageHeader
        eyebrow="Main"
        title="Persona"
        description={
          <>
            Who the NPCs think they&apos;re talking to. Each time you save the
            game, the mod snapshots your character&apos;s stats and appearance
            data; the backend writes a persona description and weaves it into
            every NPC prompt.
          </>
        }
        actions={
          <div className="flex items-center gap-2">
            {view?.generating || regenerate.isPending ? (
              <StatusPill tone="warn">Generating…</StatusPill>
            ) : view?.generated_at ? (
              <StatusPill tone="ok">
                Generated {formatTimestamp(view.generated_at)}
              </StatusPill>
            ) : (
              <StatusPill tone="idle">No persona yet</StatusPill>
            )}
            <Button
              onClick={() => regenerate.mutate()}
              disabled={
                !view?.has_capture || regenerate.isPending || view?.generating
              }
              title={
                view?.has_capture
                  ? "Re-run generation from the last capture"
                  : "Regenerate becomes available after the first in-game capture"
              }
            >
              <RefreshCw
                className={regenerate.isPending ? "animate-spin" : undefined}
              />
              {regenerate.isPending ? "Regenerating…" : "Regenerate"}
            </Button>
          </div>
        }
      />

      <Stack className="py-[var(--gap,14px)]">
        {regenerate.isError && (
          <p className="text-[13px] text-[var(--color-danger)]">
            Regenerate failed: {(regenerate.error as Error).message}
          </p>
        )}

        {query.isLoading ? (
          <EmptyState icon={<UserRound />} title="Loading persona…" />
        ) : !hasAnything ? (
          <EmptyState
            icon={<UserRound />}
            title="No capture yet"
            description="Play with the bridge running and save your game (a quicksave works). The mod will capture your character data on save and this page will fill in — no button pressing needed."
          />
        ) : (
          <>
            <Section
              title="Generated persona"
              description="Injected into every NPC prompt at SillyTavern's persona position."
            >
              {view?.description ? (
                <div className="flex flex-col gap-2.5">
                  <p className="whitespace-pre-wrap rounded-xl border border-[var(--border)] bg-[var(--color-ink-850)] px-3.5 py-3 text-[13.5px] leading-relaxed text-[var(--foreground)]">
                    {view.description}
                  </p>
                  <div className="flex flex-wrap items-center gap-2 text-[12px] text-[var(--muted-foreground)]">
                    {view.captured_at && (
                      <span>Captured {formatTimestamp(view.captured_at)}</span>
                    )}
                  </div>
                  {view.model_note && (
                    <p className="text-[12px] text-[var(--muted-foreground)]">
                      {view.model_note}
                    </p>
                  )}
                  {view.generation_error && (
                    <p className="text-[12px] text-[var(--color-danger)]">
                      Last generation attempt failed: {view.generation_error} —
                      showing the previous description.
                    </p>
                  )}
                </div>
              ) : (
                <EmptyState
                  icon={<UserRound />}
                  title="Not generated yet"
                  description={
                    view?.generation_error
                      ? `Generation failed: ${view.generation_error}. Is the LLM running? (Settings → LLM) — then hit Regenerate.`
                      : view?.generating
                        ? "A capture arrived and the description is being generated…"
                        : "A capture is stored. Hit Regenerate to produce the description."
                  }
                />
              )}
            </Section>

            <Section
              title="Character data snapshot"
              description="What the mod extracted on the last save — the raw material the description was generated from."
            >
              {statRows.length > 0 ? (
                <Table
                  head={
                    <tr>
                      <Th className="w-48">Field</Th>
                      <Th>Value</Th>
                    </tr>
                  }
                >
                  {statRows.map((row) => (
                    <tr
                      key={row.key}
                      className="transition-colors hover:bg-[var(--color-ink-850)]/60"
                    >
                      <Td className="align-top text-[12px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
                        {row.label}
                      </Td>
                      <Td className="whitespace-pre-wrap break-words text-[13px] leading-relaxed">
                        {String(row.value)}
                      </Td>
                    </tr>
                  ))}
                </Table>
              ) : (
                <EmptyState
                  icon={<UserRound />}
                  title="No data in the last capture"
                />
              )}
            </Section>

            {view?.prompt && (
              <Section
                title="Generation prompt"
                description="The exact prompt sent to the LLM for the current description."
              >
                <details className="rounded-xl border border-[var(--border)] bg-[var(--color-ink-850)]">
                  <summary className="cursor-pointer select-none px-3.5 py-2.5 text-[13px] font-medium text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
                    Show the prompt
                  </summary>
                  <pre className="overflow-x-auto whitespace-pre-wrap break-words border-t border-[var(--border)] px-3.5 py-3 text-[12.5px] leading-relaxed text-[var(--foreground)]">
                    {view.prompt}
                  </pre>
                </details>
              </Section>
            )}
          </>
        )}
      </Stack>
    </PageBody>
  );
}
