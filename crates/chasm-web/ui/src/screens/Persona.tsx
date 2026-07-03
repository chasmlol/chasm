import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { Camera, RefreshCw, UserRound } from "lucide-react";

import { personaApi, personaImageUrl, type PersonaViewDto } from "@/lib/api";
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
// The FNV mod stealth-captures the player (front screenshot + stats snapshot)
// whenever the build or outfit changes; the backend turns it into a compact
// third-person description with a vision-capable LLM (stats-only fallback)
// and injects it into NPC prompts at SillyTavern's persona slot. This page
// shows the last capture (the actual image), the generated description, when
// it was generated, the stats snapshot it used, and a Regenerate button (the
// manual test hook that re-runs generation from the last capture).
// ===========================================================================

/** Display order + labels for the stats snapshot table. */
const STAT_ROWS: { key: string; label: string }[] = [
  { key: "player_name", label: "Name" },
  { key: "level", label: "Level" },
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

/** Human label for the generation source. */
function sourceLabel(view: PersonaViewDto): string {
  if (view.source === "vision") return "Described from screenshot";
  if (view.source === "stats_only") return "Described from stats only";
  return "Not generated yet";
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
  const hasAnything = Boolean(
    view && (view.has_capture || view.has_image || view.description),
  );
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
            Who the NPCs think they&apos;re talking to. The mod quietly
            photographs your character and snapshots your stats when your build
            or outfit changes; the backend writes a persona description and
            weaves it into every NPC prompt.
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
            icon={<Camera />}
            title="No capture yet"
            description="Play with the bridge running: level up, spend skill points, pick a perk, or change your outfit. The mod will quietly capture your character and this page will fill in — no button pressing needed."
          />
        ) : (
          <>
            <div className="grid gap-[var(--gap,14px)] lg:grid-cols-2">
              <Section
                title="Last capture"
                description="The most recent stealth screenshot the mod took of your character."
              >
                {view?.has_image ? (
                  <img
                    src={personaImageUrl(
                      view.captured_at ?? view.generated_at ?? undefined,
                    )}
                    alt="Last persona capture of the player character"
                    className="w-full rounded-xl border border-[var(--border)] bg-[var(--color-ink-850)] object-contain"
                  />
                ) : (
                  <EmptyState
                    icon={<Camera />}
                    title="No screenshot stored"
                    description="The last capture arrived without an image (screenshot failed or was skipped), so the persona was generated from stats alone."
                  />
                )}
              </Section>

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
                      <StatusPill
                        tone={view.source === "vision" ? "ok" : "idle"}
                      >
                        {sourceLabel(view)}
                      </StatusPill>
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
                        Last generation attempt failed:{" "}
                        {view.generation_error} — showing the previous
                        description.
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
            </div>

            <Section
              title="Stats snapshot"
              description="What the mod extracted alongside the screenshot — the raw material the description was generated from."
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
                  title="No stats in the last capture"
                />
              )}
            </Section>
          </>
        )}
      </Stack>
    </PageBody>
  );
}
