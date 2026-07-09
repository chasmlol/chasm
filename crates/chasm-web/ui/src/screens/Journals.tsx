import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, NotebookPen, Trash2 } from "lucide-react";

import { journalsApi, type JournalEntryDto } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  EmptyState,
  PageBody,
  PageHeader,
  Stack,
  StatusPill,
} from "@/components/ui/page";

// Journals — each NPC's private, append-only inner voice. After every game save
// the journal pass writes one new entry per NPC (in character), reflecting on
// what happened since the last save and any patterns they've started to notice.
// Read-only: entries are only ever added, never edited here or by the pass.

function formatWhen(iso?: string): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString();
}

function entryWhen(entry: JournalEntryDto): string {
  if (entry.gameTime) return entry.gameTime;
  return formatWhen(entry.createdAt) ?? entry.createdAt;
}

function EntryRow({
  entry,
  characterId,
}: {
  entry: JournalEntryDto;
  characterId: string;
}) {
  const qc = useQueryClient();
  const remove = useMutation({
    mutationFn: () => journalsApi.deleteEntry(characterId, entry.createdAt),
    onSuccess: (v) => qc.setQueryData(["journals"], v),
  });
  const when = entryWhen(entry);
  const dayLabel = typeof entry.gameDay === "number" ? `Day ${entry.gameDay}` : null;
  return (
    <div className="group border-t border-[var(--line-soft)] px-5 py-4 first:border-t-0">
      <div className="mb-1.5 flex items-center gap-2 text-xs text-[var(--muted-foreground)]">
        {dayLabel && (
          <span className="font-medium text-[var(--foreground)]">{dayLabel}</span>
        )}
        <span title={formatWhen(entry.createdAt) ?? undefined}>{when}</span>
        <Button
          size="sm"
          variant="ghost"
          className="ml-auto opacity-0 transition-opacity group-hover:opacity-100"
          disabled={remove.isPending}
          title="Delete this entry"
          onClick={() => {
            if (window.confirm("Delete this journal entry? This can't be undone.")) {
              remove.mutate();
            }
          }}
        >
          <Trash2 className="size-3.5" />
        </Button>
      </div>
      <p className="whitespace-pre-wrap text-sm leading-relaxed text-[var(--foreground)]">
        {entry.text}
      </p>
    </div>
  );
}

export function Journals() {
  const query = useQuery({
    queryKey: ["journals"],
    queryFn: journalsApi.list,
    refetchInterval: (q) => (q.state.data?.passInFlight ? 3000 : false),
  });

  const view = query.data;
  const lastPass = formatWhen(view?.lastPassAt);
  const total =
    view?.characters.reduce((sum, c) => sum + c.entries.length, 0) ?? 0;

  return (
    <PageBody width="wide">
      <PageHeader
        eyebrow="Self-improvement"
        title="Journals"
        description="Each NPC's private journal. After every game save they reflect, in their own voice, on what has happened and any patterns they've noticed — the raw material the skill-creator reads to decide what habits they should pick up. The pass only ever adds entries; hover an entry to delete one by hand."
        actions={
          view?.passInFlight ? (
            <StatusPill tone="busy" pulse>
              Journaling…
            </StatusPill>
          ) : lastPass ? (
            <StatusPill tone="idle">Last pass {lastPass}</StatusPill>
          ) : undefined
        }
      />
      <div className="mt-[var(--gap,14px)] flex-1 overflow-y-auto pb-6">
        {query.isLoading ? (
          <div className="grid h-40 place-items-center text-[var(--muted-foreground)]">
            <Loader2 className="size-5 animate-spin" />
          </div>
        ) : query.isError ? (
          <EmptyState
            icon={<NotebookPen className="size-5" strokeWidth={1.75} />}
            title="Couldn't load journals."
            description="Is the chasm server running? Try reloading."
          />
        ) : total === 0 ? (
          <EmptyState
            icon={<NotebookPen className="size-5" strokeWidth={1.75} />}
            title="No journal entries yet."
            description="They appear automatically: after each game save, every NPC who was around writes a short private entry about what happened. Talk to someone in-game, do a few things, then save."
          />
        ) : (
          <Stack>
            {view!.characters.map((character) => (
              <Card key={character.characterId}>
                <CardHeader className="pb-3">
                  <CardTitle>{character.characterName}</CardTitle>
                </CardHeader>
                <CardContent className="p-0 pb-1">
                  {/* Newest entry first for a scannable timeline. */}
                  {[...character.entries].reverse().map((entry, i) => (
                    <EntryRow
                      key={`${character.characterId}@${entry.createdAt}#${i}`}
                      entry={entry}
                      characterId={character.characterId}
                    />
                  ))}
                </CardContent>
              </Card>
            ))}
          </Stack>
        )}
      </div>
    </PageBody>
  );
}
