import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { HeartHandshake, Loader2, User, Users } from "lucide-react";

import {
  relationshipsApi,
  type RelationshipCharacterDto,
  type RelationshipEntryDto,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  EmptyState,
  PageBody,
  PageHeader,
  Stack,
  StatusPill,
  TextArea,
} from "@/components/ui/page";

// Relationships — the Gamemaster's directional ledger: how each character
// currently regards the player and other NPCs. Entries appear and evolve
// automatically (the GM pass runs on every game save); this page lists every
// directional pair grouped by character and lets the user correct or clear
// the prose. Clearing removes the pair entirely (nothing is injected).

function formatWhen(iso?: string): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString();
}

function EntryRow({
  character,
  entry,
}: {
  character: RelationshipCharacterDto;
  entry: RelationshipEntryDto;
}) {
  const qc = useQueryClient();
  const [draft, setDraft] = useState(entry.text);
  const dirty = draft.trim() !== entry.text.trim();

  const save = useMutation({
    mutationFn: (text: string) =>
      relationshipsApi.save(character.characterId, entry.targetId, text),
    onSuccess: (view) => qc.setQueryData(["relationships"], view),
  });

  const when = formatWhen(entry.updatedAt ?? entry.createdAt);
  return (
    <div className="border-t border-[var(--line-soft)] px-5 py-4 first:border-t-0">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1.5 text-sm font-medium">
          {entry.targetKind === "player" ? (
            <User className="size-3.5 text-[var(--color-player)]" />
          ) : (
            <Users className="size-3.5 text-[var(--color-npc)]" />
          )}
          Toward {entry.targetName}
        </span>
        {when && (
          <span className="text-xs text-[var(--muted-foreground)]">
            updated {when}
          </span>
        )}
        <span className="ml-auto flex items-center gap-2">
          <Button
            size="sm"
            variant="ghost"
            disabled={save.isPending}
            onClick={() => {
              if (
                window.confirm(
                  `Clear ${character.characterName}'s view of ${entry.targetName}? The pair disappears until the Gamemaster observes them again.`,
                )
              ) {
                save.mutate("");
              }
            }}
          >
            Clear
          </Button>
          <Button
            size="sm"
            variant="secondary"
            disabled={!dirty || save.isPending}
            onClick={() => save.mutate(draft)}
          >
            {save.isPending ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : null}
            Save
          </Button>
        </span>
      </div>
      <TextArea
        rows={3}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        aria-label={`${character.characterName}'s view of ${entry.targetName}`}
      />
      {save.isError && (
        <p className="mt-1 text-xs text-[var(--color-danger)]">
          Save failed: {(save.error as Error).message}
        </p>
      )}
    </div>
  );
}

export function Relationships() {
  const query = useQuery({
    queryKey: ["relationships"],
    queryFn: relationshipsApi.list,
    refetchInterval: (q) => (q.state.data?.passInFlight ? 3000 : false),
  });

  const view = query.data;
  const lastPass = formatWhen(view?.lastPassAt);
  const total =
    view?.characters.reduce((sum, c) => sum + c.entries.length, 0) ?? 0;

  return (
    <PageBody width="wide">
      <PageHeader
        eyebrow="Gamemaster"
        title="Relationships"
        description="How each character currently regards the player and other NPCs. The Gamemaster reads new conversations on every game save and creates, warms, cools, or rewrites these entries on its own — edit or clear any entry to correct it."
        actions={
          view?.passInFlight ? (
            <StatusPill tone="busy" pulse>
              Gamemaster updating…
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
            icon={<HeartHandshake className="size-5" strokeWidth={1.75} />}
            title="Couldn't load relationships."
            description="Is the chasm server running? Try reloading."
          />
        ) : total === 0 ? (
          <EmptyState
            icon={<HeartHandshake className="size-5" strokeWidth={1.75} />}
            title="No relationships yet."
            description="They appear automatically: after each game save, the Gamemaster reads what was said since the last save and records how characters have come to regard the player and each other. Talk to someone in-game, then save."
          />
        ) : (
          <Stack>
            {view!.characters.map((character) => (
              <Card key={character.characterId}>
                <CardHeader className="pb-3">
                  <CardTitle>{character.characterName}</CardTitle>
                </CardHeader>
                <CardContent className="p-0 pb-1">
                  {character.entries.map((entry) => (
                    <EntryRow
                      // updatedAt in the key remounts the row (resetting the
                      // draft) when a GM pass rewrites the entry underneath.
                      key={`${character.characterId}→${entry.targetId}@${entry.updatedAt ?? ""}`}
                      character={character}
                      entry={entry}
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
