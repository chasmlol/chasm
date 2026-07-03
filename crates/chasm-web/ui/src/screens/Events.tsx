import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { History, Loader2, Search } from "lucide-react";

import { eventsApi, type EventDto } from "@/lib/api";
import { cn } from "@/lib/utils";
import { EmptyState, Field, PageBody, PageHeader } from "@/components/ui/page";

// Events — the read-only chronicle of what happened in-game: kills, deaths,
// travel, loot, conversations, quest turns, level-ups, and so on. The log
// live-follows the game (poll every 4s) with newest events at the bottom;
// type chips + text search narrow the view. Nothing here mutates anything.

/** Canonical chip ordering; unknown types are appended after these. */
const EVENT_TYPE_ORDER = [
  "combat",
  "shooting",
  "death",
  "location",
  "item",
  "conversation",
  "quest",
  "level",
  "day",
  "companion",
  "karma",
  "world",
];

/** Per-type badge palette, following the bookBadge tinted-pill pattern. */
const MUTED_BADGE =
  "border-[var(--border)] bg-[var(--color-ink-850)] text-[var(--muted-foreground)]";

const TYPE_BADGE: Record<string, string> = {
  combat:
    "border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 text-[var(--color-danger)]",
  shooting:
    "border-[var(--color-danger)]/30 bg-[var(--color-danger)]/5 text-[var(--color-danger)]",
  death:
    "border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 text-[var(--color-danger)]",
  location:
    "border-[var(--color-accent)]/40 bg-[var(--color-accent)]/10 text-[var(--color-accent)]",
  item: "border-[var(--color-npc)]/40 bg-[var(--color-npc)]/10 text-[var(--color-npc)]",
  conversation:
    "border-[var(--color-player)]/40 bg-[var(--color-player)]/10 text-[var(--color-player)]",
};

/** In-game clock when we have it; otherwise a short wall-clock fallback. */
function eventWhen(event: EventDto): string {
  if (event.gameTime) return event.gameTime;
  const date = new Date(event.realTime);
  if (Number.isNaN(date.getTime())) return event.realTime;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function matchesSearch(event: EventDto, needle: string): boolean {
  if (!needle) return true;
  if (event.summary.toLowerCase().includes(needle)) return true;
  if (event.location?.toLowerCase().includes(needle)) return true;
  return (event.actors ?? []).some((actor) =>
    actor.name.toLowerCase().includes(needle),
  );
}

function TypeChip({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-full border px-2.5 py-1 text-xs font-medium transition-colors",
        active
          ? "border-[var(--color-accent)]/40 bg-[var(--color-accent)]/10 text-[var(--color-accent)]"
          : "border-[var(--border)] bg-[var(--color-ink-850)] text-[var(--muted-foreground)] hover:text-[var(--foreground)]",
      )}
    >
      {label}
    </button>
  );
}

function EventRow({ event }: { event: EventDto }) {
  const when = eventWhen(event);
  return (
    <div className="flex items-center gap-3 border-t border-[var(--line-soft)] px-4 py-2 first:border-t-0">
      <span
        className="w-36 shrink-0 truncate text-xs tabular-nums text-[var(--muted-foreground)]"
        title={new Date(event.realTime).toLocaleString()}
      >
        {when}
      </span>
      <span
        className={cn(
          "shrink-0 rounded-full border px-2 py-0.5 text-[11px] font-medium",
          TYPE_BADGE[event.type] ?? MUTED_BADGE,
        )}
      >
        {event.type}
      </span>
      <span className="min-w-0 flex-1 truncate text-sm" title={event.summary}>
        {event.summary}
      </span>
      {event.location && (
        <span
          className="max-w-48 shrink-0 truncate text-xs text-[var(--muted-foreground)]"
          title={event.location}
        >
          {event.location}
        </span>
      )}
    </div>
  );
}

export function Events() {
  const query = useQuery({
    queryKey: ["events"],
    queryFn: eventsApi.list,
    // The log live-follows the game; poll while the page is open.
    refetchInterval: 4000,
  });

  const [search, setSearch] = useState("");
  const [activeTypes, setActiveTypes] = useState<string[]>([]);

  const view = query.data;
  const events = useMemo(() => view?.events ?? [], [view]);

  // Chips: All + one per type actually present, in canonical order.
  const presentTypes = useMemo(() => {
    const present = new Set(events.map((event) => event.type));
    return [
      ...EVENT_TYPE_ORDER.filter((type) => present.has(type)),
      ...[...present].filter((type) => !EVENT_TYPE_ORDER.includes(type)).sort(),
    ];
  }, [events]);

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return events.filter(
      (event) =>
        (activeTypes.length === 0 || activeTypes.includes(event.type)) &&
        matchesSearch(event, needle),
    );
  }, [events, activeTypes, search]);

  // Auto-follow: newest events are at the bottom; when new data arrives and
  // the list was already scrolled near the bottom, keep it pinned there.
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const stickToBottom = useRef(true);
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [filtered]);

  const toggleType = (type: string) =>
    setActiveTypes((current) =>
      current.includes(type)
        ? current.filter((t) => t !== type)
        : [...current, type],
    );

  return (
    <PageBody width="wide">
      <PageHeader
        eyebrow="Chronicle"
        title="Events"
        description="A live log of everything the bridge observed in-game — combat, travel, loot, conversations, quests, and more. Read-only; new events append at the bottom as you play."
        actions={
          view ? (
            <span className="rounded-full border border-[var(--border)] bg-[var(--color-ink-850)] px-2.5 py-1 text-xs font-medium text-[var(--muted-foreground)]">
              {filtered.length} of {view.total} events
            </span>
          ) : undefined
        }
      />

      {query.isLoading ? (
        <div className="grid h-40 place-items-center text-[var(--muted-foreground)]">
          <Loader2 className="size-5 animate-spin" />
        </div>
      ) : query.isError ? (
        <div className="mt-[var(--gap,14px)]">
          <EmptyState
            icon={<History className="size-5" strokeWidth={1.75} />}
            title="Couldn't load events."
            description="Is the chasm server running? Try reloading."
          />
        </div>
      ) : events.length === 0 ? (
        <div className="mt-[var(--gap,14px)]">
          <EmptyState
            icon={<History className="size-5" strokeWidth={1.75} />}
            title="No events yet."
            description="Events appear automatically as you play — fight, travel, pick things up, or talk to someone in-game and the log fills in here."
          />
        </div>
      ) : (
        <>
          {/* Controls: text search + type filter chips */}
          <div className="mt-[var(--gap,14px)] flex flex-wrap items-center gap-2">
            <div className="relative min-w-56 flex-1">
              <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-[var(--muted-foreground)]/70" />
              <Field
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search events…"
                className="pl-9"
              />
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <TypeChip
                label="All"
                active={activeTypes.length === 0}
                onClick={() => setActiveTypes([])}
              />
              {presentTypes.map((type) => (
                <TypeChip
                  key={type}
                  label={type}
                  active={activeTypes.includes(type)}
                  onClick={() => toggleType(type)}
                />
              ))}
            </div>
          </div>

          {/* The log — the scrollable container is what auto-follows. */}
          <div
            ref={scrollRef}
            onScroll={(e) => {
              const el = e.currentTarget;
              stickToBottom.current =
                el.scrollHeight - el.scrollTop - el.clientHeight < 48;
            }}
            className="mt-[var(--gap,14px)] mb-[var(--pad,16px)] min-h-0 flex-1 overflow-y-auto rounded-xl border border-[var(--border)]"
          >
            {filtered.length === 0 ? (
              <p className="px-4 py-10 text-center text-[13px] text-[var(--muted-foreground)]">
                No events match the current filters.
              </p>
            ) : (
              filtered.map((event) => <EventRow key={event.id} event={event} />)
            )}
          </div>
        </>
      )}
    </PageBody>
  );
}
