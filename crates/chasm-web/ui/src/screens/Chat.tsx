import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
import {
  MessagesSquare,
  BookText,
  ScrollText,
  Swords,
  Zap,
  AlertCircle,
  AlertTriangle,
  Search,
} from "lucide-react";

import {
  chatApi,
  type ChatThreadDto,
  type ChatMessageDto,
  type InjectedEntryDto,
  type OfferedActionDto,
  type ExecutedActionDto,
} from "@/lib/api";
import { PageHeader, PageBody, EmptyState, Field } from "@/components/ui/page";
import { cn } from "@/lib/utils";

// ===========================================================================
// Chat — the live conversation view (redesigned).
//
// TWO COLUMNS: a persistent conversation-list panel on the left (every
// character the user has chat history with, busiest first, with a search box),
// and the message stream on the right. Clicking a row switches the stream to
// that conversation; the active row is highlighted. This replaces the old
// header dropdown so the user can reach every character's chat, not just the
// in-scene NPC.
//
// THE KEY FEATURE is the inline per-message context strip (no click to reveal):
// under each message we show, as compact chips, what was INJECTED into that
// turn (Lore / Quest / world-info), the actions OFFERED to the model, and which
// actions actually EXECUTED — executed ones in GREEN. Sourced read-only from
// `/api/ui/v1/chat/view`, which projects each turn's recorded context.
// ===========================================================================

export function Chat() {
  const query = useQuery({
    queryKey: ["chat", "view"],
    queryFn: () => chatApi.view(),
    // The user is live in-game; keep the stream fresh without manual refresh.
    refetchInterval: 4000,
  });

  const threads = query.data?.threads ?? [];

  // Track the selected NPC thread. Default to the backend's suggestion (the
  // busiest conversation) once data arrives; fall back to the first thread.
  // Keep the selection if it's still a valid thread across refetches.
  const [selectedId, setSelectedId] = useState<string | null>(null);
  useEffect(() => {
    if (threads.length === 0) {
      if (selectedId !== null) setSelectedId(null);
      return;
    }
    const stillValid =
      selectedId && threads.some((t) => t.participant_id === selectedId);
    if (!stillValid) {
      setSelectedId(
        query.data?.default_participant_id ?? threads[0].participant_id,
      );
    }
  }, [threads, selectedId, query.data?.default_participant_id]);

  const active = useMemo<ChatThreadDto | undefined>(
    () =>
      threads.find((t) => t.participant_id === selectedId) ?? threads[0],
    [threads, selectedId],
  );

  const hasLiveChat = Boolean(query.data?.live_chat_id);

  // Jump the message stream to the bottom (latest) when a conversation is opened
  // and when it grows (live turns arrive). rAF so it runs after the new messages
  // have laid out, else scrollHeight is stale.
  const streamRef = useRef<HTMLDivElement>(null);
  const activeId = active?.participant_id;
  const messageCount = active?.messages.length ?? 0;
  useEffect(() => {
    const el = streamRef.current;
    if (!el) return;
    const id = requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
    return () => cancelAnimationFrame(id);
  }, [activeId, messageCount]);

  // Right-click a character → "Clear history". Fully clears their chat; the
  // backend also scrubs save-sync checkpoints so a game load can't restore it.
  const qc = useQueryClient();
  const liveChatId = query.data?.live_chat_id ?? null;
  const [menu, setMenu] = useState<{
    x: number;
    y: number;
    thread: ChatThreadDto;
  } | null>(null);
  const clear = useMutation({
    mutationFn: (participantId: string) =>
      chatApi.clearHistory(liveChatId!, participantId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["chat", "view"] }),
  });
  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("click", close);
    window.addEventListener("scroll", close, true);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("keydown", onKey);
    };
  }, [menu]);

  return (
    <PageBody width="full">
      <PageHeader
        eyebrow="Live"
        title="Chat"
        description="Every conversation you have chat history with. Pick a character on the left; each message shows what was injected into its turn and which actions ran."
      />

      {/* grid-rows-[minmax(0,1fr)] pins the single row to the available height so
          each column scrolls on its OWN, instead of both growing and scrolling the
          whole page (which dragged the character list along). */}
      <div className="mt-5 grid min-h-0 flex-1 gap-4 lg:grid-cols-[18rem_minmax(0,1fr)] lg:grid-rows-[minmax(0,1fr)]">
        <ConversationList
          threads={threads}
          activeId={active?.participant_id ?? null}
          onSelect={setSelectedId}
          onContext={(thread, x, y) => setMenu({ thread, x, y })}
          isLoading={query.isLoading}
        />
        <div ref={streamRef} className="min-h-0 overflow-y-auto">
          <ChatContent
            isLoading={query.isLoading}
            isError={query.isError}
            error={query.error}
            thread={active}
            hasLiveChat={hasLiveChat}
          />
        </div>
      </div>

      {menu && (
        <div
          className="fixed z-50 min-w-[10rem] overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--card)] py-1 shadow-xl"
          style={{ left: menu.x, top: menu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            type="button"
            disabled={!liveChatId || clear.isPending}
            className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[13px] text-[var(--color-danger)] hover:bg-[var(--color-ink-850)] disabled:opacity-50"
            onClick={() => {
              const t = menu.thread;
              setMenu(null);
              if (!liveChatId) return;
              if (
                window.confirm(
                  `Clear all chat history with ${t.name}? This fully clears it (ignoring save states) and can't be undone.`,
                )
              ) {
                clear.mutate(t.participant_id);
              }
            }}
          >
            <Trash2 className="size-4" strokeWidth={1.75} /> Clear history
          </button>
        </div>
      )}
    </PageBody>
  );
}

// --- conversation list panel ------------------------------------------------

function ConversationList({
  threads,
  activeId,
  onSelect,
  onContext,
  isLoading,
}: {
  threads: ChatThreadDto[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onContext: (thread: ChatThreadDto, x: number, y: number) => void;
  isLoading: boolean;
}) {
  const [search, setSearch] = useState("");

  // Filter by character name. Threads arrive already sorted by message count
  // (busiest first) from the backend, so we preserve that order here.
  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    if (!needle) return threads;
    return threads.filter((t) => t.name.toLowerCase().includes(needle));
  }, [threads, search]);

  return (
    <aside className="flex min-h-0 flex-col rounded-xl border border-[var(--border)] bg-[var(--card)]">
      <div className="border-b border-[var(--line)] p-2.5">
        <div className="relative">
          <Search
            className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-[var(--muted-foreground)]/70"
            strokeWidth={1.75}
          />
          <Field
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search characters…"
            aria-label="Search conversations by character name"
            className="h-9 pl-8"
          />
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {isLoading && threads.length === 0 ? (
          <p className="px-2 py-6 text-center text-[13px] text-[var(--muted-foreground)]">
            Loading conversations…
          </p>
        ) : threads.length === 0 ? (
          <p className="px-2 py-6 text-center text-[13px] text-[var(--muted-foreground)]">
            No conversations yet.
          </p>
        ) : filtered.length === 0 ? (
          <p className="px-2 py-6 text-center text-[13px] text-[var(--muted-foreground)]">
            No characters match “{search.trim()}”.
          </p>
        ) : (
          <ul className="flex flex-col gap-1">
            {filtered.map((thread) => (
              <li key={thread.participant_id}>
                <ConversationRow
                  thread={thread}
                  active={thread.participant_id === activeId}
                  onClick={() => onSelect(thread.participant_id)}
                  onContext={(x, y) => onContext(thread, x, y)}
                />
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}

function ConversationRow({
  thread,
  active,
  onClick,
  onContext,
}: {
  thread: ChatThreadDto;
  active: boolean;
  onClick: () => void;
  onContext: (x: number, y: number) => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      onContextMenu={(e) => {
        e.preventDefault();
        onContext(e.clientX, e.clientY);
      }}
      aria-current={active ? "true" : undefined}
      className={cn(
        "flex w-full items-center gap-2.5 rounded-lg border px-2.5 py-2 text-left transition-colors",
        active
          ? "border-[color-mix(in_srgb,var(--color-accent)_45%,var(--border))] bg-[color-mix(in_srgb,var(--color-accent)_12%,transparent)]"
          : "border-transparent hover:bg-[var(--color-ink-850)]",
      )}
    >
      <span
        className={cn(
          "relative grid size-9 shrink-0 place-items-center rounded-full text-[13px] font-semibold",
          active
            ? "bg-[color-mix(in_srgb,var(--color-accent)_22%,transparent)] text-[var(--color-accent)]"
            : "bg-[color-mix(in_srgb,var(--color-npc)_18%,transparent)] text-[var(--color-npc)]",
        )}
      >
        {thread.initial || "?"}
        {thread.present && (
          <span
            className="absolute -bottom-0.5 -right-0.5 size-2.5 rounded-full border-2 border-[var(--card)] bg-[var(--color-player)]"
            title="In scene"
          />
        )}
      </span>

      <span className="min-w-0 flex-1">
        <span className="flex items-center justify-between gap-2">
          <span className="truncate text-sm font-medium text-[var(--foreground)]">
            {thread.name}
          </span>
          <span className="shrink-0 text-[11px] tabular-nums text-[var(--muted-foreground)]">
            {thread.message_count}
          </span>
        </span>
        <span className="mt-0.5 block truncate text-[12px] text-[var(--muted-foreground)]">
          {thread.last_message_preview ||
            (thread.present ? "In scene" : "Away")}
        </span>
      </span>
    </button>
  );
}

// --- content states ---------------------------------------------------------

function ChatContent({
  isLoading,
  isError,
  error,
  thread,
  hasLiveChat,
}: {
  isLoading: boolean;
  isError: boolean;
  error: unknown;
  thread: ChatThreadDto | undefined;
  hasLiveChat: boolean;
}) {
  if (isLoading) {
    return (
      <div className="grid place-items-center py-20 text-[13px] text-[var(--muted-foreground)]">
        <span className="inline-flex items-center gap-2">
          <span className="size-2 animate-pulse rounded-full bg-[var(--color-accent)]" />
          Loading conversation…
        </span>
      </div>
    );
  }

  if (isError) {
    return (
      <EmptyState
        icon={<AlertCircle className="size-5" strokeWidth={1.75} />}
        title="Couldn’t load the conversation"
        description={
          error instanceof Error
            ? error.message
            : "The chat projection request failed. It will retry automatically."
        }
      />
    );
  }

  if (!hasLiveChat) {
    return (
      <EmptyState
        icon={<MessagesSquare className="size-5" strokeWidth={1.75} />}
        title="No live chat yet"
        description="Once the game connects and an NPC speaks, the conversation appears here."
      />
    );
  }

  if (!thread || thread.messages.length === 0) {
    return (
      <EmptyState
        icon={<MessagesSquare className="size-5" strokeWidth={1.75} />}
        title="No messages in this conversation"
        description="Nothing has been said in this thread yet."
      />
    );
  }

  return <MessageStream thread={thread} />;
}

// --- message stream ---------------------------------------------------------

function MessageStream({ thread }: { thread: ChatThreadDto }) {
  return (
    <ol className="mx-auto flex max-w-3xl flex-col gap-3 pb-6">
      {thread.messages.map((message) => (
        <MessageRow key={message.id} message={message} />
      ))}
    </ol>
  );
}

function MessageRow({ message }: { message: ChatMessageDto }) {
  const isPlayer = message.role === "player";
  const isSystem = message.role === "system";
  return (
    <li className="flex gap-3">
      <span
        className={cn(
          "mt-0.5 grid size-8 shrink-0 place-items-center rounded-full text-[12px] font-semibold",
          isPlayer
            ? "bg-[color-mix(in_srgb,var(--color-player)_22%,transparent)] text-[var(--color-player)]"
            : isSystem
              ? "bg-[var(--color-ink-700)] text-[var(--muted-foreground)]"
              : "bg-[color-mix(in_srgb,var(--color-npc)_22%,transparent)] text-[var(--color-npc)]",
        )}
      >
        {message.initial || "?"}
      </span>

      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-semibold text-[var(--foreground)]">
            {message.speaker}
          </span>
          <RoleBadge role={message.role} />
          {message.timestamp_label && (
            <time
              className="text-[11px] text-[var(--muted-foreground)]"
              dateTime={message.timestamp}
              title={message.timestamp}
            >
              {message.timestamp_label}
            </time>
          )}
        </div>

        <div className="mt-1 whitespace-pre-wrap rounded-2xl rounded-tl-sm border border-[var(--border)] bg-[var(--card)] px-3.5 py-2.5 text-[13px] leading-relaxed text-[var(--foreground)]">
          {message.text}
        </div>

        <ContextStrip message={message} />
      </div>
    </li>
  );
}

function RoleBadge({ role }: { role: string }) {
  const tone =
    role === "player"
      ? "text-[var(--color-player)]"
      : role === "system"
        ? "text-[var(--muted-foreground)]"
        : "text-[var(--color-npc)]";
  return (
    <span
      className={cn(
        "rounded-full border border-[var(--border)] bg-[var(--color-ink-850)] px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
        tone,
      )}
    >
      {role}
    </span>
  );
}

// --- the inline per-message context strip (THE key feature) -----------------

function ContextStrip({ message }: { message: ChatMessageDto }) {
  const hasLore = message.injected_lore.length > 0;
  const hasQuests = message.injected_quests.length > 0;
  const hasOffered = message.offered_actions.length > 0;
  // Executed actions that weren't in the offered set still deserve a green chip
  // (native / relayed actions). Offered-and-executed are shown via the offered
  // group (green), so here we only surface executed actions with no offered twin.
  const extraExecuted = message.executed_actions.filter((a) => !a.offered);
  const hasExtraExecuted = extraExecuted.length > 0;
  // This NPC turn was generated mid-fight — surfaced as a prominent red badge.
  const hasCombat = message.in_combat;

  if (!hasLore && !hasQuests && !hasOffered && !hasExtraExecuted && !hasCombat) {
    // Keep player turns quiet; only annotate NPC turns that genuinely recorded
    // nothing (so a missing strip never looks like a bug).
    if (message.role === "npc" && message.no_context) {
      return (
        <p className="mt-1.5 pl-0.5 text-[11px] italic text-[var(--muted-foreground)]/70">
          No turn context recorded.
        </p>
      );
    }
    return null;
  }

  return (
    <div className="mt-2 flex flex-wrap items-start gap-x-4 gap-y-2 pl-0.5">
      {hasCombat && (
        <ChipGroup
          icon={<AlertTriangle className="size-3" strokeWidth={2.5} />}
          label="In combat"
          tone="combat"
        >
          {message.combat_with.length > 0 ? (
            message.combat_with.map((name, i) => (
              <CombatChip key={`${name}-${i}`} name={name} />
            ))
          ) : (
            <CombatChip name="an enemy" />
          )}
        </ChipGroup>
      )}

      {hasLore && (
        <ChipGroup
          icon={<BookText className="size-3" strokeWidth={2} />}
          label="Lore"
          tone="lore"
        >
          {message.injected_lore.map((e, i) => (
            <InjectedChip key={`${e.id}-${i}`} entry={e} tone="lore" />
          ))}
        </ChipGroup>
      )}

      {hasQuests && (
        <ChipGroup
          icon={<ScrollText className="size-3" strokeWidth={2} />}
          label="Quests"
          tone="quest"
        >
          {message.injected_quests.map((e, i) => (
            <InjectedChip key={`${e.id}-${i}`} entry={e} tone="quest" />
          ))}
        </ChipGroup>
      )}

      {hasOffered && (
        <ChipGroup
          icon={<Swords className="size-3" strokeWidth={2} />}
          label="Actions offered"
          tone="action"
        >
          {message.offered_actions.map((a, i) => (
            <OfferedChip key={`${a.id}-${i}`} action={a} />
          ))}
        </ChipGroup>
      )}

      {hasExtraExecuted && (
        <ChipGroup
          icon={<Zap className="size-3" strokeWidth={2} />}
          label="Executed"
          tone="executed"
        >
          {extraExecuted.map((a, i) => (
            <ExecutedChip key={`${a.id}-${i}`} action={a} />
          ))}
        </ChipGroup>
      )}
    </div>
  );
}

type ChipTone = "lore" | "quest" | "action" | "executed" | "combat";

const GROUP_LABEL_TONE: Record<ChipTone, string> = {
  lore: "text-[var(--color-accent)]",
  quest: "text-[var(--color-npc)]",
  action: "text-[var(--muted-foreground)]",
  executed: "text-[var(--color-player)]",
  combat: "text-[var(--color-danger)]",
};

function ChipGroup({
  icon,
  label,
  tone,
  children,
}: {
  icon: React.ReactNode;
  label: string;
  tone: ChipTone;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span
        className={cn(
          "inline-flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wide",
          GROUP_LABEL_TONE[tone],
        )}
        title={label}
      >
        {icon}
        <span className="sr-only sm:not-sr-only">{label}</span>
      </span>
      <span className="flex flex-wrap items-center gap-1">{children}</span>
    </div>
  );
}

const CHIP_BASE =
  "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium";

function reasonDot(reason: string) {
  // constant = solid accent, keyword = npc gold, vector = player green; a tiny
  // dot keeps the activation reason visible without a second chip.
  const color =
    reason === "vector"
      ? "var(--color-player)"
      : reason === "keyword"
        ? "var(--color-npc)"
        : "var(--color-accent)";
  return (
    <span
      className="size-1.5 rounded-full"
      style={{ background: color }}
      title={reason ? `Activated: ${reason}` : undefined}
    />
  );
}

function InjectedChip({
  entry,
  tone,
}: {
  entry: InjectedEntryDto;
  tone: "lore" | "quest";
}) {
  const ring =
    tone === "lore"
      ? "border-[color-mix(in_srgb,var(--color-accent)_45%,var(--border))] text-[var(--foreground)] bg-[color-mix(in_srgb,var(--color-accent)_10%,transparent)]"
      : "border-[color-mix(in_srgb,var(--color-npc)_45%,var(--border))] text-[var(--foreground)] bg-[color-mix(in_srgb,var(--color-npc)_10%,transparent)]";
  const title =
    entry.id && entry.id !== entry.title
      ? `${entry.title} (${entry.id})${entry.reason ? ` · ${entry.reason}` : ""}`
      : `${entry.title}${entry.reason ? ` · ${entry.reason}` : ""}`;
  return (
    <span className={cn(CHIP_BASE, ring)} title={title}>
      {entry.reason && reasonDot(entry.reason)}
      <span className="max-w-[14rem] truncate">{entry.title || entry.id}</span>
    </span>
  );
}

function OfferedChip({ action }: { action: OfferedActionDto }) {
  // Executed offered actions go GREEN; the rest are a muted "offered but not
  // taken" outline. This is the at-a-glance injected-vs-executed contrast.
  const executed = action.executed;
  const cls = executed
    ? "border-[color-mix(in_srgb,var(--color-player)_55%,var(--border))] bg-[color-mix(in_srgb,var(--color-player)_16%,transparent)] text-[var(--color-player)]"
    : "border-[var(--border)] bg-[var(--color-ink-850)] text-[var(--muted-foreground)]";
  const title = `${action.title}${action.id && action.id !== action.title ? ` (${action.id})` : ""}${
    executed ? " · EXECUTED" : " · offered, not taken"
  }${action.reason ? ` · ${action.reason}` : ""}`;
  return (
    <span className={cn(CHIP_BASE, cls)} title={title}>
      {executed && <Zap className="size-3" strokeWidth={2.5} />}
      <span className="max-w-[14rem] truncate">
        {action.title || action.id}
      </span>
    </span>
  );
}

function CombatChip({ name }: { name: string }) {
  // Alarming red to match the in-prompt combat alert: this turn was spoken
  // mid-fight, with `name` being who the NPC was up against.
  return (
    <span
      className={cn(
        CHIP_BASE,
        "border-[color-mix(in_srgb,var(--color-danger)_60%,var(--border))] bg-[color-mix(in_srgb,var(--color-danger)_18%,transparent)] text-[var(--color-danger)]",
      )}
      title={`In combat with ${name}`}
    >
      <AlertTriangle className="size-3" strokeWidth={2.5} />
      <span className="max-w-[14rem] truncate">{name}</span>
    </span>
  );
}

function ExecutedChip({ action }: { action: ExecutedActionDto }) {
  const detail = [
    action.target ? `target: ${action.target}` : "",
    action.params && action.params !== "{}" ? action.params : "",
    action.reason ? action.reason : "",
  ]
    .filter(Boolean)
    .join(" · ");
  const title = `${action.label}${action.id && action.id !== action.label ? ` (${action.id})` : ""}${
    detail ? ` · ${detail}` : ""
  }`;
  return (
    <span
      className={cn(
        CHIP_BASE,
        "border-[color-mix(in_srgb,var(--color-player)_55%,var(--border))] bg-[color-mix(in_srgb,var(--color-player)_16%,transparent)] text-[var(--color-player)]",
      )}
      title={title}
    >
      <Zap className="size-3" strokeWidth={2.5} />
      <span className="max-w-[14rem] truncate">{action.label}</span>
      {action.target && (
        <span className="text-[var(--color-player)]/70">→ {action.target}</span>
      )}
    </span>
  );
}
