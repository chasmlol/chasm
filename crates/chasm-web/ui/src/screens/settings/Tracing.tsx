import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity, Loader2 } from "lucide-react";

import {
  systemApi,
  type TraceDetail,
  type TraceListEntry,
  type TraceStage,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  EmptyState,
  PageBody,
  PageHeader,
  Section,
  StatusPill,
} from "@/components/ui/page";

// Tracing — a READ-ONLY viewer of per-request traces. Lists the recent traces
// the helper wrote (GET /api/ui/v1/traces) and renders the selected one as a
// DevTools-style waterfall (GET .../traces/:id). It never mutates a trace file
// or touches the transport. Built from the shared primitives (PageHeader,
// PageBody, Section, StatusPill) so it reads like the rest of the app.

// Stage-group → accent colour for the waterfall bars + legend. Falls back to the
// muted ink for unknown groups.
const GROUP_COLOR: Record<string, string> = {
  stt: "var(--color-player)",
  llm: "var(--color-accent)",
  tts: "var(--color-npc)",
  audio: "#7c9cf5",
  anim: "#c07cf5",
  hold: "var(--color-ink-600)",
  http: "#5fb6c4",
  helper: "#5fb6c4",
  request: "#8a93a6",
};

function groupColor(group: string): string {
  return GROUP_COLOR[group] ?? "var(--color-ink-600)";
}

export function Tracing() {
  const list = useQuery({
    queryKey: ["traces"],
    queryFn: systemApi.traces,
    refetchInterval: 5000,
  });

  const [selected, setSelected] = useState<string | null>(null);

  // Auto-select the newest trace once the list loads (and whenever the current
  // selection drops out of the list).
  const entries = list.data?.traces ?? [];
  useEffect(() => {
    if (entries.length === 0) return;
    if (!selected || !entries.some((e) => e.request_id === selected)) {
      setSelected(entries[0].request_id);
    }
  }, [entries, selected]);

  return (
    <PageBody width="wide">
      <PageHeader
        eyebrow="System"
        title="Tracing"
        description="Per-request waterfalls: the STT → LLM → TTS → audio timings for each turn. Read-only."
        actions={
          <StatusPill tone="idle">
            {entries.length} {entries.length === 1 ? "trace" : "traces"}
          </StatusPill>
        }
      />

      {list.isLoading ? (
        <div className="grid flex-1 place-items-center text-[var(--muted-foreground)]">
          <Loader2 className="size-6 animate-spin" />
        </div>
      ) : list.isError ? (
        <div className="grid flex-1 place-items-center p-8 text-center">
          <p className="text-sm font-medium text-[var(--color-danger)]">
            Couldn’t load traces.
          </p>
        </div>
      ) : entries.length === 0 ? (
        <div className="flex-1 pt-[var(--gap,14px)]">
          <EmptyState
            icon={<Activity className="size-5" strokeWidth={1.75} />}
            title="No traces yet"
            description={
              list.data?.traceDir
                ? `Run a turn in-game and traces will appear here (reading ${list.data.traceDir}).`
                : "Run a turn in-game and traces will appear here."
            }
          />
        </div>
      ) : (
        <div className="grid flex-1 grid-cols-1 gap-[var(--gap,14px)] pt-[var(--gap,14px)] lg:grid-cols-[18rem_1fr]">
          <TraceList
            entries={entries}
            selected={selected}
            onSelect={setSelected}
          />
          <TraceWaterfall id={selected} />
        </div>
      )}
    </PageBody>
  );
}

function TraceList({
  entries,
  selected,
  onSelect,
}: {
  entries: TraceListEntry[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="flex flex-col overflow-hidden rounded-xl border border-[var(--border)]">
      <div className="border-b border-[var(--line)] bg-[var(--color-ink-850)] px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
        Recent requests
      </div>
      <div className="max-h-[34rem] overflow-y-auto">
        {entries.map((entry) => {
          const active = entry.request_id === selected;
          return (
            <button
              key={entry.request_id}
              onClick={() => onSelect(entry.request_id)}
              className={cn(
                "flex w-full flex-col items-start gap-0.5 border-b border-[var(--line-soft)] px-3 py-2.5 text-left transition-colors",
                active
                  ? "bg-[var(--color-accent)]/10"
                  : "hover:bg-[var(--color-ink-800)]",
              )}
            >
              <span
                className={cn(
                  "truncate font-mono text-[12px]",
                  active
                    ? "text-[var(--color-accent)]"
                    : "text-[var(--foreground)]",
                )}
              >
                {entry.request_id}
              </span>
              <span className="flex items-center gap-2 text-[12px] text-[var(--muted-foreground)]">
                <span>{formatMs(entry.total_ms)}</span>
                <span>·</span>
                <span>{entry.stage_count} stages</span>
              </span>
              {entry.started_at && (
                <span className="truncate text-[11px] text-[var(--muted-foreground)]/70">
                  {entry.started_at}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function TraceWaterfall({ id }: { id: string | null }) {
  const detail = useQuery({
    queryKey: ["trace", id],
    queryFn: () => systemApi.trace(id as string),
    enabled: Boolean(id),
  });

  if (!id || detail.isLoading) {
    return (
      <div className="grid place-items-center rounded-xl border border-[var(--border)] py-20 text-[var(--muted-foreground)]">
        <Loader2 className="size-5 animate-spin" />
      </div>
    );
  }
  if (detail.isError || !detail.data) {
    return (
      <div className="grid place-items-center rounded-xl border border-[var(--border)] py-20 text-center">
        <p className="text-sm font-medium text-[var(--color-danger)]">
          Couldn’t load this trace.
        </p>
      </div>
    );
  }

  return <TraceDetailView data={detail.data} />;
}

function TraceDetailView({ data }: { data: TraceDetail }) {
  const total = Math.max(data.totalMs, 1);

  // Distinct stage groups present, for the legend.
  const groups = Array.from(new Set(data.stages.map((s) => s.group)));

  return (
    <div className="flex flex-col gap-[var(--gap,14px)]">
      {/* Summary metrics */}
      {data.summary.metrics.length > 0 && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {data.summary.metrics.map((metric, i) => (
            <div
              key={`${metric.label}-${i}`}
              className={cn(
                "rounded-xl border px-3 py-2.5",
                metric.primary
                  ? "border-[var(--color-accent)]/40 bg-[var(--color-accent)]/5"
                  : "border-[var(--border)] bg-[var(--color-ink-850)]",
              )}
            >
              <div className="text-[11px] uppercase tracking-wider text-[var(--muted-foreground)]">
                {metric.label}
              </div>
              <div
                className={cn(
                  "mt-0.5 font-mono text-sm font-medium",
                  metric.primary && "text-[var(--color-accent)]",
                )}
              >
                {metric.value}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Waterfall */}
      <Section
        title="Waterfall"
        actions={
          <span className="font-mono text-[12px] text-[var(--muted-foreground)]">
            {formatMs(data.totalMs)} total
          </span>
        }
      >
        <div className="flex flex-col gap-1 rounded-xl border border-[var(--border)] p-3">
          {data.stages.map((stage) => (
            <StageRow key={stage.index} stage={stage} total={total} />
          ))}
        </div>
      </Section>

      {/* Legend */}
      {groups.length > 0 && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 px-1">
          {groups.map((group) => (
            <span
              key={group}
              className="inline-flex items-center gap-1.5 text-[12px] text-[var(--muted-foreground)]"
            >
              <span
                className="size-2.5 rounded-sm"
                style={{ background: groupColor(group) }}
              />
              {group}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function StageRow({ stage, total }: { stage: TraceStage; total: number }) {
  const leftPct = Math.min((stage.elapsed_ms / total) * 100, 99.4);
  const widthPct = Math.max((stage.duration_ms / total) * 100, 0.6);
  const labelLeft = leftPct > 55;

  return (
    <div className="flex items-center gap-3 py-0.5">
      <div
        className={cn(
          "w-44 shrink-0 truncate text-[12px]",
          stage.is_error
            ? "text-[var(--color-danger)]"
            : "text-[var(--foreground)]",
        )}
        title={stage.name}
      >
        {stage.name}
      </div>
      <div className="relative h-5 flex-1">
        <div
          className="absolute top-1/2 flex h-3 -translate-y-1/2 items-center rounded-[3px]"
          style={{
            left: `${leftPct}%`,
            width: `${widthPct}%`,
            minWidth: 2,
            background: stage.is_error
              ? "var(--color-danger)"
              : groupColor(stage.group),
          }}
        >
          <span
            className={cn(
              "absolute whitespace-nowrap font-mono text-[11px] text-[var(--muted-foreground)]",
              labelLeft ? "right-full mr-1.5" : "left-full ml-1.5",
            )}
          >
            {formatMs(stage.duration_ms)}
          </span>
        </div>
      </div>
    </div>
  );
}

/** Compact ms/seconds formatter, matching the backend's `format_ms` feel. */
function formatMs(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${ms.toFixed(ms < 10 ? 1 : 0)} ms`;
}
