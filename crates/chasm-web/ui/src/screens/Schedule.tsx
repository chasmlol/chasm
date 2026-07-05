import { useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CalendarClock, Clock, X } from "lucide-react";

import { schedulerApi, travelApi, type ScheduledTaskDto } from "@/lib/api";
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
// Schedule — the NPC "cronjob" board. Read-only list of scheduled tasks (owner,
// trigger, action, state) with a cancel button, plus the current in-game clock
// in the header (handy for testing time-triggered tasks). The tasks themselves
// are created by NPCs picking a scheduler action in conversation (meet_player /
// fetch_loot / schedule); chasm's tick fires them when the trigger is met.
//
// See crates/chasm-web/src/scheduler.rs for the store + tick, and the FNV action
// book's scheduler.* entries for the LLM-facing surface.
// ===========================================================================

/** Pill tone per task state. */
function statePill(state: string) {
  switch (state) {
    case "done":
      return <StatusPill tone="ok">Done</StatusPill>;
    case "active":
      return (
        <StatusPill tone="warn" pulse>
          In progress
        </StatusPill>
      );
    case "pending":
      return <StatusPill tone="idle">Pending</StatusPill>;
    case "failed":
      return <StatusPill tone="warn">Failed</StatusPill>;
    case "cancelled":
      return <StatusPill tone="idle">Cancelled</StatusPill>;
    default:
      return <StatusPill tone="idle">{state}</StatusPill>;
  }
}

function isCancellable(state: string): boolean {
  return state === "pending" || state === "active";
}

export function Schedule() {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["scheduler", "view"],
    queryFn: () => schedulerApi.view(),
    // The user is live in-game; tasks fire on their own and the clock ticks, so
    // poll to reflect firing/progress without a manual refresh.
    refetchInterval: 3000,
  });

  // Cancel routes by row kind: a travel journey cancels via the travel endpoint,
  // a scheduler task via the scheduler endpoint.
  const cancel = useMutation({
    mutationFn: (task: ScheduledTaskDto) =>
      task.kind === "journey"
        ? travelApi.cancel(task.id)
        : schedulerApi.cancel(task.id),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["scheduler", "view"] }),
  });

  const tasks = query.data?.tasks ?? [];
  const clock = query.data?.clock ?? null;

  const activeCount = useMemo(
    () => tasks.filter((t) => t.state === "pending" || t.state === "active").length,
    [tasks],
  );

  return (
    <PageBody width="wide" className="overflow-y-auto">
      <PageHeader
        eyebrow="Main"
        title="Schedule"
        description={
          <>
            Actions an NPC scheduled by adding a natural-language modifier to what
            they do: <em>"at &lt;time&gt;"</em> schedules it (e.g.{" "}
            <code className="font-mono">wave at 1am</code>) and <em>"then"</em>{" "}
            chains steps (e.g.{" "}
            <code className="font-mono">loot the body then give it to you</code>).
            chasm fires each step when it's due. Read-only here, with a cancel for
            anything still pending.
          </>
        }
        actions={
          clock ? (
            <StatusPill tone="ok">
              <Clock className="size-3.5" /> Day {clock.day} · {clock.label}
            </StatusPill>
          ) : (
            <StatusPill tone="idle">No in-game clock yet</StatusPill>
          )
        }
      />

      <Stack className="py-[var(--gap,14px)]">
        <Section
          title={`Scheduled tasks${activeCount ? ` (${activeCount} active)` : ""}`}
          description="Newest first. A step with a time fires when the in-game clock reaches it; a chained step fires once the step before it is done."
        >
          {query.isLoading ? (
            <EmptyState icon={<CalendarClock />} title="Loading schedule…" />
          ) : tasks.length === 0 ? (
            <EmptyState
              icon={<CalendarClock />}
              title="Nothing scheduled"
              description="Nothing is scheduled yet. When an NPC does an action with a time or a chain — like 'wave at 1am' or 'follow me then wait at dusk' — it shows up here."
            />
          ) : (
            <Table
              head={
                <tr>
                  <Th className="w-44">Owner</Th>
                  <Th>Task</Th>
                  <Th className="w-56">Trigger</Th>
                  <Th className="w-32">State</Th>
                  <Th className="w-20" />
                </tr>
              }
            >
              {tasks.map((task: ScheduledTaskDto) => (
                <tr
                  key={task.id}
                  className="transition-colors hover:bg-[var(--color-ink-850)]/60"
                >
                  <Td className="align-top">
                    <div className="font-medium">{task.ownerName || task.ownerNpcKey}</div>
                    <div className="text-[11px] uppercase tracking-wide text-[var(--muted-foreground)]">
                      {task.action}
                    </div>
                  </Td>
                  <Td className="align-top">
                    <div>{task.summary}</div>
                    {task.progress && (
                      <div className="mt-0.5 text-[11px] text-[var(--muted-foreground)]">
                        {task.kind === "journey"
                          ? `${task.progress} of the way`
                          : `Step ${task.progress}`}
                      </div>
                    )}
                    {task.lastError && (
                      <div className="mt-0.5 text-[11px] text-[var(--color-danger)]">
                        {task.lastError}
                      </div>
                    )}
                  </Td>
                  <Td className="align-top">
                    <div className="text-[11px] uppercase tracking-wide text-[var(--muted-foreground)]">
                      {task.triggerKind}
                    </div>
                    <div>{task.triggerDetail}</div>
                  </Td>
                  <Td className="align-top">{statePill(task.state)}</Td>
                  <Td className="align-top text-right">
                    {isCancellable(task.state) && (
                      <Button
                        variant="ghost"
                        onClick={() => cancel.mutate(task)}
                        disabled={cancel.isPending}
                        title={task.kind === "journey" ? "Cancel this journey" : "Cancel this task"}
                        className="h-7 px-2"
                      >
                        <X className="size-3.5" />
                      </Button>
                    )}
                  </Td>
                </tr>
              ))}
            </Table>
          )}
        </Section>
      </Stack>
    </PageBody>
  );
}
