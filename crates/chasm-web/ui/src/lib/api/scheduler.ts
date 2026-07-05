// Scheduler domain of the UI API — the read-only "Schedule" page.
//
// Mirrors crates/chasm-web/src/ui/scheduler.rs:
//   - GET  /api/ui/v1/scheduler            the current in-game clock + all tasks
//   - POST /api/ui/v1/scheduler/:id/cancel cancel a pending/active task
//   - POST /api/ui/v1/scheduler/event      raise a condition flag on a task (the
//     minimal condition signal, stands in for the game-event stream in testing)

import { getJson, postJson, UI_API } from "./http";

/** The current in-game clock (null before a save loads / the plugin reports it). */
export interface ClockDto {
  day: number;
  hour: number;
  /** "1:00 AM"-style label. */
  label: string;
}

/** One scheduled task row. */
export interface ScheduledTaskDto {
  id: string;
  ownerNpcKey: string;
  ownerName: string;
  characterName: string;
  /** Action alias (meet_player / fetch_loot / schedule). */
  action: string;
  /** One-line human summary. */
  summary: string;
  /** pending | active | done | cancelled | failed. */
  state: string;
  /** "In-game time" | "Condition". */
  triggerKind: string;
  /** Human trigger detail ("Day 4, 1:00 AM"). */
  triggerDetail: string;
  /** Chain progress "2/4" for composite tasks, or "63%" for a travel journey. */
  progress: string;
  lastError: string;
  createdAtMs: number;
  /** "task" (scheduler task) or "journey" (travel journey) — routes cancel. */
  kind: string;
}

export interface SchedulerViewDto {
  clock: ClockDto | null;
  tasks: ScheduledTaskDto[];
}

export interface MutationResultDto {
  ok: boolean;
  error: string;
}

export const schedulerApi = {
  /** The in-game clock + all scheduled tasks (newest first). */
  view: () => getJson<SchedulerViewDto>(`${UI_API}/scheduler`),
  /** Cancel a pending/active task. */
  cancel: (id: string) =>
    postJson<MutationResultDto>(
      `${UI_API}/scheduler/${encodeURIComponent(id)}/cancel`,
      {},
    ),
  /** Raise a condition flag on a task (e.g. "looted") — the fetch-chain test hook. */
  raiseEvent: (taskId: string, flag: string, value = true) =>
    postJson<MutationResultDto>(`${UI_API}/scheduler/event`, {
      task_id: taskId,
      flag,
      value,
    }),
};
