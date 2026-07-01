import { useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";

import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

// Live connection indicator — polls `GET /connection/status` (the real backend
// endpoint, untouched by this work) every 2s and reflects the AI-stack lifecycle
// phase, mirroring the original rail's dot + label.
const PHASE_LABEL: Record<string, string> = {
  disconnected: "Offline",
  starting: "Starting",
  connected: "Connected",
  stopping: "Stopping",
};

export function ConnectionPill() {
  const { data, isError } = useQuery({
    queryKey: ["connection-status"],
    queryFn: api.connectionStatus,
    refetchInterval: 2000,
  });

  const phase = isError ? "disconnected" : (data?.phase ?? "disconnected");
  const connected = !isError && Boolean(data?.connected);
  const starting = phase === "starting";
  const label = isError
    ? "Unreachable"
    : (PHASE_LABEL[phase] ?? "Offline");

  const dotColor = connected
    ? "var(--color-player)"
    : starting
      ? "var(--color-npc)"
      : "#3a4150";

  return (
    <div className="flex items-center gap-2 rounded-full border border-[var(--border)] bg-[var(--color-ink-850)] px-3 py-1.5">
      <span className="relative flex size-2.5">
        {(connected || starting) && (
          <motion.span
            className="absolute inline-flex size-full rounded-full"
            style={{ background: dotColor }}
            animate={{ opacity: [0.6, 0, 0.6], scale: [1, 2.2, 1] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeOut" }}
          />
        )}
        <span
          className="relative inline-flex size-2.5 rounded-full"
          style={{ background: dotColor }}
        />
      </span>
      <span
        className={cn(
          "text-xs font-medium tracking-wide",
          connected
            ? "text-[var(--color-player)]"
            : starting
              ? "text-[var(--color-npc)]"
              : "text-[var(--muted-foreground)]",
        )}
      >
        {label}
      </span>
    </div>
  );
}
