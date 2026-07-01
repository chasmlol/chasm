import type { ReactNode } from "react";

// Shared row-badge for the four book screens, so a backend `badge` label
// (e.g. "Disabled", a quest phase, "Admin") renders identically across
// Characters / Lore / Quest / Action. Returns undefined when there's no badge,
// matching <Book>'s optional `BookEntry.badge`.
export function bookBadge(label?: string): ReactNode {
  if (!label) return undefined;
  const disabled = label.toLowerCase() === "disabled";
  return (
    <span
      className={
        "shrink-0 rounded-full border px-2 py-0.5 text-[11px] font-medium " +
        (disabled
          ? "border-[var(--border)] bg-[var(--color-ink-850)] text-[var(--muted-foreground)]"
          : "border-[var(--color-accent)]/40 bg-[var(--color-accent)]/10 text-[var(--color-accent)]")
      }
    >
      {label}
    </span>
  );
}
