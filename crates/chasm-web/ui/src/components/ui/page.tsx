import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

// ===========================================================================
// Shared page primitives — the building blocks EVERY screen composes from, so
// the whole app reads consistently. They lean on the design tokens in
// index.css and the density vars the backend emits into /theme.css
// (--pad / --gap / --card-pad), so a density change reshapes spacing app-wide.
//
// Fill agents: build your screen out of these. Don't hand-roll headers,
// sections, tables, or empty states — reach for the primitive so it matches
// every other screen.
// ===========================================================================

/**
 * The standard screen header: an optional eyebrow (small uppercased kicker),
 * a title, an optional description, and an optional right-aligned actions slot
 * (e.g. a Save button or a count badge). Every content pane opens with one.
 */
export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  className,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <header
      className={cn(
        "flex items-start justify-between gap-4 border-b border-[var(--line)] pb-5",
        className,
      )}
    >
      <div className="min-w-0">
        {eyebrow && (
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--color-accent)]">
            {eyebrow}
          </p>
        )}
        <h2 className="mt-1 truncate text-2xl font-semibold tracking-tight">
          {title}
        </h2>
        {description && (
          <p className="mt-2 max-w-prose text-[13px] leading-relaxed text-[var(--muted-foreground)]">
            {description}
          </p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </header>
  );
}

/**
 * The outer wrapper for a content pane: centers + bounds the column and applies
 * density-driven padding (--pad with a comfortable fallback). All screens
 * (settings + books + chat) sit inside one of these, so the gutter is uniform.
 *
 * `width` picks the max column: "prose" (settings forms), "wide" (books /
 * tables), or "full" (chat, which manages its own internal layout).
 */
export function PageBody({
  width = "prose",
  className,
  children,
}: {
  width?: "prose" | "wide" | "full";
  className?: string;
  children: ReactNode;
}) {
  const max =
    width === "full"
      ? "max-w-none"
      : width === "wide"
        ? "max-w-5xl"
        : "max-w-3xl";
  return (
    <div
      className={cn(
        "mx-auto flex h-full flex-col",
        max,
        // Density: --pad drives the horizontal/vertical gutter; the fallback is
        // the comfortable value so the look is unchanged before /theme.css loads.
        "px-[calc(var(--pad,16px)+0.5rem)] py-[var(--pad,16px)]",
        className,
      )}
    >
      {children}
    </div>
  );
}

/**
 * A labeled block within a screen. Optional title + description, then content.
 * Use between cards/groups so vertical rhythm (via --gap) is consistent.
 */
export function Section({
  title,
  description,
  actions,
  className,
  children,
}: {
  title?: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={cn("flex flex-col", className)}>
      {(title || actions) && (
        <div className="mb-2 flex items-center justify-between gap-3">
          {title && (
            <SectionLabel>{title}</SectionLabel>
          )}
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      {description && (
        <p className="mb-2 text-[13px] leading-relaxed text-[var(--muted-foreground)]">
          {description}
        </p>
      )}
      {children}
    </section>
  );
}

/** The small uppercased label used for section/field groupings. */
export function SectionLabel({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <p
      className={cn(
        "text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted-foreground)]",
        className,
      )}
    >
      {children}
    </p>
  );
}

/**
 * A vertical stack with density-driven gap (--gap). The default container for
 * a list of cards/sections inside a PageBody.
 */
export function Stack({
  className,
  children,
}: {
  className?: string;
  children: ReactNode;
}) {
  return (
    <div className={cn("flex flex-col gap-[var(--gap,14px)]", className)}>
      {children}
    </div>
  );
}

/**
 * A labeled form row: a label (+ optional help) on the left, the control on the
 * right. The shared layout for every settings input, so forms line up across
 * screens. Pass `stacked` to put the control under the label (for wide inputs
 * like sliders / textareas).
 */
export function FormRow({
  label,
  help,
  htmlFor,
  stacked = false,
  control,
  className,
}: {
  label: ReactNode;
  help?: ReactNode;
  htmlFor?: string;
  stacked?: boolean;
  control: ReactNode;
  className?: string;
}) {
  if (stacked) {
    return (
      <div className={cn("flex flex-col gap-2", className)}>
        <div>
          <label
            htmlFor={htmlFor}
            className="block text-sm font-medium"
          >
            {label}
          </label>
          {help && (
            <p className="mt-0.5 text-[13px] text-[var(--muted-foreground)]">
              {help}
            </p>
          )}
        </div>
        {control}
      </div>
    );
  }
  return (
    <div
      className={cn(
        "flex items-start justify-between gap-4",
        className,
      )}
    >
      <div className="min-w-0">
        <label htmlFor={htmlFor} className="block text-sm font-medium">
          {label}
        </label>
        {help && (
          <p className="mt-0.5 text-[13px] text-[var(--muted-foreground)]">
            {help}
          </p>
        )}
      </div>
      <div className="shrink-0 pt-0.5">{control}</div>
    </div>
  );
}

/** A plain text/number/etc. input, themed to the chasm palette. */
export function Field({
  className,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "h-9 w-full rounded-lg border border-[var(--border)] bg-[var(--color-ink-850)] px-3 text-sm text-[var(--foreground)] outline-none transition-colors placeholder:text-[var(--muted-foreground)]/60 focus-visible:border-[var(--color-accent)] focus-visible:ring-2 focus-visible:ring-[var(--ring)]/40",
        className,
      )}
      {...props}
    />
  );
}

/** A multi-line textarea, themed to match Field. */
export function TextArea({
  className,
  ...props
}: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn(
        "w-full rounded-lg border border-[var(--border)] bg-[var(--color-ink-850)] px-3 py-2 text-sm leading-relaxed text-[var(--foreground)] outline-none transition-colors placeholder:text-[var(--muted-foreground)]/60 focus-visible:border-[var(--color-accent)] focus-visible:ring-2 focus-visible:ring-[var(--ring)]/40",
        className,
      )}
      {...props}
    />
  );
}

/** A native select, themed to match Field. */
export function Select({
  className,
  children,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        "h-9 rounded-lg border border-[var(--border)] bg-[var(--color-ink-850)] px-3 text-sm text-[var(--foreground)] outline-none transition-colors focus-visible:border-[var(--color-accent)] focus-visible:ring-2 focus-visible:ring-[var(--ring)]/40",
        className,
      )}
      {...props}
    >
      {children}
    </select>
  );
}

/**
 * A simple table primitive. Pass column headers + rows of cells; the chrome
 * (hairline borders, header styling, zebra hover) is uniform across screens.
 */
export function Table({
  head,
  children,
  className,
}: {
  head?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "overflow-hidden rounded-xl border border-[var(--border)]",
        className,
      )}
    >
      <table className="w-full border-collapse text-sm">
        {head && (
          <thead className="bg-[var(--color-ink-850)] text-[var(--muted-foreground)]">
            {head}
          </thead>
        )}
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

/** A header cell for the Table primitive. */
export function Th({
  children,
  className,
}: {
  children?: ReactNode;
  className?: string;
}) {
  return (
    <th
      className={cn(
        "border-b border-[var(--line)] px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wider",
        className,
      )}
    >
      {children}
    </th>
  );
}

/** A body cell for the Table primitive. */
export function Td({
  children,
  className,
}: {
  children?: ReactNode;
  className?: string;
}) {
  return (
    <td
      className={cn(
        "border-b border-[var(--line-soft)] px-3 py-2.5 align-middle",
        className,
      )}
    >
      {children}
    </td>
  );
}

/**
 * The standard empty / nothing-here state: a centered icon + title + hint,
 * with an optional action. Used by books with no entries, chat with no
 * messages, etc., so "empty" looks the same everywhere.
 */
export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: {
  icon?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "grid place-items-center rounded-xl border border-dashed border-[var(--border)] px-8 py-14 text-center",
        className,
      )}
    >
      <div className="max-w-sm">
        {icon && (
          <div className="mx-auto mb-3 grid size-11 place-items-center rounded-2xl border border-[var(--border)] bg-[var(--color-ink-800)] text-[var(--color-accent)]">
            {icon}
          </div>
        )}
        <p className="text-sm font-medium text-[var(--foreground)]">{title}</p>
        {description && (
          <p className="mt-1 text-[13px] leading-relaxed text-[var(--muted-foreground)]">
            {description}
          </p>
        )}
        {action && <div className="mt-4 flex justify-center">{action}</div>}
      </div>
    </div>
  );
}

/**
 * A small status pill — a colored dot + label — for model/runtime/connection
 * state. `tone` maps to a semantic color; used by the ModelPicker status and
 * anywhere a screen surfaces ready/working/error.
 */
export type StatusTone = "ok" | "warn" | "error" | "busy" | "idle";

const STATUS_DOT: Record<StatusTone, string> = {
  ok: "var(--color-player)",
  warn: "var(--color-npc)",
  error: "var(--color-danger)",
  busy: "var(--color-accent)",
  idle: "var(--color-ink-600)",
};

const STATUS_TEXT: Record<StatusTone, string> = {
  ok: "text-[var(--color-player)]",
  warn: "text-[var(--color-npc)]",
  error: "text-[var(--color-danger)]",
  busy: "text-[var(--color-accent)]",
  idle: "text-[var(--muted-foreground)]",
};

export function StatusPill({
  tone,
  children,
  pulse = false,
  className,
}: {
  tone: StatusTone;
  children: ReactNode;
  pulse?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border border-[var(--border)] bg-[var(--color-ink-850)] px-2.5 py-1 text-xs font-medium",
        STATUS_TEXT[tone],
        className,
      )}
    >
      <span
        className={cn("size-2 rounded-full", pulse && "animate-pulse")}
        style={{ background: STATUS_DOT[tone] }}
      />
      {children}
    </span>
  );
}
