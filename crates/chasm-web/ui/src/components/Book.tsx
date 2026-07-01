import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import {
  Check,
  ChevronDown,
  Loader2,
  Plus,
  RotateCcw,
  Search,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  PageBody,
  PageHeader,
  Field,
  Select,
  TextArea,
  EmptyState,
} from "@/components/ui/page";

// ===========================================================================
// Book — the SHARED layout for chasm's content books (Characters / Lore /
// Quest / Action). A searchable list of entries; each row expands in place to
// an editable detail form; save is per entry. The four books are *the same
// component* — they differ ONLY in their field schema + data, which is exactly
// what keeps "the book pages pretty aligned."
//
// ---------------------------------------------------------------------------
// PROP CONTRACT (read this if you're filling in a book screen)
// ---------------------------------------------------------------------------
// A book screen does three things and nothing else:
//   1. Fetches its entries (TanStack Query) and maps them to `BookEntry[]`.
//   2. Declares its `fields: BookField[]` — the editable schema for a row.
//   3. Implements `onSave(id, values)` (and optionally onCreate/onDelete).
// Everything else — the header, search, list, expand/collapse, the edit form,
// the per-row save/reset bar, empty + loading states — is owned HERE so all
// books look and behave identically.
//
//   <Book
//     eyebrow="Library"
//     title="Lore Book"
//     icon={<BookText .../>}
//     entries={entries}            // BookEntry[]: { id, title, subtitle?, badge?, values }
//     fields={LORE_FIELDS}         // BookField[]: the row's edit schema
//     onSave={(id, values) => mutate(...)}
//     isLoading={query.isLoading}
//   />
//
// `BookEntry.values` is a plain record keyed by `BookField.key`; the form is
// generated from `fields`, so adding a field is a one-line schema change with
// no layout work. `kind` controls the input widget.
// ===========================================================================

/** The value bag for one entry, keyed by field key. */
export type BookValues = Record<string, string | boolean | number>;

/** One row in the book list (collapsed) + its editable values. */
export interface BookEntry {
  /** Stable id used for save/expand/key. */
  id: string;
  /** Primary label shown on the collapsed row. */
  title: string;
  /** Optional secondary line (e.g. a one-line summary / category). */
  subtitle?: string;
  /** Optional small badge (e.g. a tag count, type, or state). */
  badge?: ReactNode;
  /** The editable values, keyed by field key. */
  values: BookValues;
}

/** The kind of editor a field renders as. */
export type BookFieldKind =
  | "text"
  | "textarea"
  | "number"
  | "toggle"
  | "select";

/** One editable field in an entry's detail form. */
export interface BookField {
  /** Key into `BookEntry.values`. */
  key: string;
  /** Field label. */
  label: string;
  kind: BookFieldKind;
  /** Helper text under the field. */
  help?: string;
  /** Placeholder for text/textarea/number. */
  placeholder?: string;
  /** Options for `kind: "select"`. */
  options?: { value: string; label: string }[];
  /** Rows for `kind: "textarea"` (default 4). */
  rows?: number;
  /** Make the field span the full width even in a 2-col grid (default true for textarea). */
  full?: boolean;
}

export interface BookProps {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  icon?: ReactNode;
  entries: BookEntry[];
  fields: BookField[];
  /** Persist edits for one entry. Return a promise so the row shows progress. */
  onSave: (id: string, values: BookValues) => void | Promise<unknown>;
  /** Optional: add a new entry. When provided, a "New" button appears. */
  onCreate?: () => void | Promise<unknown>;
  isLoading?: boolean;
  isError?: boolean;
  /** Word used in the search placeholder + empty copy (e.g. "characters"). */
  noun?: string;
}

export function Book({
  eyebrow = "Library",
  title,
  description,
  icon,
  entries,
  fields,
  onSave,
  onCreate,
  isLoading,
  isError,
  noun = "entries",
}: BookProps) {
  const [query, setQuery] = useState("");
  const [openId, setOpenId] = useState<string | null>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entries;
    return entries.filter(
      (e) =>
        e.title.toLowerCase().includes(q) ||
        e.subtitle?.toLowerCase().includes(q),
    );
  }, [entries, query]);

  return (
    <PageBody width="wide">
      <PageHeader
        eyebrow={eyebrow}
        title={title}
        description={description}
        actions={
          <>
            <span className="rounded-full border border-[var(--border)] bg-[var(--color-ink-850)] px-2.5 py-1 text-xs font-medium text-[var(--muted-foreground)]">
              {entries.length} {noun}
            </span>
            {onCreate && (
              <Button size="sm" onClick={() => onCreate()}>
                <Plus className="size-4" /> New
              </Button>
            )}
          </>
        }
      />

      {/* Search */}
      <div className="relative mt-[var(--gap,14px)]">
        <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-[var(--muted-foreground)]/70" />
        <Field
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={`Search ${noun}…`}
          className="pl-9"
        />
      </div>

      {/* List / states */}
      <div className="mt-[var(--gap,14px)] flex-1">
        {isLoading ? (
          <div className="grid place-items-center py-16 text-[var(--muted-foreground)]">
            <Loader2 className="size-5 animate-spin" />
          </div>
        ) : isError ? (
          <EmptyState
            icon={icon}
            title={`Couldn't load ${noun}.`}
            description="The backend returned an error. Make sure the server is running on :7341."
          />
        ) : filtered.length === 0 ? (
          <EmptyState
            icon={icon}
            title={query ? `No ${noun} match “${query}”.` : `No ${noun} yet.`}
            description={
              query
                ? "Try a different search."
                : `This book has no ${noun}. They'll appear here once added.`
            }
          />
        ) : (
          <div className="flex flex-col gap-2">
            {filtered.map((entry) => (
              <BookRow
                key={entry.id}
                entry={entry}
                fields={fields}
                open={openId === entry.id}
                onToggle={() =>
                  setOpenId((id) => (id === entry.id ? null : entry.id))
                }
                onSave={onSave}
              />
            ))}
          </div>
        )}
      </div>
    </PageBody>
  );
}

/** One expandable list row + its inline edit form. */
function BookRow({
  entry,
  fields,
  open,
  onToggle,
  onSave,
}: {
  entry: BookEntry;
  fields: BookField[];
  open: boolean;
  onToggle: () => void;
  onSave: BookProps["onSave"];
}) {
  // Local working copy of the values; reset whenever the source entry changes.
  const [values, setValues] = useState<BookValues>(entry.values);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState(0);

  // Re-sync if the upstream entry object changes identity (fresh server data).
  const initialKey = useMemo(() => JSON.stringify(entry.values), [entry.values]);
  const [syncKey, setSyncKey] = useState(initialKey);
  if (syncKey !== initialKey) {
    setSyncKey(initialKey);
    setValues(entry.values);
  }

  const dirty = useMemo(
    () => JSON.stringify(values) !== initialKey,
    [values, initialKey],
  );

  const set = (key: string, value: string | boolean | number) =>
    setValues((v) => ({ ...v, [key]: value }));

  const handleSave = async () => {
    setSaving(true);
    try {
      await onSave(entry.id, values);
      setSavedAt(Date.now());
      window.setTimeout(() => setSavedAt(0), 2000);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className={cn(
        "overflow-hidden rounded-xl border bg-[var(--card)] transition-colors",
        open ? "border-[var(--color-ink-600)]" : "border-[var(--border)]",
      )}
    >
      {/* Collapsed header row */}
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center gap-3 px-[var(--card-pad,15px)] py-3 text-left transition-colors hover:bg-[var(--color-ink-700)]/30"
      >
        <ChevronDown
          className={cn(
            "size-4 shrink-0 text-[var(--muted-foreground)] transition-transform",
            open && "rotate-180",
          )}
        />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-medium">{entry.title}</span>
            {entry.badge}
          </div>
          {entry.subtitle && (
            <p className="mt-0.5 truncate text-[13px] text-[var(--muted-foreground)]">
              {entry.subtitle}
            </p>
          )}
        </div>
      </button>

      {/* Expanded edit form */}
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
            className="overflow-hidden"
          >
            <div className="border-t border-[var(--line-soft)] px-[var(--card-pad,15px)] py-4">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                {fields.map((field) => {
                  const span =
                    field.full ?? field.kind === "textarea"
                      ? "sm:col-span-2"
                      : "";
                  return (
                    <div key={field.key} className={span}>
                      <BookFieldEditor
                        field={field}
                        value={values[field.key]}
                        onChange={(v) => set(field.key, v)}
                      />
                    </div>
                  );
                })}
              </div>

              {/* Per-entry save bar */}
              <div className="mt-4 flex items-center justify-end gap-2 border-t border-[var(--line-soft)] pt-3">
                <AnimatePresence>
                  {savedAt > 0 && (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="mr-1 flex items-center gap-1.5 text-[13px] font-medium text-[var(--color-player)]"
                    >
                      <Check className="size-4" /> Saved
                    </motion.span>
                  )}
                </AnimatePresence>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={!dirty || saving}
                  onClick={() => setValues(entry.values)}
                >
                  <RotateCcw className="size-3.5" /> Reset
                </Button>
                <Button size="sm" disabled={!dirty || saving} onClick={handleSave}>
                  {saving ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Check className="size-4" />
                  )}
                  Save
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/** Renders the right input widget for one field. */
function BookFieldEditor({
  field,
  value,
  onChange,
}: {
  field: BookField;
  value: string | boolean | number | undefined;
  onChange: (value: string | boolean | number) => void;
}) {
  const label = (
    <label className="mb-1.5 block text-[13px] font-medium">{field.label}</label>
  );
  const help = field.help && (
    <p className="mt-1 text-[12px] text-[var(--muted-foreground)]">
      {field.help}
    </p>
  );

  if (field.kind === "toggle") {
    return (
      <div className="flex items-center justify-between gap-4">
        <div>
          <span className="block text-[13px] font-medium">{field.label}</span>
          {help}
        </div>
        <Switch
          checked={Boolean(value)}
          onCheckedChange={(v) => onChange(v)}
        />
      </div>
    );
  }

  if (field.kind === "select") {
    return (
      <div>
        {label}
        <Select
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
          className="w-full"
        >
          {(field.options ?? []).map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </Select>
        {help}
      </div>
    );
  }

  if (field.kind === "textarea") {
    return (
      <div>
        {label}
        <TextArea
          rows={field.rows ?? 4}
          value={String(value ?? "")}
          placeholder={field.placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
        {help}
      </div>
    );
  }

  // text / number
  return (
    <div>
      {label}
      <Field
        type={field.kind === "number" ? "number" : "text"}
        value={value === undefined ? "" : String(value)}
        placeholder={field.placeholder}
        onChange={(e) =>
          onChange(
            field.kind === "number"
              ? Number(e.target.value)
              : e.target.value,
          )
        }
      />
      {help}
    </div>
  );
}
