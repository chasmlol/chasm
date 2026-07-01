import type { ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Check, Loader2, RotateCcw } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { PageBody, PageHeader, Stack, FormRow } from "@/components/ui/page";

// ===========================================================================
// SettingsPage — the shared wrapper for EVERY Settings screen (Interface, LLM,
// TTS, STT, Retrieval, Bridge, Profiles, Tracing). It owns the consistent
// chrome: the PageHeader, the bounded body, and the sticky save bar with
// dirty/saving/saved affordances. A screen supplies its eyebrow/title/body and
// (optionally) save state; the look + behaviour is identical across all of them.
//
// Fill agents: wrap your settings screen in <SettingsPage>. If your screen
// saves, pass the `save` block so you get the standard sticky action bar for
// free. If it's read-only (e.g. Tracing), omit `save`.
// ===========================================================================

export interface SettingsSaveState {
  /** Are there unsaved edits? Enables the Save button + "Unsaved changes". */
  dirty: boolean;
  /** Is a save in flight? Shows the spinner + disables the buttons. */
  saving?: boolean;
  /** Did the last save error? Shows a "Save failed" note. */
  error?: boolean;
  /** Flash a "Saved" confirmation (caller toggles this true briefly). */
  justSaved?: boolean;
  /** Reset edits back to the server state. */
  onReset: () => void;
  /** Persist edits. */
  onSave: () => void;
  /** Save button label (default "Save"). */
  saveLabel?: string;
}

export function SettingsPage({
  eyebrow,
  title,
  description,
  headerActions,
  save,
  children,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  headerActions?: ReactNode;
  save?: SettingsSaveState;
  children: ReactNode;
}) {
  return (
    <PageBody width="prose">
      <PageHeader
        eyebrow={eyebrow}
        title={title}
        description={description}
        actions={headerActions}
      />

      <Stack className="flex-1 pt-[var(--gap,14px)]">{children}</Stack>

      {save && <SettingsSaveBar save={save} />}
    </PageBody>
  );
}

/** The sticky save bar at the foot of a settings screen. */
function SettingsSaveBar({ save }: { save: SettingsSaveState }) {
  return (
    <div className="sticky bottom-0 z-10 mt-6 flex items-center justify-between gap-3 border-t border-[var(--line)] bg-[var(--background)]/85 py-4 backdrop-blur">
      <span className="text-[13px] text-[var(--muted-foreground)]">
        {save.dirty ? "Unsaved changes" : "Saved — applied live"}
      </span>
      <div className="flex items-center gap-2">
        <AnimatePresence>
          {save.justSaved && (
            <motion.span
              initial={{ opacity: 0, x: 6 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-1.5 text-[13px] font-medium text-[var(--color-player)]"
            >
              <Check className="size-4" /> Saved
            </motion.span>
          )}
        </AnimatePresence>
        {save.error && (
          <span className="text-[13px] text-[var(--color-danger)]">
            Save failed
          </span>
        )}
        <Button
          variant="ghost"
          size="sm"
          disabled={!save.dirty || save.saving}
          onClick={save.onReset}
        >
          <RotateCcw className="size-3.5" /> Reset
        </Button>
        <Button
          size="sm"
          disabled={!save.dirty || save.saving}
          onClick={save.onSave}
        >
          {save.saving ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <Check className="size-4" />
          )}
          {save.saveLabel ?? "Save"}
        </Button>
      </div>
    </div>
  );
}

/**
 * A labeled on/off row with a Switch on the right — the shared toggle control
 * for settings. Built on FormRow so it lines up with the other inputs.
 */
export function ToggleRow({
  label,
  help,
  checked,
  onChange,
  disabled,
}: {
  label: ReactNode;
  help?: ReactNode;
  checked: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <FormRow
      label={label}
      help={help}
      control={
        <Switch
          checked={checked}
          onCheckedChange={onChange}
          disabled={disabled}
        />
      }
    />
  );
}

/**
 * A segmented single-choice control (e.g. Comfortable / Compact), with a sliding
 * highlight pill. Generic over the option value so any settings screen can use
 * it for small enum choices.
 */
export function SegmentedControl<T extends string>({
  value,
  options,
  onChange,
  layoutId,
  className,
}: {
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (value: T) => void;
  /** Unique id so the sliding pill animates within THIS control only. */
  layoutId: string;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "inline-flex rounded-lg border border-[var(--border)] bg-[var(--color-ink-850)] p-1",
        className,
      )}
    >
      {options.map((option) => {
        const selected = value === option.value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            className={cn(
              "relative rounded-md px-4 py-1.5 text-[13px] font-medium transition-colors",
              selected
                ? "text-[var(--foreground)]"
                : "text-[var(--muted-foreground)] hover:text-[var(--foreground)]",
            )}
          >
            {selected && (
              <motion.span
                layoutId={layoutId}
                className="absolute inset-0 rounded-md bg-[var(--color-ink-600)]"
                transition={{ type: "spring", stiffness: 500, damping: 38 }}
              />
            )}
            <span className="relative">{option.label}</span>
          </button>
        );
      })}
    </div>
  );
}
