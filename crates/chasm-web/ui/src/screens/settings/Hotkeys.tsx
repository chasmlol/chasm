import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Keyboard, Loader2, RotateCcw } from "lucide-react";

import {
  systemApi,
  type HotkeysConfig,
  type HotkeysView,
} from "@/lib/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FormRow } from "@/components/ui/page";
import { SettingsPage } from "@/components/ui/settings-page";

// Hotkeys — the four in-game input bindings. Each row is a click-to-capture
// control: click, press a key, done (Escape cancels). Names are the canonical
// key names the backend validates (chasm-core/src/hotkeys.rs); the backend
// turns them into Win32 VK codes and pushes them to the game over the bridge,
// where the NVSE plugin live-polls them — no game restart needed.

/** Browser `KeyboardEvent.code` → canonical backend key name. Must stay a
 * subset of `virtual_key_code` in chasm-core/src/hotkeys.rs. Returns null for
 * keys we don't support (the capture control just keeps listening). */
function canonicalKeyName(code: string): string | null {
  if (/^Key[A-Z]$/.test(code)) return code.slice(3);
  if (/^Digit[0-9]$/.test(code)) return code.slice(5);
  if (/^F([1-9]|1[0-9]|2[0-4])$/.test(code)) return code;
  if (/^Numpad[0-9]$/.test(code)) return code;
  const map: Record<string, string> = {
    Enter: "Enter",
    NumpadEnter: "Enter",
    Space: "Space",
    Tab: "Tab",
    Backspace: "Backspace",
    AltLeft: "Alt",
    AltRight: "Alt",
    ControlLeft: "Ctrl",
    ControlRight: "Ctrl",
    ShiftLeft: "Shift",
    ShiftRight: "Shift",
    CapsLock: "CapsLock",
    ArrowLeft: "Left",
    ArrowUp: "Up",
    ArrowRight: "Right",
    ArrowDown: "Down",
    Home: "Home",
    End: "End",
    PageUp: "PageUp",
    PageDown: "PageDown",
    Insert: "Insert",
    Delete: "Delete",
    Pause: "Pause",
    ScrollLock: "ScrollLock",
    NumLock: "NumLock",
    NumpadMultiply: "NumpadMultiply",
    NumpadAdd: "NumpadAdd",
    NumpadSubtract: "NumpadSubtract",
    NumpadDecimal: "NumpadDecimal",
    NumpadDivide: "NumpadDivide",
    Semicolon: "Semicolon",
    Equal: "Equals",
    Comma: "Comma",
    Minus: "Minus",
    Period: "Period",
    Slash: "Slash",
    Backquote: "Backquote",
    BracketLeft: "LeftBracket",
    Backslash: "Backslash",
    BracketRight: "RightBracket",
    Quote: "Quote",
  };
  return map[code] ?? null;
}

/** Slightly friendlier display text for a canonical name. */
function displayKeyName(name: string): string {
  const pretty: Record<string, string> = {
    Backquote: "` (Backquote)",
    Semicolon: "; (Semicolon)",
    Equals: "= (Equals)",
    Comma: ", (Comma)",
    Minus: "- (Minus)",
    Period: ". (Period)",
    Slash: "/ (Slash)",
    Backslash: "\\ (Backslash)",
    LeftBracket: "[ (Left bracket)",
    RightBracket: "] (Right bracket)",
    Quote: "' (Quote)",
  };
  return pretty[name] ?? name;
}

const BINDING_ROWS: {
  key: keyof HotkeysConfig;
  label: string;
  help: string;
}[] = [
  {
    key: "push_to_talk",
    label: "Push to talk",
    help: "Hold to record voice input to nearby NPCs; release to send.",
  },
  {
    key: "enter_text",
    label: "Enter text",
    help: "Opens the typed-message input to nearby NPCs.",
  },
  {
    key: "todd_push_to_talk",
    label: "Todd — push to talk",
    help: "Hold to record voice input addressed to Todd (the narrator).",
  },
  {
    key: "todd_enter_text",
    label: "Todd — enter text",
    help: "Opens the typed-message input to Todd.",
  },
];

export function Hotkeys() {
  const queryClient = useQueryClient();
  const query = useQuery({ queryKey: ["hotkeys"], queryFn: systemApi.hotkeys });

  if (query.isLoading) {
    return (
      <div className="grid h-full place-items-center text-[var(--muted-foreground)]">
        <Loader2 className="size-6 animate-spin" />
      </div>
    );
  }
  if (query.isError || !query.data) {
    return (
      <div className="grid h-full place-items-center p-8 text-center">
        <div>
          <p className="text-sm font-medium text-[var(--color-danger)]">
            Couldn’t load the hotkey bindings.
          </p>
          <p className="mt-1 text-[13px] text-[var(--muted-foreground)]">
            Make sure the chasm backend is running.
          </p>
        </div>
      </div>
    );
  }

  return (
    <HotkeysForm
      data={query.data}
      onSaved={(fresh) => queryClient.setQueryData(["hotkeys"], fresh)}
    />
  );
}

/** One click-to-capture binding control. Click arms it; the next supported
 * keypress is recorded (Escape cancels, clicking away cancels). */
function KeyCaptureButton({
  value,
  capturing,
  onStartCapture,
  onCapture,
  onCancel,
  duplicate,
}: {
  value: string;
  capturing: boolean;
  onStartCapture: () => void;
  onCapture: (name: string) => void;
  onCancel: () => void;
  duplicate: boolean;
}) {
  useEffect(() => {
    if (!capturing) return;
    const onKey = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.code === "Escape") {
        onCancel();
        return;
      }
      const name = canonicalKeyName(e.code);
      if (name) onCapture(name);
      // Unsupported key: keep listening.
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [capturing, onCapture, onCancel]);

  return (
    <button
      type="button"
      onClick={() => (capturing ? onCancel() : onStartCapture())}
      onBlur={() => capturing && onCancel()}
      className={[
        "inline-flex h-9 min-w-36 items-center justify-center gap-2 rounded-lg border px-4 font-mono text-sm transition-colors",
        capturing
          ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-accent)] ring-2 ring-[var(--ring)]/40"
          : duplicate
            ? "border-[var(--color-npc)]/60 bg-[var(--color-ink-850)] text-[var(--foreground)]"
            : "border-[var(--border)] bg-[var(--color-ink-850)] text-[var(--foreground)] hover:border-[var(--color-accent)]/60",
      ].join(" ")}
    >
      <Keyboard className="size-4 opacity-60" />
      {capturing ? "Press a key…" : displayKeyName(value)}
    </button>
  );
}

function HotkeysForm({
  data,
  onSaved,
}: {
  data: HotkeysView;
  onSaved: (fresh: HotkeysView) => void;
}) {
  const initial = useMemo(() => data.config, [data.config]);
  const [form, setForm] = useState<HotkeysConfig>(initial);
  const [capturing, setCapturing] = useState<keyof HotkeysConfig | null>(null);
  const [justSaved, setJustSaved] = useState(false);

  useEffect(() => setForm(initial), [initial]);

  const dirty = useMemo(
    () => JSON.stringify(form) !== JSON.stringify(initial),
    [form, initial],
  );

  const save = useMutation({
    mutationFn: () => systemApi.saveHotkeys(form),
    onSuccess: (fresh) => {
      onSaved(fresh);
      setJustSaved(true);
      window.setTimeout(() => setJustSaved(false), 2200);
    },
  });

  const set = (key: keyof HotkeysConfig, value: string) =>
    setForm((f) => ({ ...f, [key]: value }));

  // Non-blocking duplicate detection: which key names appear on 2+ bindings.
  const duplicateNames = useMemo(() => {
    const counts = new Map<string, number>();
    for (const row of BINDING_ROWS) {
      const v = form[row.key];
      counts.set(v, (counts.get(v) ?? 0) + 1);
    }
    return new Set(
      [...counts.entries()].filter(([, n]) => n > 1).map(([v]) => v),
    );
  }, [form]);

  const duplicateRows = BINDING_ROWS.filter((row) =>
    duplicateNames.has(form[row.key]),
  );

  return (
    <SettingsPage
      eyebrow="System"
      title="Hotkeys"
      description="The in-game input bindings. Click a binding, press a key, save — a running game picks the change up within a second (no restart needed)."
      save={{
        dirty,
        saving: save.isPending,
        error: save.isError,
        justSaved,
        onReset: () => setForm(initial),
        onSave: () => save.mutate(),
        saveLabel: "Save hotkeys",
      }}
    >
      <Card>
        <CardHeader>
          <CardTitle>In-game bindings</CardTitle>
          <CardDescription>
            Letters, digits, F-keys and most standalone keys are supported
            (Escape is reserved for cancel in-game). Modifier keys like Alt
            bind as the key itself, not as a combo.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          {duplicateRows.length > 0 && (
            <div className="rounded-lg border border-[var(--color-npc)]/40 bg-[var(--color-npc)]/5 p-3 text-[13px] text-[var(--color-npc)]">
              ⚠ Duplicate binding:{" "}
              {duplicateRows.map((r) => r.label).join(", ")} share the same
              key. You can still save, but only one action will fire in-game.
            </div>
          )}
          {BINDING_ROWS.map((row) => (
            <FormRow
              key={row.key}
              label={row.label}
              help={row.help}
              control={
                <div className="flex items-center gap-2">
                  <KeyCaptureButton
                    value={form[row.key]}
                    capturing={capturing === row.key}
                    onStartCapture={() => setCapturing(row.key)}
                    onCapture={(name) => {
                      set(row.key, name);
                      setCapturing(null);
                    }}
                    onCancel={() => setCapturing(null)}
                    duplicate={duplicateNames.has(form[row.key])}
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    title={`Reset to default (${displayKeyName(data.defaults[row.key])})`}
                    onClick={() => set(row.key, data.defaults[row.key])}
                    disabled={form[row.key] === data.defaults[row.key]}
                  >
                    <RotateCcw />
                  </Button>
                </div>
              }
            />
          ))}
        </CardContent>
      </Card>
    </SettingsPage>
  );
}
