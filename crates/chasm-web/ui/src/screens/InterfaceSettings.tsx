import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Loader2 } from "lucide-react";

import {
  systemApi,
  type InterfaceForm,
  type InterfacePanel,
  type SettingsPage as SettingsPageData,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { reloadLiveTheme } from "@/lib/theme";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Section, SectionLabel } from "@/components/ui/page";
import {
  SettingsPage,
  ToggleRow,
  SegmentedControl,
} from "@/components/ui/settings-page";

// Interface settings — the appearance editor. This is the one fully-wired
// settings screen and the reference for how every settings screen is built:
// it fetches its own data, composes the SHARED primitives (SettingsPage,
// SegmentedControl, ToggleRow, Section, FormRow, Card), and uses the standard
// sticky save bar. Saving re-pulls /theme.css so accent/theme/density/font
// apply live.

// Derive the editable form state from the server's InterfacePanel view.
function formFromPanel(panel: InterfacePanel): InterfaceForm {
  return {
    theme: panel.themes.find((t) => t.selected)?.id ?? panel.themes[0]?.id ?? "",
    accent: panel.accent,
    density: panel.density,
    font_scale: panel.font_scale,
    reduce_motion: panel.reduce_motion,
    show_timestamps: panel.show_timestamps,
    show_prompt_panel: panel.show_prompt_panel,
  };
}

export function InterfaceSettings() {
  const queryClient = useQueryClient();
  const settings = useQuery({
    queryKey: ["settings", "interface"],
    queryFn: () => systemApi.settings("interface"),
  });

  if (settings.isLoading) {
    return (
      <div className="grid h-full place-items-center text-[var(--muted-foreground)]">
        <Loader2 className="size-6 animate-spin" />
      </div>
    );
  }
  if (settings.isError || !settings.data) {
    return (
      <div className="grid h-full place-items-center p-8 text-center">
        <div>
          <p className="text-sm font-medium text-[var(--color-danger)]">
            Couldn’t reach the chasm backend.
          </p>
          <p className="mt-1 text-[13px] text-[var(--muted-foreground)]">
            Make sure the server is running on :7341.
          </p>
        </div>
      </div>
    );
  }

  return (
    <InterfaceForm_
      panel={settings.data.interface}
      onSaved={(fresh) =>
        queryClient.setQueryData(["settings", "interface"], fresh)
      }
    />
  );
}

function InterfaceForm_({
  panel,
  onSaved,
}: {
  panel: InterfacePanel;
  onSaved: (fresh: SettingsPageData) => void;
}) {
  const initial = useMemo(() => formFromPanel(panel), [panel]);
  const [form, setForm] = useState<InterfaceForm>(initial);
  const [justSaved, setJustSaved] = useState(false);

  useEffect(() => setForm(initial), [initial]);

  const dirty = useMemo(
    () => JSON.stringify(form) !== JSON.stringify(initial),
    [form, initial],
  );

  const save = useMutation({
    mutationFn: () => systemApi.saveInterface(form),
    onSuccess: (fresh) => {
      // Re-pull /theme.css so the new accent/theme/font/density apply at once.
      reloadLiveTheme();
      onSaved(fresh);
      setJustSaved(true);
      window.setTimeout(() => setJustSaved(false), 2200);
    },
  });

  const set = <K extends keyof InterfaceForm>(key: K, value: InterfaceForm[K]) =>
    setForm((f) => ({ ...f, [key]: value }));

  return (
    <SettingsPage
      eyebrow="Appearance"
      title="Interface"
      description={
        <>
          Theme the control panel. Every option is emitted into a live{" "}
          <code className="rounded bg-[var(--color-ink-700)] px-1.5 py-0.5 font-mono text-[11px] text-[var(--foreground)]">
            /theme.css
          </code>{" "}
          stylesheet read fresh on each load, so changes apply immediately — no
          restart.
        </>
      }
      save={{
        dirty,
        saving: save.isPending,
        error: save.isError,
        justSaved,
        onReset: () => setForm(initial),
        onSave: () => save.mutate(),
        saveLabel: "Save appearance",
      }}
    >
      {/* Theme presets */}
      <Card>
        <CardHeader>
          <CardTitle>Theme</CardTitle>
          <CardDescription>
            Sets the base dark palette. The accent below applies on top of
            whichever preset you pick.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-3">
            {panel.themes.map((theme) => {
              const selected = form.theme === theme.id;
              return (
                <button
                  key={theme.id}
                  onClick={() => set("theme", theme.id)}
                  className={cn(
                    "group relative overflow-hidden rounded-xl border p-3 text-left transition-all",
                    selected
                      ? "border-[var(--color-accent)] ring-1 ring-[var(--color-accent)]/40"
                      : "border-[var(--border)] hover:border-[var(--color-ink-600)]",
                  )}
                >
                  <div
                    className="mb-2.5 flex h-14 items-end gap-1.5 rounded-lg p-2"
                    style={{ background: theme.bg }}
                  >
                    <span
                      className="h-full w-2/3 rounded-md"
                      style={{ background: theme.panel }}
                    />
                    <span
                      className="h-2.5 w-2.5 self-start rounded-full"
                      style={{ background: form.accent }}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[13px] font-medium">
                      {theme.label}
                    </span>
                    {selected && (
                      <Check className="size-4 text-[var(--color-accent)]" />
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Accent colour */}
      <Card>
        <CardHeader>
          <CardTitle>Accent colour</CardTitle>
          <CardDescription>
            Drives the primary button, links, focus rings and highlight badges.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-3">
            <label className="relative inline-flex size-10 cursor-pointer items-center justify-center overflow-hidden rounded-lg border border-[var(--border)]">
              <span
                className="absolute inset-0"
                style={{ background: form.accent }}
              />
              <input
                type="color"
                value={form.accent}
                onChange={(e) => set("accent", e.target.value)}
                className="absolute inset-0 cursor-pointer opacity-0"
                aria-label="Accent colour"
              />
            </label>
            <code className="rounded-md bg-[var(--color-ink-700)] px-2.5 py-1.5 font-mono text-[13px] uppercase">
              {form.accent}
            </code>
            <div className="ml-1 flex flex-wrap items-center gap-1.5">
              {panel.accents.map((accent) => {
                const selected =
                  accent.value.toLowerCase() === form.accent.toLowerCase();
                return (
                  <button
                    key={accent.value}
                    title={accent.label}
                    aria-label={accent.label}
                    onClick={() => set("accent", accent.value)}
                    className={cn(
                      "size-6 rounded-full ring-2 transition-transform hover:scale-110",
                      selected ? "ring-[var(--foreground)]" : "ring-transparent",
                    )}
                    style={{ background: accent.value }}
                  />
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Layout */}
      <Card>
        <CardHeader>
          <CardTitle>Layout</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <Section
            title="Density"
            description="Compact tightens padding across the shells, cards and message rows."
          >
            <SegmentedControl
              layoutId="density-pill"
              value={form.density}
              onChange={(v) => set("density", v)}
              options={panel.densities.map((d) => ({
                value: d.value,
                label: d.label,
              }))}
            />
          </Section>

          <div>
            <div className="flex items-baseline justify-between">
              <SectionLabel>Font scale</SectionLabel>
              <span className="font-mono text-[13px] text-[var(--color-accent)]">
                {form.font_scale}%
              </span>
            </div>
            <input
              type="range"
              min={panel.font_scale_min}
              max={panel.font_scale_max}
              step={panel.font_scale_step}
              value={form.font_scale}
              onChange={(e) => set("font_scale", Number(e.target.value))}
              className="mt-3 w-full accent-[var(--color-accent)]"
            />
            <p className="mt-1.5 text-[13px] text-[var(--muted-foreground)]">
              Scales the whole UI via the root font size ({panel.font_scale_min}–
              {panel.font_scale_max}%).
            </p>
          </div>

          <ToggleRow
            label="Show the prompt-inspector column on chat pages"
            help="When off, the right-hand prompt panel is collapsed and chat gets the extra width."
            checked={form.show_prompt_panel}
            onChange={(v) => set("show_prompt_panel", v)}
          />
        </CardContent>
      </Card>

      {/* Motion & chrome */}
      <Card>
        <CardHeader>
          <CardTitle>Motion &amp; chrome</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <ToggleRow
            label="Reduce motion"
            help="Disable transitions & animations app-wide."
            checked={form.reduce_motion}
            onChange={(v) => set("reduce_motion", v)}
          />
          <ToggleRow
            label="Show message timestamps"
            checked={form.show_timestamps}
            onChange={(v) => set("show_timestamps", v)}
          />
        </CardContent>
      </Card>
    </SettingsPage>
  );
}
