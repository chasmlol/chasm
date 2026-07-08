import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, Search } from "lucide-react";

import {
  triggersApi,
  type TriggerRuleSave,
  type TriggersViewDto,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { SettingsPage } from "@/components/ui/settings-page";
import { Switch } from "@/components/ui/switch";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, FormRow } from "@/components/ui/page";

// Triggers — witness memory + event-trigger reactions. Every game event a
// nearby NPC witnesses is written into their chat history as a narration line
// (permanent memory). Event types ENABLED here additionally make the nearest
// witnessing NPC react out loud the moment it happens — each with its own
// % chance and its own per-type cooldown, plus an optional global cooldown
// across all triggers. Conversation events are excluded entirely (the dialogue
// already IS the history).

const DEFAULT_CHANCE = 100;
const DEFAULT_COOLDOWN_SECS = 10;

/** What each event type means, for the row help text + search matching. */
const TYPE_HELP: Record<string, string> = {
  arrival:
    "An NPC finishes a travel — the traveler remembers it, and reacts if triggered ('Made it — you wanted me?').",
  combat: "A fight ends near the NPC (one summarized encounter).",
  death: "Someone dies where the NPC can see it.",
  murder: "You kill someone who wasn't hostile — a witnessed crime.",
  item: "Loot windows, equips/unequips, and consumables.",
  theft: "You take something that isn't yours (the red 'Steal' prompt).",
  pickpocket: "You go through someone's pockets while sneaking.",
  lockpick: "You pick or break open a lock.",
  hacking: "You hack a terminal (or get locked out trying).",
  shooting: "An out-of-combat shot or firing burst.",
  weapon: "You draw your weapon around people, out of combat.",
  sneak: "You start sneaking around people, out of combat.",
  location: "Arriving somewhere named.",
  trade: "You sell things to a vendor (one summarized sale).",
  repair: "You repair an item.",
  injury: "One of your limbs gets crippled.",
  rads: "Your radiation sickness reaches a new stage.",
  day: "A new in-game day starts.",
  level: "A level-up.",
  karma: "A karma class shift.",
  companion: "A companion joins or leaves.",
  quest: "A quest objective completes.",
  conversation: "Excluded from witnessing — the dialogue already is the history.",
};

interface RuleForm {
  enabled: boolean;
  chancePercent: number;
  cooldownSecs: number;
  requireSight: boolean;
}

interface TriggersForm {
  enabled: boolean;
  companionsOnly: boolean;
  globalCooldownEnabled: boolean;
  globalCooldownSecs: number;
  rules: Record<string, RuleForm>;
}

function formFrom(view: TriggersViewDto): TriggersForm {
  const rules: Record<string, RuleForm> = {};
  for (const entry of view.catalog) {
    rules[entry.type] = {
      enabled: entry.enabled,
      chancePercent: entry.chancePercent,
      cooldownSecs: entry.cooldownSecs,
      requireSight: entry.requireSight,
    };
  }
  return {
    enabled: view.enabled,
    companionsOnly: view.companionsOnly,
    globalCooldownEnabled: view.globalCooldownEnabled,
    globalCooldownSecs: view.globalCooldownSecs,
    rules,
  };
}

/** Only rules that are enabled or carry tuned knobs need persisting. */
function rulesToSave(form: TriggersForm): TriggerRuleSave[] {
  return Object.entries(form.rules)
    .filter(
      ([, rule]) =>
        rule.enabled ||
        rule.requireSight ||
        rule.chancePercent !== DEFAULT_CHANCE ||
        rule.cooldownSecs !== DEFAULT_COOLDOWN_SECS,
    )
    .map(([type, rule]) => ({
      type,
      enabled: rule.enabled,
      chancePercent: rule.chancePercent,
      cooldownSecs: rule.cooldownSecs,
      requireSight: rule.requireSight,
    }));
}

const clampChance = (value: number) =>
  Math.max(0, Math.min(100, Math.round(Number.isFinite(value) ? value : 0)));
const clampCooldown = (value: number) =>
  Math.max(0, Math.min(86_400, Math.round(Number.isFinite(value) ? value : 0)));

export function Triggers() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["triggers"],
    queryFn: triggersApi.view,
  });

  const initial = useMemo(
    () => (query.data ? formFrom(query.data) : null),
    [query.data],
  );
  const [form, setForm] = useState<TriggersForm | null>(initial);
  const [search, setSearch] = useState("");
  const [justSaved, setJustSaved] = useState(false);
  useEffect(() => setForm(initial), [initial]);

  const dirty = useMemo(
    () =>
      !!form && !!initial && JSON.stringify(form) !== JSON.stringify(initial),
    [form, initial],
  );

  const save = useMutation({
    mutationFn: () =>
      triggersApi.save({
        enabled: form!.enabled,
        companionsOnly: form!.companionsOnly,
        globalCooldownEnabled: form!.globalCooldownEnabled,
        globalCooldownSecs: form!.globalCooldownSecs,
        triggers: rulesToSave(form!),
      }),
    onSuccess: (fresh) => {
      qc.setQueryData(["triggers"], fresh);
      setJustSaved(true);
      window.setTimeout(() => setJustSaved(false), 2200);
    },
  });

  const setRule = (type: string, patch: Partial<RuleForm>) =>
    setForm((f) =>
      f
        ? {
            ...f,
            rules: { ...f.rules, [type]: { ...f.rules[type], ...patch } },
          }
        : f,
    );

  const catalog = query.data?.catalog ?? [];
  const needle = search.trim().toLowerCase();
  const visible = catalog.filter(
    (entry) =>
      !needle ||
      entry.type.toLowerCase().includes(needle) ||
      (TYPE_HELP[entry.type] ?? "").toLowerCase().includes(needle),
  );

  return (
    <SettingsPage
      eyebrow="World"
      title="Triggers"
      description="Nearby NPCs permanently remember what they see you do — every witnessed event lands in their chat history as a narration line. Enable an event type below to ALSO make the nearest witness react out loud when it happens, with a per-type % chance and cooldown."
      save={
        form
          ? {
              dirty,
              saving: save.isPending,
              error: save.isError,
              justSaved,
              onReset: () => initial && setForm(initial),
              onSave: () => form && save.mutate(),
              saveLabel: "Save triggers",
            }
          : undefined
      }
    >
      {query.isLoading && (
        <div className="flex items-center gap-2 text-sm text-[var(--muted-foreground)]">
          <Loader2 className="size-4 animate-spin" /> Loading…
        </div>
      )}

      {form && (
        <Card>
          <CardHeader>
            <CardTitle>Witness memory</CardTitle>
            <CardDescription>
              Who notices, and how often reactions are allowed at all.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <FormRow
              label="Enable witness memory"
              help="Master switch for the whole system: witnessed history lines AND trigger reactions. Off = events still reach the Events page, but NPCs see nothing."
              control={
                <Switch
                  checked={form.enabled}
                  onCheckedChange={(v) =>
                    setForm((f) => (f ? { ...f, enabled: v } : f))
                  }
                />
              }
            />
            <FormRow
              label="Companions only"
              help="Only companions witness events. Off = every mapped NPC within speaking range does."
              control={
                <Switch
                  checked={form.companionsOnly}
                  onCheckedChange={(v) =>
                    setForm((f) => (f ? { ...f, companionsOnly: v } : f))
                  }
                />
              }
            />
            <FormRow
              label="Global reaction cooldown"
              help="One shared timer across ALL trigger types: after any reaction fires, no other trigger fires until it elapses. Per-type cooldowns below still apply on top."
              control={
                <div className="flex items-center gap-3">
                  {form.globalCooldownEnabled && (
                    <label className="flex items-center gap-1.5 text-xs text-[var(--muted-foreground)]">
                      <Field
                        type="number"
                        className="w-20 text-right"
                        min={0}
                        max={86_400}
                        step={5}
                        value={form.globalCooldownSecs}
                        onChange={(e) =>
                          setForm((f) =>
                            f
                              ? {
                                  ...f,
                                  globalCooldownSecs: clampCooldown(
                                    Number(e.target.value),
                                  ),
                                }
                              : f,
                          )
                        }
                      />
                      sec
                    </label>
                  )}
                  <Switch
                    checked={form.globalCooldownEnabled}
                    onCheckedChange={(v) =>
                      setForm((f) =>
                        f ? { ...f, globalCooldownEnabled: v } : f,
                      )
                    }
                  />
                </div>
              }
            />
          </CardContent>
        </Card>
      )}

      {form && catalog.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <CardTitle>Reaction triggers</CardTitle>
                <CardDescription className="mt-1.5">
                  On = the nearest witnessing NPC speaks up immediately (through
                  normal TTS). Chance is rolled per event; the cooldown is per
                  trigger type. Off = the event is only remembered. Seen = the
                  NPC must actually see you (the game's [HIDDEN] detection, per
                  NPC) or they don't witness the event at all — leave it off
                  for things anyone could hear, like gunshots.
                </CardDescription>
              </div>
              <div className="relative w-56 shrink-0">
                <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-[var(--muted-foreground)]" />
                <Field
                  type="search"
                  className="w-full pl-8"
                  placeholder="Search events…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
              </div>
            </div>
          </CardHeader>
          <CardContent className="flex flex-col">
            {/* Column headings */}
            <div className="flex items-center gap-3 border-b border-[var(--line-soft)] pb-2 text-[11px] font-semibold uppercase tracking-wide text-[var(--muted-foreground)]">
              <span className="min-w-0 flex-1">Event</span>
              <span className="w-20 text-right">Chance %</span>
              <span className="w-20 text-right">Cooldown s</span>
              <span className="w-[46px] text-center">Seen</span>
              <span className="w-[40px] text-right">On</span>
            </div>
            {visible.length === 0 && (
              <p className="py-6 text-center text-sm text-[var(--muted-foreground)]">
                No event types match “{search}”.
              </p>
            )}
            {visible.map((entry) => {
              const rule = form.rules[entry.type] ?? {
                enabled: false,
                chancePercent: DEFAULT_CHANCE,
                cooldownSecs: DEFAULT_COOLDOWN_SECS,
                requireSight: false,
              };
              const rowDisabled = entry.excluded || !form.enabled;
              const knobsDisabled = rowDisabled || !rule.enabled;
              return (
                <div
                  key={entry.type}
                  className={cn(
                    "flex items-center gap-3 border-b border-[var(--line-soft)] py-2.5 last:border-b-0",
                    rowDisabled && "opacity-50",
                  )}
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium capitalize">
                        {entry.type}
                      </span>
                      {entry.dynamic && (
                        <span className="rounded-full border border-[var(--border)] bg-[var(--color-ink-850)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--muted-foreground)]">
                          observed
                        </span>
                      )}
                    </div>
                    <p className="mt-0.5 truncate text-xs text-[var(--muted-foreground)]">
                      {TYPE_HELP[entry.type] ??
                        "Seen in the event log; reacts like any other type."}
                    </p>
                  </div>
                  <Field
                    type="number"
                    className="w-20 text-right"
                    min={0}
                    max={100}
                    step={5}
                    disabled={knobsDisabled}
                    value={rule.chancePercent}
                    onChange={(e) =>
                      setRule(entry.type, {
                        chancePercent: clampChance(Number(e.target.value)),
                      })
                    }
                  />
                  <Field
                    type="number"
                    className="w-20 text-right"
                    min={0}
                    max={86_400}
                    step={5}
                    disabled={knobsDisabled}
                    value={rule.cooldownSecs}
                    onChange={(e) =>
                      setRule(entry.type, {
                        cooldownSecs: clampCooldown(Number(e.target.value)),
                      })
                    }
                  />
                  <div
                    className="flex w-[46px] justify-center"
                    title="Only witnessed when the NPC can actually SEE you (the game's detection state). Applies to memory too, not just reactions."
                  >
                    <Switch
                      checked={rule.requireSight && !entry.excluded}
                      disabled={rowDisabled}
                      onCheckedChange={(v) =>
                        setRule(entry.type, { requireSight: v })
                      }
                    />
                  </div>
                  <div className="flex w-[40px] justify-end">
                    <Switch
                      checked={rule.enabled && !entry.excluded}
                      disabled={rowDisabled}
                      onCheckedChange={(v) => setRule(entry.type, { enabled: v })}
                    />
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>
      )}
    </SettingsPage>
  );
}
