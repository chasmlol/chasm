import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, Sparkles, Trash2, Zap } from "lucide-react";

import {
  skillsApi,
  type SkillDto,
  type SkillOwnerDto,
  type SkillSettingsDto,
  type SkillsViewDto,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import {
  EmptyState,
  Field,
  PageBody,
  PageHeader,
  Stack,
  StatusPill,
} from "@/components/ui/page";

// Skills — the automatic, event-triggered behaviours the skill-creator writes
// for NPCs from their journals. Each skill = owner + one trigger event + one
// action, fired with no dialogue the instant the event happens in-game. This
// page lists them per NPC, lets you enable/disable or delete any, and holds the
// system's on/off switches + the per-skill cooldown.

function formatWhen(iso?: string): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString();
}

/** Friendly-ish label for a trigger event id. */
function triggerLabel(event: string): string {
  if (event === "weapon_fire") return "the player fires";
  return event.replace(/_/g, " ");
}

function SettingsCard({ view }: { view: SkillsViewDto }) {
  const qc = useQueryClient();
  const [draft, setDraft] = useState<SkillSettingsDto>(view.settings);

  // Re-seed when the server view changes (e.g. after a save round-trip).
  useEffect(() => {
    setDraft(view.settings);
  }, [view.settings]);

  const dirty = useMemo(
    () =>
      draft.journalingEnabled !== view.settings.journalingEnabled ||
      draft.skillCreationEnabled !== view.settings.skillCreationEnabled ||
      draft.skillExecutionEnabled !== view.settings.skillExecutionEnabled ||
      draft.skillCooldownSecs !== view.settings.skillCooldownSecs,
    [draft, view.settings],
  );

  const save = useMutation({
    mutationFn: (settings: SkillSettingsDto) => skillsApi.saveSettings(settings),
    onSuccess: (v) => qc.setQueryData(["skills"], v),
  });

  const Row = ({
    label,
    hint,
    checked,
    onChange,
  }: {
    label: string;
    hint: string;
    checked: boolean;
    onChange: (v: boolean) => void;
  }) => (
    <label className="flex items-start justify-between gap-4 border-t border-[var(--line-soft)] px-5 py-3 first:border-t-0">
      <span className="min-w-0">
        <span className="block text-sm font-medium">{label}</span>
        <span className="block text-xs text-[var(--muted-foreground)]">{hint}</span>
      </span>
      <Switch checked={checked} onCheckedChange={onChange} />
    </label>
  );

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle>System settings</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <Row
          label="Journaling"
          hint="After each save, every NPC writes a private journal entry."
          checked={draft.journalingEnabled}
          onChange={(v) => setDraft((d) => ({ ...d, journalingEnabled: v }))}
        />
        <Row
          label="Skill creation"
          hint="The skill-creator reads journals and creates, edits, or deletes skills."
          checked={draft.skillCreationEnabled}
          onChange={(v) => setDraft((d) => ({ ...d, skillCreationEnabled: v }))}
        />
        <Row
          label="Skill execution"
          hint="A matching in-game event slips the NPC's intention into their head and nudges them to act on it."
          checked={draft.skillExecutionEnabled}
          onChange={(v) => setDraft((d) => ({ ...d, skillExecutionEnabled: v }))}
        />
        <div className="flex items-center justify-between gap-4 border-t border-[var(--line-soft)] px-5 py-3">
          <span className="min-w-0">
            <span className="block text-sm font-medium">Per-skill cooldown</span>
            <span className="block text-xs text-[var(--muted-foreground)]">
              Minimum seconds between two firings of the same skill (0 = default).
            </span>
          </span>
          <Field
            type="number"
            min={0}
            className="w-24"
            value={String(draft.skillCooldownSecs)}
            onChange={(e) =>
              setDraft((d) => ({
                ...d,
                skillCooldownSecs: Math.max(0, Number(e.target.value) || 0),
              }))
            }
          />
        </div>
        <div className="flex items-center justify-end gap-2 border-t border-[var(--line-soft)] px-5 py-3">
          {save.isError && (
            <span className="mr-auto text-xs text-[var(--color-danger)]">
              Save failed: {(save.error as Error).message}
            </span>
          )}
          <Button
            size="sm"
            variant="secondary"
            disabled={!dirty || save.isPending}
            onClick={() => save.mutate(draft)}
          >
            {save.isPending ? <Loader2 className="size-3.5 animate-spin" /> : null}
            Save settings
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function SkillRow({ skill }: { skill: SkillDto }) {
  const qc = useQueryClient();
  const toggle = useMutation({
    mutationFn: () => skillsApi.toggle(skill.id),
    onSuccess: (v) => qc.setQueryData(["skills"], v),
  });
  const remove = useMutation({
    mutationFn: () => skillsApi.remove(skill.id),
    onSuccess: (v) => qc.setQueryData(["skills"], v),
  });

  const when = formatWhen(skill.updatedAt ?? skill.createdAt);
  return (
    <div className="border-t border-[var(--line-soft)] px-5 py-4 first:border-t-0">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className="inline-flex items-center gap-1.5 rounded-full border border-[var(--color-danger)]/30 bg-[var(--color-danger)]/5 px-2 py-0.5 text-[11px] font-medium text-[var(--color-danger)]"
          title={`trigger: ${skill.triggerEvent}`}
        >
          <Zap className="size-3" />
          when {triggerLabel(skill.triggerEvent)}
          {skill.triggerFilter ? ` (${skill.triggerFilter})` : ""}
        </span>
        <span className="ml-auto flex items-center gap-3">
          <label className="flex items-center gap-1.5 text-xs text-[var(--muted-foreground)]">
            {skill.enabled ? "On" : "Off"}
            <Switch
              checked={skill.enabled}
              disabled={toggle.isPending}
              onCheckedChange={() => toggle.mutate()}
            />
          </label>
          <Button
            size="sm"
            variant="ghost"
            disabled={remove.isPending}
            onClick={() => {
              if (window.confirm("Delete this skill? It will no longer fire.")) {
                remove.mutate();
              }
            }}
          >
            <Trash2 className="size-3.5" />
          </Button>
        </span>
      </div>
      {skill.thought && (
        <p className="mt-2 text-xs italic text-[var(--color-npc)]">
          <span className="not-italic opacity-60">thinks: </span>“{skill.thought}”
        </p>
      )}
      {skill.note && (
        <p className="mt-1 text-xs italic text-[var(--muted-foreground)]">
          {skill.note}
        </p>
      )}
      {when && (
        <p className="mt-1 text-[11px] text-[var(--muted-foreground)]">
          updated {when}
        </p>
      )}
    </div>
  );
}

function OwnerCard({ owner }: { owner: SkillOwnerDto }) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle>{owner.ownerName}</CardTitle>
      </CardHeader>
      <CardContent className="p-0 pb-1">
        {owner.skills.map((skill) => (
          <SkillRow key={skill.id} skill={skill} />
        ))}
      </CardContent>
    </Card>
  );
}

export function Skills() {
  const query = useQuery({
    queryKey: ["skills"],
    queryFn: skillsApi.list,
    refetchInterval: (q) => (q.state.data?.passInFlight ? 3000 : false),
  });

  const view = query.data;
  const lastPass = formatWhen(view?.lastPassAt);
  const totalSkills =
    view?.owners.reduce((sum, o) => sum + o.skills.length, 0) ?? 0;

  return (
    <PageBody width="wide">
      <PageHeader
        eyebrow="Self-improvement"
        title="Habits"
        description="Habits NPCs form on their own. After each save the skill-creator reads the journals and, when an NPC has clearly settled on how they respond to something that keeps happening, gives them a habit: a trigger event + their own intention. When it fires, that intention is slipped into their head and they act on it however fits the moment. Disable or delete any you don't want."
        actions={
          view?.passInFlight ? (
            <StatusPill tone="busy" pulse>
              Reviewing journals…
            </StatusPill>
          ) : lastPass ? (
            <StatusPill tone="idle">Last pass {lastPass}</StatusPill>
          ) : undefined
        }
      />
      <div className="mt-[var(--gap,14px)] flex-1 overflow-y-auto pb-6">
        {query.isLoading ? (
          <div className="grid h-40 place-items-center text-[var(--muted-foreground)]">
            <Loader2 className="size-5 animate-spin" />
          </div>
        ) : query.isError ? (
          <EmptyState
            icon={<Sparkles className="size-5" strokeWidth={1.75} />}
            title="Couldn't load skills."
            description="Is the chasm server running? Try reloading."
          />
        ) : (
          <Stack>
            {view && <SettingsCard view={view} />}
            {totalSkills === 0 ? (
              <EmptyState
                icon={<Sparkles className="size-5" strokeWidth={1.75} />}
                title="No skills yet."
                description="Skills are created automatically: when an NPC's journal shows they've settled on reacting to something that keeps happening, the skill-creator turns it into an automatic behaviour here. Give it a few saves of a clear, repeated pattern."
              />
            ) : (
              view!.owners.map((owner) => (
                <OwnerCard key={owner.ownerId} owner={owner} />
              ))
            )}
          </Stack>
        )}
      </div>
    </PageBody>
  );
}
