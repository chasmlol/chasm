import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { travelApi, type MovementSettingsDto } from "@/lib/api";
import { SettingsPage } from "@/components/ui/settings-page";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Field, FormRow } from "@/components/ui/page";

// ===========================================================================
// Travel — SETTINGS for the NPC movement system. A reusable engine that walks an
// NPC from where it stands to a destination so it ARRIVES at a scheduled in-game
// time: it measures the distance, leaves early, and (when you're away) advances
// the NPC along the route so intercepting them finds them on the road. When the
// NPC is loaded it walks with a real animated travel package; off-screen it's
// simulated invisibly.
//
// The journeys themselves are listed on the Schedule page (alongside scheduled
// actions) — this page is just the knobs. See crates/chasm-web/src/movement.rs.
// ===========================================================================

export function Travel() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["travel", "settings"],
    queryFn: () => travelApi.view(),
  });

  const panel = query.data?.settings;
  const initial: MovementSettingsDto | null = useMemo(
    () =>
      panel
        ? {
            enabled: panel.enabled,
            walkSpeed: panel.walkSpeed,
            offscreenSimulation: panel.offscreenSimulation,
            waypointStride: panel.waypointStride,
          }
        : null,
    [panel],
  );

  const [form, setForm] = useState<MovementSettingsDto | null>(initial);
  const [justSaved, setJustSaved] = useState(false);
  useEffect(() => setForm((f) => f ?? initial), [initial]);

  const dirty = useMemo(
    () => !!form && !!initial && JSON.stringify(form) !== JSON.stringify(initial),
    [form, initial],
  );

  const save = useMutation({
    mutationFn: (body: MovementSettingsDto) => travelApi.saveSettings(body),
    onSuccess: (fresh) => {
      qc.setQueryData(["travel", "settings"], (prev: unknown) =>
        prev && typeof prev === "object"
          ? { ...(prev as object), settings: fresh }
          : prev,
      );
      setForm(fresh);
      setJustSaved(true);
      window.setTimeout(() => setJustSaved(false), 2200);
    },
  });

  const set = <K extends keyof MovementSettingsDto>(
    key: K,
    value: MovementSettingsDto[K],
  ) => setForm((f) => (f ? { ...f, [key]: value } : f));

  return (
    <SettingsPage
      eyebrow="System"
      title="Travel"
      description="Settings for the NPC movement system — walk NPCs to places so they arrive on time. Active journeys appear on the Schedule page."
      save={
        form
          ? {
              dirty,
              saving: save.isPending,
              error: save.isError,
              justSaved,
              onReset: () => initial && setForm(initial),
              onSave: () => form && save.mutate(form),
              saveLabel: "Save travel settings",
            }
          : undefined
      }
    >
      {form && (
        <Card>
          <CardHeader>
            <CardTitle>Movement</CardTitle>
            <CardDescription>
              How NPCs travel. Speed is in-game time, so "arrive at 3:00 PM" is
              honoured regardless of frame rate.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <FormRow
              label="Enable travel system"
              help="When off, a scheduled travel just teleports the NPC to the destination at its time (no walking, no early departure)."
              control={
                <Switch
                  checked={form.enabled}
                  onCheckedChange={(v) => set("enabled", v)}
                />
              }
            />
            <FormRow
              label="Walk speed"
              help="Metres of world distance the NPC covers per in-game hour. Higher = faster travel, leaves later. ~1500 ≈ a brisk 1.5 km/hour."
              control={
                <Field
                  type="number"
                  value={form.walkSpeed}
                  min={1}
                  step={50}
                  onChange={(e) => set("walkSpeed", Number(e.target.value))}
                  disabled={!form.enabled}
                />
              }
            />
            <FormRow
              label="Simulate travel off-screen"
              help="Advance the NPC along the route while you're elsewhere, so intercepting them finds them on the road. Off = they simply appear at the destination on arrival."
              control={
                <Switch
                  checked={form.offscreenSimulation}
                  onCheckedChange={(v) => set("offscreenSimulation", v)}
                  disabled={!form.enabled}
                />
              }
            />
            <FormRow
              label="Waypoint stride"
              help="Minimum metres the NPC must advance before the next off-screen position update — throttles the movement stream. Lower = smoother but chattier."
              control={
                <Field
                  type="number"
                  value={form.waypointStride}
                  min={1}
                  step={5}
                  onChange={(e) => set("waypointStride", Number(e.target.value))}
                  disabled={!form.enabled || !form.offscreenSimulation}
                />
              }
            />
          </CardContent>
        </Card>
      )}
    </SettingsPage>
  );
}
