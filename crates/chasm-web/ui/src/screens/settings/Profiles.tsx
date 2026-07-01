import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BookText,
  Check,
  Layers,
  Loader2,
  ScrollText,
  Swords,
  Users,
} from "lucide-react";

import { systemApi, type ProfilesView, type UiProfile } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { EmptyState, StatusPill } from "@/components/ui/page";
import { SettingsPage } from "@/components/ui/settings-page";

// Profiles — list every drop-in game profile and activate one. Read + activate
// only; profile IMPORT (drag-and-drop portability) is a planned follow-up and is
// intentionally not built here. Built from the shared primitives (SettingsPage
// chrome, Card, StatusPill, Button, EmptyState) so it reads like every other
// settings screen. Reuses the profile cores via GET /api/ui/v1/profiles +
// POST .../profiles/select.

export function Profiles() {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["profiles"],
    queryFn: systemApi.profiles,
  });

  const activate = useMutation({
    mutationFn: (id: string) => systemApi.selectProfile(id),
    onSuccess: (fresh) => queryClient.setQueryData(["profiles"], fresh),
  });

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
            Couldn’t load profiles.
          </p>
          <p className="mt-1 text-[13px] text-[var(--muted-foreground)]">
            Make sure the chasm backend is running.
          </p>
        </div>
      </div>
    );
  }

  return (
    <ProfilesList
      data={query.data}
      activatingId={activate.isPending ? activate.variables ?? null : null}
      onActivate={(id) => activate.mutate(id)}
    />
  );
}

function ProfilesList({
  data,
  activatingId,
  onActivate,
}: {
  data: ProfilesView;
  activatingId: string | null;
  onActivate: (id: string) => void;
}) {
  return (
    <SettingsPage
      eyebrow="Content"
      title="Profiles"
      description="Each profile is a self-contained content folder — its characters, books and voices. Activate one to make it the live profile the game and chat use."
    >
      {data.profiles.length === 0 ? (
        <EmptyState
          icon={<Layers className="size-5" strokeWidth={1.75} />}
          title="No profiles found"
          description={`Drop a profile folder into ${data.profiles_dir} and it will appear here.`}
        />
      ) : (
        data.profiles.map((profile) => (
          <ProfileCard
            key={profile.id}
            profile={profile}
            activating={activatingId === profile.id}
            onActivate={() => onActivate(profile.id)}
          />
        ))
      )}

      <p className="px-1 text-[13px] text-[var(--muted-foreground)]">
        Profile import (drag-and-drop portability) is coming in a later update.
      </p>
    </SettingsPage>
  );
}

function ProfileCard({
  profile,
  activating,
  onActivate,
}: {
  profile: UiProfile;
  activating: boolean;
  onActivate: () => void;
}) {
  return (
    <Card
      className={cn(
        profile.active &&
          "border-[var(--color-accent)] ring-1 ring-[var(--color-accent)]/30",
      )}
    >
      <CardContent className="flex items-start gap-4 p-5">
        <div
          className={cn(
            "grid size-12 shrink-0 place-items-center rounded-xl border text-sm font-semibold",
            profile.active
              ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-accent)]"
              : "border-[var(--border)] bg-[var(--color-ink-800)] text-[var(--muted-foreground)]",
          )}
        >
          {profile.initials}
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="truncate text-base font-semibold tracking-tight">
              {profile.name}
            </h3>
            {profile.active && <StatusPill tone="ok">Active</StatusPill>}
          </div>
          {profile.description && (
            <p className="mt-1 text-[13px] leading-relaxed text-[var(--muted-foreground)]">
              {profile.description}
            </p>
          )}
          <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5 text-[13px] text-[var(--muted-foreground)]">
            <Count icon={<Users className="size-3.5" />} n={profile.character_count} label="characters" />
            <Count icon={<BookText className="size-3.5" />} n={profile.lorebook_count} label="lorebooks" />
            <Count icon={<ScrollText className="size-3.5" />} n={profile.quest_count} label="quests" />
            <Count icon={<Swords className="size-3.5" />} n={profile.action_count} label="actions" />
          </div>
        </div>

        <div className="shrink-0 self-center">
          {profile.active ? (
            <Button variant="ghost" size="sm" disabled>
              <Check className="size-4" /> Active
            </Button>
          ) : (
            <Button
              variant="secondary"
              size="sm"
              disabled={activating}
              onClick={onActivate}
            >
              {activating && <Loader2 className="size-4 animate-spin" />}
              Activate
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function Count({
  icon,
  n,
  label,
}: {
  icon: React.ReactNode;
  n: number;
  label: string;
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="text-[var(--muted-foreground)]/70">{icon}</span>
      <span className="font-medium text-[var(--foreground)]">{n}</span>
      {label}
    </span>
  );
}
