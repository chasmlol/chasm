import { useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AudioLines, UserPlus, UsersRound } from "lucide-react";

import {
  companionsApi,
  type CompanionOp,
  type CompanionSlotDto,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  EmptyState,
  Field,
  FormRow,
  PageBody,
  PageHeader,
  Section,
  Select,
  Stack,
  StatusPill,
  Table,
  Td,
  TextArea,
  Th,
} from "@/components/ui/page";

// ===========================================================================
// Companions — author a brand-new character and get them as a spawned, named,
// voiced follower in-game. The card lands in the Characters Book (edit the
// persona there like any character); this page owns creation + the in-game
// lifecycle (spawn / follow / dismiss / face design), talking to the NVSE
// plugin through the bridge command queue. See mod-source
// docs/companions-architecture.md for the full design.
// ===========================================================================

const STATUS_LABEL: Record<string, string> = {
  unclaimed: "Free",
  claimed: "Pending spawn",
  spawned: "In world",
  dismissed: "In holding cell",
};

function slotStatusPill(slot: CompanionSlotDto) {
  if (!slot.claimed) return <StatusPill tone="idle">Free</StatusPill>;
  if (slot.status === "spawned")
    return <StatusPill tone="ok">{STATUS_LABEL[slot.status]}</StatusPill>;
  return (
    <StatusPill tone="warn">
      {STATUS_LABEL[slot.status] ?? slot.status}
    </StatusPill>
  );
}

function voicePill(status: string) {
  switch (status) {
    case "cloned":
      return <StatusPill tone="ok">Voice cloned</StatusPill>;
    case "cloning":
      return (
        <StatusPill tone="warn" pulse>
          Cloning…
        </StatusPill>
      );
    case "failed":
      return <StatusPill tone="warn">Clone failed</StatusPill>;
    case "reference":
      return <StatusPill tone="idle">Clip saved</StatusPill>;
    default:
      return <StatusPill tone="idle">No voice</StatusPill>;
  }
}

/** Reads a File into raw base64 (no data: prefix). */
async function fileToBase64(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

const EMPTY_FORM = {
  name: "",
  description: "",
  personality: "",
  firstMessage: "",
  exampleDialogue: "",
  systemPrompt: "",
  body: "",
  faceDesign: true,
};

export function Companions() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["companions"],
    queryFn: companionsApi.view,
    refetchInterval: 4000,
  });

  const [form, setForm] = useState(EMPTY_FORM);
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const create = useMutation({
    mutationFn: async () => {
      const voiceBase64 = voiceFile ? await fileToBase64(voiceFile) : "";
      const body = form.body || view?.bodies[0]?.id || "";
      const faceDesign = form.faceDesign && (view?.inGameFaceDesign ?? false);
      return companionsApi.create({ ...form, body, faceDesign, voiceBase64 });
    },
    onSuccess: (res) => {
      setForm(EMPTY_FORM);
      setVoiceFile(null);
      setNotice(
        form.faceDesign && view?.inGameFaceDesign
          ? `${res.cardId} created. Tab into the game — the character creator will open to design their face, then they'll spawn beside you.`
          : `${res.cardId} created. They'll spawn beside you next time you're in game.`,
      );
      qc.invalidateQueries({ queryKey: ["companions"] });
      qc.invalidateQueries({ queryKey: ["books", "characters"] });
    },
    onError: (error) => setNotice(String(error)),
  });

  const op = useMutation({
    mutationFn: ({
      slot,
      action,
      name,
    }: {
      slot: number;
      action: CompanionOp;
      name?: string;
    }) => companionsApi.op(slot, action, name),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["companions"] }),
    onError: (error) => setNotice(String(error)),
  });

  const view = query.data;
  const claimed = useMemo(
    () => (view?.slots ?? []).filter((slot) => slot.claimed),
    [view],
  );
  const failedAcks = (view?.acks ?? []).filter((ack) => !ack.ok).slice(0, 3);

  const setField = (key: keyof typeof EMPTY_FORM) => (value: string | boolean) =>
    setForm((old) => ({ ...old, [key]: value }));

  const onVoiceDrop = (files: FileList | null) => {
    setDragOver(false);
    const file = files?.[0];
    if (file) setVoiceFile(file);
  };

  return (
    <PageBody>
      <PageHeader
        eyebrow="Main"
        title="Companions"
        description={
          <>
            Create a brand-new character and they join you in game as a voiced
            follower. The card lands in the Characters Book — chat, retrieval
            and prompts treat them like any NPC. What the game supports (body
            pools, in-game face design) comes from the active game profile.
          </>
        }
      />

      {view && !view.enabled ? (
        <EmptyState
          icon={<UsersRound className="size-6" strokeWidth={1.5} />}
          title="Not supported by this game profile"
          description="The active game profile declares no companion capabilities. Game-side support (a companions block in profile.json) is shipped by the game's mod."
        />
      ) : (
      <Stack>
        <Section
          title="Create companion"
          description={
            <>
              Fill the card like any character and drop in a voice clip for
              cloning.
            </>
          }
        >
          <Stack>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <FormRow
                stacked
                label="Name"
                control={
                  <Field
                    value={form.name}
                    placeholder="Companion name"
                    onChange={(e) => setField("name")(e.target.value)}
                  />
                }
              />
              <FormRow
                stacked
                label="Body"
                help="Picks the in-game template pool (declared by the game profile)."
                control={
                  <Select
                    value={form.body || view?.bodies[0]?.id || ""}
                    onChange={(e) => setField("body")(e.target.value)}
                  >
                    {(view?.bodies ?? []).map((b) => (
                      <option key={b.id} value={b.id}>
                        {b.label} — {b.free} free
                      </option>
                    ))}
                  </Select>
                }
              />
            </div>
            <FormRow
              stacked
              label="Description"
              help="Who they are — background, role, what they know."
              control={
                <TextArea
                  rows={4}
                  value={form.description}
                  onChange={(e) => setField("description")(e.target.value)}
                />
              }
            />
            <FormRow
              stacked
              label="Personality"
              control={
                <TextArea
                  rows={2}
                  value={form.personality}
                  onChange={(e) => setField("personality")(e.target.value)}
                />
              }
            />
            <FormRow
              stacked
              label="First message"
              help="Their opening line when you first talk."
              control={
                <TextArea
                  rows={2}
                  value={form.firstMessage}
                  onChange={(e) => setField("firstMessage")(e.target.value)}
                />
              }
            />
            <FormRow
              stacked
              label="Example dialogue"
              control={
                <TextArea
                  rows={3}
                  value={form.exampleDialogue}
                  onChange={(e) => setField("exampleDialogue")(e.target.value)}
                />
              }
            />
            <FormRow
              stacked
              label="System prompt"
              help="Optional — how they speak and behave. Leave blank for the default."
              control={
                <TextArea
                  rows={3}
                  value={form.systemPrompt}
                  onChange={(e) => setField("systemPrompt")(e.target.value)}
                />
              }
            />
            <FormRow
              stacked
              label="Voice"
              help={
                view?.voiceHint ||
                "Drop an audio clip (~10-20s of clean speech; WAV/FLAC/OGG). It runs through the voice-clone pipeline and becomes their spoken voice."
              }
              control={
                <div
                  role="button"
                  tabIndex={0}
                  onClick={() => fileInput.current?.click()}
                  onKeyDown={(e) => e.key === "Enter" && fileInput.current?.click()}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    onVoiceDrop(e.dataTransfer.files);
                  }}
                  className={`flex cursor-pointer items-center gap-3 rounded-lg border border-dashed px-4 py-3 text-sm transition-colors ${
                    dragOver
                      ? "border-[var(--color-accent)] bg-[var(--color-ink-850)]"
                      : "border-[var(--border)]"
                  }`}
                >
                  <AudioLines className="size-5 shrink-0" strokeWidth={1.75} />
                  <span className="text-[var(--muted-foreground)]">
                    {voiceFile
                      ? `${voiceFile.name} (${Math.round(voiceFile.size / 1024)} KB)`
                      : "Drop a voice clip here, or click to browse"}
                  </span>
                  <input
                    ref={fileInput}
                    type="file"
                    accept="audio/*"
                    className="hidden"
                    onChange={(e) => onVoiceDrop(e.target.files)}
                  />
                </div>
              }
            />
            {view?.inGameFaceDesign && (
              <FormRow
                label="Design face in game"
                help={
                  view.faceDesignHint ||
                  "Opens the game's character creator next time you're in game."
                }
                control={
                  <input
                    type="checkbox"
                    className="size-4 accent-[var(--color-accent)]"
                    checked={form.faceDesign}
                    onChange={(e) => setField("faceDesign")(e.target.checked)}
                  />
                }
              />
            )}
            <div className="flex items-center gap-3">
              <Button
                disabled={!form.name.trim() || create.isPending}
                onClick={() => create.mutate()}
              >
                <UserPlus className="size-4" strokeWidth={1.75} />
                {create.isPending ? "Creating…" : "Create companion"}
              </Button>
              {notice && (
                <p className="text-[13px] text-[var(--muted-foreground)]">{notice}</p>
              )}
            </div>
          </Stack>
        </Section>

        <Section
          title="Your companions"
          description="Live pool state from the game plugin. Commands queue while the game is closed and run when you're back in."
        >
          {claimed.length === 0 ? (
            <EmptyState
              icon={<UsersRound className="size-6" strokeWidth={1.5} />}
              title="No companions yet"
              description="Create one above — it becomes a character card plus an in-game follower."
            />
          ) : (
            <Table
              head={
                <tr>
                  <Th>Name</Th>
                  <Th>Status</Th>
                  <Th>Face</Th>
                  <Th>Voice</Th>
                  <Th>Actions</Th>
                </tr>
              }
            >
              {claimed.map((slot) => (
                <tr key={slot.slot}>
                  <Td>
                    <span className="font-medium">{slot.name || slot.npcKey}</span>
                    <span className="ml-2 text-xs text-[var(--muted-foreground)]">
                      {slot.body || "?"} · slot {slot.slot + 1}
                      {slot.hasCard ? "" : " · card missing"}
                    </span>
                  </Td>
                  <Td>{slotStatusPill(slot)}</Td>
                  <Td>
                    {slot.faceDesigned ? (
                      <StatusPill tone="ok">Designed</StatusPill>
                    ) : (
                      <StatusPill tone="idle">Template</StatusPill>
                    )}
                  </Td>
                  <Td>{voicePill(slot.voiceStatus)}</Td>
                  <Td>
                    <div className="flex flex-wrap gap-1.5">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => op.mutate({ slot: slot.slot, action: "summon" })}
                      >
                        Summon
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => op.mutate({ slot: slot.slot, action: "dismiss" })}
                      >
                        Stop following
                      </Button>
                      {view?.inGameFaceDesign && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => op.mutate({ slot: slot.slot, action: "face_design" })}
                        >
                          Design face
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => {
                          const name = window.prompt(
                            "New name (the floating in-game name updates on the next load or crosshair refresh):",
                            slot.name,
                          );
                          if (name && name.trim()) {
                            op.mutate({ slot: slot.slot, action: "rename", name: name.trim() });
                          }
                        }}
                      >
                        Rename
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => op.mutate({ slot: slot.slot, action: "despawn" })}
                      >
                        Send home
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => {
                          if (
                            window.confirm(
                              `Release ${slot.name || "this companion"}? The slot is freed; the character card stays in your book.`,
                            )
                          ) {
                            op.mutate({ slot: slot.slot, action: "release" });
                          }
                        }}
                      >
                        Release
                      </Button>
                    </div>
                  </Td>
                </tr>
              ))}
            </Table>
          )}
          {failedAcks.length > 0 && (
            <div className="mt-3 flex flex-col gap-1">
              {failedAcks.map((ack) => (
                <p
                  key={ack.requestId}
                  className="text-[13px] text-[var(--muted-foreground)]"
                >
                  ⚠ {ack.op} failed: {ack.error || "unknown error"}
                </p>
              ))}
            </div>
          )}
        </Section>
      </Stack>
      )}
    </PageBody>
  );
}
