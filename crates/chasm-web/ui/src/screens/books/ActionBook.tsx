import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Swords } from "lucide-react";

import { booksApi, type BookEntryDto } from "@/lib/api";
import { Book, type BookEntry, type BookField } from "@/components/Book";
import { bookBadge } from "./badge";

// Action Book — the actions NPCs can take (follow, attack, spawn, …). Same
// shared <Book>; differs only in fields/data. Fields map to the action entry's
// on-disk shape (comment/actionId/riskTier/key/content/scopes/disable) via the
// backend; Scope toggles the entry's `global` scope membership (admin gating).
const ACTION_FIELDS: BookField[] = [
  { key: "title", label: "Title", kind: "text", placeholder: "Action title" },
  {
    key: "actionId",
    label: "Action id",
    kind: "text",
    placeholder: "e.g. movement.follow_target",
    help: "The canonical id the engine dispatches.",
  },
  {
    key: "keys",
    label: "Trigger keywords",
    kind: "text",
    placeholder: "follow, follow me, come with",
    help: "Comma-separated phrases that surface this action from dialogue.",
  },
  {
    key: "description",
    label: "Description",
    kind: "textarea",
    rows: 4,
    help: "What the action does in-game.",
  },
  {
    key: "riskTier",
    label: "Risk tier",
    kind: "select",
    options: [
      { value: "low", label: "Low" },
      { value: "medium", label: "Medium" },
      { value: "high", label: "High" },
    ],
  },
  {
    key: "scope",
    label: "Scope",
    kind: "select",
    options: [
      { value: "any", label: "Any NPC" },
      { value: "admin", label: "Admin only" },
    ],
    help: "Whether every NPC can take this action or only the admin.",
  },
  { key: "enabled", label: "Enabled", kind: "toggle" },
];

function toEntry(dto: BookEntryDto): BookEntry {
  return {
    id: dto.id,
    title: dto.title,
    subtitle: dto.subtitle,
    badge: bookBadge(dto.badge),
    values: dto.values,
  };
}

export function ActionBook() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["books", "action"],
    queryFn: () => booksApi.list("action"),
  });

  const save = useMutation({
    mutationFn: ({ id, values }: { id: string; values: BookEntry["values"] }) =>
      booksApi.save("action", id, values),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["books", "action"] }),
  });

  return (
    <Book
      eyebrow="Library"
      title="Action Book"
      description="The actions NPCs can take in-game. Each row expands to edit; save per action."
      icon={<Swords className="size-5" strokeWidth={1.75} />}
      noun="actions"
      entries={(query.data?.entries ?? []).map(toEntry)}
      fields={ACTION_FIELDS}
      isLoading={query.isLoading}
      isError={query.isError}
      onSave={(id, values) => save.mutateAsync({ id, values })}
    />
  );
}
