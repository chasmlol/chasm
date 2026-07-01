import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ScrollText } from "lucide-react";

import { booksApi, type BookEntryDto } from "@/lib/api";
import { Book, type BookEntry, type BookField } from "@/components/Book";
import { bookBadge } from "./badge";

// Quest Book — quest entries + their lifecycle phase. Same shared <Book>;
// differs only in fields/data. Fields map to the quest entry's on-disk shape
// (comment/questName/questId/phase/offerSummary/content) via the backend.
const QUEST_FIELDS: BookField[] = [
  { key: "title", label: "Title", kind: "text", placeholder: "Entry title" },
  {
    key: "questName",
    label: "Quest name",
    kind: "text",
    placeholder: "In-game quest name",
  },
  {
    key: "questId",
    label: "Quest id",
    kind: "text",
    placeholder: "e.g. VCG02",
    help: "The game's quest identifier.",
  },
  {
    key: "status",
    label: "Phase",
    kind: "select",
    options: [
      { value: "available", label: "Available" },
      { value: "active", label: "Active" },
      { value: "complete", label: "Complete" },
      { value: "failed", label: "Failed" },
    ],
  },
  {
    key: "keys",
    label: "Trigger keywords",
    kind: "text",
    placeholder: "comma, separated, keys",
    help: "Comma-separated terms that surface this quest in dialogue.",
  },
  {
    key: "offerSummary",
    label: "Offer summary",
    kind: "textarea",
    rows: 3,
    help: "How an NPC pitches the quest to the player.",
  },
  {
    key: "description",
    label: "Description",
    kind: "textarea",
    rows: 5,
    help: "What the quest is about and how it advances.",
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

export function QuestBook() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["books", "quest"],
    queryFn: () => booksApi.list("quest"),
  });

  const save = useMutation({
    mutationFn: ({ id, values }: { id: string; values: BookEntry["values"] }) =>
      booksApi.save("quest", id, values),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["books", "quest"] }),
  });

  return (
    <Book
      eyebrow="Library"
      title="Quest Book"
      description="Quest lines and their phase. Each row expands to edit; save per quest."
      icon={<ScrollText className="size-5" strokeWidth={1.75} />}
      noun="quests"
      entries={(query.data?.entries ?? []).map(toEntry)}
      fields={QUEST_FIELDS}
      isLoading={query.isLoading}
      isError={query.isError}
      onSave={(id, values) => save.mutateAsync({ id, values })}
    />
  );
}
