import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BookText } from "lucide-react";

import { booksApi, type BookEntryDto } from "@/lib/api";
import { Book, type BookEntry, type BookField } from "@/components/Book";
import { bookBadge } from "./badge";

// Lore Book — world facts injected by retrieval. Same shared <Book>; differs
// only in fields/data. Fields map to the lorebook entry's on-disk shape
// (comment/key/content/disable) via crates/chasm-web/src/ui/books.rs.
const LORE_FIELDS: BookField[] = [
  { key: "title", label: "Title", kind: "text", placeholder: "Entry title" },
  {
    key: "keys",
    label: "Trigger keywords",
    kind: "text",
    placeholder: "comma, separated, keys",
    help: "Comma-separated terms that activate this entry in retrieval.",
  },
  {
    key: "content",
    label: "Content",
    kind: "textarea",
    rows: 6,
    help: "The lore text injected into the prompt when this entry matches.",
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

export function LoreBook() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["books", "lore"],
    queryFn: () => booksApi.list("lore"),
  });

  const save = useMutation({
    mutationFn: ({ id, values }: { id: string; values: BookEntry["values"] }) =>
      booksApi.save("lore", id, values),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["books", "lore"] }),
  });

  return (
    <Book
      eyebrow="Library"
      title="Lore Book"
      description="World facts injected into prompts by retrieval. Each row expands to edit; save per entry."
      icon={<BookText className="size-5" strokeWidth={1.75} />}
      noun="entries"
      entries={(query.data?.entries ?? []).map(toEntry)}
      fields={LORE_FIELDS}
      isLoading={query.isLoading}
      isError={query.isError}
      onSave={(id, values) => save.mutateAsync({ id, values })}
    />
  );
}
