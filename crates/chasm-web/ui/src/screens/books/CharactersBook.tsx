import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Users } from "lucide-react";

import { booksApi, type BookEntryDto } from "@/lib/api";
import { Book, type BookEntry, type BookField } from "@/components/Book";
import { bookBadge } from "./badge";

// Characters Book — every character card (the .png personas) + the editable
// prompt fields. Uses the SHARED <Book>; only the field schema + data mapping
// live here. The system prompt is the key editable field; saving re-embeds the
// persona into the PNG card (see crates/chasm-web/src/ui/books.rs).
const CHARACTER_FIELDS: BookField[] = [
  { key: "name", label: "Name", kind: "text", placeholder: "Character name" },
  {
    key: "systemPrompt",
    label: "System prompt",
    kind: "textarea",
    rows: 6,
    help: "The instruction that defines how this character speaks and behaves.",
  },
  {
    key: "description",
    label: "Description",
    kind: "textarea",
    rows: 5,
    help: "Who the character is — background, role, and what they know.",
  },
  {
    key: "personality",
    label: "Personality",
    kind: "textarea",
    rows: 3,
  },
  {
    key: "scenario",
    label: "Scenario",
    kind: "textarea",
    rows: 3,
    help: "The situation the character is in when the player meets them.",
  },
  {
    key: "firstMessage",
    label: "First message",
    kind: "textarea",
    rows: 3,
    help: "The character's opening line.",
  },
  {
    key: "exampleDialogue",
    label: "Example dialogue",
    kind: "textarea",
    rows: 4,
    help: "Sample exchanges that anchor the character's voice.",
  },
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

export function CharactersBook() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["books", "characters"],
    queryFn: () => booksApi.list("characters"),
  });

  const save = useMutation({
    mutationFn: ({ id, values }: { id: string; values: BookEntry["values"] }) =>
      booksApi.save("characters", id, values),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["books", "characters"] }),
  });

  return (
    <Book
      eyebrow="Library"
      title="Characters Book"
      description="The cast of NPCs and their personas. Each row expands to edit; save per character."
      icon={<Users className="size-5" strokeWidth={1.75} />}
      noun="characters"
      entries={(query.data?.entries ?? []).map(toEntry)}
      fields={CHARACTER_FIELDS}
      isLoading={query.isLoading}
      isError={query.isError}
      onSave={(id, values) => save.mutateAsync({ id, values })}
    />
  );
}
