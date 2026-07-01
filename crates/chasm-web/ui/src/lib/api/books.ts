// Content-books domain of the UI API (Characters / Lore / Quest / Action).
//
// STUB. The four book screens render via the shared <Book> component
// (src/components/Book.tsx); each maps its backend list to BookEntry[] and
// saves per entry. The endpoints below return `{ entries: [] }` until a fill
// agent implements them in crates/chasm-web/src/ui/books.rs.
//
// Fill agents: keep the per-book shape so each screen stays an isolated edit —
// add the real fields to the *Entry interfaces and implement load/save here +
// in the matching backend module.

import { getJson, postJson, UI_API } from "./http";

/** A generic book entry as returned by the UI API (values are book-specific). */
export interface BookEntryDto {
  id: string;
  title: string;
  subtitle?: string;
  /** Optional small badge label (e.g. "Disabled", a quest phase, "Admin"). */
  badge?: string;
  /** Book-specific editable fields, keyed by field key. */
  values: Record<string, string | boolean | number>;
}

export interface BookListDto {
  entries: BookEntryDto[];
}

/** The four book kinds, used as the path segment. */
export type BookKind = "characters" | "lore" | "quest" | "action";

export const booksApi = {
  list: (kind: BookKind) =>
    getJson<BookListDto>(`${UI_API}/books/${kind}`),
  save: (kind: BookKind, id: string, values: BookEntryDto["values"]) =>
    postJson<BookEntryDto>(`${UI_API}/books/${kind}/${id}`, { values }),
};
