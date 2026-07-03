// Barrel for the chasm UI API.
//
// The client is split into per-domain modules so fill agents edit only their
// own file:
//   - system.ts   settings round-trip + connection status   (systemApi)
//   - books.ts    Characters / Lore / Quest / Action         (booksApi)   [stub]
//   - models.ts   LLM / TTS / STT / Retrieval                 (modelsApi)  [stub]
//   - chat.ts     live-chat projection                        (chatApi)    [stub]
//
// This index re-exports each domain's types and composes them into one flat
// `api` object so existing call sites (`api.settings(...)`, `api.connectionStatus`)
// keep working. Prefer the namespaced objects (systemApi/booksApi/…) in new
// code; the flat `api` is the back-compat surface.

export * from "./system";
export * from "./books";
export * from "./models";
export * from "./config";
export * from "./chat";
export * from "./gamestate";
export * from "./globals";

import { systemApi } from "./system";
import { booksApi } from "./books";
import { modelsApi } from "./models";
import { configApi } from "./config";
import { chatApi } from "./chat";
import { gamestateApi } from "./gamestate";
import { globalsApi } from "./globals";

export {
  systemApi,
  booksApi,
  modelsApi,
  configApi,
  chatApi,
  gamestateApi,
  globalsApi,
};

/**
 * Flat convenience surface. The system methods are hoisted to the top level for
 * back-compat with the original single-file client; the per-domain objects are
 * also attached so `api.books.list(...)` etc. work.
 */
export const api = {
  ...systemApi,
  books: booksApi,
  models: modelsApi,
  config: configApi,
  chat: chatApi,
  gamestate: gamestateApi,
  globals: globalsApi,
};
