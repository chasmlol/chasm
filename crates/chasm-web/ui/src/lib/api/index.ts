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
export * from "./companions";
export * from "./models";
export * from "./config";
export * from "./providers";
export * from "./tts";
export * from "./chat";
export * from "./gamestate";
export * from "./globals";
export * from "./persona";
export * from "./relationships";

import { systemApi } from "./system";
import { booksApi } from "./books";
import { companionsApi } from "./companions";
import { modelsApi } from "./models";
import { configApi } from "./config";
import { providersApi } from "./providers";
import { ttsApi } from "./tts";
import { chatApi } from "./chat";
import { gamestateApi } from "./gamestate";
import { globalsApi } from "./globals";
import { personaApi } from "./persona";
import { relationshipsApi } from "./relationships";

export {
  systemApi,
  booksApi,
  companionsApi,
  modelsApi,
  configApi,
  providersApi,
  ttsApi,
  chatApi,
  gamestateApi,
  globalsApi,
  personaApi,
  relationshipsApi,
};

/**
 * Flat convenience surface. The system methods are hoisted to the top level for
 * back-compat with the original single-file client; the per-domain objects are
 * also attached so `api.books.list(...)` etc. work.
 */
export const api = {
  ...systemApi,
  books: booksApi,
  companions: companionsApi,
  models: modelsApi,
  config: configApi,
  providers: providersApi,
  tts: ttsApi,
  chat: chatApi,
  gamestate: gamestateApi,
  globals: globalsApi,
  persona: personaApi,
  relationships: relationshipsApi,
};
