// Shared fetch helpers for the chasm UI JSON API.
//
// Every per-domain api module (system, books, models, chat) imports these so
// error handling + headers are uniform. Endpoints all live under the
// `/api/ui/v1/*` namespace added by the UI work (crates/chasm-web/src/ui),
// except the read-only `/connection/status` the UI shares with the backend.
// The UI NEVER calls `/api/headless/*` or `/api/game/*`.

export async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${url} → ${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export async function postJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${url} → ${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export async function putJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${url} → ${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

/** Base path for the UI JSON API. */
export const UI_API = "/api/ui/v1";
