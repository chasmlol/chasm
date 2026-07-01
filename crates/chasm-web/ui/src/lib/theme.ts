// Live, settings-driven theme.
//
// The backend serves `/theme.css` fresh on every request, generated from the
// Interface settings (--accent, --bg, --panel, --line, density --pad/--gap, and
// `html{font-size}`). We load it as a separate stylesheet so the new app themes
// from the exact same source the old UI uses; the app's design tokens in
// index.css reference these vars (with static fallbacks), so accent / theme
// preset / font scale all apply.

const LINK_ID = "live-theme";

/** Inject the `/theme.css` <link> once, at app startup. */
export function ensureLiveTheme(): void {
  if (document.getElementById(LINK_ID)) return;
  const link = document.createElement("link");
  link.id = LINK_ID;
  link.rel = "stylesheet";
  // Absolute path: the app is served under /app/, but /theme.css lives at the origin root.
  link.href = "/theme.css";
  document.head.appendChild(link);
}

/**
 * Re-fetch `/theme.css` (cache-busted) so a just-saved appearance applies live,
 * without a full page reload.
 */
export function reloadLiveTheme(): void {
  const link = document.getElementById(LINK_ID) as HTMLLinkElement | null;
  if (link) {
    link.href = `/theme.css?t=${Date.now()}`;
  } else {
    ensureLiveTheme();
  }
}
