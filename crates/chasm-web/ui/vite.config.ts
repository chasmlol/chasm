import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// chasm UI build.
//
// Serving model: the production bundle is served BY the existing axum backend on
// :7341 under the `/app/` path (see crates/chasm-web/src/ui.rs). `base`
// must therefore be `/app/` so every emitted asset URL is prefixed to match.
//
// Dev model: `vite dev` on :5173 with HMR. All backend calls (the UI JSON API,
// the connection status, the headless/game contract) are proxied to the running
// axum server on :7341, so the React app talks to the real backend in dev too.
const API_TARGET = process.env.CHASM_API_TARGET ?? "http://127.0.0.1:7341";

export default defineConfig({
  base: "/app/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Keep the bundle debuggable in the desktop WebView2; chasm runs locally so
    // a sourcemap costs nothing meaningful and makes the premium UI inspectable.
    sourcemap: true,
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      // UI JSON API (added by this work) — read/save settings, etc.
      "/api/ui": API_TARGET,
      // Top-level model-stack control + status lights.
      "/api/stack": API_TARGET,
      // Voice-clone status/trigger (samples served via the existing /voices).
      "/api/voices": API_TARGET,
      // Existing backend contracts the UI reads (never written by the UI).
      "/connection": API_TARGET,
      // Live-chat clear-history POST (top-level route, not under /api/ui).
      "/live": API_TARGET,
      "/api/headless": API_TARGET,
      "/api/game": API_TARGET,
      // Dynamic theme stylesheet + voices, served by axum.
      "/theme.css": API_TARGET,
      "/voices": API_TARGET,
      "/health": API_TARGET,
    },
  },
});
