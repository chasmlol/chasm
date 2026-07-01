import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";

import {
  systemApi,
  type BridgeConfig,
  type BridgeConnection,
  type BridgeView,
} from "@/lib/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, FormRow, StatusPill, type StatusTone } from "@/components/ui/page";
import { SettingsPage } from "@/components/ui/settings-page";

// Bridge — the bridge/connection CONFIGURATION (the plain launcher fields that
// describe how the FNV helper is wired) plus the live game connection status.
// Reuses the same AppSettings read/save path the Interface screen uses via
// GET /api/ui/v1/settings/bridge + POST .../bridge/save, and surfaces the
// read-only /connection/status projection in the header. It edits config ONLY —
// it never drives the transport or the AI-stack lifecycle.

const PHASE_LABEL: Record<string, string> = {
  disconnected: "Offline",
  starting: "Starting",
  connected: "Connected",
  stopping: "Stopping",
};

function connectionTone(conn: BridgeConnection): StatusTone {
  if (conn.connected) return "ok";
  if (conn.phase === "starting" || conn.phase === "stopping") return "warn";
  return "idle";
}

export function Bridge() {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["bridge"],
    queryFn: systemApi.bridge,
    // Keep the connection status reasonably live while the page is open.
    refetchInterval: 4000,
  });

  if (query.isLoading) {
    return (
      <div className="grid h-full place-items-center text-[var(--muted-foreground)]">
        <Loader2 className="size-6 animate-spin" />
      </div>
    );
  }
  if (query.isError || !query.data) {
    return (
      <div className="grid h-full place-items-center p-8 text-center">
        <div>
          <p className="text-sm font-medium text-[var(--color-danger)]">
            Couldn’t load the bridge config.
          </p>
          <p className="mt-1 text-[13px] text-[var(--muted-foreground)]">
            Make sure the chasm backend is running.
          </p>
        </div>
      </div>
    );
  }

  return (
    <BridgeForm
      data={query.data}
      onSaved={(fresh) => queryClient.setQueryData(["bridge"], fresh)}
    />
  );
}

function BridgeForm({
  data,
  onSaved,
}: {
  data: BridgeView;
  onSaved: (fresh: BridgeView) => void;
}) {
  const initial = useMemo(() => data.config, [data.config]);
  const [form, setForm] = useState<BridgeConfig>(initial);
  const [justSaved, setJustSaved] = useState(false);

  useEffect(() => setForm(initial), [initial]);

  const dirty = useMemo(
    () => JSON.stringify(form) !== JSON.stringify(initial),
    [form, initial],
  );

  const save = useMutation({
    mutationFn: () => systemApi.saveBridge(form),
    onSuccess: (fresh) => {
      onSaved(fresh);
      setJustSaved(true);
      window.setTimeout(() => setJustSaved(false), 2200);
    },
  });

  const set = <K extends keyof BridgeConfig>(key: K, value: BridgeConfig[K]) =>
    setForm((f) => ({ ...f, [key]: value }));

  const conn = data.connection;
  const lastSeen =
    conn.last_seen_secs == null
      ? "never seen"
      : `last heartbeat ${conn.last_seen_secs.toFixed(1)}s ago`;

  return (
    <SettingsPage
      eyebrow="System"
      title="Bridge"
      description="How chasm connects to the game: the helper config + transport paths. These are configuration only — chasm starts the AI stack itself when the game connects."
      headerActions={
        <StatusPill tone={connectionTone(conn)} pulse={conn.connected}>
          {PHASE_LABEL[conn.phase] ?? "Offline"}
        </StatusPill>
      }
      save={{
        dirty,
        saving: save.isPending,
        error: save.isError,
        justSaved,
        onReset: () => setForm(initial),
        onSave: () => save.mutate(),
        saveLabel: "Save bridge config",
      }}
    >
      {/* Live connection */}
      <Card>
        <CardHeader>
          <CardTitle>Connection</CardTitle>
          <CardDescription>
            Whether the in-game plugin is currently talking to chasm. Read-only —
            driven by the game's heartbeat.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-3">
            <StatusPill tone={connectionTone(conn)} pulse={conn.connected}>
              {PHASE_LABEL[conn.phase] ?? "Offline"}
            </StatusPill>
            <span className="text-[13px] text-[var(--muted-foreground)]">
              {lastSeen}
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Helper paths */}
      <Card>
        <CardHeader>
          <CardTitle>Helper</CardTitle>
          <CardDescription>
            Paths to the FNV bridge helper. Leave any field blank to use the
            built-in default. The bridge now runs in-process; the Node helper
            fields are only used if you point chasm at an external helper.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <FormRow
            stacked
            label="Helper config path"
            htmlFor="helper_config"
            help="The nvbridge.config.json. Also where the traces directory is discovered from."
            control={
              <Field
                id="helper_config"
                value={form.helper_config}
                placeholder="Built-in default"
                onChange={(e) => set("helper_config", e.target.value)}
              />
            }
          />
          <FormRow
            stacked
            label="Helper script path"
            htmlFor="helper_script"
            help="nvbridge-helper.mjs. If the resolved path doesn't exist, the external helper is skipped."
            control={
              <Field
                id="helper_script"
                value={form.helper_script}
                placeholder="Built-in default"
                onChange={(e) => set("helper_script", e.target.value)}
              />
            }
          />
          <FormRow
            stacked
            label="node.exe path"
            htmlFor="helper_node"
            help="Node runtime for the external helper. Blank = default / PATH."
            control={
              <Field
                id="helper_node"
                value={form.helper_node}
                placeholder="Built-in default / PATH"
                onChange={(e) => set("helper_node", e.target.value)}
              />
            }
          />
          <FormRow
            stacked
            label="Helper working directory"
            htmlFor="helper_cwd"
            help="Blank = the helper script's folder."
            control={
              <Field
                id="helper_cwd"
                value={form.helper_cwd}
                placeholder="Helper script's folder"
                onChange={(e) => set("helper_cwd", e.target.value)}
              />
            }
          />
        </CardContent>
      </Card>

      {/* Tracing root */}
      <Card>
        <CardHeader>
          <CardTitle>Traces directory</CardTitle>
          <CardDescription>
            Override where per-request trace files are read from. Blank =
            auto-discover from the helper config (the Tracing screen reads these).
          </CardDescription>
        </CardHeader>
        <CardContent>
          <FormRow
            stacked
            label="Trace directory override"
            htmlFor="trace_dir"
            control={
              <Field
                id="trace_dir"
                value={form.trace_dir}
                placeholder="Auto-discover"
                onChange={(e) => set("trace_dir", e.target.value)}
              />
            }
          />
        </CardContent>
      </Card>
    </SettingsPage>
  );
}
