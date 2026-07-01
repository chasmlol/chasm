import { useMutation, useQuery } from "@tanstack/react-query";
import { CheckCircle2, Download, Loader2, RefreshCw } from "lucide-react";

import { systemApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { PageBody, PageHeader, Section, StatusPill } from "@/components/ui/page";

// Updates — checks GitHub for a newer chasm release and self-updates.
// GET /api/app/version reports current vs latest; "Update & restart" POSTs
// /api/app/update/install and the backend downloads the installer, runs it
// silently, and relaunches chasm. It's backend-driven on purpose: this UI runs
// in the Tauri webview, which can't open external links or call Tauri APIs, so
// the download+install must happen server-side (the backend runs on this machine).

export function Updates() {
  const check = useQuery({
    queryKey: ["app-version"],
    queryFn: systemApi.appVersion,
    staleTime: 0,
    gcTime: 0,
    refetchOnWindowFocus: false,
  });
  const install = useMutation({ mutationFn: systemApi.installUpdate });

  const data = check.data;
  const started = install.data?.started === true;
  const failed = install.isError || install.data?.started === false;

  return (
    <PageBody width="prose">
      <PageHeader
        eyebrow="System"
        title="Updates"
        description="Check for a newer version of chasm. When one is available, chasm can download it, install it, and restart itself."
        actions={
          data ? (
            data.update_available ? (
              <StatusPill tone="warn">Update available</StatusPill>
            ) : (
              <StatusPill tone="ok">Up to date</StatusPill>
            )
          ) : null
        }
      />

      <Section className="flex-1 pt-[var(--gap,14px)]">
        <div className="flex flex-col gap-4 rounded-xl border border-[var(--border)] p-[var(--card-pad,16px)]">
          <div className="flex items-center justify-between gap-4">
            <div className="min-w-0">
              <p className="text-sm font-medium">Current version</p>
              <p className="mt-0.5 font-mono text-[13px] text-[var(--muted-foreground)]">
                {data ? `v${data.current}` : "…"}
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => check.refetch()}
              disabled={check.isFetching || install.isPending}
            >
              {check.isFetching ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <RefreshCw className="size-4" />
              )}
              Check for updates
            </Button>
          </div>

          {check.isError && (
            <p className="text-[13px] text-[var(--color-danger)]">
              Couldn’t reach GitHub. Check your connection and try again.
            </p>
          )}

          {data && data.update_available && data.latest && (
            <div className="flex flex-col gap-3 rounded-lg border border-[var(--color-npc)]/40 bg-[var(--color-npc)]/5 p-4">
              <div className="flex items-center gap-2 text-sm font-medium text-[var(--color-npc)]">
                <Download className="size-4" />
                Update available: v{data.latest}
              </div>

              {started ? (
                <p className="text-[13px] text-[var(--muted-foreground)]">
                  Downloading and installing v{data.latest}… chasm will close and
                  reopen on its own in a moment. If it doesn’t reopen, launch chasm
                  again from your Start menu.
                </p>
              ) : (
                <>
                  <p className="text-[13px] text-[var(--muted-foreground)]">
                    chasm will download the update, install it, and restart
                    automatically.
                  </p>
                  <div>
                    <Button
                      size="sm"
                      onClick={() => install.mutate()}
                      disabled={install.isPending}
                    >
                      {install.isPending ? (
                        <Loader2 className="size-4 animate-spin" />
                      ) : (
                        <Download className="size-4" />
                      )}
                      {install.isPending ? "Downloading…" : "Update & restart"}
                    </Button>
                  </div>
                  {failed && (
                    <p className="text-[13px] text-[var(--color-danger)]">
                      {install.data?.error ?? "Couldn’t start the update."} You can
                      also update manually from
                      github.com/chasmlol/chasm/releases.
                    </p>
                  )}
                </>
              )}
            </div>
          )}

          {data && !data.update_available && !check.isFetching && (
            <div className="flex items-center gap-2 text-[13px] text-[var(--color-player)]">
              <CheckCircle2 className="size-4" />
              You’re on the latest version.
            </div>
          )}
        </div>
      </Section>
    </PageBody>
  );
}
