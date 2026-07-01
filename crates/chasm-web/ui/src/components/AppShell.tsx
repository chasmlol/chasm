import { NavLink, Outlet } from "react-router-dom";

import { cn } from "@/lib/utils";
import { NAV_GROUPS } from "@/lib/nav";
import { ConnectionPill } from "@/components/ConnectionPill";
import { StackControls } from "@/components/StackControls";

// ===========================================================================
// AppShell — the single, persistent application shell. A left sidebar that is
// ALWAYS visible and is the ONLY navigation; clicking a nav item swaps just the
// right-hand content pane (the <Outlet/>). There are NO in-page navigation
// buttons anywhere — every destination lives in this sidebar.
//
// Driven by NAV_GROUPS (src/lib/nav.tsx) + react-router's NavLink, so the active
// state and routing are automatic. The ConnectionPill lives in the sidebar
// header. Density-driven spacing flows through via the shared primitives the
// screens use.
// ===========================================================================

export function AppShell() {
  return (
    <div className="flex h-full overflow-hidden">
      <Sidebar />
      {/* Content pane — the ONLY thing that changes between nav items. */}
      <main className="min-w-0 flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}

function Sidebar() {
  return (
    <aside className="flex w-[244px] shrink-0 flex-col border-r border-[var(--line)] bg-[var(--color-ink-850)]">
      {/* Header: brand + live connection indicator */}
      <div className="flex items-center justify-between gap-2 px-4 pb-3 pt-4">
        <div className="flex items-center gap-2.5">
          <div className="grid size-9 place-items-center rounded-xl bg-[var(--color-ink-600)] text-lg font-extrabold text-[var(--color-accent)] shadow-[0_8px_24px_-14px_var(--color-accent)]">
            c
          </div>
          <div>
            <h1 className="text-[15px] font-semibold leading-none tracking-tight">
              chasm
            </h1>
            <p className="mt-1 text-[10px] font-medium uppercase tracking-[0.16em] text-[var(--muted-foreground)]">
              NPC engine
            </p>
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-2 px-4 pb-3">
        <ConnectionPill />
        <StackControls />
      </div>

      {/* Nav groups */}
      <nav className="flex-1 overflow-y-auto px-3 py-2">
        {NAV_GROUPS.map((group) => (
          <div key={group.label} className="mb-4">
            <p className="px-2 pb-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted-foreground)]/80">
              {group.label}
            </p>
            <div className="flex flex-col gap-0.5">
              {group.items.map((item) => (
                <NavLink
                  key={item.key}
                  to={item.path}
                  className={({ isActive }) =>
                    cn(
                      "group flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm transition-colors",
                      isActive
                        ? "bg-[var(--color-ink-700)] text-[var(--foreground)] shadow-[0_1px_0_rgba(255,255,255,0.04)_inset]"
                        : "text-[var(--muted-foreground)] hover:bg-[var(--color-ink-700)]/50 hover:text-[var(--foreground)]",
                    )
                  }
                >
                  {({ isActive }) => (
                    <>
                      <item.icon
                        className={cn(
                          "size-4 shrink-0",
                          isActive
                            ? "text-[var(--color-accent)]"
                            : "text-[var(--muted-foreground)] group-hover:text-[var(--foreground)]",
                        )}
                        strokeWidth={1.75}
                      />
                      <span className="font-medium">{item.label}</span>
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>
    </aside>
  );
}
