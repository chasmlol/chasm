import type { ReactNode } from "react";
import { Construction } from "lucide-react";

import { PageBody, PageHeader, EmptyState } from "@/components/ui/page";

// The shared "<Name> — coming soon" placeholder. Every stubbed screen renders
// one of these INSIDE the shared page layout, so a not-yet-filled screen still
// looks like part of the app (header + bounded body + empty state) and is
// reachable via the sidebar. Fill agents replace the screen body; the shell,
// routing, and nav are already done.
export function ComingSoon({
  eyebrow,
  title,
  description,
  icon,
  note,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  icon?: ReactNode;
  /** Extra line shown in the empty-state body. */
  note?: ReactNode;
}) {
  return (
    <PageBody width="wide">
      <PageHeader eyebrow={eyebrow} title={title} description={description} />
      <div className="flex-1 pt-[var(--gap,14px)]">
        <EmptyState
          icon={icon ?? <Construction className="size-5" strokeWidth={1.75} />}
          title={
            <span>
              {title} — coming soon
            </span>
          }
          description={
            note ??
            "This screen is part of the redesign and hasn't been filled in yet. The shell, routing, and shared components are in place."
          }
        />
      </div>
    </PageBody>
  );
}
