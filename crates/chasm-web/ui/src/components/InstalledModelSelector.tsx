import { useMutation, useQueryClient } from "@tanstack/react-query";

import {
  modelsApi,
  type ModelDomain,
  type ModelDto,
  type ModelSettingsDto,
} from "@/lib/api";
import { ModelPicker, type ModelCard } from "@/components/ModelPicker";
import { EmptyState } from "@/components/ui/page";

// ===========================================================================
// InstalledModelSelector — lists the INSTALLED models for a domain and lets the
// user pick which one is ACTIVE. Used by the LLM / STT Local views above the
// recommended/placement list so the user can see which installed model is live
// and switch between installed ones.
//
// The active model is `selected_id` from GET /models/:domain; selecting a card
// POSTs /models/:domain/select {id}, writes the fresh payload into the cache,
// and invalidates ["models", domain] so every consumer refreshes. When nothing
// is installed yet, it shows guidance instead (the placement flow below handles
// actually adding one).
// ===========================================================================

function toCard(dto: ModelDto): ModelCard {
  return {
    id: dto.id,
    name: dto.name,
    description: dto.description,
    installed: dto.installed,
    recommended: dto.recommended,
    meta: dto.meta,
    status: dto.status,
  };
}

export interface InstalledModelSelectorProps {
  /** Which model domain to select the active model for. */
  domain: ModelDomain;
  /** The full model list from GET /models/:domain (unfiltered). */
  models: ModelDto[];
  /** The active model id (`selected_id` from the payload). */
  selectedId?: string;
  isLoading?: boolean;
  isError?: boolean;
  /** Section title / description overrides. */
  title?: string;
  description?: string;
  /** Copy for the empty state when nothing is installed yet. */
  emptyTitle?: string;
  emptyDescription?: string;
}

export function InstalledModelSelector({
  domain,
  models,
  selectedId,
  isLoading,
  isError,
  title = "Active model",
  description = "Pick which installed model is active. Switching applies on the next request.",
  emptyTitle = "No model installed yet",
  emptyDescription = "Place a model below to make it available, then it appears here to activate.",
}: InstalledModelSelectorProps) {
  const qc = useQueryClient();
  const key = ["models", domain];

  const select = useMutation({
    mutationFn: (id: string) => modelsApi.select(domain, id),
    onSuccess: (fresh: ModelSettingsDto) => {
      qc.setQueryData(key, fresh);
      qc.invalidateQueries({ queryKey: key });
    },
  });

  const installed = models.filter((m) => m.installed);

  // Only guide-to-placement when the fetch succeeded and there's genuinely
  // nothing installed — while loading / on error, let ModelPicker show its own
  // spinner / error so we don't flash the empty state.
  if (!isLoading && !isError && installed.length === 0) {
    return (
      <EmptyState title={emptyTitle} description={emptyDescription} />
    );
  }

  return (
    <ModelPicker
      title={title}
      description={description}
      models={installed.map(toCard)}
      selectedId={selectedId}
      onSelect={(id) => select.mutate(id)}
      isLoading={isLoading}
      isError={isError}
    />
  );
}
