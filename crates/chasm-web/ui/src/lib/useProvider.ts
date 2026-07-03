import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  providersApi,
  type ProviderCapability,
  type ProviderConfigForm,
} from "@/lib/api";

// Shared wiring for a capability's provider picker + API config, used by the
// LLM / STT / TTS settings screens so the select + save-config flow is
// identical. The GET payload is the single source of truth; select and
// saveConfig both return the fresh payload and write it back into the cache.

export function useProvider(capability: ProviderCapability) {
  const qc = useQueryClient();
  const key = ["providers", capability];

  const query = useQuery({
    queryKey: key,
    queryFn: () => providersApi.get(capability),
  });

  const select = useMutation({
    mutationFn: (provider: string) => providersApi.select(capability, provider),
    onSuccess: (fresh) => qc.setQueryData(key, fresh),
  });

  const [justSaved, setJustSaved] = useState(false);
  const saveConfig = useMutation({
    mutationFn: (form: ProviderConfigForm) =>
      providersApi.saveConfig(capability, form),
    onSuccess: (fresh) => {
      qc.setQueryData(key, fresh);
      setJustSaved(true);
      window.setTimeout(() => setJustSaved(false), 2200);
    },
  });

  const view = query.data;
  const selectedId = view?.selected ?? "local";
  const selectedProvider = view?.providers.find((p) => p.id === selectedId);
  const isLocal = selectedProvider?.kind === "local";

  return {
    query,
    view,
    selectedId,
    selectedProvider,
    isLocal,
    select,
    saveConfig,
    justSaved,
    saveError: saveConfig.isError
      ? (saveConfig.error as Error)?.message || "Save failed"
      : null,
  };
}
