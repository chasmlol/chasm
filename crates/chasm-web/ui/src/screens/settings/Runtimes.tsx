import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { modelsApi, type ModelDto } from "@/lib/api";
import { SettingsPage } from "@/components/ui/settings-page";
import { ModelPicker, type ModelCard } from "@/components/ModelPicker";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

// Runtimes settings — picks the managed LLM runtime that serves :5001.
//
//   * koboldcpp (default): LLM + Whisper STT in one process, single KV slot.
//   * llama.cpp (llama-server): multiple prompt-cache slots, so group-scene
//     speaker swaps reuse each speaker's cached prompt instead of paying a
//     full reprocess. llama-server has NO Whisper, so voice input then needs
//     the Parakeet STT engine (Settings → STT).
//
// Selecting a runtime persists it and live-swaps the process serving :5001
// (same selected model). The compatibility card below surfaces the
// whisper-requires-koboldcpp interaction so voice input never dies silently.

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

export function Runtimes() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["models", "runtime"],
    queryFn: () => modelsApi.get("runtime"),
  });
  // The STT selection, to surface the whisper/koboldcpp dependency: the
  // Parakeet card's id is "parakeet"; anything else selected = a whisper model.
  const stt = useQuery({
    queryKey: ["models", "stt"],
    queryFn: () => modelsApi.get("stt"),
  });

  const select = useMutation({
    mutationFn: (id: string) => modelsApi.select("runtime", id),
    onSuccess: (fresh) => qc.setQueryData(["models", "runtime"], fresh),
  });
  const download = useMutation({
    mutationFn: (id: string) => modelsApi.download("runtime", id),
    onSuccess: (fresh) => qc.setQueryData(["models", "runtime"], fresh),
  });

  const selectedRuntime = query.data?.selected_id ?? "koboldcpp";
  const sttSelection = stt.data?.selected_id;
  const sttIsParakeet = sttSelection === "parakeet";
  const showWhisperWarning =
    selectedRuntime === "llamacpp" && !sttIsParakeet && !stt.isLoading;

  return (
    <SettingsPage
      eyebrow="AI"
      title="Runtimes"
      description="The local server that runs the selected LLM. Switching runtimes reloads the model in place — same port, same model, no other changes."
    >
      <ModelPicker
        models={(query.data?.models ?? []).map(toCard)}
        selectedId={query.data?.selected_id}
        folder={query.data?.folder ? { path: query.data.folder } : undefined}
        isLoading={query.isLoading}
        isError={query.isError}
        downloadingId={download.isPending ? download.variables : null}
        onSelect={(id) => select.mutate(id)}
        onDownload={(id) => download.mutate(id)}
      />

      <Card>
        <CardHeader>
          <CardTitle>Voice input compatibility</CardTitle>
          <CardDescription>
            Whisper runs inside koboldcpp. llama.cpp has no Whisper, so with
            that runtime voice input needs the Parakeet engine (Settings →
            STT), which runs on its own port and works with either runtime.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {showWhisperWarning ? (
            <p className="text-sm text-[var(--color-npc)]">
              llama.cpp is selected but STT is still set to a Whisper model —
              voice input will be unavailable. Select Parakeet under Settings →
              STT (or switch back to koboldcpp) to keep push-to-talk working.
            </p>
          ) : (
            <p className="text-sm text-[var(--muted-foreground)]">
              {selectedRuntime === "llamacpp"
                ? "STT is set to Parakeet — voice input keeps working on llama.cpp."
                : "koboldcpp is selected — both Whisper and Parakeet work."}
            </p>
          )}
        </CardContent>
      </Card>
    </SettingsPage>
  );
}
