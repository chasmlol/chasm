import { useEffect, useMemo, useState } from "react";
import { Check, Eye, EyeOff, Loader2 } from "lucide-react";

import type {
  ProviderCapability,
  ProviderConfigForm,
  ProviderDto,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, FormRow, Select } from "@/components/ui/page";

// ===========================================================================
// ApiProviderConfig — the config form for a selected HOSTED API provider. Shows
// an API-key field (password + reveal), a model field (text + <datalist> of the
// provider's suggested ids), a base-URL field (only when needs_base_url), and a
// voice field (only when needs_voice: a <select> of suggested voices PLUS a
// free-text custom id). Persists via POST .../config with an explicit Save.
//
// Model + voice are editable text (not hard dropdowns) because hosted ids
// rotate. The key pre-fills from config.api_key and is never logged.
// ===========================================================================

export interface ApiProviderConfigProps {
  capability: ProviderCapability;
  provider: ProviderDto;
  /** Persist the edited fields. Only changed fields are sent. */
  onSave: (form: ProviderConfigForm) => void;
  saving?: boolean;
  /** A readable backend error from the last save, if any. */
  error?: string | null;
  justSaved?: boolean;
}

/** Sentinel select value that switches the voice control to a custom text id. */
const CUSTOM_VOICE = "__custom__";

export function ApiProviderConfig({
  capability,
  provider,
  onSave,
  saving,
  error,
  justSaved,
}: ApiProviderConfigProps) {
  const initial = provider.config;
  const [apiKey, setApiKey] = useState(initial.api_key ?? "");
  const [model, setModel] = useState(initial.model ?? "");
  const [baseUrl, setBaseUrl] = useState(initial.base_url ?? "");
  const [voice, setVoice] = useState(initial.voice ?? "");
  const [routing, setRouting] = useState(initial.routing ?? "");
  const [showKey, setShowKey] = useState(false);
  const hasRouting = (provider.routing_options ?? []).length > 0;

  // Reset the form when the provider (or its saved config) changes underneath us.
  useEffect(() => {
    setApiKey(initial.api_key ?? "");
    setModel(initial.model ?? "");
    setBaseUrl(initial.base_url ?? "");
    setVoice(initial.voice ?? "");
    setRouting(initial.routing ?? "");
    // provider.id keys the reset; config values are the source of truth.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider.id, initial.api_key, initial.model, initial.base_url, initial.voice, initial.routing]);

  const listId = `models-${capability}-${provider.id}`;

  // Does the current voice string match one of the suggested voices?
  const voiceIsKnown = useMemo(
    () => provider.voices.some((v) => v.id === voice),
    [provider.voices, voice],
  );
  // When the current voice isn't a known suggestion (and is non-empty), the
  // <select> shows "Custom…" and a text input carries the actual id.
  const [voiceMode, setVoiceMode] = useState<"known" | "custom">(
    voice && !voiceIsKnown ? "custom" : "known",
  );
  useEffect(() => {
    setVoiceMode(voice && !voiceIsKnown ? "custom" : "known");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider.id]);

  const dirty = useMemo(
    () =>
      apiKey !== (initial.api_key ?? "") ||
      model !== (initial.model ?? "") ||
      (provider.needs_base_url && baseUrl !== (initial.base_url ?? "")) ||
      (provider.needs_voice && voice !== (initial.voice ?? "")) ||
      (hasRouting && routing !== (initial.routing ?? "")),
    [apiKey, model, baseUrl, voice, routing, hasRouting, initial, provider],
  );

  const submit = () => {
    const form: ProviderConfigForm = { provider: provider.id };
    if (apiKey !== (initial.api_key ?? "")) form.apiKey = apiKey;
    if (model !== (initial.model ?? "")) form.model = model;
    if (provider.needs_base_url && baseUrl !== (initial.base_url ?? ""))
      form.baseUrl = baseUrl;
    if (provider.needs_voice && voice !== (initial.voice ?? ""))
      form.voice = voice;
    if (hasRouting && routing !== (initial.routing ?? "")) form.routing = routing;
    onSave(form);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{provider.name} configuration</CardTitle>
        <CardDescription>
          Your key stays on this machine and is sent only to {provider.name}.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <FormRow
          stacked
          label="API key"
          htmlFor={`${listId}-key`}
          help="Stored locally; used to authenticate requests to the provider."
          control={
            <div className="relative">
              <Field
                id={`${listId}-key`}
                type={showKey ? "text" : "password"}
                autoComplete="off"
                spellCheck={false}
                className="pr-10 font-mono"
                placeholder="sk-…"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
              />
              <button
                type="button"
                onClick={() => setShowKey((s) => !s)}
                aria-label={showKey ? "Hide API key" : "Show API key"}
                className="absolute inset-y-0 right-0 grid w-10 place-items-center text-[var(--muted-foreground)] transition-colors hover:text-[var(--foreground)]"
              >
                {showKey ? (
                  <EyeOff className="size-4" />
                ) : (
                  <Eye className="size-4" />
                )}
              </button>
            </div>
          }
        />

        <FormRow
          stacked
          label="Model"
          htmlFor={`${listId}-model`}
          help="The model id to request. Suggestions below; type any id the provider offers."
          control={
            <>
              <Field
                id={`${listId}-model`}
                list={listId}
                autoComplete="off"
                spellCheck={false}
                placeholder={provider.default_model || "model id"}
                value={model}
                onChange={(e) => setModel(e.target.value)}
              />
              <datalist id={listId}>
                {provider.models.map((m) => (
                  <option key={m} value={m} />
                ))}
              </datalist>
            </>
          }
        />

        {hasRouting && (
          <FormRow
            stacked
            label="Routing"
            htmlFor={`${listId}-routing`}
            help="How OpenRouter picks a provider for your model. Speed routes to the fastest (best for live dialogue); Price to the cheapest; Balanced uses OpenRouter's default."
            control={
              <Select
                id={`${listId}-routing`}
                className="w-full"
                value={routing || "speed"}
                onChange={(e) => setRouting(e.target.value)}
              >
                {(provider.routing_options ?? []).map((o) => (
                  <option key={o.id} value={o.id}>
                    {o.label}
                  </option>
                ))}
              </Select>
            }
          />
        )}

        {provider.needs_base_url && (
          <FormRow
            stacked
            label="Base URL"
            htmlFor={`${listId}-base`}
            help="The API endpoint. Leave blank to use the provider default."
            control={
              <Field
                id={`${listId}-base`}
                autoComplete="off"
                spellCheck={false}
                className="font-mono"
                placeholder={provider.default_base_url || "https://…"}
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
              />
            }
          />
        )}

        {provider.needs_voice && (
          <FormRow
            stacked
            label="Voice"
            help="Pick a suggested voice, or choose Custom to enter a voice id directly."
            control={
              <div className="flex flex-col gap-2">
                <Select
                  className="w-full"
                  value={voiceMode === "custom" ? CUSTOM_VOICE : voice}
                  onChange={(e) => {
                    const val = e.target.value;
                    if (val === CUSTOM_VOICE) {
                      setVoiceMode("custom");
                    } else {
                      setVoiceMode("known");
                      setVoice(val);
                    }
                  }}
                >
                  <option value="">Default</option>
                  {provider.voices.map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.label}
                    </option>
                  ))}
                  <option value={CUSTOM_VOICE}>Custom…</option>
                </Select>
                {voiceMode === "custom" && (
                  <Field
                    autoComplete="off"
                    spellCheck={false}
                    className="font-mono"
                    placeholder="voice id"
                    value={voice}
                    onChange={(e) => setVoice(e.target.value)}
                  />
                )}
              </div>
            }
          />
        )}

        {error && (
          <p className="text-[13px] text-[var(--color-danger)]">{error}</p>
        )}

        <div className="flex items-center justify-end gap-3">
          {justSaved && (
            <span className="flex items-center gap-1.5 text-[13px] font-medium text-[var(--color-player)]">
              <Check className="size-4" /> Saved
            </span>
          )}
          <Button size="sm" disabled={!dirty || saving} onClick={submit}>
            {saving ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Check className="size-4" />
            )}
            Save connection
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
