// Skills domain — the event-triggered NPC "skills" authored by the
// self-improving-NPC skill-creator pass and fired (no LLM) by the executor.
//
// GET  /api/ui/v1/skills            → grouped skills + allowed actions + settings
// POST /api/ui/v1/skills/settings   → save the journaling/creation/execution toggles
// POST /api/ui/v1/skills/:id/toggle → enable/disable one skill
// POST /api/ui/v1/skills/:id/delete → remove one skill

import { getJson, postJson, UI_API } from "./http";

export interface SkillActionDto {
  actionId: string;
  target?: string;
}

export interface SkillDto {
  id: string;
  triggerEvent: string;
  triggerFilter?: string;
  actions: SkillActionDto[];
  note: string;
  thought: string;
  enabled: boolean;
  createdAt?: string;
  updatedAt?: string;
}

export interface SkillOwnerDto {
  ownerId: string;
  ownerName: string;
  skills: SkillDto[];
}

export interface AllowedActionDto {
  actionId: string;
  title: string;
  description: string;
}

export interface SkillSettingsDto {
  journalingEnabled: boolean;
  skillCreationEnabled: boolean;
  skillExecutionEnabled: boolean;
  skillCooldownSecs: number;
}

export interface SkillsViewDto {
  owners: SkillOwnerDto[];
  allowedActions: AllowedActionDto[];
  settings: SkillSettingsDto;
  lastPassAt?: string;
  passInFlight: boolean;
}

export const skillsApi = {
  list: () => getJson<SkillsViewDto>(`${UI_API}/skills`),
  saveSettings: (settings: SkillSettingsDto) =>
    postJson<SkillsViewDto>(`${UI_API}/skills/settings`, settings),
  toggle: (id: string) =>
    postJson<SkillsViewDto>(`${UI_API}/skills/${encodeURIComponent(id)}/toggle`, {}),
  remove: (id: string) =>
    postJson<SkillsViewDto>(`${UI_API}/skills/${encodeURIComponent(id)}/delete`, {}),
};
