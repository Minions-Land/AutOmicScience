import type { Tool } from '../toolset/Tool.js';

export interface Skill {
  name: string;
  description: string;
  /** Injected into the system prompt when the skill is active. */
  instructions: string;
  tools?: Tool[];
}

export interface SkillLoader {
  load(pathOrName: string): Promise<Skill>;
}
