import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FileSkillLoader } from '../../../skill/SkillLoader.js';
import type { Skill } from '../../../skill/Skill.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** Search dirs containing built-in scMAS skills. */
export const SCMAS_SKILL_DIRS: string[] = [HERE];

/** Default loader pre-wired to the plugin's skill directory. */
export function scmasSkillLoader(): FileSkillLoader {
  return new FileSkillLoader(SCMAS_SKILL_DIRS);
}

/** Convenience: load the canonical `scmas-annotation` skill. */
export async function loadScmasAnnotationSkill(): Promise<Skill> {
  return scmasSkillLoader().load('scmas-annotation');
}
