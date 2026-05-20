import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FileSkillLoader } from './SkillLoader.js';
import type { Skill } from './Skill.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BUILTIN_DIR = path.join(HERE, 'builtin');

/** Search dirs containing MedrixAI's built-in skill markdown. */
export const BUILTIN_SKILL_DIRS: string[] = [BUILTIN_DIR];

/** Loader pre-wired to the built-in skill directory. */
export function builtinSkillLoader(): FileSkillLoader {
  return new FileSkillLoader(BUILTIN_SKILL_DIRS);
}

/** Load the canonical `annotation-pipeline` skill. */
export async function loadAnnotationPipelineSkill(): Promise<Skill> {
  return builtinSkillLoader().load('annotation-pipeline');
}
