import os from 'node:os';
import path from 'node:path';
import { z } from 'zod';
import type { Agent } from '../agent/Agent.js';
import { builtinSkillLoader } from '../skill/BuiltinSkills.js';
import { BUILTIN_SKILL_DIRS } from '../skill/BuiltinSkills.js';
import { FileSkillLoader } from '../skill/SkillLoader.js';
import type { Skill } from '../skill/Skill.js';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';

export interface SkillToolSetOptions {
  agent: Agent;
  rootDir?: string;
  searchDirs?: string[];
}

export function skillToolSet(opts: SkillToolSetOptions): ToolSet {
  const rootDir = opts.rootDir ?? process.cwd();
  const loader = new FileSkillLoader(opts.searchDirs ?? defaultSkillDirs(rootDir));

  return new ToolSet('skill', [
    defineTool<Record<string, never>, { skills: SkillSummary[] }>({
      name: 'list_available_skills',
      aliases: ['list_skills', 'skills'],
      operation: 'read',
      description: 'List available AutOmicScience skills from built-in, project, and user skill directories.',
      searchHint: 'skill skills capability prompt workflow built-in project user list',
      parameters: z.object({}),
      isReadOnly: () => true,
      execute: async () => {
        const active = new Set(opts.agent.listSkills().map((skill) => skill.name));
        return {
          skills: (await loader.list()).map((skill) => ({
            name: skill.name,
            description: skill.description,
            source: skill.source,
            active: active.has(skill.name),
          })),
        };
      },
    }),

    defineTool<Record<string, never>, { skills: SkillSummary[] }>({
      name: 'list_active_skills',
      operation: 'read',
      description: 'List skills currently loaded into the active AutOmicScience agent prompt.',
      searchHint: 'active loaded skill current prompt',
      parameters: z.object({}),
      isReadOnly: () => true,
      execute: async () => ({
        skills: opts.agent.listSkills().map((skill) => toSkillSummary(skill, true)),
      }),
    }),

    defineTool<{ name: string }, SkillSummary & { instructions: string }>({
      name: 'read_skill',
      operation: 'read',
      description: 'Read a AutOmicScience skill by name or path and return its instructions.',
      searchHint: 'skill read inspect instructions prompt markdown',
      parameters: z.object({
        name: z.string().describe('Skill name or path, for example annotation-pipeline.'),
      }),
      isReadOnly: () => true,
      execute: async ({ name }) => {
        const active = opts.agent.listSkills().find((skill) => skill.name === name);
        const skill = active ?? await loader.read(name);
        return {
          ...toSkillSummary(skill, Boolean(active)),
          instructions: skill.instructions,
        };
      },
    }),

    defineTool<{ name: string }, { loaded: SkillSummary; tools_added: string[] }>({
      name: 'load_skill',
      operation: 'task',
      description: 'Load a AutOmicScience skill into the active agent. Skill instructions are injected into the next model call.',
      searchHint: 'skill load activate use prompt workflow',
      parameters: z.object({
        name: z.string().describe('Skill name or path, for example annotation-pipeline.'),
      }),
      isReadOnly: () => false,
      execute: async ({ name }) => {
        const skill = name === 'annotation-pipeline'
          ? await builtinSkillLoader().load('annotation-pipeline')
          : await loader.load(name);
        opts.agent.addSkill(skill);
        for (const tool of skill.tools ?? []) {
          if (!opts.agent.listTools().some((item) => item.name === tool.name)) {
            opts.agent.addTool(tool);
          }
        }
        return {
          loaded: toSkillSummary(skill, true),
          tools_added: (skill.tools ?? []).map((tool) => tool.name),
        };
      },
    }),

    defineTool<{ name: string }, { removed: boolean; name: string }>({
      name: 'remove_skill',
      operation: 'task',
      description: 'Remove a loaded skill from the active AutOmicScience agent prompt.',
      searchHint: 'skill unload remove deactivate',
      parameters: z.object({
        name: z.string().describe('Loaded skill name.'),
      }),
      isReadOnly: () => false,
      execute: async ({ name }) => ({ name, removed: opts.agent.removeSkill(name) }),
    }),
  ]);
}

function defaultSkillDirs(rootDir: string): string[] {
  return [
    ...BUILTIN_SKILL_DIRS,
    path.join(rootDir, 'skills'),
    path.join(rootDir, '.aos', 'skills'),
    path.join(os.homedir(), '.aos', 'skills'),
  ];
}

interface SkillSummary {
  name: string;
  description: string;
  source?: string;
  active: boolean;
}

function toSkillSummary(skill: Pick<Skill, 'name' | 'description'> & { source?: string }, active: boolean): SkillSummary {
  return {
    name: skill.name,
    description: skill.description,
    source: skill.source,
    active,
  };
}
