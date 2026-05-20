/**
 * scMAS plugin for PantheonOS-ts
 *
 * Ports the CanChen_MAS multi-stage single-cell annotation pipeline into
 * PantheonOS-ts plugin shape. Heavy scientific compute remains in the
 * vendored Python module under `_import_scMAS/CanChen_MAS/` and is invoked
 * via subprocess Tools; LLM stages are reimplemented as PantheonOS-ts
 * Agents driving a `Sequential` team.
 */

export * as schemas from './schemas/index.js';
export * from './tools/index.js';
export * from './agents/index.js';
export * from './team/index.js';
export {
  scmasSkillLoader,
  loadScmasAnnotationSkill,
  SCMAS_SKILL_DIRS,
} from './skills/index.js';

import { scmasToolSet } from './tools/index.js';
import { loadScmasAnnotationSkill } from './skills/index.js';
import { createScmasPipeline } from './team/ScmasPipeline.js';
import {
  createStage2SelectorAgent,
  createStage3AdapterAgent,
  createStage4AdjudicatorAgent,
  type ScmasAgentOptions,
} from './agents/StageAgents.js';
import type { ToolSet } from '../../toolset/ToolSet.js';
import type { Skill } from '../../skill/Skill.js';
import type { Agent } from '../../agent/Agent.js';
import type { Sequential } from '../../team/Sequential.js';
import type { BridgeOptions } from './tools/PythonBridge.js';

/** Minimal host interface so plugins can register without coupling to a concrete host. */
export interface PluginHost {
  registerTool?(toolset: ToolSet): void;
  registerAgent?(agent: Agent): void;
  registerSkill?(skill: Skill): void;
  registerTeam?(team: Sequential): void;
}

/**
 * Eagerly construct the full plugin surface and register it with a host.
 * Useful for runtime plugin loaders. Returns the constructed objects so
 * callers can keep references for direct invocation as well.
 */
export async function registerScmas(
  host: PluginHost = {},
  opts: ScmasAgentOptions,
): Promise<{
  toolset: ToolSet;
  skill: Skill;
  agents: { stage2: Agent; stage3: Agent; stage4: Agent };
  team: Sequential;
}> {
  const toolset = scmasToolSet(opts.bridge as BridgeOptions | undefined);
  const skill = await loadScmasAnnotationSkill();
  const stage2 = await createStage2SelectorAgent(opts);
  const stage3 = await createStage3AdapterAgent(opts);
  const stage4 = await createStage4AdjudicatorAgent(opts);
  const team = await createScmasPipeline(opts);

  host.registerTool?.(toolset);
  host.registerSkill?.(skill);
  host.registerAgent?.(stage2);
  host.registerAgent?.(stage3);
  host.registerAgent?.(stage4);
  host.registerTeam?.(team);

  return { toolset, skill, agents: { stage2, stage3, stage4 }, team };
}
