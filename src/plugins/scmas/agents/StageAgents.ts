import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Agent } from '../../../agent/Agent.js';
import type { AgentOptions } from '../../../agent/AgentOptions.js';
import { ToolSet } from '../../../toolset/ToolSet.js';
import { stageToolSet } from '../tools/StageTools.js';
import type { BridgeOptions } from '../tools/PythonBridge.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROMPTS_DIR = path.resolve(HERE, '..', 'prompts');

async function readPrompt(filename: string): Promise<string> {
  return fs.readFile(path.join(PROMPTS_DIR, filename), 'utf8');
}

export interface ScmasAgentOptions extends Omit<AgentOptions, 'systemPrompt' | 'toolset'> {
  /** Bridge config for any tools the agent ends up calling. */
  bridge?: BridgeOptions;
  /** Override the default toolset (Stage tools). */
  toolset?: ToolSet;
  /** Override the system-prompt markdown contents (rare; mostly for tests). */
  systemPromptOverride?: string;
}

/**
 * Stage-2 selector LLM agent. Returns JSON conforming to
 * `Stage2SelectionResponse`. The deterministic Python reviewer in
 * `select-models` remains the source of truth and rejects any non-conforming
 * payload.
 */
export async function createStage2SelectorAgent(opts: ScmasAgentOptions): Promise<Agent> {
  const systemPrompt = opts.systemPromptOverride ?? (await readPrompt('stage2_selector.system.md'));
  return new Agent({
    name: 'scmas-stage2-selector',
    ...opts,
    systemPrompt,
    toolset: opts.toolset ?? stageToolSet(opts.bridge),
  });
}

/**
 * Stage-3 adapter LLM agent. Emits an AdapterSpec object whose actions are
 * drawn from `ALLOWED_ACTIONS`.
 */
export async function createStage3AdapterAgent(opts: ScmasAgentOptions): Promise<Agent> {
  const systemPrompt = opts.systemPromptOverride ?? (await readPrompt('stage3_adapter.system.md'));
  return new Agent({
    name: 'scmas-stage3-adapter',
    ...opts,
    systemPrompt,
    toolset: opts.toolset ?? stageToolSet(opts.bridge),
  });
}

/**
 * Stage-4 low-consistency cell adjudicator LLM agent.
 */
export async function createStage4AdjudicatorAgent(opts: ScmasAgentOptions): Promise<Agent> {
  const systemPrompt = opts.systemPromptOverride ?? (await readPrompt('stage4_adjudicator.system.md'));
  return new Agent({
    name: 'scmas-stage4-adjudicator',
    ...opts,
    systemPrompt,
    toolset: opts.toolset ?? stageToolSet(opts.bridge),
  });
}
