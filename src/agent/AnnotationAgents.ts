import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Agent } from './Agent.js';
import type { AgentOptions } from './AgentOptions.js';
import { ToolSet } from '../toolset/ToolSet.js';
import { annotationStageToolSet } from '../toolset/AnnotationStageTools.js';
import type { BridgeOptions } from '../bridge/PythonBridge.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROMPTS_DIR = path.join(HERE, 'prompts');

async function readPrompt(filename: string): Promise<string> {
  return fs.readFile(path.join(PROMPTS_DIR, filename), 'utf8');
}

/**
 * Options shared by every built-in annotation-pipeline agent. Extends the
 * core AgentOptions; the toolset defaults to `annotationStageToolSet()`
 * so the agent can drive the pipeline end-to-end.
 */
export interface BuiltinAgentOptions extends Omit<AgentOptions, 'systemPrompt' | 'toolset'> {
  /** Bridge config for any subprocess tools the agent ends up calling. */
  bridge?: BridgeOptions;
  /** Override the default toolset. */
  toolset?: ToolSet;
  /** Override the system-prompt markdown (mostly useful in tests). */
  systemPromptOverride?: string;
}

/**
 * Selector agent — picks `top_k` (source, model) execution pairs for a
 * profiled query. Output is JSON conforming to `SelectorResponse`.
 */
export async function createSelectorAgent(opts: BuiltinAgentOptions): Promise<Agent> {
  const systemPrompt = opts.systemPromptOverride ?? (await readPrompt('selector.system.md'));
  return new Agent({
    name: 'medrix-selector',
    ...opts,
    systemPrompt,
    toolset: opts.toolset ?? annotationStageToolSet(opts.bridge),
  });
}

/**
 * Adapter agent — emits an `AdapterSpec` whose actions are drawn from
 * `ALLOWED_ACTIONS`. The deterministic Python reviewer rejects unknown
 * actions, mutated immutable fields, missing paths, or unsafe keys.
 */
export async function createAdapterAgent(opts: BuiltinAgentOptions): Promise<Agent> {
  const systemPrompt = opts.systemPromptOverride ?? (await readPrompt('adapter.system.md'));
  return new Agent({
    name: 'medrix-adapter',
    ...opts,
    systemPrompt,
    toolset: opts.toolset ?? annotationStageToolSet(opts.bridge),
  });
}

/**
 * Adjudicator agent — resolves low-consistency cell groups by picking
 * exactly one `selected_label` from `allowed_labels` per group.
 */
export async function createAdjudicatorAgent(opts: BuiltinAgentOptions): Promise<Agent> {
  const systemPrompt = opts.systemPromptOverride ?? (await readPrompt('adjudicator.system.md'));
  return new Agent({
    name: 'medrix-adjudicator',
    ...opts,
    systemPrompt,
    toolset: opts.toolset ?? annotationStageToolSet(opts.bridge),
  });
}
