import { Sequential } from '../../../team/Sequential.js';
import {
  createStage2SelectorAgent,
  createStage3AdapterAgent,
  createStage4AdjudicatorAgent,
  type ScmasAgentOptions,
} from '../agents/StageAgents.js';

/**
 * `Sequential` team chaining Stage-2 → Stage-3 → Stage-4 LLM agents.
 *
 * The deterministic Python tooling (Stage-1 eval, scDesign3, data prep, the
 * non-LLM portions of Stage 2/3/4) is invoked by tools the agents can call;
 * the team only orchestrates the LLM portions of the pipeline.
 */
export async function createScmasPipeline(opts: ScmasAgentOptions): Promise<Sequential> {
  const stage2 = await createStage2SelectorAgent(opts);
  const stage3 = await createStage3AdapterAgent(opts);
  const stage4 = await createStage4AdjudicatorAgent(opts);
  return new Sequential([stage2, stage3, stage4], 'scmas-pipeline');
}
