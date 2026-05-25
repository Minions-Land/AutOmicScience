import { Sequential } from './Sequential.js';
import {
  createSelectorAgent,
  createAdapterAgent,
  createAdjudicatorAgent,
  type BuiltinAgentOptions,
} from '../agent/AnnotationAgents.js';

/**
 * Built-in annotation pipeline:
 *   Selector → Adapter → Adjudicator
 *
 * The deterministic Python tooling (data prep, synthetic generation,
 * benchmarking, the non-LLM portions of selection/adapt/consensus) is
 * invoked by tools the agents call; the team only orchestrates the
 * LLM portions of the pipeline.
 */
export async function createAnnotationPipeline(opts: BuiltinAgentOptions): Promise<Sequential> {
  const selector = await createSelectorAgent(opts);
  const adapter = await createAdapterAgent(opts);
  const adjudicator = await createAdjudicatorAgent(opts);
  return new Sequential([selector, adapter, adjudicator], 'annotation-pipeline');
}
