import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';
import { runPython, type BridgeOptions, type CliFlag } from '../bridge/PythonBridge.js';

/**
 * Annotation-pipeline orchestration tools.
 *
 * These wrap the deterministic Python entry points for query profiling,
 * source/model selection, execution planning, adapter+execute, and
 * cross-model consensus. The LLM portions of selection/adaptation/
 * adjudication are also exposed here so an LLM-driven Novaeve Agent can
 * call them directly. The Python subprocess is the source of truth.
 */

const ProfileQueryArgs = z.object({
  datasetId: z.string().min(1),
  input: z.string().optional(),
  outputDir: z.string().optional(),
  maxCells: z.number().int().positive().default(20_000),
  seed: z.number().int().default(3028),
});
type ProfileQueryArgs = z.infer<typeof ProfileQueryArgs>;

const SelectSourcesArgs = z.object({
  queryProfile: z.string().min(1),
  outputDir: z.string().optional(),
  artifactBundle: z.string().optional(),
  preparedSourceRoot: z.string().optional(),
  capabilityDir: z.string().optional(),
  topK: z.number().int().positive().optional(),
  numModels: z.number().int().positive().optional(),
  minSharedGenes: z.number().int().positive().default(30),
  maxSourceProfileCells: z.number().int().positive().default(20_000),
  maxQueryCells: z.number().int().positive().default(5_000),
  maxReferenceCells: z.number().int().positive().default(1_000),
  k: z.number().int().positive().default(15),
  device: z.string().default(''),
  batchSize: z.number().int().positive().default(16),
  llmMode: z.enum(['required', 'optional', 'off']).default('required'),
  llmModel: z.string().optional(),
  llmMaxCandidates: z.number().int().positive().default(80),
  llmRetryLimit: z.number().int().nonnegative().default(2),
  selectionStrategy: z.enum(['batch', 'iterative', 'one_by_one']).default('one_by_one'),
  selectionObjective: z
    .enum(['unified_rank', 'consensus', 'best_single_ablation'])
    .default('unified_rank'),
  iterativeExcludeScope: z.enum(['model', 'family']).default('family'),
  excludedModels: z.array(z.string()).default([]),
  seed: z.number().int().default(3028),
});
type SelectSourcesArgs = z.infer<typeof SelectSourcesArgs>;

const RunPlanArgs = z.object({
  plan: z.string().min(1),
  outputDir: z.string().optional(),
  maxQueryCells: z.number().int().nonnegative().default(0),
  maxReferenceCells: z.number().int().nonnegative().default(0),
  minSharedGenes: z.number().int().nonnegative().default(0),
  k: z.number().int().nonnegative().default(0),
  device: z.string().optional(),
  batchSize: z.number().int().nonnegative().default(0),
});
type RunPlanArgs = z.infer<typeof RunPlanArgs>;

const InspectContractsArgs = z.object({
  capabilityDir: z.string().optional(),
  registry: z.string().optional(),
  outputDir: z.string().optional(),
});
type InspectContractsArgs = z.infer<typeof InspectContractsArgs>;

const AdaptAndExecuteArgs = z.object({
  plan: z.string().min(1),
  mode: z.enum(['subset', 'full']).default('subset'),
  resume: z.boolean().default(false),
  outputDir: z.string().optional(),
  capabilityDir: z.string().optional(),
  registry: z.string().optional(),
  retryLimit: z.number().int().nonnegative().default(2),
  llmMode: z.enum(['required', 'optional', 'off']).default('required'),
  llmModel: z.string().optional(),
  llmRetryLimit: z.number().int().nonnegative().default(2),
});
type AdaptAndExecuteArgs = z.infer<typeof AdaptAndExecuteArgs>;

const ConsensusArgs = z.object({
  executionSummary: z.string().min(1),
  mode: z.enum(['subset', 'full']).default('subset'),
  outputDir: z.string().optional(),
  seed: z.number().int().default(3028),
  llmPolicyMode: z.enum(['required', 'optional', 'off']).default('off'),
  llmModel: z.string().optional(),
  llmCellAdjudicationMode: z.enum(['required', 'optional', 'off']).default('off'),
  llmCellMaxGroups: z.number().int().positive().default(120),
  llmCellBatchSize: z.number().int().positive().default(12),
  modelScope: z.enum(['selected', 'completed']).default('selected'),
  skipReferenceGeometry: z.boolean().default(false),
  executionStrategy: z.enum(['selected_only', 'benchmark_all']).default('selected_only'),
  maxProbeCells: z.number().int().positive().default(5000),
});
type ConsensusArgs = z.infer<typeof ConsensusArgs>;

export function annotationStageToolSet(opt: BridgeOptions = {}): ToolSet {
  return new ToolSet('annotation-stage', [
    defineTool<ProfileQueryArgs, unknown>({
      name: 'annotate_profile_query',
      description: 'Build a query profile for downstream source/model selection.',
      parameters: ProfileQueryArgs,
      execute: async (a) =>
        runPython(
          'profile-query',
          [
            ['--dataset-id', a.datasetId],
            ['--input', a.input],
            ['--output-dir', a.outputDir],
            ['--max-cells', a.maxCells],
            ['--seed', a.seed],
          ],
          opt,
        ),
    }),
    defineTool<SelectSourcesArgs, unknown>({
      name: 'annotate_select_sources',
      description: 'Select no-training source+model pairs for a profiled query dataset.',
      parameters: SelectSourcesArgs,
      execute: async (a) => {
        const flags: CliFlag[] = [
          ['--query-profile', a.queryProfile],
          ['--output-dir', a.outputDir],
          ['--artifact-bundle', a.artifactBundle],
          ['--prepared-source-root', a.preparedSourceRoot],
          ['--capability-dir', a.capabilityDir],
          ['--top-k', a.topK],
          ['--num-models', a.numModels],
          ['--min-shared-genes', a.minSharedGenes],
          ['--max-source-profile-cells', a.maxSourceProfileCells],
          ['--max-query-cells', a.maxQueryCells],
          ['--max-reference-cells', a.maxReferenceCells],
          ['--k', a.k],
          ['--device', a.device],
          ['--batch-size', a.batchSize],
          ['--llm-mode', a.llmMode],
          ['--llm-model', a.llmModel],
          ['--llm-max-candidates', a.llmMaxCandidates],
          ['--llm-retry-limit', a.llmRetryLimit],
          ['--selection-strategy', a.selectionStrategy],
          ['--selection-objective', a.selectionObjective],
          ['--iterative-exclude-scope', a.iterativeExcludeScope],
          ['--seed', a.seed],
        ];
        for (const m of a.excludedModels) flags.push(['--exclude-model', m]);
        return runPython('select-models', flags, opt);
      },
    }),
    defineTool<RunPlanArgs, unknown>({
      name: 'annotate_run_plan',
      description: 'Run subset no-training execution from a selected execution plan.',
      parameters: RunPlanArgs,
      execute: async (a) =>
        runPython(
          'run-cross-species-plan',
          [
            ['--plan', a.plan],
            ['--output-dir', a.outputDir],
            ['--max-query-cells', a.maxQueryCells],
            ['--max-reference-cells', a.maxReferenceCells],
            ['--min-shared-genes', a.minSharedGenes],
            ['--k', a.k],
            ['--device', a.device],
            ['--batch-size', a.batchSize],
          ],
          opt,
        ),
    }),
    defineTool<InspectContractsArgs, unknown>({
      name: 'annotate_inspect_contracts',
      description: 'Inspect capability cards, registry artifacts, and wrapper signatures.',
      parameters: InspectContractsArgs,
      execute: async (a) =>
        runPython(
          'inspect-model-contracts',
          [
            ['--capability-dir', a.capabilityDir],
            ['--registry', a.registry],
            ['--output-dir', a.outputDir],
          ],
          opt,
        ),
    }),
    defineTool<AdaptAndExecuteArgs, unknown>({
      name: 'annotate_adapt_and_execute',
      description: 'Generate adapter specs and execute the whitelist plan.',
      parameters: AdaptAndExecuteArgs,
      execute: async (a) =>
        runPython(
          'adapt-and-execute',
          [
            ['--plan', a.plan],
            ['--mode', a.mode],
            ['--resume', a.resume],
            ['--output-dir', a.outputDir],
            ['--capability-dir', a.capabilityDir],
            ['--registry', a.registry],
            ['--retry-limit', a.retryLimit],
            ['--llm-mode', a.llmMode],
            ['--llm-model', a.llmModel],
            ['--llm-retry-limit', a.llmRetryLimit],
          ],
          opt,
        ),
    }),
    defineTool<ConsensusArgs, unknown>({
      name: 'annotate_run_consensus',
      description: 'Reference-enhanced consensus fusion across all completed adapters.',
      parameters: ConsensusArgs,
      execute: async (a) =>
        runPython(
          'run-consensus',
          [
            ['--stage3-summary', a.executionSummary],
            ['--mode', a.mode],
            ['--output-dir', a.outputDir],
            ['--seed', a.seed],
            ['--llm-policy-mode', a.llmPolicyMode],
            ['--llm-model', a.llmModel],
            ['--llm-cell-adjudication-mode', a.llmCellAdjudicationMode],
            ['--llm-cell-max-groups', a.llmCellMaxGroups],
            ['--llm-cell-batch-size', a.llmCellBatchSize],
            ['--model-scope', a.modelScope],
            ['--skip-reference-geometry', a.skipReferenceGeometry],
            ['--execution-strategy', a.executionStrategy],
            ['--max-probe-cells', a.maxProbeCells],
          ],
          opt,
        ),
    }),
  ]);
}

/** Union of every annotation-pipeline toolset useful to one agent. */
export function annotationToolSet(opt: BridgeOptions = {}): ToolSet {
  return new ToolSet('annotation').merge(annotationStageToolSet(opt));
}
