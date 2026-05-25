import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';
import { runPython, type BridgeOptions } from '../bridge/PythonBridge.js';

const BioMasPreflightArgs = z.object({
  outputDir: z.string().optional().describe('Directory for preflight report files.'),
  rscriptPath: z.string().optional().describe('Optional explicit Rscript path.'),
  includeFoundation: z.boolean().default(true).describe('Check torch/transformers/safetensors readiness.'),
  includeNotebook: z.boolean().default(true).describe('Check Jupyter kernel dependencies.'),
});
type BioMasPreflightArgs = z.infer<typeof BioMasPreflightArgs>;

const TinyDemoArgs = z.object({
  outputDir: z.string().optional().describe('Output directory for synthetic tiny demo assets.'),
  cellsPerLabel: z.number().int().positive().default(8).describe('Reference cells per synthetic label.'),
  seed: z.number().int().default(3028),
});
type TinyDemoArgs = z.infer<typeof TinyDemoArgs>;

const RunTinyDemoArgs = TinyDemoArgs.extend({
  topK: z.number().int().positive().default(1).describe('Number of selected expression-baseline methods to execute.'),
});
type RunTinyDemoArgs = z.infer<typeof RunTinyDemoArgs>;

const WorkflowPlanArgs = z.object({
  goal: z.string().min(1).describe('User scientific or engineering goal.'),
  inputKind: z
    .enum(['unknown', 'h5ad', 'npz', '10x_mtx', 'spatial_h5ad', 'prepared_bundle', 'tiny_demo'])
    .default('unknown'),
  hasRealData: z.boolean().default(false).describe('Whether the user supplied real biological data paths.'),
  hasFoundationWeights: z.boolean().default(false).describe('Whether foundation model checkpoint paths are configured.'),
  allowLLMSelection: z.boolean().default(true).describe('Whether selector/adaptation stages may call an LLM.'),
  allowSyntheticTinyDemo: z.boolean().default(true).describe('Whether to use local synthetic tiny demo as smoke test only.'),
});
type WorkflowPlanArgs = z.infer<typeof WorkflowPlanArgs>;

export function bioMasToolSet(opt: BridgeOptions = {}): ToolSet {
  return new ToolSet('bio-mas', [
    defineTool<BioMasPreflightArgs, unknown>({
      name: 'bio_mas_preflight',
      description:
        'Inspect AutOmicScience bioinformatics MAS readiness: Python modules, Rscript, real datasets, ' +
        'foundation checkpoints, and synthetic tiny-demo assets. Reports missing data honestly.',
      operation: 'read',
      parameters: BioMasPreflightArgs,
      isReadOnly: () => true,
      execute: async (a) =>
        runPython(
          'bio-mas-preflight',
          [
            ['--output-dir', a.outputDir],
            ['--rscript-path', a.rscriptPath],
            ['--no-foundation', !a.includeFoundation],
            ['--no-notebook', !a.includeNotebook],
          ],
          opt,
        ),
    }),
    defineTool<TinyDemoArgs, unknown>({
      name: 'bio_mas_create_tiny_demo',
      description:
        'Create clearly marked synthetic tiny single-cell data for local smoke tests. ' +
        'This is not real biological data and must not be used for scientific conclusions.',
      operation: 'write',
      parameters: TinyDemoArgs,
      isReadOnly: () => false,
      isDestructive: () => false,
      execute: async (a) =>
        runPython(
          'create-tiny-bio-demo',
          [
            ['--output-dir', a.outputDir],
            ['--cells-per-label', a.cellsPerLabel],
            ['--seed', a.seed],
          ],
          opt,
        ),
    }),
    defineTool<RunTinyDemoArgs, unknown>({
      name: 'bio_mas_run_tiny_demo',
      description:
        'Run a real local MAS smoke chain on synthetic tiny data: profile query, select expression-baseline source/model pair, execute label transfer, and return metrics paths.',
      operation: 'write',
      parameters: RunTinyDemoArgs,
      isReadOnly: () => false,
      isDestructive: () => false,
      execute: async (a) =>
        runPython(
          'run-tiny-bio-mas-demo',
          [
            ['--output-dir', a.outputDir],
            ['--cells-per-label', a.cellsPerLabel],
            ['--seed', a.seed],
            ['--top-k', a.topK],
          ],
          { timeoutMs: 10 * 60 * 1000, ...opt },
        ),
    }),
    defineTool<WorkflowPlanArgs, BioMasWorkflowPlan>({
      name: 'bio_mas_plan_workflow',
      description:
        'Return a AutOmicScience engineering execution plan for the bioinformatics MAS: agents, tools, dependencies, expected artifacts, and honest blockers.',
      operation: 'read',
      parameters: WorkflowPlanArgs,
      isReadOnly: () => true,
      execute: async (a) => buildWorkflowPlan(a),
    }),
  ]);
}

export interface BioMasWorkflowPlan {
  goal: string;
  masStyle: string;
  agents: { name: string; responsibility: string; primaryTools: string[] }[];
  phases: { phase: string; actions: string[]; expectedArtifacts: string[]; canRunNow: boolean; blockers: string[] }[];
  smokeTest: { recommended: boolean; tool: string; warning: string };
  productionRequirements: string[];
  correctFeedback: string[];
}

function buildWorkflowPlan(args: WorkflowPlanArgs): BioMasWorkflowPlan {
  const hasUsableInput = args.hasRealData || args.inputKind === 'tiny_demo';
  const llmBlockers = args.allowLLMSelection ? [] : ['LLM selector/adaptor disabled; use llm_mode=off deterministic fallback only.'];
  const realDataBlockers = args.hasRealData
    ? []
    : ['No real biological data path supplied. Production annotation cannot start; only synthetic tiny smoke test is appropriate.'];
  const foundationBlockers = args.hasFoundationWeights
    ? []
    : ['Foundation model checkpoints are not configured; Geneformer/scGPT/Nicheformer/UCE execution should be treated as unavailable.'];

  return {
    goal: args.goal,
    masStyle:
      'AutOmicScience engineering style: explicit tool contracts, project instructions, permission-aware execution, staged artifacts, preflight, traceable failures, and specialist agents rather than one monolithic prompt.',
    agents: [
      {
        name: 'BioMAS Coordinator',
        responsibility: 'Maintains the execution plan, checks blockers, decides whether to run tiny demo or production workflow.',
        primaryTools: ['bio_mas_preflight', 'bio_mas_plan_workflow', 'task_*', 'search_tools'],
      },
      {
        name: 'Data Steward',
        responsibility: 'Validates h5ad/npz/prepared-bundle inputs, builds catalogs, prepares references and source bundles.',
        primaryTools: ['bio_build_dataset_catalog', 'bio_prepare_sources', 'annotate_profile_query'],
      },
      {
        name: 'Selector',
        responsibility: 'Selects execution-ready source/model pairs from gene overlap, capability cards, and benchmark provenance.',
        primaryTools: ['annotate_select_sources'],
      },
      {
        name: 'Adapter Executor',
        responsibility: 'Inspects contracts, generates safe adapter specs, runs whitelisted execution.',
        primaryTools: ['annotate_inspect_contracts', 'annotate_adapt_and_execute', 'annotate_run_plan'],
      },
      {
        name: 'Consensus Reviewer',
        responsibility: 'Fuses completed model predictions, records low-confidence cases, and produces a report.',
        primaryTools: ['annotate_run_consensus', 'bench_label_transfer_smoke'],
      },
    ],
    phases: [
      {
        phase: '0. Preflight',
        actions: ['Run bio_mas_preflight', 'Record missing Python modules, Rscript, real datasets, and foundation checkpoints.'],
        expectedArtifacts: ['runs/bio_mas_preflight/bio_mas_preflight.json'],
        canRunNow: true,
        blockers: [],
      },
      {
        phase: '1. Tiny smoke test',
        actions: ['Run bio_mas_run_tiny_demo when no real data is available or before production execution.'],
        expectedArtifacts: ['tiny_demo_manifest.json', 'query_profile.json', 'selected_execution_plan.yaml', 'metrics.csv', 'predictions.csv'],
        canRunNow: args.allowSyntheticTinyDemo,
        blockers: args.allowSyntheticTinyDemo ? [] : ['Synthetic tiny demo disabled by caller.'],
      },
      {
        phase: '2. Real data preparation',
        actions: ['Provide h5ad/npz/prepared bundle paths', 'Run bio_prepare_sources or annotate_profile_query', 'Build dataset catalog.'],
        expectedArtifacts: ['source_manifest.json', 'query_profile.json', 'dataset_catalog.json'],
        canRunNow: args.hasRealData,
        blockers: realDataBlockers,
      },
      {
        phase: '3. Source/model selection',
        actions: ['Run annotate_select_sources', 'Use LLM selection when enabled; otherwise record deterministic fallback.'],
        expectedArtifacts: ['candidate_pairs.csv', 'selection_report.md', 'selected_execution_plan.yaml', 'review.json'],
        canRunNow: hasUsableInput && (args.allowLLMSelection || args.inputKind === 'tiny_demo'),
        blockers: [...realDataBlockers, ...llmBlockers],
      },
      {
        phase: '4. Execution and consensus',
        actions: ['Run annotate_run_plan for expression baselines', 'Run foundation models only when weights exist', 'Run annotate_run_consensus after stage 3.'],
        expectedArtifacts: ['metrics.csv', 'predictions.csv', 'skips_and_failures.csv', 'run_summary.json', 'consensus outputs'],
        canRunNow: hasUsableInput,
        blockers: args.hasFoundationWeights ? realDataBlockers : [...realDataBlockers, ...foundationBlockers],
      },
    ],
    smokeTest: {
      recommended: !args.hasRealData || args.inputKind === 'tiny_demo',
      tool: 'bio_mas_run_tiny_demo',
      warning: 'Tiny data is synthetic, generated locally, and only validates wiring. It is not evidence for biological performance.',
    },
    productionRequirements: [
      'A real query h5ad/npz or a standard prepared bundle with gene names and optional labels.',
      'Reference/source data paths via AOS_MAS_* variables or prepared source bundles.',
      'Foundation checkpoints for Geneformer/scGPT/Nicheformer/UCE if those methods are selected.',
      'Rscript and scDesign3 R packages only when running synthetic generation.',
      'An LLM API key/base URL when llm_mode is required or optional.',
    ],
    correctFeedback: [
      'Preflight returns ok=true for core Python modules; missing real data/checkpoints are listed separately.',
      'Tiny smoke returns synthetic_tiny_demo=true and writes metrics/predictions with at least one metric row.',
      'Production runs should create selected_execution_plan.yaml before execution and skips_and_failures.csv for non-runnable pairs.',
      'If assets are missing, the correct behavior is a structured missing/blocker report, not a fabricated result.',
    ],
  };
}
