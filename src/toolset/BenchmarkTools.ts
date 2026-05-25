import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';
import { runPython, type BridgeOptions, type CliFlag } from '../bridge/PythonBridge.js';

/**
 * Benchmark + label-transfer tooling: build benchmark NPZs, run the model
 * registry, smoke-test no-training label transfer, and run the UCE 33L IMA
 * label-transfer flow. All deterministic — no LLM.
 */

const PrepareBenchmarkDatasetsArgs = z.object({
  outputDir: z.string().optional(),
  noNewSynthetic: z.boolean().default(false),
  maxCells: z.number().int().nonnegative().default(0),
  seed: z.number().int().default(3028),
});
type PrepareBenchmarkDatasetsArgs = z.infer<typeof PrepareBenchmarkDatasetsArgs>;

const LabelTransferSmokeArgs = z.object({
  outputDir: z.string().optional(),
  syntheticRoot: z.string().optional(),
  sources: z.array(z.string()).default([]),
  variants: z.array(z.string()).default([]),
  methods: z.array(z.string()).default([]),
  maxReferenceCells: z.number().int().positive().default(500),
  maxQueryCells: z.number().int().positive().default(200),
  minSharedGenes: z.number().int().positive().default(50),
  k: z.number().int().positive().default(15),
  device: z.string().default(''),
  batchSize: z.number().int().positive().default(16),
  seed: z.number().int().default(3028),
  includeExistingSeaad: z.boolean().default(false),
});
type LabelTransferSmokeArgs = z.infer<typeof LabelTransferSmokeArgs>;

const EvaluateArgs = z.object({
  outputDir: z.string().optional(),
  registry: z.string().optional(),
  datasetManifest: z.string().optional(),
  noNewSynthetic: z.boolean().default(false),
  models: z.array(z.string()).default([]),
  datasets: z.array(z.string()).default([]),
  device: z.string().default(''),
  batchSize: z.number().int().positive().default(512),
  numWorkers: z.number().int().nonnegative().default(0),
  prepareOnly: z.boolean().default(false),
  maxCells: z.number().int().nonnegative().default(0),
  seed: z.number().int().default(3028),
});
type EvaluateArgs = z.infer<typeof EvaluateArgs>;

const UceLabelTransferArgs = z.object({
  datasetId: z.string().min(1),
  queryPath: z.string().min(1),
  outputDir: z.string().min(1),
  stage3Summary: z.string().optional(),
  maxQueryCells: z.number().int().nonnegative().default(0),
  maxReferenceCellsPerLabel: z.number().int().positive().default(5000),
  k: z.number().int().positive().default(25),
  minVoteShare: z.number().min(0).max(1).default(0.5),
  device: z.string().default(''),
  batchSize: z.number().int().positive().default(64),
  queryChunkSize: z.number().int().positive().default(8192),
  sampleSize: z.number().int().positive().default(1024),
  padLength: z.number().int().positive().default(1536),
  seed: z.number().int().default(3028),
  rebuildReferenceCache: z.boolean().default(false),
});
type UceLabelTransferArgs = z.infer<typeof UceLabelTransferArgs>;

export function benchmarkToolSet(opt: BridgeOptions = {}): ToolSet {
  return new ToolSet('benchmark', [
    defineTool<PrepareBenchmarkDatasetsArgs, unknown>({
      name: 'bench_prepare_datasets',
      description: 'Convert benchmark inputs to model-specific SEA-AD 140-gene NPZs.',
      parameters: PrepareBenchmarkDatasetsArgs,
      execute: async (a) =>
        runPython(
          'prepare-eval-datasets',
          [
            ['--output-dir', a.outputDir],
            ['--no-new-synthetic', a.noNewSynthetic],
            ['--max-cells', a.maxCells],
            ['--seed', a.seed],
          ],
          opt,
        ),
    }),
    defineTool<LabelTransferSmokeArgs, unknown>({
      name: 'bench_label_transfer_smoke',
      description: 'No-training label transfer over synthetic variants.',
      parameters: LabelTransferSmokeArgs,
      execute: async (a) => {
        const flags: CliFlag[] = [
          ['--output-dir', a.outputDir],
          ['--synthetic-root', a.syntheticRoot],
          ['--max-reference-cells', a.maxReferenceCells],
          ['--max-query-cells', a.maxQueryCells],
          ['--min-shared-genes', a.minSharedGenes],
          ['--k', a.k],
          ['--device', a.device],
          ['--batch-size', a.batchSize],
          ['--seed', a.seed],
          ['--include-existing-seaad', a.includeExistingSeaad],
        ];
        for (const s of a.sources) flags.push(['--source', s]);
        for (const v of a.variants) flags.push(['--variant', v]);
        for (const m of a.methods) flags.push(['--method', m]);
        return runPython('raw-label-transfer-smoke', flags, opt);
      },
    }),
    defineTool<EvaluateArgs, unknown>({
      name: 'bench_evaluate',
      description: 'Run the model registry over prepared real/synthetic datasets.',
      parameters: EvaluateArgs,
      execute: async (a) => {
        const flags: CliFlag[] = [
          ['--output-dir', a.outputDir],
          ['--registry', a.registry],
          ['--dataset-manifest', a.datasetManifest],
          ['--no-new-synthetic', a.noNewSynthetic],
          ['--device', a.device],
          ['--batch-size', a.batchSize],
          ['--num-workers', a.numWorkers],
          ['--prepare-only', a.prepareOnly],
          ['--max-cells', a.maxCells],
          ['--seed', a.seed],
        ];
        for (const m of a.models) flags.push(['--model', m]);
        for (const d of a.datasets) flags.push(['--dataset', d]);
        return runPython('evaluate', flags, opt);
      },
    }),
    defineTool<UceLabelTransferArgs, unknown>({
      name: 'bench_uce_label_transfer',
      description: 'Run UCE 33L query embeddings against the IMA embedding reference.',
      parameters: UceLabelTransferArgs,
      execute: async (a) =>
        runPython(
          'run-uce-ima-transfer',
          [
            ['--dataset-id', a.datasetId],
            ['--query-path', a.queryPath],
            ['--output-dir', a.outputDir],
            ['--stage3-summary', a.stage3Summary],
            ['--max-query-cells', a.maxQueryCells],
            ['--max-reference-cells-per-label', a.maxReferenceCellsPerLabel],
            ['--k', a.k],
            ['--min-vote-share', a.minVoteShare],
            ['--device', a.device],
            ['--batch-size', a.batchSize],
            ['--query-chunk-size', a.queryChunkSize],
            ['--sample-size', a.sampleSize],
            ['--pad-length', a.padLength],
            ['--seed', a.seed],
            ['--rebuild-reference-cache', a.rebuildReferenceCache],
          ],
          opt,
        ),
    }),
  ]);
}
