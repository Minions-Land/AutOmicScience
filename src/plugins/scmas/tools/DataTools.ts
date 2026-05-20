import { z } from 'zod';
import { defineTool } from '../../../toolset/Tool.js';
import { ToolSet } from '../../../toolset/ToolSet.js';
import { runScmas, type BridgeOptions } from './PythonBridge.js';

type Flag = string | [string, string | number | boolean | null | undefined];

const BuildLabelMapsArgs = z.object({ force: z.boolean().default(false) });
type BuildLabelMapsArgs = z.infer<typeof BuildLabelMapsArgs>;

const BuildReferenceArgs = z.object({
  maxCellsPerSource: z.number().int().positive().default(10_000),
  seed: z.number().int().default(3028),
  maxCellsSeaadReference: z.number().int().positive().default(100_000),
  maxCellsSeaadTest: z.number().int().positive().default(50_000),
  includeSmartseq: z.boolean().default(false),
  output: z.string().optional(),
});
type BuildReferenceArgs = z.infer<typeof BuildReferenceArgs>;

const BuildSeaadTestArgs = z.object({
  maxCells: z.number().int().positive().default(50_000),
  seed: z.number().int().default(3028),
  output: z.string().optional(),
});
type BuildSeaadTestArgs = z.infer<typeof BuildSeaadTestArgs>;

const PrepareSourcesArgs = z.object({
  maxCellsPerSource: z.number().int().positive().default(10_000),
  seed: z.number().int().default(3028),
  maxGenesPerSource: z.number().int().nonnegative().default(0),
  includeSmartseq: z.boolean().default(false),
  includeSeaadReference: z.boolean().default(false),
  sources: z.array(z.string()).default([]),
  outputRoot: z.string().optional(),
});
type PrepareSourcesArgs = z.infer<typeof PrepareSourcesArgs>;

const BuildDatasetCatalogArgs = z.object({
  outputDir: z.string().optional(),
  noShapeProbe: z.boolean().default(false),
});
type BuildDatasetCatalogArgs = z.infer<typeof BuildDatasetCatalogArgs>;

/**
 * Wrappers around the deterministic data-prep subcommands of `python -m scmas ...`.
 */
export function dataToolSet(opt: BridgeOptions = {}): ToolSet {
  return new ToolSet('scmas-data', [
    defineTool<BuildLabelMapsArgs, unknown>({
      name: 'scmas_build_label_maps',
      description: 'Build SEA-AD label/gene maps (deterministic).',
      parameters: BuildLabelMapsArgs,
      execute: async ({ force }) => runScmas('build-label-maps', [['--force', force]], opt),
    }),
    defineTool<BuildReferenceArgs, unknown>({
      name: 'scmas_build_reference',
      description: 'Build merged human/mouse reference h5ad and SEA-AD real test h5ad.',
      parameters: BuildReferenceArgs,
      execute: async (a) =>
        runScmas(
          'build-reference',
          [
            ['--max-cells-per-source', a.maxCellsPerSource],
            ['--seed', a.seed],
            ['--max-cells-seaad-reference', a.maxCellsSeaadReference],
            ['--max-cells-seaad-test', a.maxCellsSeaadTest],
            ['--include-smartseq', a.includeSmartseq],
            ['--output', a.output],
          ],
          opt,
        ),
    }),
    defineTool<BuildSeaadTestArgs, unknown>({
      name: 'scmas_build_seaad_test',
      description: 'Build SEA-AD MERFISH held-out donor test h5ad only.',
      parameters: BuildSeaadTestArgs,
      execute: async (a) =>
        runScmas(
          'build-seaad-test',
          [
            ['--max-cells', a.maxCells],
            ['--seed', a.seed],
            ['--output', a.output],
          ],
          opt,
        ),
    }),
    defineTool<PrepareSourcesArgs, unknown>({
      name: 'scmas_prepare_sources',
      description: 'Prepare standard bundles used as scDesign3 source data.',
      parameters: PrepareSourcesArgs,
      execute: async (a) => {
        const flags: Flag[] = [
          ['--max-cells-per-source', a.maxCellsPerSource],
          ['--seed', a.seed],
          ['--max-genes-per-source', a.maxGenesPerSource],
          ['--include-smartseq', a.includeSmartseq],
          ['--include-seaad-reference', a.includeSeaadReference],
          ['--output-root', a.outputRoot],
        ];
        for (const s of a.sources) flags.push(['--source', s]);
        return runScmas('prepare-sources', flags, opt);
      },
    }),
    defineTool<BuildDatasetCatalogArgs, unknown>({
      name: 'scmas_build_dataset_catalog',
      description: 'Write a dataset role/source table for stage-1 planning and smoke runs.',
      parameters: BuildDatasetCatalogArgs,
      execute: async (a) =>
        runScmas(
          'build-dataset-catalog',
          [
            ['--output-dir', a.outputDir],
            ['--no-shape-probe', a.noShapeProbe],
          ],
          opt,
        ),
    }),
  ]);
}
