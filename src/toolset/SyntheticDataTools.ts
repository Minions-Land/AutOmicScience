import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';
import { runPython, type BridgeOptions } from '../bridge/PythonBridge.js';

/**
 * scDesign3 R-driven synthetic data generation: config writing, preflight
 * checks, and execution. All operations go through the Python bridge,
 * which in turn invokes `Rscript`.
 */

const WriteConfigsArgs = z.object({
  preparedSourceRoot: z.string().optional(),
  configRoot: z.string().optional(),
  targetTotal: z.number().int().positive().default(20_000),
  nCores: z.number().int().positive().default(8),
  seed: z.number().int().default(3028),
});
type WriteConfigsArgs = z.infer<typeof WriteConfigsArgs>;

const PreflightArgs = z.object({ rscriptPath: z.string().default('Rscript') });
type PreflightArgs = z.infer<typeof PreflightArgs>;

const GenerateArgs = z.object({
  configManifest: z.string().optional(),
  rscriptPath: z.string().default('Rscript'),
  forceRefit: z.boolean().default(false),
  dryRun: z.boolean().default(false),
});
type GenerateArgs = z.infer<typeof GenerateArgs>;

export function syntheticDataToolSet(opt: BridgeOptions = {}): ToolSet {
  return new ToolSet('synthetic-data', [
    defineTool<WriteConfigsArgs, unknown>({
      name: 'synth_write_configs',
      description: 'Write synthetic-generation config JSONs for prepared sources.',
      parameters: WriteConfigsArgs,
      execute: async (a) =>
        runPython(
          'write-scdesign3-configs',
          [
            ['--prepared-source-root', a.preparedSourceRoot],
            ['--config-root', a.configRoot],
            ['--target-total', a.targetTotal],
            ['--n-cores', a.nCores],
            ['--seed', a.seed],
          ],
          opt,
        ),
    }),
    defineTool<PreflightArgs, unknown>({
      name: 'synth_preflight',
      description: 'Check the synthetic-generation R runner and environment.',
      parameters: PreflightArgs,
      execute: async (a) =>
        runPython('preflight-scdesign3', [['--rscript-path', a.rscriptPath]], opt),
    }),
    defineTool<GenerateArgs, unknown>({
      name: 'synth_generate',
      description: 'Run all generated synthetic-generation configs.',
      parameters: GenerateArgs,
      execute: async (a) =>
        runPython(
          'run-scdesign3',
          [
            ['--config-manifest', a.configManifest],
            ['--rscript-path', a.rscriptPath],
            ['--force-refit', a.forceRefit],
            ['--dry-run', a.dryRun],
          ],
          opt,
        ),
    }),
  ]);
}
