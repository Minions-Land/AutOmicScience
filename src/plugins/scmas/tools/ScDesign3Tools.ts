import { z } from 'zod';
import { defineTool } from '../../../toolset/Tool.js';
import { ToolSet } from '../../../toolset/ToolSet.js';
import { runScmas, type BridgeOptions } from './PythonBridge.js';

const WriteScDesign3ConfigsArgs = z.object({
  preparedSourceRoot: z.string().optional(),
  configRoot: z.string().optional(),
  targetTotal: z.number().int().positive().default(20_000),
  nCores: z.number().int().positive().default(8),
  seed: z.number().int().default(3028),
});
type WriteScDesign3ConfigsArgs = z.infer<typeof WriteScDesign3ConfigsArgs>;

const PreflightScDesign3Args = z.object({ rscriptPath: z.string().default('Rscript') });
type PreflightScDesign3Args = z.infer<typeof PreflightScDesign3Args>;

const RunScDesign3Args = z.object({
  configManifest: z.string().optional(),
  rscriptPath: z.string().default('Rscript'),
  forceRefit: z.boolean().default(false),
  dryRun: z.boolean().default(false),
});
type RunScDesign3Args = z.infer<typeof RunScDesign3Args>;

/**
 * scDesign3 R-driven synthetic generation pipeline (config writing, preflight,
 * actual generation). All operations are subprocess calls into the vendored
 * Python module which in turn invokes `Rscript`.
 */
export function scDesign3ToolSet(opt: BridgeOptions = {}): ToolSet {
  return new ToolSet('scmas-scdesign3', [
    defineTool<WriteScDesign3ConfigsArgs, unknown>({
      name: 'scmas_write_scdesign3_configs',
      description: 'Write scDesign3 config JSON files for prepared sources.',
      parameters: WriteScDesign3ConfigsArgs,
      execute: async (a) =>
        runScmas(
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
    defineTool<PreflightScDesign3Args, unknown>({
      name: 'scmas_preflight_scdesign3',
      description: 'Check scDesign3 runner and R environment.',
      parameters: PreflightScDesign3Args,
      execute: async (a) =>
        runScmas('preflight-scdesign3', [['--rscript-path', a.rscriptPath]], opt),
    }),
    defineTool<RunScDesign3Args, unknown>({
      name: 'scmas_run_scdesign3',
      description: 'Run generated scDesign3 configs.',
      parameters: RunScDesign3Args,
      execute: async (a) =>
        runScmas(
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
