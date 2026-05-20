import type { Command } from 'commander';
import { runScmas, type BridgeOptions } from './tools/PythonBridge.js';

/**
 * Adds an `scmas` subcommand group to the top-level pantheon-ts CLI.
 *
 * Each subcommand is a thin shell over `python -m scmas <name> ...`. The
 * Python implementation is the source of truth; this surface exists so
 * pantheon-ts users can run the pipeline without context-switching into
 * the Python project.
 */
export function registerScmasCli(program: Command): void {
  const scmas = program
    .command('scmas')
    .description('scMAS multi-stage single-cell annotation (vendored Python).');

  // Helper: passthrough a known scmas subcommand with the user's argv tail.
  const passthrough = (name: string, summary: string) => {
    scmas
      .command(`${name} [args...]`)
      .description(`${summary} (passthrough to \`python -m scmas ${name}\`)`)
      .allowUnknownOption(true)
      .action(async (args: string[] = []) => {
        const result = await runScmas(name, args.map((a) => a as string), {} as BridgeOptions);
        if (result.stdout) process.stdout.write(result.stdout);
        if (result.stderr) process.stderr.write(result.stderr);
        process.exit(result.exitCode);
      });
  };

  passthrough('build-label-maps', 'Build SEA-AD label/gene maps');
  passthrough('build-reference', 'Build merged reference + SEA-AD test h5ad');
  passthrough('build-seaad-test', 'Build SEA-AD held-out donor test h5ad');
  passthrough('prepare-sources', 'Prepare scDesign3 source bundles');
  passthrough('build-dataset-catalog', 'Write dataset role/source catalog');
  passthrough('write-scdesign3-configs', 'Write scDesign3 config JSONs');
  passthrough('preflight-scdesign3', 'Check scDesign3 R environment');
  passthrough('run-scdesign3', 'Run scDesign3 generation configs');
  passthrough('prepare-eval-datasets', 'Prepare model-specific eval NPZs');
  passthrough('evaluate', 'Run Stage-1 evaluation');
  passthrough('raw-label-transfer-smoke', 'No-training label-transfer smoke test');
  passthrough('profile-query', 'Stage-2 prerequisite query profile');
  passthrough('select-models', 'Stage-2 LLM-driven (source, model) selection');
  passthrough('run-cross-species-plan', 'Subset cross-species execution');
  passthrough('inspect-model-contracts', 'Inspect capability cards / registry');
  passthrough('adapt-and-execute', 'Stage-3 LLM adapter spec + execution');
  passthrough('run-consensus', 'Stage-4 consensus + optional adjudication');
  passthrough('run-uce-ima-transfer', 'UCE 33L IMA label transfer');
}
