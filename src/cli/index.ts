#!/usr/bin/env node
import { Command } from 'commander';
import readline from 'node:readline';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { config as dotenvConfig } from 'dotenv';
import { Agent } from '../agent/Agent.js';
import { Repl } from '../repl/Repl.js';
import { defaultModel } from '../provider/ModelSelector.js';
import { fileToolSet, shellToolSet, webToolSet } from '../toolset/BuiltinToolSets.js';
import { ToolSet } from '../toolset/ToolSet.js';
import { runPython } from '../bridge/PythonBridge.js';

// Load env from .env and ~/.novaeve/.env
dotenvConfig();
dotenvConfig({ path: path.join(os.homedir(), '.novaeve', '.env') });

const program = new Command();
program
  .name('novaeve')
  .description('Novaeve-Agent CLI')
  .version('0.1.0');

program
  .command('cli')
  .description('Start an interactive REPL with a default agent.')
  .option('-m, --model <model>', 'Model id (e.g. gpt-4o-mini, claude-3-5-sonnet-latest)')
  .option('--no-tools', 'Disable built-in toolsets')
  .action(async (opts) => {
    const toolset = opts.tools !== false
      ? new ToolSet('builtin').merge(fileToolSet()).merge(shellToolSet()).merge(webToolSet())
      : new ToolSet('empty');
    const agent = new Agent({
      name: 'novaeve',
      model: opts.model ?? defaultModel(),
      toolset,
      systemPrompt:
        'You are Novaeve, a helpful multi-tool AI assistant. Use tools when useful.',
    });
    await new Repl({ agent }).start();
  });

program
  .command('ui')
  .description('Launch the UI (not yet implemented).')
  .action(() => {
    console.log('UI not yet implemented, use cli');
  });

program
  .command('setup')
  .description('Interactive API-key setup. Writes to ~/.novaeve/.env')
  .action(async () => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    const ask = (q: string) =>
      new Promise<string>((resolve) => rl.question(q, (a) => resolve(a.trim())));

    console.log('Novaeve setup. Leave blank to skip a key.');
    const openai = await ask('OPENAI_API_KEY: ');
    const anthropic = await ask('ANTHROPIC_API_KEY: ');
    const google = await ask('GOOGLE_API_KEY: ');
    const model = await ask(`NOVAEVE_MODEL [${defaultModel()}]: `);
    const nats = await ask('NATS_URL [nats://localhost:4222]: ');
    rl.close();

    const dir = path.join(os.homedir(), '.novaeve');
    await fs.mkdir(dir, { recursive: true });
    const lines: string[] = [];
    if (openai) lines.push(`OPENAI_API_KEY=${openai}`);
    if (anthropic) lines.push(`ANTHROPIC_API_KEY=${anthropic}`);
    if (google) lines.push(`GOOGLE_API_KEY=${google}`);
    lines.push(`NOVAEVE_MODEL=${model || defaultModel()}`);
    lines.push(`NATS_URL=${nats || 'nats://localhost:4222'}`);
    const envPath = path.join(dir, '.env');
    await fs.writeFile(envPath, lines.join('\n') + '\n');
    console.log(`Wrote ${envPath}`);
  });

const annotate = program
  .command('annotate')
  .description('Built-in single-cell annotation pipeline (delegates to the vendored bridge).');

const ANNOTATE_SUBCOMMANDS: { name: string; summary: string }[] = [
  { name: 'build-label-maps', summary: 'Build SEA-AD label/gene maps' },
  { name: 'build-reference', summary: 'Build merged reference + SEA-AD test h5ad' },
  { name: 'build-seaad-test', summary: 'Build SEA-AD held-out donor test h5ad' },
  { name: 'prepare-sources', summary: 'Prepare reference source bundles' },
  { name: 'build-dataset-catalog', summary: 'Write dataset role/source catalog' },
  { name: 'write-scdesign3-configs', summary: 'Write synthetic-generation configs' },
  { name: 'preflight-scdesign3', summary: 'Check synthetic-generation R environment' },
  { name: 'run-scdesign3', summary: 'Run synthetic generation configs' },
  { name: 'prepare-eval-datasets', summary: 'Prepare model-specific benchmark NPZs' },
  { name: 'evaluate', summary: 'Run capability evaluation across the registry' },
  { name: 'raw-label-transfer-smoke', summary: 'No-training label-transfer smoke test' },
  { name: 'profile-query', summary: 'Build a query profile for selection' },
  { name: 'select-models', summary: 'LLM-driven (source, model) selection' },
  { name: 'run-cross-species-plan', summary: 'Subset cross-species execution' },
  { name: 'inspect-model-contracts', summary: 'Inspect capability cards / registry' },
  { name: 'adapt-and-execute', summary: 'LLM adapter spec + whitelist execution' },
  { name: 'run-consensus', summary: 'Consensus + optional adjudication' },
  { name: 'run-uce-ima-transfer', summary: 'UCE 33L IMA label transfer' },
];

for (const { name, summary } of ANNOTATE_SUBCOMMANDS) {
  annotate
    .command(`${name} [args...]`)
    .description(summary)
    .allowUnknownOption(true)
    .action(async (args: string[] = []) => {
      const result = await runPython(name, args);
      if (result.stdout) process.stdout.write(result.stdout);
      if (result.stderr) process.stderr.write(result.stderr);
      process.exit(result.exitCode);
    });
}

program.parseAsync(process.argv).catch((err) => {
  console.error(err);
  process.exit(1);
});
