#!/usr/bin/env node
import { Command } from 'commander';
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
import { DevServer } from '../ui/Server.js';
import { LocalStore } from '../store/LocalStore.js';
import { SetupWizard } from './SetupWizard.js';

// Load env from .env and ~/.medrix/.env
dotenvConfig();
dotenvConfig({ path: path.join(os.homedir(), '.medrix', '.env') });

const program = new Command();
program
  .name('medrix')
  .description('MedrixAI CLI')
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
      name: 'medrix',
      model: opts.model ?? defaultModel(),
      toolset,
      systemPrompt:
        'You are MedrixAI, a helpful multi-tool AI assistant. Use tools when useful.',
    });
    await new Repl({ agent }).start();
  });

program
  .command('serve')
  .description('Start the MedrixAI UI server.')
  .option('-p, --port <port>', 'Port to listen on', '3000')
  .action(async (opts) => {
    const port = parseInt(opts.port, 10);
    const server = new DevServer();
    await server.start(port);
    console.log(`MedrixAI UI server running on http://localhost:${port}`);
    process.on('SIGINT', async () => {
      await server.stop();
      process.exit(0);
    });
  });

program
  .command('setup')
  .description('Interactive API-key setup. Writes to ~/.medrix/.env')
  .option('--no-validate', 'Skip live API key validation')
  .option('--non-interactive', 'Use existing env without prompting')
  .action(async (opts) => {
    const wizard = new SetupWizard({
      skipValidation: opts.validate === false,
      nonInteractive: !!opts.nonInteractive,
    });
    const result = await wizard.run();
    console.log(`Configured providers: ${result.providers.join(', ') || '(none)'}`);
    console.log(`Default model: ${result.defaultModel}`);
  });

// --- Store commands ---
const store = program
  .command('store')
  .description('Manage the Novaeve package store.');

store
  .command('search <query>')
  .description('Search the store for agents, skills, tools, or teams.')
  .action(async (query: string) => {
    const s = new LocalStore();
    const results = await s.search(query);
    if (results.length === 0) {
      console.log('No results found.');
      return;
    }
    for (const r of results) {
      console.log(`${r.id} (${r.category}) v${r.version} — ${r.description}`);
    }
  });

store
  .command('install <id>')
  .description('Install a store entry by id.')
  .action(async (id: string) => {
    const s = new LocalStore();
    await s.install(id);
    console.log(`Installed: ${id}`);
  });

store
  .command('publish <json>')
  .description('Publish an entry (pass JSON string).')
  .action(async (json: string) => {
    const s = new LocalStore();
    const entry = JSON.parse(json);
    await s.publish(entry);
    console.log(`Published: ${entry.id}`);
  });

store
  .command('list')
  .description('List all store entries.')
  .option('-c, --category <category>', 'Filter by category')
  .action(async (opts) => {
    const s = new LocalStore();
    const entries = await s.list(opts.category);
    if (entries.length === 0) {
      console.log('Store is empty.');
      return;
    }
    for (const e of entries) {
      console.log(`${e.id} (${e.category}) v${e.version} — ${e.description}`);
    }
  });

// --- Evolve command ---
program
  .command('evolve <config>')
  .description('Run evolutionary optimization from a config JSON file.')
  .action(async (configPath: string) => {
    const raw = await fs.readFile(configPath, 'utf-8');
    const config = JSON.parse(raw);
    console.log('Evolution config loaded:', config);
    console.log('Evolution execution not yet wired — use the Evolver API directly.');
  });

// --- Annotate commands ---
const annotate = program
  .command('annotate')
  .description('Built-in single-cell annotation pipeline (delegates to the bridge runtime).');

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
