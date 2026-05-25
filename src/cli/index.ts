#!/usr/bin/env node
import { Command } from 'commander';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { config as dotenvConfig } from 'dotenv';
import { Agent } from '../agent/Agent.js';
import { Repl } from '../repl/Repl.js';
import { defaultModel } from '../provider/ModelSelector.js';
import { createDefaultToolSet } from '../toolset/BuiltinToolSets.js';
import { ToolSet } from '../toolset/ToolSet.js';
import { FilePermissionStore, PermissionManager } from '../permissions/index.js';
import type { PermissionMode } from '../permissions/index.js';
import { PluginLoader } from '../plugin/index.js';
import { CommandRegistry, FileCommandLoader } from '../commands/index.js';
import { FileTaskManager } from '../task/index.js';
import { runStructuredAgent, startStructuredIO } from './structuredIO.js';
import { runPython } from '../bridge/PythonBridge.js';
import { DevServer } from '../ui/Server.js';
import { LocalStore } from '../store/LocalStore.js';
import { SetupWizard } from './SetupWizard.js';
import { AOS_SYSTEM_PROMPT } from '../agent/prompts/AOSSystemPrompt.js';

dotenvConfig();
dotenvConfig({ path: path.join(os.homedir(), '.aos', '.env'), override: false });

const program = new Command();
program
  .name('aos')
  .description('AutOmicScience CLI')
  .version('0.1.0');

program
  .command('cli')
  .description('Start an interactive REPL with a default AOS agent.')
  .option('-m, --model <model>', 'Model id, for example gpt-5.5, gpt-5.4, gemini-2.5-flash, or anthropic/<provider-model-id>')
  .option('--no-tools', 'Disable built-in toolsets')
  .option('--permission-mode <mode>', 'Permission mode: default, plan, auto, bypassPermissions', 'default')
  .option('--plugin <nameOrPath...>', 'Load one or more local plugins')
  .action(async (opts) => {
    const permissionStore = new FilePermissionStore();
    const permissionManager = await permissionStore.createManager({ mode: opts.permissionMode as PermissionMode });
    const taskManager = new FileTaskManager();
    const toolset = opts.tools !== false
      ? createDefaultToolSet({ rootDir: process.cwd(), permissionManager, taskManager })
      : new ToolSet('empty');
    const commands = new CommandRegistry();
    await loadCommandFiles(commands);
    const agent = new Agent({
      name: 'aos',
      model: opts.model ?? defaultModel(),
      toolset,
      systemPrompt: AOS_SYSTEM_PROMPT,
    });
    if (opts.plugin?.length) {
      const loader = new PluginLoader([
        path.join(os.homedir(), '.aos', 'plugins'),
        path.join(process.cwd(), 'plugins'),
      ]);
      for (const target of opts.plugin) {
        const plugin = await loader.load(target);
        agent.addPlugin(plugin);
        for (const command of plugin.commands) commands.register({ ...command, source: plugin.manifest.name });
      }
    }
    await new Repl({ agent, commands }).start();
    await permissionStore.persistManager(permissionManager);
  });

program
  .command('run <input>')
  .description('Run AutOmicScience once and print either final text or NDJSON events.')
  .option('-m, --model <model>', 'Model id')
  .option('--json', 'Print NDJSON AgentEvents instead of final text')
  .option('--max-iterations <n>', 'Maximum agent tool-use iterations', '8')
  .option('--permission-mode <mode>', 'Permission mode: default, plan, auto, bypassPermissions', 'default')
  .action(async (input: string, opts) => {
    const permissionStore = new FilePermissionStore();
    const permissionManager = await permissionStore.createManager({ mode: opts.permissionMode as PermissionMode });
    const taskManager = new FileTaskManager();
    const agent = new Agent({
      name: 'aos',
      model: opts.model ?? defaultModel(),
      toolset: createDefaultToolSet({ rootDir: process.cwd(), permissionManager, taskManager }),
      projectInstructions: { cwd: process.cwd() },
      systemPrompt: AOS_SYSTEM_PROMPT,
      maxIterations: parseInt(opts.maxIterations, 10) || 8,
    });
    if (opts.json) await runStructuredAgent(agent, input);
    else console.log(await agent.runToText(input));
    await permissionStore.persistManager(permissionManager);
  });

program
  .command('stdio')
  .description('Run AutOmicScience as an NDJSON structured IO process.')
  .option('-m, --model <model>', 'Model id')
  .option('--permission-mode <mode>', 'Permission mode: default, plan, auto, bypassPermissions', 'default')
  .action(async (opts) => {
    const permissionStore = new FilePermissionStore();
    const permissionManager = await permissionStore.createManager({ mode: opts.permissionMode as PermissionMode });
    const taskManager = new FileTaskManager();
    const agent = new Agent({
      name: 'aos',
      model: opts.model ?? defaultModel(),
      toolset: createDefaultToolSet({ rootDir: process.cwd(), permissionManager, taskManager }),
      projectInstructions: { cwd: process.cwd() },
      systemPrompt: AOS_SYSTEM_PROMPT,
    });
    await startStructuredIO(agent);
    await permissionStore.persistManager(permissionManager);
  });

program
  .command('serve')
  .description('Start the AutOmicScience UI server.')
  .option('-p, --port <port>', 'Port to listen on', '3000')
  .option('--no-aos-compat', 'Disable AutOmicScience frontend-compatible NATS/RPC service')
  .option('--aos-id-hash <id>', 'Stable id hash for AutOmicScience frontend auto-connect', 'automic-science')
  .option('--aos-data-dir <dir>', 'Data directory for AOS-compatible chats, installs, and NATS state')
  .action(async (opts) => {
    const port = parseInt(opts.port, 10);
    const server = new DevServer({
      rootDir: process.cwd(),
      enableAOSCompat: opts.aosCompat !== false,
      aosServiceIdHash: opts.aosIdHash,
      aosCompatDataDir: opts.aosDataDir,
    });
    await server.start(port);
    console.log(`AutOmicScience UI server running on http://localhost:${port}`);
    process.on('SIGINT', async () => {
      await server.stop();
      process.exit(0);
    });
  });

program
  .command('setup')
  .description('Interactive API key setup. Writes to ~/.aos/.env')
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

const store = program
  .command('store')
  .description('Manage the AutOmicScience package store.');

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
      console.log(`${r.id} (${r.category}) v${r.version} - ${r.description}`);
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
  .description('Publish an entry. Pass a JSON string.')
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
      console.log(`${e.id} (${e.category}) v${e.version} - ${e.description}`);
    }
  });

program
  .command('evolve <config>')
  .description('Run evolutionary optimization from a config JSON file.')
  .action(async (configPath: string) => {
    const raw = await fs.readFile(configPath, 'utf-8');
    const config = JSON.parse(raw);
    console.log('Evolution config loaded:', config);
    console.log('Evolution execution is not wired in this CLI yet; use the Evolver API directly.');
  });

const annotate = program
  .command('annotate')
  .description('Built-in single-cell annotation pipeline. Delegates to the bridge runtime.');

const ANNOTATE_SUBCOMMANDS: { name: string; summary: string }[] = [
  { name: 'build-label-maps', summary: 'Build SEA-AD label/gene maps' },
  { name: 'build-reference', summary: 'Build merged reference plus SEA-AD test h5ad' },
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
  { name: 'select-models', summary: 'LLM-driven source/model selection' },
  { name: 'run-cross-species-plan', summary: 'Subset cross-species execution' },
  { name: 'inspect-model-contracts', summary: 'Inspect capability cards and registry' },
  { name: 'adapt-and-execute', summary: 'LLM adapter spec plus whitelist execution' },
  { name: 'run-consensus', summary: 'Consensus plus optional adjudication' },
  { name: 'run-uce-ima-transfer', summary: 'UCE 33L IMA label transfer' },
  { name: 'bio-mas-preflight', summary: 'Inspect bio MAS dependencies, data, checkpoints, R, and tiny-demo assets' },
  { name: 'create-tiny-bio-demo', summary: 'Create clearly marked synthetic tiny data for local smoke tests' },
  { name: 'run-tiny-bio-mas-demo', summary: 'Run profile, select, and execute on synthetic tiny demo data' },
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

async function loadCommandFiles(commands: CommandRegistry): Promise<void> {
  const loader = new FileCommandLoader([
    path.join(os.homedir(), '.aos', 'commands'),
    path.join(process.cwd(), '.aos', 'commands'),
    path.join(process.cwd(), 'commands'),
  ]);
  for (const command of await loader.loadAll()) commands.register(command);
}
