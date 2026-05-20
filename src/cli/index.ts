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
import { registerScmasCli } from '../plugins/scmas/cli.js';

// Load env from .env and ~/.pantheon/.env
dotenvConfig();
dotenvConfig({ path: path.join(os.homedir(), '.pantheon', '.env') });

const program = new Command();
program
  .name('pantheon-ts')
  .description('PantheonOS TypeScript CLI')
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
      name: 'pantheon',
      model: opts.model ?? defaultModel(),
      toolset,
      systemPrompt:
        'You are PantheonOS, a helpful multi-tool AI assistant. Use tools when useful.',
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
  .description('Interactive API-key setup. Writes to ~/.pantheon/.env')
  .action(async () => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    const ask = (q: string) =>
      new Promise<string>((resolve) => rl.question(q, (a) => resolve(a.trim())));

    console.log('PantheonOS setup. Leave blank to skip a key.');
    const openai = await ask('OPENAI_API_KEY: ');
    const anthropic = await ask('ANTHROPIC_API_KEY: ');
    const google = await ask('GOOGLE_API_KEY: ');
    const model = await ask(`PANTHEON_MODEL [${defaultModel()}]: `);
    const nats = await ask('NATS_URL [nats://localhost:4222]: ');
    rl.close();

    const dir = path.join(os.homedir(), '.pantheon');
    await fs.mkdir(dir, { recursive: true });
    const lines: string[] = [];
    if (openai) lines.push(`OPENAI_API_KEY=${openai}`);
    if (anthropic) lines.push(`ANTHROPIC_API_KEY=${anthropic}`);
    if (google) lines.push(`GOOGLE_API_KEY=${google}`);
    lines.push(`PANTHEON_MODEL=${model || defaultModel()}`);
    lines.push(`NATS_URL=${nats || 'nats://localhost:4222'}`);
    const envPath = path.join(dir, '.env');
    await fs.writeFile(envPath, lines.join('\n') + '\n');
    console.log(`Wrote ${envPath}`);
  });

registerScmasCli(program);

program.parseAsync(process.argv).catch((err) => {
  console.error(err);
  process.exit(1);
});
