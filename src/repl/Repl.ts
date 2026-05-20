import readline from 'node:readline';
import { Agent } from '../agent/Agent.js';
import type { AgentEvent } from '../types.js';

export interface ReplOptions {
  agent: Agent;
  prompt?: string;
}

export class Repl {
  constructor(private readonly opts: ReplOptions) {}

  async start(): Promise<void> {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    const prompt = this.opts.prompt ?? `${this.opts.agent.name}> `;
    process.stdout.write(`MedrixAI REPL. Type 'exit' or Ctrl-D to quit.\n`);
    rl.setPrompt(prompt);
    rl.prompt();
    rl.on('line', async (line) => {
      const input = line.trim();
      if (!input) return rl.prompt();
      if (input === 'exit' || input === 'quit') {
        rl.close();
        return;
      }
      try {
        for await (const ev of this.opts.agent.run(input)) {
          this.renderEvent(ev);
        }
        process.stdout.write('\n');
      } catch (err) {
        process.stderr.write(`\n[error] ${(err as Error).message}\n`);
      }
      rl.prompt();
    });
    await new Promise<void>((resolve) => {
      rl.on('close', async () => {
        await this.opts.agent.close();
        resolve();
      });
    });
  }

  private renderEvent(ev: AgentEvent): void {
    switch (ev.type) {
      case 'text':
        process.stdout.write(String(ev.data ?? ''));
        break;
      case 'tool_call': {
        const c = ev.data as { name: string; arguments: unknown };
        process.stdout.write(`\n[tool_call] ${c.name} ${JSON.stringify(c.arguments)}\n`);
        break;
      }
      case 'tool_result': {
        const r = ev.data as { content: string };
        process.stdout.write(`[tool_result] ${truncate(r.content, 200)}\n`);
        break;
      }
      case 'done':
        // Already streamed.
        break;
      default:
        break;
    }
  }
}

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n)}…` : s;
}
