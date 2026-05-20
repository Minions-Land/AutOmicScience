import readline from 'node:readline';
import { Agent } from '../agent/Agent.js';
import type { Team } from '../team/Team.js';
import type { AgentEvent } from '../types.js';
import { FileSessionStore } from '../session/FileSessionStore.js';
import type { SessionStore } from '../session/SessionStore.js';
import { join } from 'path';
import { homedir } from 'os';

export interface ReplOptions {
  agent?: Agent;
  team?: Team;
  prompt?: string;
  sessionStore?: SessionStore;
  historyFile?: string;
  showToolCalls?: boolean;
}

interface SlashCommand {
  name: string;
  description: string;
  handler: (args: string) => Promise<string | void>;
}

export class Repl {
  private readonly agent: Agent | undefined;
  private readonly team: Team | undefined;
  private readonly sessionStore: SessionStore;
  private readonly historyFile: string;
  private showToolCalls: boolean;
  private chatId: string;
  private commands: Map<string, SlashCommand> = new Map();
  private running = false;
  private currentAbort: AbortController | null = null;

  constructor(private readonly opts: ReplOptions) {
    this.agent = opts.agent;
    this.team = opts.team;
    this.sessionStore = opts.sessionStore ?? new FileSessionStore(join(homedir(), '.medrix', 'sessions'));
    this.historyFile = opts.historyFile ?? join(homedir(), '.medrix', 'repl_history');
    this.showToolCalls = opts.showToolCalls ?? true;
    this.chatId = `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    this.registerBuiltinCommands();
  }

  private registerBuiltinCommands(): void {
    this.registerCommand({ name: 'help', description: 'Show available commands', handler: () => this.cmdHelp() });
    this.registerCommand({ name: 'clear', description: 'Clear conversation history', handler: () => this.cmdClear() });
    this.registerCommand({ name: 'history', description: 'Show conversation history', handler: () => this.cmdHistory() });
    this.registerCommand({ name: 'save', description: 'Save current session', handler: (a) => this.cmdSave(a) });
    this.registerCommand({ name: 'load', description: 'Load a saved session', handler: (a) => this.cmdLoad(a) });
    this.registerCommand({ name: 'sessions', description: 'List saved sessions', handler: () => this.cmdSessions() });
    this.registerCommand({ name: 'model', description: 'Show or change model', handler: (a) => this.cmdModel(a) });
    this.registerCommand({ name: 'tools', description: 'List available tools', handler: () => this.cmdTools() });
    this.registerCommand({ name: 'verbose', description: 'Toggle tool call display', handler: () => this.cmdVerbose() });
    this.registerCommand({ name: 'cancel', description: 'Cancel current operation', handler: () => this.cmdCancel() });
    this.registerCommand({ name: 'export', description: 'Export conversation to file', handler: (a) => this.cmdExport(a) });
    this.registerCommand({ name: 'new', description: 'Start a new conversation', handler: () => this.cmdNew() });
  }

  registerCommand(cmd: SlashCommand): void {
    this.commands.set(cmd.name, cmd);
  }

  async start(): Promise<void> {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: true,
    });
    const name = this.agent?.name ?? 'medrix';
    const prompt = this.opts.prompt ?? `${name}> `;
    this.running = true;

    process.stdout.write(`MedrixAI REPL v1.0\n`);
    process.stdout.write(`Type /help for commands, 'exit' or Ctrl-D to quit.\n\n`);

    // Handle Ctrl-C to cancel current operation
    process.on('SIGINT', () => {
      if (this.currentAbort) {
        this.currentAbort.abort();
        this.currentAbort = null;
        process.stdout.write('\n[cancelled]\n');
      } else {
        process.stdout.write('\nUse "exit" to quit.\n');
      }
      rl.prompt();
    });

    rl.setPrompt(prompt);
    rl.prompt();

    rl.on('line', async (line) => {
      const input = line.trim();
      if (!input) { rl.prompt(); return; }
      if (input === 'exit' || input === 'quit') { rl.close(); return; }

      if (input.startsWith('/')) {
        const [cmdName, ...rest] = input.slice(1).split(/\s+/);
        const cmd = this.commands.get(cmdName);
        if (cmd) {
          try {
            const result = await cmd.handler(rest.join(' '));
            if (result) process.stdout.write(`${result}\n`);
          } catch (err) {
            process.stderr.write(`[error] ${(err as Error).message}\n`);
          }
        } else {
          process.stdout.write(`Unknown command: /${cmdName}. Type /help for available commands.\n`);
        }
        rl.prompt();
        return;
      }

      // Run agent/team
      this.currentAbort = new AbortController();
      try {
        if (this.team) {
          for await (const ev of this.team.run(input)) {
            this.renderEvent(ev);
          }
        } else if (this.agent) {
          for await (const ev of this.agent.run(input)) {
            this.renderEvent(ev);
          }
        }
        process.stdout.write('\n');
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          process.stderr.write(`\n[error] ${err.message}\n`);
        }
      }
      this.currentAbort = null;
      rl.prompt();
    });

    await new Promise<void>((resolve) => {
      rl.on('close', async () => {
        this.running = false;
        if (this.agent) await this.agent.close();
        resolve();
      });
    });
  }

  private renderEvent(ev: AgentEvent): void {
    switch (ev.type) {
      case 'text':
        process.stdout.write(String(ev.data ?? ''));
        break;
      case 'tool_call':
        if (this.showToolCalls) {
          const c = ev.data as { name: string; arguments: unknown };
          process.stdout.write(`\n  [tool] ${c.name}(${truncate(JSON.stringify(c.arguments), 100)})\n`);
        }
        break;
      case 'tool_result':
        if (this.showToolCalls) {
          const r = ev.data as { content: string };
          process.stdout.write(`  [result] ${truncate(r.content, 150)}\n`);
        }
        break;
      case 'agent_start': {
        const d = ev.data as { name?: string };
        if (d?.name) process.stdout.write(`\n[${d.name}] `);
        break;
      }
      case 'agent_done':
        break;
      case 'plan':
        process.stdout.write(`\n[plan]\n${ev.data}\n`);
        break;
      case 'done':
        break;
      default:
        break;
    }
  }

  // --- Slash Commands ---

  private async cmdHelp(): Promise<string> {
    const lines = ['Available commands:'];
    for (const [name, cmd] of this.commands) {
      lines.push(`  /${name.padEnd(12)} ${cmd.description}`);
    }
    return lines.join('\n');
  }

  private async cmdClear(): Promise<string> {
    if (this.agent) (this.agent as any).messages = [];
    return 'Conversation cleared.';
  }

  private async cmdHistory(): Promise<string> {
    if (!this.agent) return 'No agent attached.';
    const msgs = (this.agent as any).messages ?? [];
    if (msgs.length === 0) return 'No messages in history.';
    return msgs.map((m: any, i: number) => `${i + 1}. [${m.role}] ${truncate(m.content, 80)}`).join('\n');
  }

  private async cmdSave(name: string): Promise<string> {
    const sessionName = name || `session_${Date.now()}`;
    const messages = this.agent ? (this.agent as any).messages ?? [] : [];
    await this.sessionStore.save(sessionName, { chatId: this.chatId, messages, savedAt: new Date().toISOString() });
    return `Session saved as "${sessionName}".`;
  }

  private async cmdLoad(name: string): Promise<string> {
    if (!name) return 'Usage: /load <session-name>';
    const session = await this.sessionStore.load(name);
    if (!session) return `Session "${name}" not found.`;
    if (this.agent && session.messages) {
      (this.agent as any).messages = session.messages;
    }
    this.chatId = session.chatId ?? this.chatId;
    return `Session "${name}" loaded (${session.messages?.length ?? 0} messages).`;
  }

  private async cmdSessions(): Promise<string> {
    const sessions = await this.sessionStore.list();
    if (sessions.length === 0) return 'No saved sessions.';
    return sessions.map((s, i) => `${i + 1}. ${s}`).join('\n');
  }

  private async cmdModel(args: string): Promise<string> {
    if (!this.agent) return 'No agent attached.';
    if (!args) return `Current model: ${(this.agent as any).model ?? 'unknown'}`;
    (this.agent as any).model = args;
    return `Model changed to: ${args}`;
  }

  private async cmdTools(): Promise<string> {
    if (!this.agent) return 'No agent attached.';
    const tools = (this.agent as any).toolSet?.tools ?? [];
    if (tools.length === 0) return 'No tools registered.';
    return tools.map((t: any) => `  ${t.name}: ${t.description ?? ''}`).join('\n');
  }

  private async cmdVerbose(): Promise<string> {
    this.showToolCalls = !this.showToolCalls;
    return `Tool call display: ${this.showToolCalls ? 'ON' : 'OFF'}`;
  }

  private async cmdCancel(): Promise<string> {
    if (this.currentAbort) {
      this.currentAbort.abort();
      this.currentAbort = null;
      return 'Operation cancelled.';
    }
    return 'Nothing to cancel.';
  }

  private async cmdExport(filename: string): Promise<string> {
    const { writeFileSync } = await import('fs');
    const target = filename || `medrix_export_${Date.now()}.md`;
    const messages = this.agent ? (this.agent as any).messages ?? [] : [];
    const md = messages.map((m: any) => `## ${m.role}\n\n${m.content}\n`).join('\n---\n\n');
    writeFileSync(target, md, 'utf-8');
    return `Exported ${messages.length} messages to ${target}`;
  }

  private async cmdNew(): Promise<string> {
    if (this.agent) (this.agent as any).messages = [];
    this.chatId = `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    return 'New conversation started.';
  }
}

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n)}…` : s;
}
