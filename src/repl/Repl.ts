import { promises as fs } from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { homedir } from 'node:os';
import { Agent } from '../agent/Agent.js';
import type { Skill } from '../skill/Skill.js';
import { builtinSkillLoader } from '../skill/BuiltinSkills.js';
import { FileSkillLoader } from '../skill/SkillLoader.js';
import type { Team } from '../team/Team.js';
import type { AgentEvent, Message } from '../types.js';
import { FileSessionStore } from '../session/FileSessionStore.js';
import type { SessionStore } from '../session/SessionStore.js';
import { CommandRegistry } from '../commands/index.js';
import { PluginLoader } from '../plugin/index.js';

export interface ReplOptions {
  agent?: Agent;
  team?: Team;
  prompt?: string;
  sessionStore?: SessionStore;
  historyFile?: string;
  showToolCalls?: boolean;
  commands?: CommandRegistry;
}

export class Repl {
  private readonly agent: Agent | undefined;
  private readonly team: Team | undefined;
  private readonly sessionStore: SessionStore;
  private readonly historyFile: string;
  private showToolCalls: boolean;
  private chatId: string;
  private commands: CommandRegistry;
  private running = false;
  private currentAbort: AbortController | null = null;
  private lastUserInput: string | null = null;
  private lastAssistantOutput = '';
  private pluginLoader: PluginLoader;

  constructor(private readonly opts: ReplOptions) {
    this.agent = opts.agent;
    this.team = opts.team;
    this.sessionStore = opts.sessionStore ?? new FileSessionStore(path.join(homedir(), '.aos', 'sessions'));
    this.historyFile = opts.historyFile ?? path.join(homedir(), '.aos', 'repl_history');
    this.showToolCalls = opts.showToolCalls ?? true;
    this.chatId = `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    this.commands = opts.commands ?? new CommandRegistry();
    this.pluginLoader = new PluginLoader([
      path.join(homedir(), '.aos', 'plugins'),
      path.join(process.cwd(), 'plugins'),
    ]);
    this.registerBuiltinCommands();
  }

  private registerBuiltinCommands(): void {
    this.registerCommand({ name: 'help', description: 'Show available commands', handler: () => this.cmdHelp() });
    this.registerCommand({ name: 'clear', description: 'Clear conversation history', handler: () => this.cmdClear() });
    this.registerCommand({ name: 'history', description: 'Show conversation history', handler: ({ args }) => this.cmdHistory(args) });
    this.registerCommand({ name: 'save', description: 'Save current session', handler: ({ args }) => this.cmdSave(args) });
    this.registerCommand({ name: 'load', description: 'Load a saved session', handler: ({ args }) => this.cmdLoad(args) });
    this.registerCommand({ name: 'sessions', description: 'List saved sessions', handler: () => this.cmdSessions() });
    this.registerCommand({ name: 'delete-session', description: 'Delete a saved session', handler: ({ args }) => this.cmdDeleteSession(args) });
    this.registerCommand({ name: 'model', description: 'Show or change model chain', handler: ({ args }) => this.cmdModel(args) });
    this.registerCommand({ name: 'tools', description: 'List available tools', handler: () => this.cmdTools() });
    this.registerCommand({ name: 'skills', description: 'List/load/remove active skills', handler: ({ args }) => this.cmdSkills(args) });
    this.registerCommand({ name: 'plugins', description: 'List/load local plugins', handler: ({ args }) => this.cmdPlugins(args) });
    this.registerCommand({ name: 'memory', description: 'Remember or recall key/value memory', handler: ({ args }) => this.cmdMemory(args) });
    this.registerCommand({ name: 'context', description: 'Show context and token estimate', handler: ({ args }) => this.cmdContext(args) });
    this.registerCommand({ name: 'compact', description: 'Compact conversation history', handler: ({ args }) => this.cmdCompact(args) });
    this.registerCommand({ name: 'rewind', description: 'Remove recent turns from history', handler: ({ args }) => this.cmdRewind(args) });
    this.registerCommand({ name: 'retry', description: 'Retry the last user prompt', handler: () => this.cmdRetry() });
    this.registerCommand({ name: 'status', description: 'Show agent and REPL status', handler: () => this.cmdStatus() });
    this.registerCommand({ name: 'verbose', description: 'Toggle tool call display', handler: () => this.cmdVerbose() });
    this.registerCommand({ name: 'cancel', description: 'Cancel current operation', handler: () => this.cmdCancel() });
    this.registerCommand({ name: 'export', description: 'Export conversation to a markdown/json file', handler: ({ args }) => this.cmdExport(args) });
    this.registerCommand({ name: 'export-session', description: 'Export a saved session as markdown/jsonl/bundle', handler: ({ args }) => this.cmdExportSession(args) });
    this.registerCommand({ name: 'import-session', description: 'Import a session bundle', handler: ({ args }) => this.cmdImportSession(args) });
    this.registerCommand({ name: 'new', description: 'Start a new conversation', handler: () => this.cmdNew() });
  }

  registerCommand(cmd: Parameters<CommandRegistry['register']>[0]): void {
    this.commands.register(cmd);
  }

  async start(): Promise<void> {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: true,
      historySize: 1000,
    });
    const name = this.agent?.name ?? 'aos';
    const prompt = this.opts.prompt ?? `${name}> `;
    this.running = true;

    await this.loadLineHistory(rl).catch(() => {});

    process.stdout.write('AutOmicScience REPL v1.0\n');
    process.stdout.write("Type /help for commands, 'exit' or Ctrl-D to quit.\n\n");

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
            const result = await this.commands.run(cmdName, rest.join(' '));
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

      await this.runInput(input);
      rl.prompt();
    });

    await new Promise<void>((resolve) => {
      rl.on('close', async () => {
        this.running = false;
        await this.saveLineHistory(rl).catch(() => {});
        if (this.agent) await this.agent.close();
        resolve();
      });
    });
  }

  private async runInput(input: string): Promise<void> {
    this.currentAbort = new AbortController();
    this.lastUserInput = input;
    this.lastAssistantOutput = '';
    try {
      if (this.team) {
        for await (const ev of this.team.run(input)) {
          this.renderEvent(ev);
        }
      } else if (this.agent) {
        for await (const ev of this.agent.run(input, { signal: this.currentAbort.signal })) {
          this.renderEvent(ev);
        }
      }
      process.stdout.write('\n');
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        process.stderr.write(`\n[error] ${err.message}\n`);
      }
    } finally {
      this.currentAbort = null;
    }
  }

  private renderEvent(ev: AgentEvent): void {
    switch (ev.type) {
      case 'text': {
        const text = String(ev.data ?? '');
        this.lastAssistantOutput += text;
        process.stdout.write(text);
        break;
      }
      case 'tool_call':
        if (this.showToolCalls) {
          const calls = Array.isArray(ev.data) ? ev.data : [ev.data];
          for (const call of calls as { name?: string; arguments?: unknown }[]) {
            process.stdout.write(`\n  [tool] ${call.name ?? 'unknown'}(${truncate(JSON.stringify(call.arguments ?? {}), 100)})\n`);
          }
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
      case 'plan':
        process.stdout.write(`\n[plan]\n${ev.data}\n`);
        break;
      default:
        break;
    }
  }

  private async cmdHelp(): Promise<string> {
    const lines = ['Available commands:'];
    for (const cmd of this.commands.list()) {
      lines.push(`  /${cmd.name.padEnd(15)} ${cmd.description}`);
    }
    return lines.join('\n');
  }

  private async cmdClear(): Promise<string> {
    if (this.agent) await this.agent.clearHistory();
    return 'Conversation cleared.';
  }

  private async cmdHistory(args: string): Promise<string> {
    if (!this.agent) return 'No agent attached.';
    const limit = parsePositiveInt(args) ?? 50;
    const msgs = await this.agent.getHistory(limit);
    if (msgs.length === 0) return 'No messages in history.';
    return msgs
      .map((m, i) => `${i + 1}. [${m.role}] ${truncate(messageText(m), 120)}`)
      .join('\n');
  }

  private async cmdSave(name: string): Promise<string> {
    if (!this.agent) return 'No agent attached.';
    const sessionName = name || `session_${Date.now()}`;
    const messages = await this.agent.getHistory();
    await this.sessionStore.save(sessionName, {
      chatId: this.chatId,
      messages,
      savedAt: new Date().toISOString(),
      metadata: { agent: this.agent.name, models: this.agent.modelsList },
    });
    return `Session saved as "${sessionName}".`;
  }

  private async cmdLoad(name: string): Promise<string> {
    if (!this.agent) return 'No agent attached.';
    if (!name) return 'Usage: /load <session-name>';
    const session = await this.sessionStore.load(name);
    if (!session) return `Session "${name}" not found.`;
    await this.agent.setHistory(session.messages ?? []);
    this.chatId = session.chatId ?? this.chatId;
    return `Session "${name}" loaded (${session.messages?.length ?? 0} messages).`;
  }

  private async cmdSessions(): Promise<string> {
    const sessions = await this.sessionStore.list();
    if (sessions.length === 0) return 'No saved sessions.';
    return sessions.map((s, i) => `${i + 1}. ${s}`).join('\n');
  }

  private async cmdDeleteSession(name: string): Promise<string> {
    if (!name) return 'Usage: /delete-session <session-name>';
    await this.sessionStore.delete(name);
    return `Session "${name}" deleted.`;
  }

  private async cmdModel(args: string): Promise<string> {
    if (!this.agent) return 'No agent attached.';
    const trimmed = args.trim();
    if (!trimmed) return `Current models: ${this.agent.modelsList.join(' -> ')}`;
    const models = trimmed.split(',').map((m) => m.trim()).filter(Boolean);
    this.agent.setModel(models.length > 1 ? models : models[0]);
    return `Model chain changed to: ${this.agent.modelsList.join(' -> ')}`;
  }

  private async cmdTools(): Promise<string> {
    if (!this.agent) return 'No agent attached.';
    const tools = this.agent.listTools();
    if (tools.length === 0) return 'No tools registered.';
    return tools.map((t) => `  ${t.name}: ${t.description ?? ''}`).join('\n');
  }

  private async cmdSkills(args: string): Promise<string> {
    if (!this.agent) return 'No agent attached.';
    const [action, ...rest] = args.trim().split(/\s+/).filter(Boolean);
    if (!action || action === 'list') {
      const skills = this.agent.listSkills();
      if (skills.length === 0) return 'No active skills.';
      return skills.map((skill) => `  ${skill.name}: ${skill.description}`).join('\n');
    }
    if (action === 'load') {
      const target = rest.join(' ');
      if (!target) return 'Usage: /skills load <name-or-path>';
      const skill = await this.loadSkill(target);
      this.agent.addSkill(skill);
      return `Loaded skill: ${skill.name}`;
    }
    if (action === 'remove') {
      const name = rest[0];
      if (!name) return 'Usage: /skills remove <name>';
      return this.agent.removeSkill(name) ? `Removed skill: ${name}` : `Skill not active: ${name}`;
    }
    return 'Usage: /skills [list|load|remove] [name-or-path]';
  }

  private async cmdPlugins(args: string): Promise<string> {
    const [action, ...rest] = args.trim().split(/\s+/).filter(Boolean);
    if (!action || action === 'list') {
      const plugins = await this.pluginLoader.discover();
      return plugins.length === 0 ? 'No plugins found.' : plugins.map((name) => `  ${name}`).join('\n');
    }
    if (action === 'load') {
      const target = rest.join(' ');
      if (!target) return 'Usage: /plugins load <name-or-path>';
      const plugin = await this.pluginLoader.load(target);
      if (this.agent) this.agent.addPlugin(plugin);
      for (const command of plugin.commands) this.commands.register({ ...command, source: plugin.manifest.name });
      return `Loaded plugin: ${plugin.manifest.name} (${plugin.skills.length} skills, ${plugin.tools.length} tools, ${plugin.commands.length} commands)`;
    }
    return 'Usage: /plugins [list|load] [name-or-path]';
  }

  private async cmdMemory(args: string): Promise<string> {
    if (!this.agent) return 'No agent attached.';
    const [action, key, ...valueParts] = args.trim().split(/\s+/).filter(Boolean);
    if (action === 'remember') {
      if (!key || valueParts.length === 0) return 'Usage: /memory remember <key> <value>';
      await this.agent.remember(key, valueParts.join(' '));
      return `Remembered: ${key}`;
    }
    if (action === 'recall') {
      if (!key) return 'Usage: /memory recall <key>';
      const value = await this.agent.recall(key);
      return value === null ? `No memory for: ${key}` : value;
    }
    return 'Usage: /memory remember <key> <value> | /memory recall <key>';
  }

  private async cmdContext(args: string): Promise<string> {
    if (!this.agent) return 'No agent attached.';
    const limit = parsePositiveInt(args) ?? 12;
    const snapshot = await this.agent.snapshot(limit);
    const tokens = snapshot.recentMessages.reduce((sum, message) => sum + estimateMessageTokens(message), 0);
    return [
      'Context:',
      `  Agent: ${snapshot.name}`,
      `  Models: ${snapshot.models.join(' -> ')}`,
      `  Messages: ${snapshot.messageCount}`,
      `  Recent estimate: ${tokens} tokens (${snapshot.recentMessages.length} messages)`,
      `  Tools: ${snapshot.toolCount}`,
      `  Skills: ${snapshot.skillCount}`,
    ].join('\n');
  }

  private async cmdCompact(args: string): Promise<string> {
    if (!this.agent) return 'No agent attached.';
    const before = (await this.agent.getHistory()).length;
    const maxTokens = parsePositiveInt(args) ?? 8000;
    const afterMessages = await this.agent.compactHistory(maxTokens);
    return `Compacted history: ${before} -> ${afterMessages.length} messages.`;
  }

  private async cmdRewind(args: string): Promise<string> {
    if (!this.agent) return 'No agent attached.';
    const turns = parsePositiveInt(args) ?? 1;
    const history = await this.agent.getHistory();
    const trimmed = rewindTurns(history, turns);
    await this.agent.setHistory(trimmed);
    return `Rewound ${history.length - trimmed.length} messages.`;
  }

  private async cmdRetry(): Promise<string | void> {
    if (!this.lastUserInput) return 'No previous user prompt to retry.';
    if (this.agent) {
      const history = await this.agent.getHistory();
      const trimmed = trimLastAssistantResponse(history);
      await this.agent.setHistory(trimmed);
    }
    await this.runInput(this.lastUserInput);
  }

  private async cmdStatus(): Promise<string> {
    const base = [
      'REPL Status:',
      `  Running: ${this.running ? 'yes' : 'no'}`,
      `  Tool display: ${this.showToolCalls ? 'on' : 'off'}`,
      `  Active chat: ${this.chatId}`,
    ];
    if (this.agent) {
      const snapshot = await this.agent.snapshot(0);
      base.push(
        `  Agent: ${snapshot.name}`,
        `  Models: ${snapshot.models.join(' -> ')}`,
        `  Messages: ${snapshot.messageCount}`,
        `  Tools: ${snapshot.toolCount}`,
        `  Skills: ${snapshot.skillCount}`,
      );
    }
    return base.join('\n');
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

  private async cmdExport(args: string): Promise<string> {
    if (!this.agent) return 'No agent attached.';
    const parts = args.trim().split(/\s+/).filter(Boolean);
    const format = parts[0] === 'json' || parts[0] === 'markdown' ? parts.shift()! : 'markdown';
    const target = parts.join(' ') || `aos_export_${Date.now()}.${format === 'json' ? 'json' : 'md'}`;
    const messages = await this.agent.getHistory();
    if (format === 'json') {
      await fs.writeFile(target, JSON.stringify({ chatId: this.chatId, messages }, null, 2), 'utf-8');
    } else {
      const md = messages.map((m) => `## ${m.role}\n\n${messageText(m)}\n`).join('\n---\n\n');
      await fs.writeFile(target, md, 'utf-8');
    }
    return `Exported ${messages.length} messages to ${target}`;
  }

  private async cmdExportSession(args: string): Promise<string> {
    const [sessionId, format = 'bundle', targetArg] = args.trim().split(/\s+/).filter(Boolean);
    if (!sessionId) return 'Usage: /export-session <session-name> [markdown|jsonl|bundle] [target]';
    const target = targetArg ?? `${sessionId}.${format === 'markdown' ? 'md' : format === 'jsonl' ? 'jsonl' : 'bundle'}`;
    if (format === 'markdown') {
      if (!this.sessionStore.exportMarkdown) return 'Session store does not support markdown export.';
      await this.sessionStore.exportMarkdown(sessionId, target);
    } else if (format === 'jsonl') {
      if (!this.sessionStore.exportJsonl) return 'Session store does not support JSONL export.';
      await this.sessionStore.exportJsonl(sessionId, target);
    } else if (format === 'bundle') {
      if (!this.sessionStore.exportBundle) return 'Session store does not support bundle export.';
      await this.sessionStore.exportBundle(sessionId, target);
    } else {
      return 'Usage: /export-session <session-name> [markdown|jsonl|bundle] [target]';
    }
    return `Exported session "${sessionId}" to ${target}`;
  }

  private async cmdImportSession(args: string): Promise<string> {
    const [bundlePath, sessionId] = args.trim().split(/\s+/).filter(Boolean);
    if (!bundlePath) return 'Usage: /import-session <bundle-path> [session-name]';
    if (!this.sessionStore.importBundle) return 'Session store does not support bundle import.';
    const imported = await this.sessionStore.importBundle(bundlePath, sessionId);
    return `Imported session "${imported}".`;
  }

  private async cmdNew(): Promise<string> {
    if (this.agent) await this.agent.clearHistory();
    this.chatId = `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    this.lastUserInput = null;
    this.lastAssistantOutput = '';
    return 'New conversation started.';
  }

  private async loadSkill(target: string): Promise<Skill> {
    try {
      return await builtinSkillLoader().load(target);
    } catch {
      const loader = new FileSkillLoader([
        path.join(homedir(), '.aos', 'skills'),
        path.join(process.cwd(), 'skills'),
      ]);
      return loader.load(target);
    }
  }

  private async loadLineHistory(rl: readline.Interface): Promise<void> {
    const raw = await fs.readFile(this.historyFile, 'utf-8');
    const lines = raw.split('\n').filter(Boolean).slice(-1000);
    (rl as unknown as { history: string[] }).history = [...lines].reverse();
  }

  private async saveLineHistory(rl: readline.Interface): Promise<void> {
    const history = [...((rl as unknown as { history?: string[] }).history ?? [])].reverse();
    await fs.mkdir(path.dirname(this.historyFile), { recursive: true });
    await fs.writeFile(this.historyFile, history.join('\n'), 'utf-8');
  }
}

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n)}...` : s;
}

function messageText(message: Message): string {
  if (typeof message.content === 'string') return message.content;
  return message.content
    .map((part) => (part.type === 'text' ? part.text : `[image:${part.mediaType ?? 'unknown'}]`))
    .join('\n');
}

function estimateMessageTokens(message: Message): number {
  return Math.ceil(messageText(message).length / 4);
}

function parsePositiveInt(value: string): number | undefined {
  const n = Number.parseInt(value.trim(), 10);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

function rewindTurns(history: Message[], turns: number): Message[] {
  let remaining = [...history];
  for (let i = 0; i < turns && remaining.length > 0; i++) {
    while (remaining.length > 0 && remaining[remaining.length - 1].role !== 'user') {
      remaining.pop();
    }
    if (remaining.length > 0) remaining.pop();
  }
  return remaining;
}

function trimLastAssistantResponse(history: Message[]): Message[] {
  const remaining = [...history];
  while (remaining.length > 0) {
    const role = remaining[remaining.length - 1].role;
    if (role === 'assistant' || role === 'tool') remaining.pop();
    else break;
  }
  return remaining;
}
