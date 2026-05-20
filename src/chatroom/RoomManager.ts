import { existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync, unlinkSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { homedir } from 'node:os';
import { EventEmitter } from 'node:events';
import type { Message } from '../types.js';
import { NatsThread, type ChunkHook, type StepMessageHook, type ThreadState } from './Thread.js';
import { NatsStreamAdapter } from './Stream.js';
import { ProjectManager, type ProjectInfo } from './Projects.js';
import {
  exportChatBundle,
  importChatBundle,
  exportChatToMarkdown,
  exportChatToJSON,
  type ExportOptions,
  type ExportResult,
  type ImportResult,
} from './Export.js';

// --- Types ---

export interface RoomManagerOptions {
  /** Room name identifier. */
  name?: string;
  /** Data directory for persistence. Default: ~/.medrix */
  dataDir?: string;
  /** NATS URL for streaming. Default: nats://localhost:4222 */
  natsUrl?: string;
  /** Enable NATS streaming for real-time message publishing. */
  enableNatsStreaming?: boolean;
  /** Active project/workspace path. */
  workspacePath?: string;
}

export interface AgentRegistration {
  name: string;
  description: string;
  icon?: string;
  model?: string;
  capabilities?: string[];
  /** Handler function that processes messages and returns a response. */
  handler: (messages: Message[], context?: ChatContext) => Promise<string>;
}

export interface ChatInfo {
  id: string;
  name: string;
  running: boolean;
  lastActivityDate: string | null;
  project: Record<string, unknown> | null;
  workspaceMode: string | null;
  workspacePath: string | null;
  messageCount: number;
}

export interface ChatContext {
  chatId: string;
  agentName?: string;
  workspacePath?: string;
  variables?: Record<string, unknown>;
}

export interface ChatResult {
  success: boolean;
  message?: string;
  response?: string;
  chatId?: string;
}

export type SlashCommandHandler = (
  args: string,
  context: { chatId: string; roomManager: RoomManager },
) => Promise<string>;

export interface Permission {
  agentName: string;
  canTalkTo: string[]; // '*' means all
  canBeCalledBy: string[]; // '*' means all
}

// --- Memory (simple JSONL-based persistence) ---

interface ChatMemory {
  id: string;
  name: string;
  messages: Message[];
  metadata: Record<string, unknown>;
  extraData: Record<string, unknown>;
}

class MemoryManager {
  private memoryDir: string;
  private cache: Map<string, ChatMemory> = new Map();

  constructor(memoryDir: string) {
    this.memoryDir = memoryDir;
    mkdirSync(memoryDir, { recursive: true });
  }

  get path(): string {
    return this.memoryDir;
  }

  listMemories(): string[] {
    if (!existsSync(this.memoryDir)) return [];
    return readdirSync(this.memoryDir)
      .filter((f) => f.endsWith('.jsonl'))
      .map((f) => f.replace('.jsonl', ''));
  }

  getMemory(chatId: string): ChatMemory {
    if (this.cache.has(chatId)) return this.cache.get(chatId)!;

    const jsonlPath = join(this.memoryDir, `${chatId}.jsonl`);
    const metaPath = join(this.memoryDir, `${chatId}.meta.json`);

    if (!existsSync(jsonlPath)) {
      throw new Error(`Chat '${chatId}' not found`);
    }

    const messages: Message[] = [];
    const lines = readFileSync(jsonlPath, 'utf-8').split('\n').filter((l) => l.trim());
    for (const line of lines) {
      try {
        messages.push(JSON.parse(line) as Message);
      } catch { /* skip malformed */ }
    }

    let metadata: Record<string, unknown> = {};
    let extraData: Record<string, unknown> = {};
    if (existsSync(metaPath)) {
      try {
        const meta = JSON.parse(readFileSync(metaPath, 'utf-8'));
        metadata = meta;
        extraData = meta.extra_data ?? meta.extraData ?? {};
      } catch { /* ignore */ }
    }

    const memory: ChatMemory = {
      id: chatId,
      name: (metadata.name as string) ?? 'Untitled',
      messages,
      metadata,
      extraData,
    };
    this.cache.set(chatId, memory);
    return memory;
  }

  newMemory(name?: string): ChatMemory {
    const id = crypto.randomUUID();
    const memory: ChatMemory = {
      id,
      name: name ?? 'New Chat',
      messages: [],
      metadata: { name: name ?? 'New Chat', id },
      extraData: {},
    };
    this.cache.set(id, memory);
    this.save(id);
    return memory;
  }

  save(chatId: string): void {
    const memory = this.cache.get(chatId);
    if (!memory) return;

    const jsonlPath = join(this.memoryDir, `${chatId}.jsonl`);
    const metaPath = join(this.memoryDir, `${chatId}.meta.json`);

    const jsonlContent = memory.messages
      .map((m) => JSON.stringify(m))
      .join('\n') + '\n';
    writeFileSync(jsonlPath, jsonlContent, 'utf-8');

    const meta = {
      ...memory.metadata,
      name: memory.name,
      id: memory.id,
      extra_data: memory.extraData,
    };
    writeFileSync(metaPath, JSON.stringify(meta, null, 2), 'utf-8');
  }

  deleteMemory(chatId: string): void {
    this.cache.delete(chatId);
    const jsonlPath = join(this.memoryDir, `${chatId}.jsonl`);
    const metaPath = join(this.memoryDir, `${chatId}.meta.json`);
    if (existsSync(jsonlPath)) unlinkSync(jsonlPath);
    if (existsSync(metaPath)) unlinkSync(metaPath);
  }

  updateName(chatId: string, name: string): void {
    const memory = this.getMemory(chatId);
    memory.name = name;
    memory.metadata.name = name;
    this.save(chatId);
  }

  addMessage(chatId: string, message: Message): void {
    const memory = this.getMemory(chatId);
    memory.messages.push(message);
    this.save(chatId);
  }

  getMessages(chatId: string, limit?: number): Message[] {
    const memory = this.getMemory(chatId);
    if (limit) return memory.messages.slice(-limit);
    return memory.messages;
  }
}

// --- RoomManager ---

/**
 * RoomManager is the central orchestrator for the MedrixAI chatroom.
 *
 * Responsibilities:
 * - Agent registration and lifecycle management
 * - Message routing (direct, broadcast, to-agent)
 * - Slash command handling (/help, /agents, /clear, /export, /thread, etc.)
 * - File/attachment handling
 * - Permission system (who can talk to whom)
 * - Message history with pagination
 * - Room state persistence
 * - Thread management
 * - Integration with NATS streaming
 */
export class RoomManager extends EventEmitter {
  readonly name: string;
  private dataDir: string;
  private memoryDir: string;
  private natsUrl: string;
  private enableNatsStreaming: boolean;

  // Core state
  private agents: Map<string, AgentRegistration> = new Map();
  private permissions: Map<string, Permission> = new Map();
  private threads: Map<string, NatsThread> = new Map();
  private activeChats: Map<string, { threadId: string; running: boolean }> = new Map();
  private slashCommands: Map<string, SlashCommandHandler> = new Map();

  // Managers
  private memoryManager: MemoryManager;
  private projectManager: ProjectManager;
  private natsAdapter: NatsStreamAdapter | null = null;

  constructor(opts?: RoomManagerOptions) {
    super();
    this.name = opts?.name ?? 'medrix-chatroom';
    this.dataDir = opts?.dataDir ?? join(homedir(), '.medrix');
    this.memoryDir = join(this.dataDir, 'memory');
    this.natsUrl = opts?.natsUrl ?? process.env.NATS_URL ?? 'nats://localhost:4222';
    this.enableNatsStreaming = opts?.enableNatsStreaming ?? false;

    // Initialize managers
    this.memoryManager = new MemoryManager(this.memoryDir);
    this.projectManager = new ProjectManager(opts?.workspacePath);

    // Initialize NATS streaming if enabled
    if (this.enableNatsStreaming) {
      this.natsAdapter = new NatsStreamAdapter();
    }

    // Register built-in slash commands
    this.registerBuiltinCommands();
  }

  // ═══════════════════════════════════════════════════════════════
  // Agent Registration
  // ═══════════════════════════════════════════════════════════════

  /** Register an agent with the room. */
  registerAgent(registration: AgentRegistration): void {
    this.agents.set(registration.name, registration);
    // Default permission: can talk to everyone
    if (!this.permissions.has(registration.name)) {
      this.permissions.set(registration.name, {
        agentName: registration.name,
        canTalkTo: ['*'],
        canBeCalledBy: ['*'],
      });
    }
    this.emit('agent:registered', registration.name);
  }

  /** Deregister an agent from the room. */
  deregisterAgent(name: string): boolean {
    const existed = this.agents.delete(name);
    this.permissions.delete(name);
    if (existed) {
      this.emit('agent:deregistered', name);
    }
    return existed;
  }

  /** Get info about a registered agent. */
  getAgent(name: string): AgentRegistration | undefined {
    return this.agents.get(name);
  }

  /** List all registered agents. */
  listAgents(): Array<{ name: string; description: string; icon?: string; model?: string; capabilities?: string[] }> {
    return Array.from(this.agents.values()).map((a) => ({
      name: a.name,
      description: a.description,
      icon: a.icon,
      model: a.model,
      capabilities: a.capabilities,
    }));
  }

  // ═══════════════════════════════════════════════════════════════
  // Permission System
  // ═══════════════════════════════════════════════════════════════

  /** Set permissions for an agent. */
  setPermission(permission: Permission): void {
    this.permissions.set(permission.agentName, permission);
  }

  /** Check if agent A can send messages to agent B. */
  canCommunicate(fromAgent: string, toAgent: string): boolean {
    const fromPerm = this.permissions.get(fromAgent);
    const toPerm = this.permissions.get(toAgent);

    if (!fromPerm || !toPerm) return false;

    const canSend = fromPerm.canTalkTo.includes('*') || fromPerm.canTalkTo.includes(toAgent);
    const canReceive = toPerm.canBeCalledBy.includes('*') || toPerm.canBeCalledBy.includes(fromAgent);

    return canSend && canReceive;
  }

  // ═══════════════════════════════════════════════════════════════
  // Chat Lifecycle
  // ═══════════════════════════════════════════════════════════════

  /** Create a new chat session. */
  async createChat(opts?: {
    name?: string;
    projectName?: string;
    workspacePath?: string;
    workspaceMode?: 'project' | 'isolated';
  }): Promise<ChatResult & { chatId: string }> {
    const memory = this.memoryManager.newMemory(opts?.name);

    // Set project metadata
    const project: Record<string, unknown> = {};
    if (opts?.projectName) project.name = opts.projectName;
    if (opts?.workspacePath) {
      project.workspace_path = opts.workspacePath;
      project.workspace_mode = 'isolated';
      mkdirSync(opts.workspacePath, { recursive: true });
    } else {
      project.workspace_mode = opts?.workspaceMode ?? 'project';
    }

    memory.extraData.project = project;
    memory.extraData.last_activity_date = new Date().toISOString();
    this.memoryManager.save(memory.id);

    this.emit('chat:created', memory.id);

    return {
      success: true,
      chatId: memory.id,
      message: 'Chat created successfully',
    };
  }

  /** Delete a chat session. */
  async deleteChat(chatId: string): Promise<ChatResult> {
    try {
      // Stop any running thread
      await this.stopChat(chatId);

      // Check for isolated workspace to clean up
      const memory = this.memoryManager.getMemory(chatId);
      const project = memory.extraData.project as Record<string, unknown> | undefined;
      if (project?.workspace_mode === 'isolated' && project.workspace_path) {
        const wsPath = project.workspace_path as string;
        if (existsSync(wsPath) && wsPath.includes('.medrix')) {
          rmSync(wsPath, { recursive: true, force: true });
        }
      }

      this.memoryManager.deleteMemory(chatId);
      this.emit('chat:deleted', chatId);
      return { success: true, message: 'Chat deleted successfully' };
    } catch (e) {
      return { success: false, message: String(e) };
    }
  }

  /** List all chats, optionally filtered by project. */
  async listChats(projectName?: string): Promise<ChatInfo[]> {
    const ids = this.memoryManager.listMemories();
    const chats: ChatInfo[] = [];

    for (const id of ids) {
      try {
        const memory = this.memoryManager.getMemory(id);
        const project = memory.extraData.project as Record<string, unknown> | undefined;

        // Filter by project
        if (projectName !== undefined) {
          const chatProject = project?.name as string | undefined;
          if (chatProject !== projectName) continue;
        }

        chats.push({
          id,
          name: memory.name,
          running: this.activeChats.has(id),
          lastActivityDate: (memory.extraData.last_activity_date as string) ?? null,
          project: (project as Record<string, unknown>) ?? null,
          workspaceMode: (project?.workspace_mode as string) ?? null,
          workspacePath: (project?.workspace_path as string) ?? null,
          messageCount: memory.messages.length,
        });
      } catch {
        // Skip corrupted entries
      }
    }

    // Sort by last activity (most recent first)
    chats.sort((a, b) => {
      const aDate = a.lastActivityDate ? new Date(a.lastActivityDate).getTime() : 0;
      const bDate = b.lastActivityDate ? new Date(b.lastActivityDate).getTime() : 0;
      return bDate - aDate;
    });

    return chats;
  }

  /** Get messages for a chat with optional pagination. */
  async getChatMessages(chatId: string, opts?: { limit?: number; offset?: number }): Promise<Message[]> {
    const messages = this.memoryManager.getMessages(chatId);
    if (opts?.offset || opts?.limit) {
      const start = opts.offset ?? 0;
      const end = opts.limit ? start + opts.limit : undefined;
      return messages.slice(start, end);
    }
    return messages;
  }

  /** Update the name of a chat. */
  async updateChatName(chatId: string, name: string): Promise<ChatResult> {
    try {
      this.memoryManager.updateName(chatId, name);
      return { success: true, message: 'Chat name updated' };
    } catch (e) {
      return { success: false, message: String(e) };
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // Message Routing & Chat Execution
  // ═══════════════════════════════════════════════════════════════

  /**
   * Send a message to a chat and get a response from the assigned agent.
   * This is the main entry point for chat interaction.
   */
  async chat(
    chatId: string,
    messages: Message[],
    opts?: {
      agentName?: string;
      contextVariables?: Record<string, unknown>;
      onChunk?: ChunkHook;
      onStepMessage?: StepMessageHook;
    },
  ): Promise<ChatResult> {
    // Check if chat is already running
    if (this.activeChats.get(chatId)?.running) {
      return { success: false, message: 'Chat is already running' };
    }

    // Get memory
    let memory: ChatMemory;
    try {
      memory = this.memoryManager.getMemory(chatId);
    } catch {
      return { success: false, message: `Chat '${chatId}' not found` };
    }

    // Check for slash commands in the last user message
    const lastMsg = messages[messages.length - 1];
    if (lastMsg?.role === 'user' && typeof lastMsg.content === 'string' && lastMsg.content.startsWith('/')) {
      const cmdResult = await this.handleSlashCommand(lastMsg.content, chatId);
      if (cmdResult !== null) {
        // Store the command and response
        this.memoryManager.addMessage(chatId, lastMsg);
        const responseMsg: Message = { role: 'assistant', content: cmdResult };
        this.memoryManager.addMessage(chatId, responseMsg);
        return { success: true, response: cmdResult, chatId };
      }
    }

    // Mark as running
    this.activeChats.set(chatId, { threadId: chatId, running: true });
    memory.extraData.running = true;
    memory.extraData.last_activity_date = new Date().toISOString();

    try {
      // Add messages to memory
      for (const msg of messages) {
        this.memoryManager.addMessage(chatId, msg);
      }

      // Determine which agent to use
      const agentName = opts?.agentName ?? this.getActiveAgent(chatId);
      const agent = agentName ? this.agents.get(agentName) : this.getDefaultAgent();

      if (!agent) {
        return { success: false, message: 'No agent available to handle this chat', chatId };
      }

      // Build context
      const project = memory.extraData.project as Record<string, unknown> | undefined;
      const context: ChatContext = {
        chatId,
        agentName: agent.name,
        workspacePath: (project?.workspace_path as string) ?? undefined,
        variables: opts?.contextVariables,
      };

      // Set up NATS streaming hooks
      if (this.natsAdapter) {
        const [chunkHook, stepHook] = this.natsAdapter.createHooks(chatId);
        if (opts?.onChunk) {
          const originalChunk = opts.onChunk;
          opts.onChunk = async (chunk) => {
            await chunkHook(chunk);
            await originalChunk(chunk);
          };
        }
        if (opts?.onStepMessage) {
          const originalStep = opts.onStepMessage;
          opts.onStepMessage = async (step) => {
            await stepHook(step);
            await originalStep(step);
          };
        }
      }

      // Get full conversation history for the agent
      const fullHistory = this.memoryManager.getMessages(chatId);

      // Call the agent handler
      const response = await agent.handler(fullHistory, context);

      // Store the response
      const responseMsg: Message = { role: 'assistant', content: response, name: agent.name };
      this.memoryManager.addMessage(chatId, responseMsg);

      // Notify via NATS
      if (this.natsAdapter) {
        await this.natsAdapter.publishChatFinished(chatId);
      }

      this.emit('chat:message', { chatId, agentName: agent.name, response });

      return { success: true, response, chatId };
    } catch (e) {
      return { success: false, message: String(e), chatId };
    } finally {
      // Mark as not running
      this.activeChats.set(chatId, { threadId: chatId, running: false });
      memory.extraData.running = false;
      this.memoryManager.save(chatId);
    }
  }

  /** Route a message to a specific agent (direct message). */
  async routeToAgent(
    chatId: string,
    agentName: string,
    message: Message,
    fromAgent?: string,
  ): Promise<ChatResult> {
    // Check permissions
    if (fromAgent && !this.canCommunicate(fromAgent, agentName)) {
      return {
        success: false,
        message: `Agent '${fromAgent}' does not have permission to talk to '${agentName}'`,
      };
    }

    const agent = this.agents.get(agentName);
    if (!agent) {
      return { success: false, message: `Agent '${agentName}' not found` };
    }

    return this.chat(chatId, [message], { agentName });
  }

  /** Broadcast a message to all agents in the room. */
  async broadcast(chatId: string, message: Message): Promise<Map<string, string>> {
    const responses = new Map<string, string>();
    const promises = Array.from(this.agents.entries()).map(async ([name, agent]) => {
      try {
        const history = this.memoryManager.getMessages(chatId);
        const response = await agent.handler([...history, message], { chatId, agentName: name });
        responses.set(name, response);
      } catch (e) {
        responses.set(name, `Error: ${e}`);
      }
    });
    await Promise.allSettled(promises);
    return responses;
  }

  /** Stop a running chat. */
  async stopChat(chatId: string): Promise<ChatResult> {
    const active = this.activeChats.get(chatId);
    if (!active?.running) {
      return { success: false, message: 'Chat is not running' };
    }

    // Signal thread to stop
    const thread = this.threads.get(chatId);
    if (thread) {
      thread.stop();
    }

    this.activeChats.set(chatId, { ...active, running: false });
    this.emit('chat:stopped', chatId);
    return { success: true, message: 'Chat stopped' };
  }

  // ═══════════════════════════════════════════════════════════════
  // Active Agent Management
  // ═══════════════════════════════════════════════════════════════

  /** Set the active agent for a chat. */
  setActiveAgent(chatId: string, agentName: string): ChatResult {
    if (!this.agents.has(agentName)) {
      return { success: false, message: `Agent '${agentName}' not found` };
    }
    const memory = this.memoryManager.getMemory(chatId);
    memory.extraData.active_agent = agentName;
    this.memoryManager.save(chatId);
    return { success: true, message: `Active agent set to '${agentName}'` };
  }

  /** Get the active agent for a chat. */
  getActiveAgent(chatId: string): string | null {
    try {
      const memory = this.memoryManager.getMemory(chatId);
      return (memory.extraData.active_agent as string) ?? null;
    } catch {
      return null;
    }
  }

  /** Get the default agent (first registered). */
  private getDefaultAgent(): AgentRegistration | undefined {
    return this.agents.values().next().value;
  }

  // ═══════════════════════════════════════════════════════════════
  // Slash Commands
  // ═══════════════════════════════════════════════════════════════

  /** Register a custom slash command. */
  registerCommand(name: string, handler: SlashCommandHandler): void {
    this.slashCommands.set(name.startsWith('/') ? name : `/${name}`, handler);
  }

  /** Handle a slash command. Returns response string or null if not a command. */
  private async handleSlashCommand(input: string, chatId: string): Promise<string | null> {
    const trimmed = input.trim();
    if (!trimmed.startsWith('/')) return null;

    const spaceIdx = trimmed.indexOf(' ');
    const command = spaceIdx === -1 ? trimmed : trimmed.slice(0, spaceIdx);
    const args = spaceIdx === -1 ? '' : trimmed.slice(spaceIdx + 1).trim();

    const handler = this.slashCommands.get(command);
    if (!handler) return null;

    try {
      return await handler(args, { chatId, roomManager: this });
    } catch (e) {
      return `Error executing ${command}: ${e}`;
    }
  }

  /** Register all built-in slash commands. */
  private registerBuiltinCommands(): void {
    // /help - List available commands
    this.slashCommands.set('/help', async () => {
      const commands = Array.from(this.slashCommands.keys()).sort();
      const lines = ['Available commands:', ''];
      for (const cmd of commands) {
        lines.push(`  ${cmd}`);
      }
      lines.push('', 'Type a command to execute it.');
      return lines.join('\n');
    });

    // /agents - List registered agents
    this.slashCommands.set('/agents', async () => {
      const agents = this.listAgents();
      if (agents.length === 0) return 'No agents registered.';
      const lines = ['Registered agents:', ''];
      for (const a of agents) {
        const icon = a.icon ? `${a.icon} ` : '';
        lines.push(`  ${icon}${a.name} - ${a.description}`);
        if (a.model) lines.push(`    Model: ${a.model}`);
      }
      return lines.join('\n');
    });

    // /clear - Clear chat history
    this.slashCommands.set('/clear', async (_args, ctx) => {
      const memory = this.memoryManager.getMemory(ctx.chatId);
      memory.messages = [];
      this.memoryManager.save(ctx.chatId);
      return 'Chat history cleared.';
    });

    // /export - Export chat
    this.slashCommands.set('/export', async (args, ctx) => {
      const format = args.trim() || 'markdown';
      const outputDir = join(this.dataDir, 'exports');
      mkdirSync(outputDir, { recursive: true });

      let result: ExportResult;
      if (format === 'json') {
        const outputPath = join(outputDir, `${ctx.chatId}.json`);
        result = exportChatToJSON(this.memoryDir, ctx.chatId, outputPath);
      } else if (format === 'bundle') {
        const bundleDir = join(outputDir, ctx.chatId);
        result = exportChatBundle(this.memoryDir, ctx.chatId, bundleDir);
      } else {
        const outputPath = join(outputDir, `${ctx.chatId}.md`);
        result = exportChatToMarkdown(this.memoryDir, ctx.chatId, outputPath);
      }

      if (result.success) {
        return `Exported to: ${result.bundlePath}\n(${result.stats?.messages ?? 0} messages)`;
      }
      return `Export failed: ${result.message}`;
    });

    // /import - Import chat
    this.slashCommands.set('/import', async (args) => {
      const bundlePath = args.trim();
      if (!bundlePath) return 'Usage: /import <path-to-bundle>';
      if (!existsSync(bundlePath)) return `Bundle not found: ${bundlePath}`;

      const result = importChatBundle(this.memoryDir, bundlePath, this.dataDir);
      if (result.success) {
        return `Imported: ${result.chatName} (ID: ${result.chatId})`;
      }
      return `Import failed: ${result.message}`;
    });

    // /thread - Create or manage threads
    this.slashCommands.set('/thread', async (args, ctx) => {
      const parts = args.trim().split(' ');
      const subcommand = parts[0] || 'list';

      switch (subcommand) {
        case 'new': {
          const title = parts.slice(1).join(' ') || undefined;
          const thread = new NatsThread(this.name, crypto.randomUUID(), { title });
          this.threads.set(thread.id, thread);
          return `Created thread: ${thread.id}${title ? ` (${title})` : ''}`;
        }
        case 'list': {
          const threads = Array.from(this.threads.values());
          if (threads.length === 0) return 'No active threads.';
          const lines = ['Active threads:', ''];
          for (const t of threads) {
            const title = t.metadata.title ?? 'Untitled';
            lines.push(`  [${t.state}] ${t.id.slice(0, 8)}... - ${title} (${t.messages.length} msgs)`);
          }
          return lines.join('\n');
        }
        case 'close': {
          const threadId = parts[1];
          const thread = this.findThread(threadId);
          if (!thread) return `Thread not found: ${threadId}`;
          await thread.closeThread();
          return `Thread ${thread.id.slice(0, 8)}... closed.`;
        }
        case 'archive': {
          const threadId = parts[1];
          const thread = this.findThread(threadId);
          if (!thread) return `Thread not found: ${threadId}`;
          await thread.archive();
          return `Thread ${thread.id.slice(0, 8)}... archived.`;
        }
        default:
          return 'Usage: /thread [new|list|close|archive] [args]';
      }
    });

    // /name - Rename current chat
    this.slashCommands.set('/name', async (args, ctx) => {
      const newName = args.trim();
      if (!newName) return 'Usage: /name <new-name>';
      this.memoryManager.updateName(ctx.chatId, newName);
      return `Chat renamed to: ${newName}`;
    });

    // /status - Show room status
    this.slashCommands.set('/status', async () => {
      const agents = this.agents.size;
      const chats = this.memoryManager.listMemories().length;
      const threads = this.threads.size;
      const running = Array.from(this.activeChats.values()).filter((a) => a.running).length;
      return [
        'Room Status:',
        `  Name: ${this.name}`,
        `  Agents: ${agents}`,
        `  Chats: ${chats}`,
        `  Active threads: ${threads}`,
        `  Running chats: ${running}`,
        `  NATS streaming: ${this.enableNatsStreaming ? 'enabled' : 'disabled'}`,
      ].join('\n');
    });

    // /project - Project management
    this.slashCommands.set('/project', async (args) => {
      const parts = args.trim().split(' ');
      const subcommand = parts[0] || 'info';

      switch (subcommand) {
        case 'list': {
          const projects = this.projectManager.listProjects();
          if (projects.length === 0) return 'No projects registered.';
          const lines = ['Projects:', ''];
          for (const p of projects) {
            const active = p.isActive ? ' [ACTIVE]' : '';
            const exists = p.exists ? '' : ' [MISSING]';
            lines.push(`  ${p.name}${active}${exists}`);
            lines.push(`    Path: ${p.path}`);
          }
          return lines.join('\n');
        }
        case 'switch': {
          const path = parts.slice(1).join(' ');
          if (!path) return 'Usage: /project switch <path>';
          const info = this.projectManager.setActive(path);
          if (!info) return `Project not found: ${path}`;
          return `Switched to project: ${info.name}`;
        }
        case 'register': {
          const path = parts.slice(1).join(' ');
          if (!path) return 'Usage: /project register <path>';
          if (!existsSync(path)) return `Directory not found: ${path}`;
          const info = this.projectManager.register(path);
          return `Registered project: ${info.name} (${info.path})`;
        }
        case 'info':
        default: {
          const active = this.projectManager.activeProject;
          if (!active) return 'No active project.';
          return `Active project: ${active.name}\nPath: ${active.path}`;
        }
      }
    });

    // /switch - Switch active agent
    this.slashCommands.set('/switch', async (args, ctx) => {
      const agentName = args.trim();
      if (!agentName) {
        const agents = this.listAgents();
        return 'Usage: /switch <agent-name>\nAvailable: ' + agents.map((a) => a.name).join(', ');
      }
      const result = this.setActiveAgent(ctx.chatId, agentName);
      return result.message ?? 'Done';
    });
  }

  /** Find a thread by full or partial ID. */
  private findThread(idOrPrefix: string): NatsThread | undefined {
    if (!idOrPrefix) return undefined;
    // Exact match
    if (this.threads.has(idOrPrefix)) return this.threads.get(idOrPrefix);
    // Prefix match
    for (const [id, thread] of this.threads) {
      if (id.startsWith(idOrPrefix)) return thread;
    }
    return undefined;
  }

  // ═══════════════════════════════════════════════════════════════
  // Thread Management
  // ═══════════════════════════════════════════════════════════════

  /** Create a new thread in the room. */
  createThread(opts?: { title?: string; parentThreadId?: string }): NatsThread {
    const thread = new NatsThread(this.name, crypto.randomUUID(), {
      title: opts?.title,
      parentThreadId: opts?.parentThreadId,
    });
    this.threads.set(thread.id, thread);

    // Link to parent if specified
    if (opts?.parentThreadId) {
      const parent = this.threads.get(opts.parentThreadId);
      if (parent) {
        parent.addChildThread(thread.id);
      }
    }

    this.emit('thread:created', thread.id);
    return thread;
  }

  /** Get a thread by ID. */
  getThread(threadId: string): NatsThread | undefined {
    return this.threads.get(threadId);
  }

  /** List all threads with optional state filter. */
  listThreads(state?: ThreadState): NatsThread[] {
    const threads = Array.from(this.threads.values());
    if (state) return threads.filter((t) => t.state === state);
    return threads;
  }

  /** Remove a thread (must be closed or archived). */
  removeThread(threadId: string): boolean {
    const thread = this.threads.get(threadId);
    if (!thread) return false;
    if (thread.state === 'open') return false; // Must close first

    // Unlink from parent
    if (thread.parentThreadId) {
      const parent = this.threads.get(thread.parentThreadId);
      if (parent) parent.removeChildThread(threadId);
    }

    this.threads.delete(threadId);
    this.emit('thread:removed', threadId);
    return true;
  }

  // ═══════════════════════════════════════════════════════════════
  // Export / Import
  // ═══════════════════════════════════════════════════════════════

  /** Export a chat to a bundle. */
  async exportChat(chatId: string, opts?: ExportOptions & { outputPath?: string; format?: 'bundle' | 'markdown' | 'json' }): Promise<ExportResult> {
    const outputDir = join(this.dataDir, 'exports');
    mkdirSync(outputDir, { recursive: true });

    const format = opts?.format ?? 'bundle';
    const outputPath = opts?.outputPath ?? join(outputDir, chatId);

    switch (format) {
      case 'markdown':
        return exportChatToMarkdown(this.memoryDir, chatId, outputPath + '.md', opts);
      case 'json':
        return exportChatToJSON(this.memoryDir, chatId, outputPath + '.json', opts);
      case 'bundle':
      default:
        return exportChatBundle(this.memoryDir, chatId, outputPath, opts);
    }
  }

  /** Import a chat from a bundle. */
  async importChat(bundlePath: string, targetRoot?: string): Promise<ImportResult> {
    return importChatBundle(this.memoryDir, bundlePath, targetRoot ?? this.dataDir);
  }

  // ═══════════════════════════════════════════════════════════════
  // Project Management (delegated)
  // ═══════════════════════════════════════════════════════════════

  /** List all registered projects. */
  listProjects(): ReturnType<ProjectManager['listProjects']> {
    return this.projectManager.listProjects();
  }

  /** Get the active project. */
  getActiveProject(): ProjectInfo | null {
    return this.projectManager.activeProject;
  }

  /** Register a new project. */
  registerProject(path: string, name?: string): ProjectInfo {
    return this.projectManager.register(path, name);
  }

  /** Remove a project from registry. */
  removeProject(path: string): boolean {
    return this.projectManager.remove(path);
  }

  /** Switch active project. */
  switchProject(path: string): ProjectInfo | null {
    return this.projectManager.setActive(path);
  }

  // ═══════════════════════════════════════════════════════════════
  // Message History & Revert
  // ═══════════════════════════════════════════════════════════════

  /** Revert chat to a specific message index (deletes that message and all after). */
  async revertToMessage(chatId: string, messageIndex: number): Promise<ChatResult> {
    try {
      const memory = this.memoryManager.getMemory(chatId);
      if (messageIndex < 0 || messageIndex >= memory.messages.length) {
        return { success: false, message: 'Invalid message index' };
      }
      memory.messages = memory.messages.slice(0, messageIndex);
      this.memoryManager.save(chatId);
      return { success: true, message: `Reverted to message index ${messageIndex}` };
    } catch (e) {
      return { success: false, message: String(e) };
    }
  }

  /** Get message count for a chat. */
  getMessageCount(chatId: string): number {
    try {
      return this.memoryManager.getMessages(chatId).length;
    } catch {
      return 0;
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // NATS Streaming Integration
  // ═══════════════════════════════════════════════════════════════

  /** Connect the NATS stream adapter. */
  async connectStreaming(): Promise<void> {
    if (this.natsAdapter) {
      await this.natsAdapter.ensureConnected(this.natsUrl);
    }
  }

  /** Disconnect streaming. */
  async disconnectStreaming(): Promise<void> {
    if (this.natsAdapter) {
      await this.natsAdapter.close();
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // Lifecycle
  // ═══════════════════════════════════════════════════════════════

  /** Initialize the room manager (connect streaming, etc). */
  async start(): Promise<void> {
    if (this.enableNatsStreaming) {
      await this.connectStreaming();
    }
    this.emit('room:started');
  }

  /** Shut down the room manager gracefully. */
  async stop(): Promise<void> {
    // Stop all running chats
    for (const [chatId, state] of this.activeChats) {
      if (state.running) {
        await this.stopChat(chatId);
      }
    }

    // Close all threads
    for (const thread of this.threads.values()) {
      await thread.close();
    }
    this.threads.clear();

    // Disconnect streaming
    await this.disconnectStreaming();

    this.emit('room:stopped');
  }

  /** Get room status summary. */
  getStatus(): Record<string, unknown> {
    return {
      name: this.name,
      agents: this.agents.size,
      chats: this.memoryManager.listMemories().length,
      activeThreads: this.threads.size,
      runningChats: Array.from(this.activeChats.values()).filter((a) => a.running).length,
      natsStreaming: this.enableNatsStreaming,
      activeProject: this.projectManager.activeProject?.name ?? null,
    };
  }
}
