import type { AgentEvent, ChatOptions, Message, TextContent, ImageContent } from '../types.js';
import { ToolSet } from '../toolset/ToolSet.js';
import { InMemoryMemory } from '../memory/InMemoryMemory.js';
import type { Memory } from '../memory/Memory.js';
import type { LLMProvider } from '../provider/Provider.js';
import { parseModelString } from '../provider/Provider.js';
import { providerForModel } from '../provider/ModelSelector.js';
import type { MessageCompressor } from './compression/Compressor.js';
import { SummaryCompressor } from './compression/SummaryCompressor.js';
import type { Skill } from '../skill/Skill.js';
import type { McpPlugin } from '../mcp/McpPlugin.js';
import type { AgentOptions } from './AgentOptions.js';
import { logger } from '../utils/logger.js';
import { uid } from '../utils/misc.js';
import type { Tool } from '../toolset/Tool.js';
import { z } from '../toolset/Tool.js';
import type { ExecutionContext } from '../types.js';
import type { LoadedPlugin } from '../plugin/index.js';
import type { HookManager } from '../hooks/index.js';
import { formatProjectInstructions, loadProjectInstructions } from '../project/index.js';
import type { ProjectInstructionOptions } from '../project/index.js';
import { AOS_SYSTEM_PROMPT } from './prompts/AOSSystemPrompt.js';
import { skillToolSet } from '../toolset/SkillTools.js';

export { parseModelString };

function estimateTokens(msg: Message): number {
  const content = msg.content;
  if (typeof content === 'string') return Math.ceil(content.length / 4);
  let total = 0;
  for (const part of content as (TextContent | ImageContent)[]) {
    if (part.type === 'text') total += Math.ceil(part.text.length / 4);
    else total += 85;
  }
  return total;
}

function totalTokens(messages: Message[]): number {
  return messages.reduce((sum, m) => sum + estimateTokens(m), 0);
}

function isAbortError(err: unknown): boolean {
  return err instanceof Error && (err.name === 'AbortError' || err.message === 'The operation was aborted');
}

function createAbortError(): Error {
  const err = new Error('The operation was aborted');
  err.name = 'AbortError';
  return err;
}

export interface AgentSnapshot {
  name: string;
  models: string[];
  model: string;
  usesThinking: boolean;
  toolCount: number;
  skillCount: number;
  messageCount: number;
  recentMessages: Message[];
  skills: Pick<Skill, 'name' | 'description'>[];
  tools: { name: string; description: string }[];
}

export function smartFunc<T>(
  description: string,
  fn: (input: string, context: ExecutionContext) => Promise<T>,
): Tool {
  return {
    name: description.toLowerCase().replace(/\s+/g, '_').slice(0, 64),
    description,
    parameters: z.object({
      input: z.string().describe('Natural language input for the function'),
    }),
    execute: async (args: Record<string, unknown>, context: ExecutionContext) => {
      const input = String(args['input'] ?? '');
      const result = await fn(input, context);
      return { content: typeof result === 'string' ? result : JSON.stringify(result) };
    },
  };
}

export class Agent {
  public readonly name: string;
  private models: string[];
  private provider?: LLMProvider;
  private toolset: ToolSet;
  private skills: Skill[];
  private mcpPlugins: McpPlugin[];
  private memory: Memory;
  private compressor: MessageCompressor;
  private baseSystemPrompt: string;
  private temperature?: number;
  private maxTokens?: number;
  private maxIterations: number;
  private maxRetries: number;
  private maxHistoryTokens: number;
  private responseFormat?: AgentOptions['responseFormat'];
  private projectInstructions?: boolean | ProjectInstructionOptions;
  private projectInstructionsText = '';
  private skillSearchDirs?: string[];
  private skillRootDir: string;
  private hooks?: HookManager;
  private mcpReady = false;

  private onToolCall?: (name: string, args: unknown) => void | Promise<void>;
  private onToolResult?: (name: string, result: unknown) => void | Promise<void>;
  private onMessage?: (msg: Message) => void | Promise<void>;
  private onBeforeRun?: (input: string | Message[]) => void | Promise<void>;
  private onAfterRun?: (result: string) => void | Promise<void>;
  private onError?: (error: Error) => void | Promise<void>;

  constructor(opts: AgentOptions) {
    this.name = opts.name ?? 'agent';
    this.models = Array.isArray(opts.model) ? [...opts.model] : [opts.model];
    if (this.models.length === 0) throw new Error('Agent requires at least one model');
    this.provider = opts.provider;
    this.toolset = opts.toolset ?? new ToolSet(this.name);
    this.skills = [];
    this.mcpPlugins = opts.mcpPlugins ?? [];
    this.memory = opts.memory ?? new InMemoryMemory();
    this.compressor = opts.compressor ?? new SummaryCompressor({ strategy: 'summarize', recentTurnsToKeep: 2 });
    this.baseSystemPrompt = opts.systemPrompt ?? AOS_SYSTEM_PROMPT;
    this.temperature = opts.temperature;
    this.maxTokens = opts.maxTokens;
    this.maxIterations = opts.maxIterations ?? 8;
    this.maxRetries = opts.maxRetries ?? 2;
    this.maxHistoryTokens = opts.maxHistoryTokens ?? 100_000;
    this.responseFormat = opts.responseFormat;
    this.projectInstructions = opts.projectInstructions;
    this.skillSearchDirs = opts.skillSearchDirs;
    this.skillRootDir = typeof opts.projectInstructions === 'object' && opts.projectInstructions.cwd
      ? opts.projectInstructions.cwd
      : process.cwd();
    this.hooks = opts.hooks;
    this.onToolCall = opts.onToolCall;
    this.onToolResult = opts.onToolResult;
    this.onMessage = opts.onMessage;
    this.onBeforeRun = opts.onBeforeRun;
    this.onAfterRun = opts.onAfterRun;
    this.onError = opts.onError;

    for (const skill of opts.skills ?? []) {
      this.addSkill(skill);
    }
    if (!this.toolset.has('list_available_skills')) {
      this.toolset.merge(skillToolSet({
        agent: this,
        rootDir: this.skillRootDir,
        searchDirs: this.skillSearchDirs,
      }));
    }
  }

  clone(overrides?: Partial<AgentOptions>): Agent {
    return new Agent({
      name: this.name,
      model: [...this.models],
      provider: this.provider,
      toolset: this.toolset,
      skills: [...this.skills],
      skillSearchDirs: this.skillSearchDirs,
      mcpPlugins: [...this.mcpPlugins],
      memory: new InMemoryMemory(),
      compressor: this.compressor,
      systemPrompt: this.baseSystemPrompt,
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      maxIterations: this.maxIterations,
      maxRetries: this.maxRetries,
      maxHistoryTokens: this.maxHistoryTokens,
      responseFormat: this.responseFormat,
      projectInstructions: this.projectInstructions,
      hooks: this.hooks,
      onToolCall: this.onToolCall,
      onToolResult: this.onToolResult,
      onMessage: this.onMessage,
      onBeforeRun: this.onBeforeRun,
      onAfterRun: this.onAfterRun,
      onError: this.onError,
      ...overrides,
    });
  }

  get model(): string {
    return parseModelString(this.models[0]).base;
  }

  get usesThinking(): boolean {
    return parseModelString(this.models[0]).extendedThinking;
  }

  get modelsList(): string[] {
    return [...this.models];
  }

  setModel(model: string | string[]): void {
    const next = Array.isArray(model) ? [...model] : [model];
    if (next.length === 0) throw new Error('Agent requires at least one model');
    this.models = next;
  }

  addSkill(skill: Skill): this {
    this.skills = this.skills.filter((s) => s.name !== skill.name);
    this.skills.push(skill);
    return this;
  }

  removeSkill(name: string): boolean {
    const before = this.skills.length;
    this.skills = this.skills.filter((s) => s.name !== name);
    return this.skills.length !== before;
  }

  listSkills(): Skill[] {
    return [...this.skills];
  }

  addTool(tool: Tool, replace = false): this {
    if (replace && this.toolset.has(tool.name)) {
      this.toolset.unregister(tool.name);
    }
    if (this.toolset.has(tool.name)) {
      throw new Error(`Tool '${tool.name}' is already registered in agent '${this.name}'`);
    }
    this.toolset.register(tool);
    return this;
  }

  removeTool(name: string): boolean {
    return this.toolset.unregister(name);
  }

  listTools(): { name: string; description: string }[] {
    return this.toolset.list().map((tool) => ({ name: tool.name, description: tool.description }));
  }

  async executeTool(name: string, args: unknown, metadata?: Record<string, unknown>) {
    return this.toolset.execute(name, args, {
      agentName: this.name,
      metadata,
    });
  }

  addPlugin(plugin: LoadedPlugin): this {
    for (const skill of plugin.skills) this.addSkill(skill);
    for (const tool of plugin.tools) this.addTool(tool, true);
    if (plugin.hooks && this.hooks) {
      plugin.hooks(this.hooks);
    }
    return this;
  }

  async getHistory(limit?: number): Promise<Message[]> {
    return this.memory.recent(limit ?? Number.MAX_SAFE_INTEGER);
  }

  async setHistory(messages: Message[]): Promise<void> {
    await this.memory.clear();
    for (const message of messages) {
      await this.memory.append(message);
    }
  }

  async clearHistory(): Promise<void> {
    await this.memory.clear();
  }

  async remember(key: string, value: string): Promise<void> {
    if (!this.memory.remember) throw new Error('This memory backend does not support long-term storage');
    await this.memory.remember(key, value);
  }

  async recall(key: string): Promise<string | null> {
    if (!this.memory.recall) return null;
    return this.memory.recall(key);
  }

  async compactHistory(maxTokens: number = this.maxHistoryTokens): Promise<Message[]> {
    const current = await this.memory.recent(Number.MAX_SAFE_INTEGER);
    if (current.length === 0) return current;
    const compressed = await this.compressor.compress(current, maxTokens);
    await this.memory.clear();
    for (const message of compressed) {
      await this.memory.append(message);
    }
    return compressed;
  }

  async snapshot(limit = 12): Promise<AgentSnapshot> {
    const recentMessages = await this.memory.recent(limit);
    const allMessages = await this.memory.recent(Number.MAX_SAFE_INTEGER);
    return {
      name: this.name,
      models: [...this.models],
      model: this.model,
      usesThinking: this.usesThinking,
      toolCount: this.toolset.size(),
      skillCount: this.skills.length,
      messageCount: allMessages.length,
      recentMessages,
      skills: this.skills.map((skill) => ({ name: skill.name, description: skill.description })),
      tools: this.listTools(),
    };
  }

  async ready(): Promise<void> {
    if (this.mcpReady) return;
    if (this.projectInstructions) {
      const opts = typeof this.projectInstructions === 'object' ? this.projectInstructions : {};
      this.projectInstructionsText = formatProjectInstructions(await loadProjectInstructions(opts));
    }
    for (const p of this.mcpPlugins) {
      await p.connect();
      const tools = await p.getTools();
      for (const t of tools) {
        if (!this.toolset.has(t.name)) this.toolset.register(t);
      }
    }
    this.mcpReady = true;
  }

  async close(): Promise<void> {
    for (const p of this.mcpPlugins) {
      try {
        await p.disconnect();
      } catch (e) {
        logger.warn(`MCP plugin '${p.name}' disconnect failed:`, (e as Error).message);
      }
    }
    this.mcpReady = false;
  }

  private buildSystemPrompt(): string {
    const parts = [this.baseSystemPrompt];
    if (this.projectInstructionsText) parts.push(this.projectInstructionsText);
    for (const skill of this.skills) {
      parts.push(`\n## Skill: ${skill.name}\n${skill.instructions}`);
      if (skill.tools && skill.tools.length > 0) {
        for (const t of skill.tools) {
          if (!this.toolset.has(t.name)) this.toolset.register(t);
        }
      }
    }
    return parts.join('\n');
  }

  private resolveProvider(model: string): LLMProvider {
    const { base } = parseModelString(model);
    return this.provider ?? providerForModel(base);
  }

  private async maybeCompressHistory(): Promise<void> {
    const recent = await this.memory.recent(Number.MAX_SAFE_INTEGER);
    const used = totalTokens(recent);
    if (used <= this.maxHistoryTokens) return;
    logger.warn(`History token estimate ${used} exceeds ${this.maxHistoryTokens}; compacting history.`);
    await this.compactHistory(this.maxHistoryTokens);
  }

  async *run(input: string | Message[], opts?: { signal?: AbortSignal }): AsyncGenerator<AgentEvent> {
    if (opts?.signal?.aborted) throw createAbortError();

    await this.ready();
    if (this.onBeforeRun) await this.onBeforeRun(input);
    await this.hooks?.emit('agent:beforeRun', { input });

    const userMessages: Message[] =
      typeof input === 'string' ? [{ role: 'user', content: input }] : [...input];

    try {
      for (const m of userMessages) {
        if (opts?.signal?.aborted) throw createAbortError();
        await this.memory.append(m);
        if (this.onMessage) await this.onMessage(m);
        await this.hooks?.emit('message', { message: m });
      }

      const tools = this.toolset.toOpenAITools();
      const baseChatOpts: Omit<ChatOptions, 'model'> = {
        tools: tools.length > 0 ? tools : undefined,
        temperature: this.temperature,
        maxTokens: this.maxTokens,
        stream: true,
        extendedThinking: this.usesThinking,
        responseFormat: this.responseFormat,
      };

      let finalText = '';

      for (let iter = 0; iter < this.maxIterations; iter++) {
        await this.maybeCompressHistory();
        if (opts?.signal?.aborted) throw createAbortError();

        const recent = await this.memory.recent(Number.MAX_SAFE_INTEGER);
        const messages: Message[] = [
          { role: 'system', content: this.buildSystemPrompt() },
          ...recent,
        ];

        const result = await this.callWithFallback(messages, baseChatOpts, opts?.signal);
        let assistantText = '';
        const toolCalls = result.toolCalls;

        for (const chunk of result.textChunks) {
          assistantText += chunk;
          yield { type: 'text', data: chunk };
        }

        const assistantMsg: Message = {
          role: 'assistant',
          content: assistantText,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        };
        await this.memory.append(assistantMsg);
        if (this.onMessage) await this.onMessage(assistantMsg);
        await this.hooks?.emit('message', { message: assistantMsg });

        if (toolCalls.length === 0) {
          finalText = assistantText;
          break;
        }

        yield { type: 'tool_call', data: toolCalls };

        const toolResults = await Promise.all(
          toolCalls.map(async (call) => {
            if (opts?.signal?.aborted) throw createAbortError();
            if (this.onToolCall) await this.onToolCall(call.name, call.arguments);
            await this.hooks?.emit('tool:beforeCall', { name: call.name, args: call.arguments });

            const tr = await this.toolset.execute(call.name, call.arguments, {
              agentName: this.name,
              signal: opts?.signal,
              metadata: { tool_call_id: call.id },
            });
            const fixed = { ...tr, tool_call_id: call.id };

            if (this.onToolResult) await this.onToolResult(call.name, fixed);
            await this.hooks?.emit('tool:afterCall', { name: call.name, result: fixed });
            return { call, fixed };
          }),
        );

        for (const { call, fixed } of toolResults) {
          yield { type: 'tool_result', data: fixed };
          const toolMsg: Message = {
            role: 'tool',
            tool_call_id: call.id,
            name: call.name,
            content: fixed.content,
          };
          await this.memory.append(toolMsg);
          if (this.onMessage) await this.onMessage(toolMsg);
          await this.hooks?.emit('message', { message: toolMsg });
        }
      }

      yield { type: 'done', data: finalText };
      if (this.onAfterRun) await this.onAfterRun(finalText);
      await this.hooks?.emit('agent:afterRun', { result: finalText });
    } catch (err) {
      if (this.onError && err instanceof Error && !isAbortError(err)) {
        await this.onError(err);
      }
      if (err instanceof Error && !isAbortError(err)) {
        await this.hooks?.emit('agent:error', { error: err });
      }
      throw err;
    }
  }

  private async callWithFallback(
    messages: Message[],
    base: Omit<ChatOptions, 'model'>,
    signal?: AbortSignal,
  ): Promise<{ textChunks: string[]; toolCalls: { id: string; name: string; arguments: Record<string, unknown> }[] }> {
    let lastErr: unknown;
    for (const model of this.models) {
      for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
        try {
          const provider = this.resolveProvider(model);
          const { extendedThinking } = parseModelString(model);
          const opts: ChatOptions = { ...base, model, signal, extendedThinking: extendedThinking || base.extendedThinking };
          const textChunks: string[] = [];
          const toolCalls: { id: string; name: string; arguments: Record<string, unknown> }[] = [];
          for await (const ev of provider.chat(messages, opts)) {
            if (signal?.aborted) throw createAbortError();
            if (ev.type === 'text' && ev.text) textChunks.push(ev.text);
            else if (ev.type === 'tool_call' && ev.toolCall) {
              toolCalls.push({
                id: ev.toolCall.id || uid('call'),
                name: ev.toolCall.name,
                arguments: ev.toolCall.arguments,
              });
            }
          }
          return { textChunks, toolCalls };
        } catch (err) {
          if (signal?.aborted || isAbortError(err)) throw err;
          lastErr = err;
          if (attempt < this.maxRetries) {
            const delay = Math.pow(2, attempt) * 500;
            logger.warn(
              `Model '${model}' attempt ${attempt + 1} failed: ${(err as Error).message}. Retrying in ${delay}ms.`,
            );
            await new Promise((res) => setTimeout(res, delay));
          } else {
            logger.warn(`Model '${model}' failed after ${this.maxRetries + 1} attempts. Trying next model.`);
          }
        }
      }
    }
    throw lastErr ?? new Error('All models failed');
  }

  async runToText(input: string | Message[], opts?: { signal?: AbortSignal }): Promise<string> {
    let final = '';
    for await (const ev of this.run(input, opts)) {
      if (ev.type === 'done') final = String(ev.data ?? '');
    }
    return final;
  }
}
