import type { AgentEvent, ChatOptions, Message, TextContent, ImageContent } from '../types.js';
import { ToolSet } from '../toolset/ToolSet.js';
import { InMemoryMemory } from '../memory/InMemoryMemory.js';
import type { Memory } from '../memory/Memory.js';
import type { LLMProvider } from '../provider/Provider.js';
import { providerForModel } from '../provider/ModelSelector.js';
import type { Skill } from '../skill/Skill.js';
import type { McpPlugin } from '../mcp/McpPlugin.js';
import type { AgentOptions } from './AgentOptions.js';
import { logger } from '../utils/logger.js';
import { uid } from '../utils/misc.js';
import type { Tool } from '../toolset/Tool.js';
import { z } from '../toolset/Tool.js';
import type { ExecutionContext } from '../types.js';

// ── Token counting helpers ──────────────────────────────────────────────────

function estimateTokens(msg: Message): number {
  const content = msg.content;
  if (typeof content === 'string') return Math.ceil(content.length / 4);
  // multimodal: text parts only (images counted as ~85 tokens each)
  let total = 0;
  for (const part of content as (TextContent | ImageContent)[]) {
    if (part.type === 'text') total += Math.ceil(part.text.length / 4);
    else total += 85; // rough vision token estimate
  }
  return total;
}

function totalTokens(messages: Message[]): number {
  return messages.reduce((sum, m) => sum + estimateTokens(m), 0);
}

// ── Think suffix parsing ────────────────────────────────────────────────────

import { parseModelString } from '../provider/Provider.js';
export { parseModelString };

// ── smartFunc factory ───────────────────────────────────────────────────────

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

// ── Agent ───────────────────────────────────────────────────────────────────

export class Agent {
  public readonly name: string;
  private models: string[];
  private provider?: LLMProvider;
  private toolset: ToolSet;
  private skills: Skill[];
  private mcpPlugins: McpPlugin[];
  private memory: Memory;
  private baseSystemPrompt: string;
  private temperature?: number;
  private maxTokens?: number;
  private maxIterations: number;
  private maxRetries: number;
  private maxHistoryTokens: number;
  private responseFormat?: AgentOptions['responseFormat'];
  private mcpReady = false;

  // Hooks
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
    this.skills = opts.skills ?? [];
    this.mcpPlugins = opts.mcpPlugins ?? [];
    this.memory = opts.memory ?? new InMemoryMemory();
    this.baseSystemPrompt = opts.systemPrompt ?? 'You are a helpful AI assistant.';
    this.temperature = opts.temperature;
    this.maxTokens = opts.maxTokens;
    this.maxIterations = opts.maxIterations ?? 8;
    this.maxRetries = opts.maxRetries ?? 2;
    this.maxHistoryTokens = opts.maxHistoryTokens ?? 100_000;
    this.responseFormat = opts.responseFormat;
    this.onToolCall = opts.onToolCall;
    this.onToolResult = opts.onToolResult;
    this.onMessage = opts.onMessage;
    this.onBeforeRun = opts.onBeforeRun;
    this.onAfterRun = opts.onAfterRun;
    this.onError = opts.onError;
  }

  /** Create a copy of this agent with optional overrides. */
  clone(overrides?: Partial<AgentOptions>): Agent {
    return new Agent({
      name: this.name,
      model: [...this.models],
      provider: this.provider,
      toolset: this.toolset,
      skills: [...this.skills],
      mcpPlugins: [...this.mcpPlugins],
      memory: new InMemoryMemory(),
      systemPrompt: this.baseSystemPrompt,
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      maxIterations: this.maxIterations,
      maxRetries: this.maxRetries,
      maxHistoryTokens: this.maxHistoryTokens,
      responseFormat: this.responseFormat,
      onToolCall: this.onToolCall,
      onToolResult: this.onToolResult,
      onMessage: this.onMessage,
      onBeforeRun: this.onBeforeRun,
      onAfterRun: this.onAfterRun,
      onError: this.onError,
      ...overrides,
    });
  }

  /** Get the primary model (with +think suffix stripped). */
  get model(): string {
    return parseModelString(this.models[0]).base;
  }

  /** Check if this agent uses extended thinking. */
  get usesThinking(): boolean {
    return parseModelString(this.models[0]).extendedThinking;
  }

  /** Connect MCP plugins and merge their tools into this agent's toolset. */
  async ready(): Promise<void> {
    if (this.mcpReady) return;
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

  /** Auto-truncate history if over token budget using simple tail-truncation. */
  private async maybeCompressHistory(): Promise<void> {
    const recent = await this.memory.recent();
    const used = totalTokens(recent);
    if (used <= this.maxHistoryTokens) return;

    // Drop oldest non-system messages until under budget
    logger.warn(
      `History token estimate ${used} exceeds ${this.maxHistoryTokens}; trimming oldest messages.`,
    );
    // We can't mutate memory directly — clear and re-append trimmed set
    const systemMessages = recent.filter((m) => m.role === 'system');
    const nonSystem = recent.filter((m) => m.role !== 'system');

    let budget = this.maxHistoryTokens - totalTokens(systemMessages);
    const kept: Message[] = [];
    // Walk from newest to oldest
    for (let i = nonSystem.length - 1; i >= 0; i--) {
      const t = estimateTokens(nonSystem[i]);
      if (budget - t < 0) break;
      budget -= t;
      kept.unshift(nonSystem[i]);
    }

    // Rebuild memory with trimmed history
    if ('clear' in this.memory && typeof (this.memory as unknown as { clear: () => Promise<void> }).clear === 'function') {
      await (this.memory as unknown as { clear: () => Promise<void> }).clear();
      for (const m of [...systemMessages, ...kept]) {
        await this.memory.append(m);
      }
    }
  }

  /** Run the agent on input. Yields a stream of AgentEvents. */
  async *run(input: string | Message[]): AsyncGenerator<AgentEvent> {
    await this.ready();
    if (this.onBeforeRun) await this.onBeforeRun(input);

    const userMessages: Message[] =
      typeof input === 'string' ? [{ role: 'user', content: input }] : [...input];
    for (const m of userMessages) {
      await this.memory.append(m);
      if (this.onMessage) await this.onMessage(m);
    }

    const tools = this.toolset.toOpenAITools();
    const baseChatOpts: Omit<ChatOptions, 'model'> = {
      tools: tools.length > 0 ? tools : undefined,
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      stream: true,
      responseFormat: this.responseFormat,
    };

    let finalText = '';

    for (let iter = 0; iter < this.maxIterations; iter++) {
      // Auto-truncate if history is too long
      await this.maybeCompressHistory();

      const recent = await this.memory.recent();
      const messages: Message[] = [
        { role: 'system', content: this.buildSystemPrompt() },
        ...recent,
      ];

      const result = await this.callWithFallback(messages, baseChatOpts);
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

      if (toolCalls.length === 0) {
        finalText = assistantText;
        break;
      }

      // Execute tools in parallel (feature 4)
      yield { type: 'tool_call', data: toolCalls };

      const toolResults = await Promise.all(
        toolCalls.map(async (call) => {
          // Fire onToolCall hook
          if (this.onToolCall) await this.onToolCall(call.name, call.arguments);

          const tr = await this.toolset.execute(call.name, call.arguments, {
            agentName: this.name,
            metadata: { tool_call_id: call.id },
          });
          const fixed = { ...tr, tool_call_id: call.id };

          // Fire onToolResult hook
          if (this.onToolResult) await this.onToolResult(call.name, fixed);

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
      }
    }

    yield { type: 'done', data: finalText };
    if (this.onAfterRun) await this.onAfterRun(finalText);
  }

  private async callWithFallback(
    messages: Message[],
    base: Omit<ChatOptions, 'model'>,
  ): Promise<{ textChunks: string[]; toolCalls: { id: string; name: string; arguments: Record<string, unknown> }[] }> {
    let lastErr: unknown;
    for (const model of this.models) {
      // Retry with exponential backoff (feature 6)
      for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
        try {
          const provider = this.resolveProvider(model);
          const opts: ChatOptions = { ...base, model };
          const textChunks: string[] = [];
          const toolCalls: { id: string; name: string; arguments: Record<string, unknown> }[] = [];
          for await (const ev of provider.chat(messages, opts)) {
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
          lastErr = err;
          if (attempt < this.maxRetries) {
            const delay = Math.pow(2, attempt) * 500; // 500ms, 1000ms, 2000ms
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

  /** Convenience: run to completion and return the final text. */
  async runToText(input: string | Message[]): Promise<string> {
    let final = '';
    for await (const ev of this.run(input)) {
      if (ev.type === 'done') final = String(ev.data ?? '');
    }
    return final;
  }
}
