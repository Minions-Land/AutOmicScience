import type { AgentEvent, ChatOptions, Message } from '../types.js';
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
  private mcpReady = false;

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
    return this.provider ?? providerForModel(model);
  }

  /** Run the agent on input. Yields a stream of AgentEvents. */
  async *run(input: string | Message[]): AsyncGenerator<AgentEvent> {
    await this.ready();

    const userMessages: Message[] =
      typeof input === 'string' ? [{ role: 'user', content: input }] : [...input];
    for (const m of userMessages) await this.memory.append(m);

    const tools = this.toolset.toOpenAITools();
    const baseChatOpts: Omit<ChatOptions, 'model'> = {
      tools: tools.length > 0 ? tools : undefined,
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      stream: true,
    };

    let finalText = '';

    for (let iter = 0; iter < this.maxIterations; iter++) {
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

      if (toolCalls.length === 0) {
        finalText = assistantText;
        break;
      }

      // Execute tools.
      for (const call of toolCalls) {
        yield { type: 'tool_call', data: call };
        const tr = await this.toolset.execute(call.name, call.arguments, {
          agentName: this.name,
          metadata: { tool_call_id: call.id },
        });
        const fixed = { ...tr, tool_call_id: call.id };
        yield { type: 'tool_result', data: fixed };
        await this.memory.append({
          role: 'tool',
          tool_call_id: call.id,
          name: call.name,
          content: fixed.content,
        });
      }
    }

    yield { type: 'done', data: finalText };
  }

  private async callWithFallback(
    messages: Message[],
    base: Omit<ChatOptions, 'model'>,
  ): Promise<{ textChunks: string[]; toolCalls: { id: string; name: string; arguments: Record<string, unknown> }[] }> {
    let lastErr: unknown;
    for (const model of this.models) {
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
        logger.warn(`Model '${model}' failed: ${(err as Error).message}. Trying fallback.`);
        lastErr = err;
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
