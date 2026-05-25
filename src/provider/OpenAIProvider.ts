import type { ChatOptions, Message } from '../types.js';
import { type LLMProvider, type ProviderStreamChunk, parseModelString } from './Provider.js';
import { safeJsonParse, uid } from '../utils/misc.js';

export interface OpenAIProviderOptions {
  apiKey?: string;
  baseURL?: string;
}

/**
 * Streaming OpenAI Chat Completions provider with tool calls.
 * Lazy-imports the `openai` SDK.
 */
export class OpenAIProvider implements LLMProvider {
  public readonly name = 'openai';
  public readonly supportsTools = true;
  private client: any = null;
  private opts: OpenAIProviderOptions;

  constructor(opts: OpenAIProviderOptions = {}) {
    this.opts = opts;
  }

  private async ensureClient(): Promise<any> {
    if (this.client) return this.client;
    const mod: any = await import('openai');
    const OpenAI = mod.default ?? mod.OpenAI;
    this.client = new OpenAI({
      apiKey: this.opts.apiKey ?? process.env.AOS_OPENAI_API_KEY ?? process.env.OPENAI_API_KEY,
      baseURL: this.opts.baseURL ?? openAICompatibleBaseURL(),
    });
    return this.client;
  }

  async *chat(messages: Message[], options: ChatOptions): AsyncGenerator<ProviderStreamChunk> {
    const client = await this.ensureClient();
    const { base } = parseModelString(options.model);

    const stream = await client.chat.completions.create({
      model: base,
      messages: messages.map(toOpenAIMessage),
      tools: options.tools && options.tools.length > 0 ? options.tools : undefined,
      tool_choice: options.tools && options.tools.length > 0 ? 'auto' : undefined,
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      stream: true,
    }, options.signal ? { signal: options.signal } : undefined);

    // Aggregate tool-call deltas by index.
    const toolBuf: Map<number, { id: string; name: string; argText: string }> = new Map();
    let finishReason: string | undefined;

    for await (const chunk of stream) {
      const choice = chunk.choices?.[0];
      if (!choice) continue;
      const delta = choice.delta ?? {};
      if (typeof delta.content === 'string' && delta.content.length > 0) {
        yield { type: 'text', text: delta.content };
      }
      if (Array.isArray(delta.tool_calls)) {
        for (const tc of delta.tool_calls) {
          const idx = tc.index ?? 0;
          const cur = toolBuf.get(idx) ?? { id: tc.id ?? uid('call'), name: '', argText: '' };
          if (tc.id) cur.id = tc.id;
          if (tc.function?.name) cur.name += tc.function.name;
          if (tc.function?.arguments) cur.argText += tc.function.arguments;
          toolBuf.set(idx, cur);
        }
      }
      if (choice.finish_reason) finishReason = choice.finish_reason;
    }

    for (const tc of toolBuf.values()) {
      yield {
        type: 'tool_call',
        toolCall: {
          id: tc.id,
          name: tc.name,
          arguments: safeJsonParse<Record<string, unknown>>(tc.argText || '{}', {}),
        },
      };
    }

    yield { type: 'done', finishReason };
  }
}

function openAICompatibleBaseURL(): string | undefined {
  const raw =
    process.env.AOS_OPENAI_BASE_URL ??
    process.env.OPENAI_BASE_URL ??
    process.env.NEWAPI_BASE_URL;
  if (!raw) return undefined;
  const trimmed = raw.trim().replace(/\/+$/, '');
  return trimmed.endsWith('/v1') ? trimmed : `${trimmed}/v1`;
}

function toOpenAIMessage(m: Message): any {
  if (m.role === 'tool') {
    return { role: 'tool', tool_call_id: m.tool_call_id, content: m.content };
  }
  if (m.role === 'assistant' && m.tool_calls && m.tool_calls.length > 0) {
    return {
      role: 'assistant',
      content: m.content || null,
      tool_calls: m.tool_calls.map((tc) => ({
        id: tc.id,
        type: 'function',
        function: { name: tc.name, arguments: JSON.stringify(tc.arguments ?? {}) },
      })),
    };
  }
  return { role: m.role, content: m.content, name: m.name };
}
