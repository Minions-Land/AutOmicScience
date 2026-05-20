import type { ChatOptions, Message } from '../types.js';
import { type LLMProvider, type ProviderStreamChunk, parseModelString } from './Provider.js';
import { uid } from '../utils/misc.js';

export interface AnthropicProviderOptions {
  apiKey?: string;
}

/**
 * Thin Anthropic adapter. Recognizes the `+think` suffix and forwards
 * extended-thinking via the `thinking` parameter.
 */
export class AnthropicProvider implements LLMProvider {
  public readonly name = 'anthropic';
  public readonly supportsTools = true;
  private client: any = null;
  private opts: AnthropicProviderOptions;

  constructor(opts: AnthropicProviderOptions = {}) {
    this.opts = opts;
  }

  private async ensureClient(): Promise<any> {
    if (this.client) return this.client;
    const mod: any = await import('@anthropic-ai/sdk');
    const Anthropic = mod.default ?? mod.Anthropic;
    this.client = new Anthropic({ apiKey: this.opts.apiKey ?? process.env.ANTHROPIC_API_KEY });
    return this.client;
  }

  async *chat(messages: Message[], options: ChatOptions): AsyncGenerator<ProviderStreamChunk> {
    const client = await this.ensureClient();
    const { base, extendedThinking } = parseModelString(options.model);
    const useThinking = extendedThinking || options.extendedThinking;

    const system = messages.filter((m) => m.role === 'system').map((m) => m.content).join('\n\n');
    const convo = messages.filter((m) => m.role !== 'system').map(toAnthropicMessage);

    const tools = (options.tools ?? []).map((t) => ({
      name: t.function.name,
      description: t.function.description,
      input_schema: t.function.parameters,
    }));

    const params: any = {
      model: base,
      system: system || undefined,
      messages: convo,
      max_tokens: options.maxTokens ?? 4096,
      temperature: useThinking ? 1 : options.temperature,
      tools: tools.length > 0 ? tools : undefined,
      stream: true,
    };
    if (useThinking) {
      params.thinking = { type: 'enabled', budget_tokens: 4096 };
    }

    const stream = await client.messages.create(params);

    let stopReason: string | undefined;
    const toolBuf = new Map<number, { id: string; name: string; argText: string }>();

    for await (const event of stream) {
      if (event.type === 'content_block_start' && event.content_block?.type === 'tool_use') {
        toolBuf.set(event.index, {
          id: event.content_block.id ?? uid('call'),
          name: event.content_block.name,
          argText: '',
        });
      } else if (event.type === 'content_block_delta') {
        const d = event.delta;
        if (d?.type === 'text_delta' && d.text) {
          yield { type: 'text', text: d.text };
        } else if (d?.type === 'input_json_delta' && d.partial_json) {
          const cur = toolBuf.get(event.index);
          if (cur) cur.argText += d.partial_json;
        }
      } else if (event.type === 'message_delta') {
        if (event.delta?.stop_reason) stopReason = event.delta.stop_reason;
      }
    }

    for (const tc of toolBuf.values()) {
      let parsed: Record<string, unknown> = {};
      try {
        parsed = tc.argText ? JSON.parse(tc.argText) : {};
      } catch {
        parsed = { _raw: tc.argText };
      }
      yield { type: 'tool_call', toolCall: { id: tc.id, name: tc.name, arguments: parsed } };
    }

    yield { type: 'done', finishReason: stopReason };
  }
}

function toAnthropicMessage(m: Message): any {
  if (m.role === 'tool') {
    return {
      role: 'user',
      content: [
        {
          type: 'tool_result',
          tool_use_id: m.tool_call_id,
          content: m.content,
        },
      ],
    };
  }
  if (m.role === 'assistant' && m.tool_calls && m.tool_calls.length > 0) {
    const blocks: any[] = [];
    if (m.content) blocks.push({ type: 'text', text: m.content });
    for (const tc of m.tool_calls) {
      blocks.push({ type: 'tool_use', id: tc.id, name: tc.name, input: tc.arguments });
    }
    return { role: 'assistant', content: blocks };
  }
  return { role: m.role === 'assistant' ? 'assistant' : 'user', content: m.content };
}
