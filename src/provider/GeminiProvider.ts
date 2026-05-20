import type { ChatOptions, Message } from '../types.js';
import { type LLMProvider, type ProviderStreamChunk, parseModelString } from './Provider.js';
import { uid } from '../utils/misc.js';

export interface GeminiProviderOptions {
  apiKey?: string;
}

/** Thin Google Generative AI adapter. */
export class GeminiProvider implements LLMProvider {
  public readonly name = 'gemini';
  public readonly supportsTools = true;
  private genAI: any = null;
  private opts: GeminiProviderOptions;

  constructor(opts: GeminiProviderOptions = {}) {
    this.opts = opts;
  }

  private async ensure(): Promise<any> {
    if (this.genAI) return this.genAI;
    const mod: any = await import('@google/generative-ai');
    const { GoogleGenerativeAI } = mod;
    this.genAI = new GoogleGenerativeAI(this.opts.apiKey ?? process.env.GOOGLE_API_KEY ?? '');
    return this.genAI;
  }

  async *chat(messages: Message[], options: ChatOptions): AsyncGenerator<ProviderStreamChunk> {
    const genAI = await this.ensure();
    const { base } = parseModelString(options.model);

    const system = messages.filter((m) => m.role === 'system').map((m) => m.content).join('\n\n');
    const history = messages
      .filter((m) => m.role !== 'system')
      .map((m) => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
      }));

    const tools =
      options.tools && options.tools.length > 0
        ? [
            {
              functionDeclarations: options.tools.map((t) => ({
                name: t.function.name,
                description: t.function.description,
                parameters: t.function.parameters,
              })),
            },
          ]
        : undefined;

    const model = genAI.getGenerativeModel({
      model: base,
      systemInstruction: system || undefined,
      tools,
    });

    const result = await model.generateContentStream({ contents: history });

    const calls: { id: string; name: string; arguments: Record<string, unknown> }[] = [];
    for await (const chunk of result.stream) {
      const text = typeof chunk.text === 'function' ? chunk.text() : '';
      if (text) yield { type: 'text', text };
      const fnCalls = chunk.functionCalls?.() ?? [];
      for (const c of fnCalls) {
        calls.push({ id: uid('call'), name: c.name, arguments: c.args ?? {} });
      }
    }
    for (const c of calls) yield { type: 'tool_call', toolCall: c };
    yield { type: 'done' };
  }
}
