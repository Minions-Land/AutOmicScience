import type { ChatOptions, Message, ToolCall } from '../types.js';

export interface ProviderStreamChunk {
  type: 'text' | 'tool_call' | 'done';
  text?: string;
  toolCall?: ToolCall;
  finishReason?: string;
}

export interface LLMProvider {
  name: string;
  supportsTools: boolean;
  /** Streams provider events. Implementations MUST emit a final 'done' chunk. */
  chat(messages: Message[], options: ChatOptions): AsyncGenerator<ProviderStreamChunk>;
}

/** Strip the `+think[:level]` suffix; return base model and thinking config. */
export function parseModelString(model: string): { base: string; extendedThinking: boolean; thinkingLevel?: 'low' | 'medium' | 'high' } {
  const thinkMatch = model.match(/^(.+)\+think(?::(\w+))?$/);
  if (thinkMatch) {
    const level = thinkMatch[2] as 'low' | 'medium' | 'high' | undefined;
    return { base: thinkMatch[1], extendedThinking: true, thinkingLevel: level };
  }
  return { base: model, extendedThinking: false };
}
