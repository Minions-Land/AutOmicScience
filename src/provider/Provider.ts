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

/** Strip the `+think` suffix; return base model and an extendedThinking flag. */
export function parseModelString(model: string): { base: string; extendedThinking: boolean } {
  if (model.endsWith('+think')) {
    return { base: model.slice(0, -'+think'.length), extendedThinking: true };
  }
  return { base: model, extendedThinking: false };
}
