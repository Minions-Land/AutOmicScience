import type { Message } from '../types.js';
import { extractTextContent } from './messageFormatter.js';

const MODEL_CONTEXT_WINDOWS: Record<string, number> = {
  'gpt-4o': 128_000,
  'gpt-4o-mini': 128_000,
  'gpt-4-turbo': 128_000,
  'gpt-4': 8_192,
  'gpt-3.5-turbo': 16_385,
  'o1': 200_000,
  'o1-mini': 128_000,
  'o3': 200_000,
  'o3-mini': 200_000,
  'o4-mini': 200_000,
  'gemini-2.5-pro': 1_000_000,
  'gemini-2.5-flash': 1_000_000,
  'gemini-2.0-flash': 1_000_000,
};

export function estimateTokens(text: string): number {
  if (!text) return 0;

  let cjkChars = 0;
  let asciiChars = 0;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (
      (code >= 0x4e00 && code <= 0x9fff) ||
      (code >= 0x3040 && code <= 0x30ff) ||
      (code >= 0xac00 && code <= 0xd7af)
    ) {
      cjkChars++;
    } else if (code < 128) {
      asciiChars++;
    }
  }
  const otherChars = text.length - cjkChars - asciiChars;
  return Math.max(1, Math.ceil(cjkChars * 0.6 + asciiChars * 0.25 + otherChars * 0.5));
}

export function estimateMessagesTokens(messages: Message[]): number {
  let total = 0;
  for (const msg of messages) {
    total += 4;
    total += estimateTokens(extractTextContent(msg));
    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        total += estimateTokens(tc.name + JSON.stringify(tc.arguments ?? {}));
      }
    }
  }
  return total;
}

export function fitWithinBudget(messages: Message[], budget: number): Message[] {
  const total = estimateMessagesTokens(messages);
  if (total <= budget) return messages;

  const system = messages.filter((m) => m.role === 'system');
  const nonSystem = messages.filter((m) => m.role !== 'system');
  const result = [...nonSystem];
  let currentTotal = total;

  while (currentTotal > budget && result.length > 2) {
    const removed = result.shift()!;
    currentTotal -= 4 + estimateTokens(extractTextContent(removed));
  }

  return [...system, ...result];
}

export function tokenBudgetForModel(model: string): number {
  const providerPrefix = model.match(/^(openai|anthropic|gemini)\//)?.[1];
  const base = model.replace(/^(openai|anthropic|gemini)\//, '').replace(/\+think(:\w+)?$/, '');
  const lower = base.toLowerCase();

  if (MODEL_CONTEXT_WINDOWS[lower]) return MODEL_CONTEXT_WINDOWS[lower];

  for (const [key, value] of Object.entries(MODEL_CONTEXT_WINDOWS)) {
    if (lower.startsWith(key)) return value;
  }

  if (providerPrefix === 'anthropic') return 200_000;
  if (lower.startsWith('gemini')) return 1_000_000;
  if (lower.startsWith('gpt-4')) return 128_000;

  return 128_000;
}
