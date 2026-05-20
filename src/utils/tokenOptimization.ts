import type { Message } from '../types.js';
import { extractTextContent } from './messageFormatter.js';

/** Known model context window sizes (input tokens). */
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
  'claude-sonnet-4-5': 200_000,
  'claude-opus-4': 200_000,
  'claude-3-5-sonnet': 200_000,
  'claude-3-haiku': 200_000,
  'gemini-2.5-pro': 1_000_000,
  'gemini-2.5-flash': 1_000_000,
  'gemini-2.0-flash': 1_000_000,
};

/**
 * Estimate token count for a string using the ~4 chars/token heuristic.
 * Accounts for CJK characters which use more tokens per character.
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;

  let cjkChars = 0;
  let asciiChars = 0;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (
      (code >= 0x4e00 && code <= 0x9fff) || // CJK Unified
      (code >= 0x3040 && code <= 0x30ff) || // Hiragana + Katakana
      (code >= 0xac00 && code <= 0xd7af)    // Hangul
    ) {
      cjkChars++;
    } else if (code < 128) {
      asciiChars++;
    }
  }
  const otherChars = text.length - cjkChars - asciiChars;

  // CJK ~1.7 chars/token, ASCII ~4 chars/token, other ~2 chars/token
  const tokens = cjkChars * 0.6 + asciiChars * 0.25 + otherChars * 0.5;
  return Math.max(1, Math.ceil(tokens));
}

/** Estimate total tokens for an array of messages. */
export function estimateMessagesTokens(messages: Message[]): number {
  let total = 0;
  for (const msg of messages) {
    // Per-message overhead (role, framing)
    total += 4;
    total += estimateTokens(extractTextContent(msg));
    // Tool calls add tokens
    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        total += estimateTokens(tc.name + JSON.stringify(tc.arguments ?? {}));
      }
    }
  }
  return total;
}

/**
 * Trim messages to fit within a token budget.
 * Preserves system messages and the most recent messages.
 * Drops oldest non-system messages first.
 */
export function fitWithinBudget(messages: Message[], budget: number): Message[] {
  const total = estimateMessagesTokens(messages);
  if (total <= budget) return messages;

  const system = messages.filter((m) => m.role === 'system');
  const nonSystem = messages.filter((m) => m.role !== 'system');

  // Keep at least the last 2 messages
  const minKeep = 2;
  const result = [...nonSystem];
  let currentTotal = total;

  while (currentTotal > budget && result.length > minKeep) {
    const removed = result.shift()!;
    currentTotal -= 4 + estimateTokens(extractTextContent(removed));
  }

  return [...system, ...result];
}

/**
 * Return the context window size for a known model.
 * Falls back to 128k for unknown models.
 */
export function tokenBudgetForModel(model: string): number {
  // Strip provider prefix and thinking suffix
  const base = model.replace(/^(openai|anthropic|gemini)\//, '').replace(/\+think(:\w+)?$/, '');
  const lower = base.toLowerCase();

  // Exact match
  if (MODEL_CONTEXT_WINDOWS[lower]) return MODEL_CONTEXT_WINDOWS[lower];

  // Prefix match
  for (const [key, value] of Object.entries(MODEL_CONTEXT_WINDOWS)) {
    if (lower.startsWith(key)) return value;
  }

  // Heuristic defaults by provider prefix
  if (lower.startsWith('claude')) return 200_000;
  if (lower.startsWith('gemini')) return 1_000_000;
  if (lower.startsWith('gpt-4')) return 128_000;

  return 128_000;
}
