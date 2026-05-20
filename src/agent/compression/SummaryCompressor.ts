import type { Message } from '../../types.js';
import type { MessageCompressor } from './Compressor.js';

/**
 * Simple compressor that truncates older messages to fit the token budget.
 * Placeholder for an LLM-driven summarization compressor.
 *
 * Strategy: keep the system message (if any) and the most recent messages
 * that fit within the approximate token budget (estimated at 4 chars/token).
 */
export class SummaryCompressor implements MessageCompressor {
  private readonly charsPerToken: number;

  constructor(charsPerToken = 4) {
    this.charsPerToken = charsPerToken;
  }

  /**
   * Compress by keeping the system message and trimming from the front.
   * @param messages - The full message history.
   * @param maxTokens - Target token budget.
   */
  async compress(messages: Message[], maxTokens: number): Promise<Message[]> {
    const maxChars = maxTokens * this.charsPerToken;

    // Always keep the system message if present
    const systemMsg = messages.find((m) => m.role === 'system');
    const nonSystem = messages.filter((m) => m.role !== 'system');

    let budget = maxChars;
    if (systemMsg) {
      budget -= systemMsg.content.length;
    }

    // Keep messages from the end until budget is exhausted
    const kept: Message[] = [];
    for (let i = nonSystem.length - 1; i >= 0; i--) {
      const msg = nonSystem[i];
      const len = msg.content.length;
      if (budget - len < 0 && kept.length > 0) break;
      budget -= len;
      kept.unshift(msg);
    }

    if (systemMsg) {
      return [systemMsg, ...kept];
    }
    return kept;
  }
}
