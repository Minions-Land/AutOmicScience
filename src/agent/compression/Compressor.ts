import type { Message } from '../../types.js';

/**
 * Compresses a message history to fit within a token budget.
 */
export interface MessageCompressor {
  /**
   * Compress the message array to fit within maxTokens.
   * @param messages - The full message history.
   * @param maxTokens - Target token budget.
   * @returns A compressed message array that fits the budget.
   */
  compress(messages: Message[], maxTokens: number): Promise<Message[]>;
}
