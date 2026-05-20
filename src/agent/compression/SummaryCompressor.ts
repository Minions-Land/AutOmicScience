import type { Message } from '../../types.js';
import type { MessageCompressor } from './Compressor.js';

/**
 * Compressor that manages message history within a token budget.
 *
 * Strategy:
 * 1. If messages already fit within maxTokens, return as-is.
 * 2. Otherwise, keep the system message and the last 2 user/assistant turns intact.
 * 3. Replace older messages with a single summary message.
 *
 * Currently uses truncation-based summarization. The point where an LLM
 * summarization call would plug in is marked with a comment below.
 */
export class SummaryCompressor implements MessageCompressor {
  private readonly charsPerToken: number;

  constructor(charsPerToken = 4) {
    this.charsPerToken = charsPerToken;
  }

  /**
   * Compress messages to fit within the token budget.
   * @param messages - The full message history.
   * @param maxTokens - Target token budget.
   */
  async compress(messages: Message[], maxTokens: number): Promise<Message[]> {
    const maxChars = maxTokens * this.charsPerToken;

    // Check if messages already fit
    const totalChars = messages.reduce((sum, m) => sum + m.content.length, 0);
    if (totalChars <= maxChars) {
      return messages;
    }

    // Separate system message
    const systemMsg = messages.find((m) => m.role === 'system');
    const nonSystem = messages.filter((m) => m.role !== 'system');

    // Keep the last 2 user/assistant conversation turns (up to 4 messages)
    const recentTurns: Message[] = [];
    let turnCount = 0;
    for (let i = nonSystem.length - 1; i >= 0 && turnCount < 2; i--) {
      const msg = nonSystem[i];
      recentTurns.unshift(msg);
      if (msg.role === 'user') turnCount++;
    }

    // The older messages that will be summarized
    const olderMessages = nonSystem.slice(0, nonSystem.length - recentTurns.length);

    if (olderMessages.length === 0) {
      // Nothing to compress — just keep system + recent
      return systemMsg ? [systemMsg, ...recentTurns] : recentTurns;
    }

    // Compute budget for the summary
    let usedChars = recentTurns.reduce((sum, m) => sum + m.content.length, 0);
    if (systemMsg) usedChars += systemMsg.content.length;
    const summaryBudget = Math.max(maxChars - usedChars, 200);

    // --- LLM SUMMARIZATION PLUG-IN POINT ---
    // In production, replace this truncation with an LLM call:
    //   const summary = await llm.summarize(olderMessages, summaryBudget);
    // For now, we truncate the concatenated older messages to fit the budget.
    const concatenated = olderMessages
      .map((m) => `[${m.role}]: ${m.content}`)
      .join('\n');
    const truncated = concatenated.slice(0, summaryBudget);
    // --- END LLM PLUG-IN POINT ---

    const summaryMessage: Message = {
      role: 'system',
      content: `[Prior context summary]: ${truncated}`,
    };

    const result: Message[] = [];
    if (systemMsg) result.push(systemMsg);
    result.push(summaryMessage);
    result.push(...recentTurns);
    return result;
  }
}
