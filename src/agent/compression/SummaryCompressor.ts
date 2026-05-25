import type { Message } from '../../types.js';
import type { MessageCompressor } from './Compressor.js';

export type CompressionStrategy = 'truncate' | 'summarize' | 'sliding-window';

export interface SummaryCompressorOptions {
  /** Characters per token estimate. Default: 4. */
  charsPerToken?: number;
  /** Compression strategy. Default: 'summarize'. */
  strategy?: CompressionStrategy;
  /** Number of recent turns to always keep intact. Default: 2. */
  recentTurnsToKeep?: number;
  /** For sliding-window: window size in messages. Default: 20. */
  windowSize?: number;
  /** Optional LLM summarizer function. If not provided, falls back to truncation. */
  summarizer?: (messages: Message[], maxTokens: number) => Promise<string>;
  /** Messages marked as pinned (by index) are never removed. */
  pinnedIndices?: Set<number>;
}

/**
 * Compressor that manages message history within a token budget.
 *
 * Strategies:
 * - truncate: Drop oldest messages, keeping recent turns.
 * - summarize: Use an LLM to summarize older messages into a compact block.
 * - sliding-window: Keep a fixed window of recent messages, drop the rest.
 *
 * Features:
 * - LLM-driven summarization (pluggable summarizer function)
 * - Token budget awareness
 * - Preserve pinned messages (never removed)
 * - Multiple strategy support
 */
export class SummaryCompressor implements MessageCompressor {
  private readonly charsPerToken: number;
  private readonly strategy: CompressionStrategy;
  private readonly recentTurnsToKeep: number;
  private readonly windowSize: number;
  private readonly summarizer?: (messages: Message[], maxTokens: number) => Promise<string>;
  private readonly pinnedIndices: Set<number>;

  constructor(opts: SummaryCompressorOptions = {}) {
    this.charsPerToken = opts.charsPerToken ?? 4;
    this.strategy = opts.strategy ?? 'summarize';
    this.recentTurnsToKeep = opts.recentTurnsToKeep ?? 2;
    this.windowSize = opts.windowSize ?? 20;
    this.summarizer = opts.summarizer;
    this.pinnedIndices = opts.pinnedIndices ?? new Set();
  }

  async compress(messages: Message[], maxTokens: number): Promise<Message[]> {
    const maxChars = maxTokens * this.charsPerToken;

    // Check if messages already fit
    if (this.totalChars(messages) <= maxChars) {
      return messages;
    }

    switch (this.strategy) {
      case 'truncate':
        return this.truncateStrategy(messages, maxChars);
      case 'sliding-window':
        return this.slidingWindowStrategy(messages, maxChars);
      case 'summarize':
      default:
        return this.summarizeStrategy(messages, maxChars, maxTokens);
    }
  }

  // ── Truncate Strategy ─────────────────────────────────────────────────────

  private truncateStrategy(messages: Message[], maxChars: number): Message[] {
    const { systemMsg, pinned, recent, older } = this.partitionMessages(messages);

    if (older.length === 0) {
      return this.assembleResult(systemMsg, pinned, [], recent);
    }

    // Compute budget for older messages
    let usedChars = this.messagesChars(recent) + this.messagesChars(pinned);
    if (systemMsg) usedChars += this.msgChars(systemMsg);
    const summaryBudget = Math.max(maxChars - usedChars, 200);

    // Truncate concatenated older messages
    const concatenated = older
      .map((m) => `[${m.role}]: ${this.getContent(m)}`)
      .join('\n');
    const truncated = concatenated.slice(0, summaryBudget);

    const summaryMessage: Message = {
      role: 'system',
      content: `[Prior context summary]: ${truncated}`,
    };

    return this.assembleResult(systemMsg, pinned, [summaryMessage], recent);
  }

  // ── Sliding Window Strategy ───────────────────────────────────────────────

  private slidingWindowStrategy(messages: Message[], maxChars: number): Message[] {
    const systemMsg = messages.find((m) => m.role === 'system');
    const nonSystem = messages.filter((m) => m.role !== 'system');

    // Keep pinned messages + last N messages
    const pinned = nonSystem.filter((_, i) => this.pinnedIndices.has(i));
    const window = nonSystem.slice(-this.windowSize);

    // Merge pinned that aren't already in window
    const windowSet = new Set(window);
    const extraPinned = pinned.filter((m) => !windowSet.has(m));

    const result: Message[] = [];
    if (systemMsg) result.push(systemMsg);
    result.push(...extraPinned);
    result.push(...window);

    // If still over budget, trim from the start of the window
    while (this.totalChars(result) > maxChars && result.length > 2) {
      const idx = systemMsg ? 1 : 0;
      if (!this.pinnedIndices.has(idx)) {
        result.splice(idx + (extraPinned.length), 1);
      } else {
        break;
      }
    }

    return result;
  }

  // ── Summarize Strategy ────────────────────────────────────────────────────

  private async summarizeStrategy(
    messages: Message[],
    maxChars: number,
    maxTokens: number,
  ): Promise<Message[]> {
    const { systemMsg, pinned, recent, older } = this.partitionMessages(messages);

    if (older.length === 0) {
      return this.assembleResult(systemMsg, pinned, [], recent);
    }

    // Compute budget for summary
    let usedChars = this.messagesChars(recent) + this.messagesChars(pinned);
    if (systemMsg) usedChars += this.msgChars(systemMsg);
    const summaryBudgetChars = Math.max(maxChars - usedChars, 200);
    const summaryBudgetTokens = Math.floor(summaryBudgetChars / this.charsPerToken);

    let summaryText: string;

    if (this.summarizer) {
      // Use LLM-driven summarization
      try {
        summaryText = await this.summarizer(older, summaryBudgetTokens);
      } catch {
        // Fallback to truncation if LLM fails
        summaryText = this.fallbackTruncate(older, summaryBudgetChars);
      }
    } else {
      // No summarizer provided — use truncation fallback
      summaryText = this.fallbackTruncate(older, summaryBudgetChars);
    }

    const summaryMessage: Message = {
      role: 'system',
      content: `[Conversation summary]: ${summaryText}`,
    };

    return this.assembleResult(systemMsg, pinned, [summaryMessage], recent);
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  private partitionMessages(messages: Message[]): {
    systemMsg: Message | undefined;
    pinned: Message[];
    recent: Message[];
    older: Message[];
  } {
    const systemMsg = messages.find((m) => m.role === 'system');
    const nonSystem = messages.filter((m) => m.role !== 'system');

    // Identify pinned messages
    const pinned: Message[] = [];
    const unpinned: Message[] = [];
    for (let i = 0; i < nonSystem.length; i++) {
      const originalIdx = messages.indexOf(nonSystem[i]);
      if (this.pinnedIndices.has(originalIdx)) {
        pinned.push(nonSystem[i]);
      } else {
        unpinned.push(nonSystem[i]);
      }
    }

    // Keep recent turns
    const recentTurns: Message[] = [];
    let turnCount = 0;
    for (let i = unpinned.length - 1; i >= 0 && turnCount < this.recentTurnsToKeep; i--) {
      recentTurns.unshift(unpinned[i]);
      if (unpinned[i].role === 'user') turnCount++;
    }

    const older = unpinned.slice(0, unpinned.length - recentTurns.length);

    return { systemMsg, pinned, recent: recentTurns, older };
  }

  private assembleResult(
    systemMsg: Message | undefined,
    pinned: Message[],
    summaryMessages: Message[],
    recent: Message[],
  ): Message[] {
    const result: Message[] = [];
    if (systemMsg) result.push(systemMsg);
    result.push(...summaryMessages);
    result.push(...pinned);
    result.push(...recent);
    return result;
  }

  private fallbackTruncate(messages: Message[], maxChars: number): string {
    const concatenated = messages
      .map((m) => `[${m.role}]: ${this.getContent(m)}`)
      .join('\n');
    return concatenated.slice(0, maxChars);
  }

  private getContent(msg: Message): string {
    if (typeof msg.content === 'string') return msg.content;
    return (msg.content as Array<{ type: string; text?: string }>)
      .filter((p) => p.type === 'text')
      .map((p) => p.text ?? '')
      .join(' ');
  }

  private msgChars(msg: Message): number {
    return this.getContent(msg).length;
  }

  private messagesChars(msgs: Message[]): number {
    return msgs.reduce((sum, m) => sum + this.msgChars(m), 0);
  }

  private totalChars(msgs: Message[]): number {
    return this.messagesChars(msgs);
  }
}
