import type { Message } from '../types.js';
import type { Memory } from './Memory.js';

export class InMemoryMemory implements Memory {
  private messages: Message[] = [];
  private store = new Map<string, string>();

  constructor(private readonly maxMessages = 200) {}

  async append(message: Message): Promise<void> {
    this.messages.push(message);
    if (this.messages.length > this.maxMessages) {
      // Drop oldest non-system messages.
      const sys = this.messages.filter((m) => m.role === 'system');
      const rest = this.messages.filter((m) => m.role !== 'system');
      const trimmed = rest.slice(rest.length - (this.maxMessages - sys.length));
      this.messages = [...sys, ...trimmed];
    }
  }

  async recent(limit?: number): Promise<Message[]> {
    if (limit === undefined) return [...this.messages];
    return this.messages.slice(-limit);
  }

  async clear(): Promise<void> {
    this.messages = [];
  }

  async remember(key: string, value: string): Promise<void> {
    this.store.set(key, value);
  }

  async recall(key: string): Promise<string | null> {
    return this.store.get(key) ?? null;
  }
}
