import type { Message } from '../types.js';

export interface Memory {
  /** Append a message to short-term memory. */
  append(message: Message): Promise<void>;
  /** Return the recent message window for prompting. */
  recent(limit?: number): Promise<Message[]>;
  /** Clear short-term memory. */
  clear(): Promise<void>;
  /** Long-term store: store a fact or summary by key. Optional. */
  remember?(key: string, value: string): Promise<void>;
  /** Long-term store: retrieve a fact by key. Optional. */
  recall?(key: string): Promise<string | null>;
}
