import type { Message } from '../types.js';

/**
 * Persistent session storage — save and restore conversation histories.
 */
export interface SessionStore {
  /** Save a session's messages. Overwrites any existing session with the same id. */
  save(sessionId: string, messages: Message[]): Promise<void>;
  /** Load a session's messages by id. Returns null if not found. */
  load(sessionId: string): Promise<Message[] | null>;
  /** List all stored session ids. */
  list(): Promise<string[]>;
  /** Delete a session by id. */
  delete(sessionId: string): Promise<void>;
}
