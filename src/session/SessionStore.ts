import type { Message } from '../types.js';

export interface SessionData {
  chatId?: string;
  messages?: Message[];
  savedAt?: string;
  metadata?: Record<string, unknown>;
}

export interface SessionStore {
  save(sessionId: string, data: SessionData | Message[]): Promise<void>;
  load(sessionId: string): Promise<SessionData | null>;
  list(): Promise<string[]>;
  delete(sessionId: string): Promise<void>;
}
