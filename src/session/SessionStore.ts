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
  exportMarkdown?(sessionId: string, outputPath: string): Promise<void>;
  exportJsonl?(sessionId: string, outputPath: string): Promise<void>;
  exportBundle?(sessionId: string, outputDir: string): Promise<void>;
  importBundle?(bundlePath: string, sessionId?: string): Promise<string>;
}
