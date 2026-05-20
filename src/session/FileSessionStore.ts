import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import type { Message } from '../types.js';
import type { SessionStore } from './SessionStore.js';

const SESSIONS_DIR = path.join(os.homedir(), '.medrix', 'sessions');

/**
 * File-based session store implementation.
 * Stores each session as a JSON file at ~/.medrix/sessions/<id>.json.
 */
export class FileSessionStore implements SessionStore {
  private dir: string;

  constructor(dir?: string) {
    this.dir = dir ?? SESSIONS_DIR;
  }

  private filePath(sessionId: string): string {
    // Sanitize session id to prevent path traversal
    const safe = sessionId.replace(/[^a-zA-Z0-9_-]/g, '_');
    return path.join(this.dir, `${safe}.json`);
  }

  /** Save a session's messages to disk. */
  async save(sessionId: string, messages: Message[]): Promise<void> {
    await fs.mkdir(this.dir, { recursive: true });
    const data = JSON.stringify(messages, null, 2);
    await fs.writeFile(this.filePath(sessionId), data, 'utf-8');
  }

  /** Load a session's messages from disk. Returns null if not found. */
  async load(sessionId: string): Promise<Message[] | null> {
    try {
      const raw = await fs.readFile(this.filePath(sessionId), 'utf-8');
      return JSON.parse(raw) as Message[];
    } catch {
      return null;
    }
  }

  /** List all stored session ids. */
  async list(): Promise<string[]> {
    try {
      const files = await fs.readdir(this.dir);
      return files
        .filter((f) => f.endsWith('.json'))
        .map((f) => f.slice(0, -5));
    } catch {
      return [];
    }
  }

  /** Delete a session file. */
  async delete(sessionId: string): Promise<void> {
    try {
      await fs.unlink(this.filePath(sessionId));
    } catch {
      // Ignore if file doesn't exist
    }
  }
}
