import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import type { SessionStore, SessionData } from './SessionStore.js';
import type { Message } from '../types.js';

const SESSIONS_DIR = path.join(os.homedir(), '.medrix', 'sessions');

export class FileSessionStore implements SessionStore {
  private dir: string;

  constructor(dir?: string) {
    this.dir = dir ?? SESSIONS_DIR;
  }

  private filePath(sessionId: string): string {
    const safe = sessionId.replace(/[^a-zA-Z0-9_-]/g, '_');
    return path.join(this.dir, `${safe}.json`);
  }

  async save(sessionId: string, data: SessionData | Message[]): Promise<void> {
    await fs.mkdir(this.dir, { recursive: true });
    const toStore: SessionData = Array.isArray(data) ? { messages: data, savedAt: new Date().toISOString() } : data;
    await fs.writeFile(this.filePath(sessionId), JSON.stringify(toStore, null, 2), 'utf-8');
  }

  async load(sessionId: string): Promise<SessionData | null> {
    try {
      const raw = await fs.readFile(this.filePath(sessionId), 'utf-8');
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return { messages: parsed };
      return parsed as SessionData;
    } catch {
      return null;
    }
  }

  async list(): Promise<string[]> {
    try {
      const files = await fs.readdir(this.dir);
      return files.filter((f) => f.endsWith('.json')).map((f) => f.slice(0, -5));
    } catch {
      return [];
    }
  }

  async delete(sessionId: string): Promise<void> {
    try { await fs.unlink(this.filePath(sessionId)); } catch { /* ignore */ }
  }
}
