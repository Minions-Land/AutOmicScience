import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import type { SessionStore, SessionData } from './SessionStore.js';
import type { Message } from '../types.js';

const SESSIONS_DIR = path.join(os.homedir(), '.aos', 'sessions');

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

  async exportMarkdown(sessionId: string, outputPath: string): Promise<void> {
    const session = await this.load(sessionId);
    if (!session) throw new Error(`Session not found: ${sessionId}`);
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await fs.writeFile(outputPath, sessionToMarkdown(sessionId, session), 'utf-8');
  }

  async exportJsonl(sessionId: string, outputPath: string): Promise<void> {
    const session = await this.load(sessionId);
    if (!session) throw new Error(`Session not found: ${sessionId}`);
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    const lines = (session.messages ?? []).map((message) => JSON.stringify(message));
    await fs.writeFile(outputPath, lines.join('\n') + (lines.length ? '\n' : ''), 'utf-8');
  }

  async exportBundle(sessionId: string, outputDir: string): Promise<void> {
    const session = await this.load(sessionId);
    if (!session) throw new Error(`Session not found: ${sessionId}`);
    await fs.mkdir(outputDir, { recursive: true });
    await fs.writeFile(path.join(outputDir, 'session.json'), JSON.stringify(session, null, 2), 'utf-8');
    await fs.writeFile(path.join(outputDir, 'session.md'), sessionToMarkdown(sessionId, session), 'utf-8');
    await this.exportJsonl(sessionId, path.join(outputDir, 'session.jsonl'));
    await fs.writeFile(path.join(outputDir, 'manifest.json'), JSON.stringify({
      version: '1.0',
      sessionId,
      chatId: session.chatId,
      exportedAt: new Date().toISOString(),
      messageCount: session.messages?.length ?? 0,
      metadata: session.metadata ?? {},
    }, null, 2), 'utf-8');
  }

  async importBundle(bundlePath: string, sessionId?: string): Promise<string> {
    const raw = await fs.readFile(path.join(bundlePath, 'session.json'), 'utf-8');
    const session = JSON.parse(raw) as SessionData;
    const id = sessionId ?? session.chatId ?? path.basename(bundlePath);
    await this.save(id, session);
    return id;
  }
}

function sessionToMarkdown(sessionId: string, session: SessionData): string {
  const parts = [
    `# ${sessionId}`,
    '',
    `> Exported: ${new Date().toISOString()}`,
    `> Messages: ${session.messages?.length ?? 0}`,
    '',
    '---',
    '',
  ];
  for (const message of session.messages ?? []) {
    parts.push(`## ${message.role}`, '', messageToText(message), '', '---', '');
  }
  return parts.join('\n');
}

function messageToText(message: Message): string {
  if (typeof message.content === 'string') return message.content;
  return message.content.map((part) => part.type === 'text' ? part.text : `[image:${part.mediaType ?? 'unknown'}]`).join('\n');
}
