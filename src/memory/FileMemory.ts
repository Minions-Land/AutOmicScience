import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import type { Message } from '../types.js';
import type { Memory } from './Memory.js';

/**
 * File-backed Memory implementation.
 *
 * Short-term messages are stored as JSON Lines (JSONL) at:
 *   <dir>/<agentName>.jsonl
 * (default dir: ~/.aos/memory/agents)
 *
 * Long-term key/value facts are stored as a JSON sidecar:
 *   <dir>/<agentName>.kv.json
 *
 * Append is O(1) on the file (single fs.appendFile). recent() reads the file
 * and returns the trailing N parsed messages. clear() truncates the file.
 *
 * The implementation is robust to malformed lines (they're skipped with a
 * warning) and creates parent directories on demand.
 */
export class FileMemory implements Memory {
  private readonly dir: string;
  private readonly agentName: string;
  private readonly jsonlPath: string;
  private readonly kvPath: string;
  private kvCache: Record<string, string> | null = null;
  private dirEnsured = false;

  constructor(agentName: string, dir?: string) {
    if (!agentName) throw new Error('FileMemory: agentName is required');
    this.agentName = agentName;
    this.dir = dir ?? path.join(os.homedir(), '.aos', 'memory', 'agents');
    const safeName = agentName.replace(/[^a-zA-Z0-9._-]/g, '_');
    this.jsonlPath = path.join(this.dir, `${safeName}.jsonl`);
    this.kvPath = path.join(this.dir, `${safeName}.kv.json`);
  }

  /** Path to the underlying JSONL file (useful for tests / introspection). */
  get filePath(): string {
    return this.jsonlPath;
  }

  /** Agent name this memory is bound to. */
  get name(): string {
    return this.agentName;
  }

  private async ensureDir(): Promise<void> {
    if (this.dirEnsured) return;
    await fs.mkdir(this.dir, { recursive: true });
    this.dirEnsured = true;
  }

  async append(message: Message): Promise<void> {
    await this.ensureDir();
    const line = JSON.stringify(message) + '\n';
    await fs.appendFile(this.jsonlPath, line, 'utf8');
  }

  async recent(limit = 100): Promise<Message[]> {
    let raw: string;
    try {
      raw = await fs.readFile(this.jsonlPath, 'utf8');
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return [];
      throw err;
    }
    if (!raw) return [];
    const lines = raw.split('\n');
    // Walk backwards to collect up to `limit` valid messages without parsing the whole file twice.
    const collected: Message[] = [];
    for (let i = lines.length - 1; i >= 0 && collected.length < limit; i--) {
      const line = lines[i];
      if (!line) continue;
      try {
        collected.push(JSON.parse(line) as Message);
      } catch {
        // Skip malformed line.
      }
    }
    collected.reverse();
    return collected;
  }

  async clear(): Promise<void> {
    await this.ensureDir();
    try {
      await fs.truncate(this.jsonlPath, 0);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        // Nothing to clear.
        return;
      }
      throw err;
    }
  }

  // --- Long-term key/value store ---------------------------------------------

  private async loadKv(): Promise<Record<string, string>> {
    if (this.kvCache) return this.kvCache;
    try {
      const raw = await fs.readFile(this.kvPath, 'utf8');
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        this.kvCache = parsed as Record<string, string>;
      } else {
        this.kvCache = {};
      }
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        this.kvCache = {};
      } else {
        // Corrupt JSON — start fresh rather than crash.
        this.kvCache = {};
      }
    }
    return this.kvCache;
  }

  private async saveKv(): Promise<void> {
    if (!this.kvCache) return;
    await this.ensureDir();
    const tmp = `${this.kvPath}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(this.kvCache, null, 2), 'utf8');
    await fs.rename(tmp, this.kvPath);
  }

  async remember(key: string, value: string): Promise<void> {
    const kv = await this.loadKv();
    kv[key] = value;
    await this.saveKv();
  }

  async recall(key: string): Promise<string | null> {
    const kv = await this.loadKv();
    return Object.prototype.hasOwnProperty.call(kv, key) ? kv[key] : null;
  }
}
