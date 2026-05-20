import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import type { Store } from './Store.js';
import type { StoreEntry } from './StoreEntry.js';

const STORE_DIR = path.join(os.homedir(), '.novaeve', 'store');
const REGISTRY_FILE = path.join(STORE_DIR, 'registry.json');

/**
 * Local filesystem-backed store implementation.
 * Uses ~/.novaeve/store/registry.json as a JSON registry.
 */
export class LocalStore implements Store {
  private entries: StoreEntry[] = [];
  private loaded = false;

  private async ensureLoaded(): Promise<void> {
    if (this.loaded) return;
    try {
      await fs.mkdir(STORE_DIR, { recursive: true });
      const raw = await fs.readFile(REGISTRY_FILE, 'utf-8');
      this.entries = JSON.parse(raw) as StoreEntry[];
    } catch {
      this.entries = [];
    }
    this.loaded = true;
  }

  private async persist(): Promise<void> {
    await fs.mkdir(STORE_DIR, { recursive: true });
    await fs.writeFile(REGISTRY_FILE, JSON.stringify(this.entries, null, 2));
  }

  /** Search entries by query string (case-insensitive substring match). */
  async search(query: string): Promise<StoreEntry[]> {
    await this.ensureLoaded();
    const q = query.toLowerCase();
    return this.entries.filter(
      (e) =>
        e.name.toLowerCase().includes(q) ||
        e.description.toLowerCase().includes(q) ||
        e.id.toLowerCase().includes(q),
    );
  }

  /** Install an entry by id (no-op stub — logs the install). */
  async install(id: string): Promise<void> {
    await this.ensureLoaded();
    const entry = this.entries.find((e) => e.id === id);
    if (!entry) {
      throw new Error(`Store entry not found: ${id}`);
    }
    // Stub: in a real implementation, download and install the package
  }

  /** Publish a new entry to the local store. */
  async publish(entry: StoreEntry): Promise<void> {
    await this.ensureLoaded();
    const idx = this.entries.findIndex((e) => e.id === entry.id);
    if (idx >= 0) {
      this.entries[idx] = entry;
    } else {
      this.entries.push(entry);
    }
    await this.persist();
  }

  /** List all entries, optionally filtered by category. */
  async list(category?: StoreEntry['category']): Promise<StoreEntry[]> {
    await this.ensureLoaded();
    if (!category) return [...this.entries];
    return this.entries.filter((e) => e.category === category);
  }
}
