import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import type { Store } from './Store.js';
import type { StoreEntry } from './StoreEntry.js';
import { PackageInstaller } from './PackageInstaller.js';
import type { PackageType } from './PackageInstaller.js';

const STORE_DIR = path.join(os.homedir(), '.medrix', 'store');
const REGISTRY_FILE = path.join(STORE_DIR, 'registry.json');

export class LocalStore implements Store {
  private entries: StoreEntry[] = [];
  private loaded = false;
  private readonly installer = new PackageInstaller();

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

  async search(query: string, opts?: { type?: string; limit?: number }): Promise<StoreEntry[]> {
    await this.ensureLoaded();
    const q = query.toLowerCase();
    let results = this.entries.filter(
      (e) =>
        e.name.toLowerCase().includes(q) ||
        e.description.toLowerCase().includes(q) ||
        e.id.toLowerCase().includes(q) ||
        (e.tags ?? []).some((t) => t.toLowerCase().includes(q)),
    );
    if (opts?.type) results = results.filter((e) => e.category === opts.type);
    if (opts?.limit) results = results.slice(0, opts.limit);
    return results;
  }

  async install(id: string, version?: string): Promise<string[]> {
    await this.ensureLoaded();
    const entry = this.entries.find((e) => e.id === id);
    if (!entry) throw new Error(`Store entry not found: ${id}`);
    const content = (entry as any).content ?? `# ${entry.name}\n\n${entry.description}`;
    const files = entry.files;
    return this.installer.install(entry.category as PackageType, entry.name, content, files);
  }

  async uninstall(id: string): Promise<string[]> {
    await this.ensureLoaded();
    const entry = this.entries.find((e) => e.id === id);
    if (!entry) throw new Error(`Store entry not found: ${id}`);
    return this.installer.uninstall(entry.category as PackageType, entry.name);
  }

  async publish(entry: StoreEntry & { content: string }): Promise<void> {
    await this.ensureLoaded();
    const idx = this.entries.findIndex((e) => e.id === entry.id);
    const stored = { ...entry, updatedAt: new Date().toISOString() };
    if (idx >= 0) {
      this.entries[idx] = stored;
    } else {
      (stored as any).createdAt = new Date().toISOString();
      this.entries.push(stored);
    }
    await this.persist();
  }

  async list(category?: StoreEntry['category']): Promise<StoreEntry[]> {
    await this.ensureLoaded();
    if (!category) return [...this.entries];
    return this.entries.filter((e) => e.category === category);
  }

  async getPackage(id: string): Promise<StoreEntry | undefined> {
    await this.ensureLoaded();
    return this.entries.find((e) => e.id === id);
  }
}
