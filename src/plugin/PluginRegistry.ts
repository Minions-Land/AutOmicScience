import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';

export interface PluginRegistryEntry {
  target: string;
  name?: string;
  enabled: boolean;
  loadedAt?: string;
}

export class AOSPluginRegistry {
  constructor(private readonly filePath = path.join(os.homedir(), '.aos', 'plugins.json')) {}

  async list(): Promise<PluginRegistryEntry[]> {
    try {
      return JSON.parse(await fs.readFile(this.filePath, 'utf-8')) as PluginRegistryEntry[];
    } catch {
      return [];
    }
  }

  async add(entry: PluginRegistryEntry): Promise<void> {
    const entries = (await this.list()).filter((item) => item.target !== entry.target);
    entries.push(entry);
    await this.save(entries);
  }

  async setEnabled(target: string, enabled: boolean): Promise<void> {
    const entries = await this.list();
    const entry = entries.find((item) => item.target === target);
    if (entry) entry.enabled = enabled;
    await this.save(entries);
  }

  async save(entries: PluginRegistryEntry[]): Promise<void> {
    await fs.mkdir(path.dirname(this.filePath), { recursive: true });
    await fs.writeFile(this.filePath, JSON.stringify(entries, null, 2), 'utf-8');
  }
}
