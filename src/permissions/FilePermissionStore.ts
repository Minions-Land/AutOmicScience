import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { PermissionManager } from './PermissionManager.js';
import type { PermissionManagerOptions, PermissionMode, PermissionRule } from './PermissionManager.js';

export interface PermissionStoreData {
  mode: PermissionMode;
  askFallback?: 'allow' | 'deny';
  rules: PermissionRule[];
}

export class FilePermissionStore {
  constructor(private readonly filePath = path.join(os.homedir(), '.aos', 'permissions.json')) {}

  async load(): Promise<PermissionStoreData> {
    try {
      const raw = await fs.readFile(this.filePath, 'utf-8');
      const parsed = JSON.parse(raw) as PermissionStoreData;
      return {
        mode: parsed.mode ?? 'default',
        askFallback: parsed.askFallback ?? 'deny',
        rules: parsed.rules ?? [],
      };
    } catch {
      return { mode: 'default', askFallback: 'deny', rules: [] };
    }
  }

  async save(data: PermissionStoreData): Promise<void> {
    await fs.mkdir(path.dirname(this.filePath), { recursive: true });
    await fs.writeFile(this.filePath, JSON.stringify(data, null, 2), 'utf-8');
  }

  async createManager(overrides: Partial<PermissionManagerOptions> = {}): Promise<PermissionManager> {
    const data = await this.load();
    return new PermissionManager({
      mode: overrides.mode ?? data.mode,
      askFallback: overrides.askFallback ?? data.askFallback ?? 'deny',
      rules: overrides.rules ?? data.rules,
    });
  }

  async persistManager(manager: PermissionManager): Promise<void> {
    await this.save({
      mode: manager.getMode(),
      askFallback: 'deny',
      rules: manager.listRules(),
    });
  }
}
