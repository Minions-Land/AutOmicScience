import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { InMemoryTaskManager } from './InMemoryTaskManager.js';
import type { TaskSpec } from './TaskSpec.js';
import type { TaskStatus, TaskResult } from './TaskStatus.js';

export class FileTaskManager extends InMemoryTaskManager {
  constructor(private readonly filePath = path.join(os.homedir(), '.aos', 'tasks.json')) {
    super();
  }

  override async submit(spec: TaskSpec): Promise<string> {
    const id = await super.submit(spec);
    await this.persist();
    this.onComplete(id).finally(() => {
      this.persist().catch(() => {});
    }).catch(() => {});
    return id;
  }

  override async cancel(id: string): Promise<void> {
    await super.cancel(id);
    await this.persist();
  }

  async loadSnapshots(): Promise<TaskStatus[]> {
    try {
      const raw = await fs.readFile(this.filePath, 'utf-8');
      return JSON.parse(raw) as TaskStatus[];
    } catch {
      return [];
    }
  }

  async persist(): Promise<void> {
    await fs.mkdir(path.dirname(this.filePath), { recursive: true });
    await fs.writeFile(this.filePath, JSON.stringify(await this.list(), null, 2), 'utf-8');
  }

  async history(): Promise<{ live: TaskStatus[]; persisted: TaskStatus[] }> {
    return {
      live: await this.list(),
      persisted: await this.loadSnapshots(),
    };
  }

  override async onComplete(id: string): Promise<TaskResult> {
    const result = await super.onComplete(id);
    await this.persist();
    return result;
  }
}
