import { promises as fs } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import type { RegisteredCommand } from './CommandRegistry.js';

export class FileCommandLoader {
  constructor(private readonly searchDirs: string[] = []) {}

  async discover(): Promise<string[]> {
    const names: string[] = [];
    for (const dir of this.searchDirs) {
      const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => []);
      for (const entry of entries) {
        if (entry.isFile() && /\.(md|js|mjs)$/.test(entry.name)) {
          names.push(path.basename(entry.name, path.extname(entry.name)));
        }
      }
    }
    return [...new Set(names)].sort();
  }

  async loadAll(): Promise<RegisteredCommand[]> {
    const commands: RegisteredCommand[] = [];
    for (const dir of this.searchDirs) {
      const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => []);
      for (const entry of entries) {
        if (!entry.isFile() || !/\.(md|js|mjs)$/.test(entry.name)) continue;
        commands.push(await this.loadFile(path.join(dir, entry.name)));
      }
    }
    return commands;
  }

  async loadFile(filePath: string): Promise<RegisteredCommand> {
    const full = path.resolve(filePath);
    if (full.endsWith('.md')) return this.loadPromptCommand(full);
    const mod = await import(pathToFileURL(full).href);
    const command = mod.default ?? mod.command ?? mod;
    if (!command?.name || !command?.handler) {
      throw new Error(`Command module ${full} did not export { name, handler }`);
    }
    return { ...command, source: full };
  }

  private async loadPromptCommand(filePath: string): Promise<RegisteredCommand> {
    const raw = await fs.readFile(filePath, 'utf-8');
    const { meta, body } = parseFrontMatter(raw);
    const name = meta.name ?? path.basename(filePath, '.md');
    return {
      name,
      description: meta.description ?? `Prompt command loaded from ${path.basename(filePath)}`,
      kind: 'prompt',
      source: filePath,
      handler: ({ args }) => body.replace(/\{\{\s*args\s*\}\}/g, args).trim(),
    };
  }
}

function parseFrontMatter(raw: string): { meta: Record<string, string>; body: string } {
  const match = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/.exec(raw);
  if (!match) return { meta: {}, body: raw };
  const meta: Record<string, string> = {};
  for (const line of match[1].split('\n')) {
    const idx = line.indexOf(':');
    if (idx < 0) continue;
    meta[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
  }
  return { meta, body: match[2] };
}
