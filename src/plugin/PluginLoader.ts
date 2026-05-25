import { promises as fs } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import type { Skill } from '../skill/index.js';
import type { Tool } from '../toolset/index.js';
import type { RegisteredCommand } from '../commands/index.js';
import type { HookManager } from '../hooks/index.js';

export interface AOSPluginManifest {
  name: string;
  version?: string;
  description?: string;
  entry?: string;
  skills?: string[];
  commands?: string[];
  tools?: string[];
}

export interface LoadedPlugin {
  manifest: AOSPluginManifest;
  path: string;
  skills: Skill[];
  tools: Tool[];
  commands: RegisteredCommand[];
  hooks?: (hooks: HookManager) => void;
}

export class PluginLoader {
  constructor(private readonly searchDirs: string[] = []) {}

  async load(nameOrPath: string): Promise<LoadedPlugin> {
    const pluginPath = await this.resolve(nameOrPath);
    const manifest = await this.readManifest(pluginPath);
    const entry = manifest.entry ? path.join(pluginPath, manifest.entry) : path.join(pluginPath, 'index.js');
    const exports = await this.loadEntry(entry);

    return {
      manifest,
      path: pluginPath,
      skills: normalizeArray<Skill>(exports.skills ?? exports.skill),
      tools: normalizeArray<Tool>(exports.tools ?? exports.tool),
      commands: normalizeArray<RegisteredCommand>(exports.commands ?? exports.command),
      hooks: typeof exports.hooks === 'function' ? exports.hooks as (hooks: HookManager) => void : undefined,
    };
  }

  async discover(): Promise<string[]> {
    const found: string[] = [];
    for (const dir of this.searchDirs) {
      const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => []);
      for (const entry of entries) {
        if (!entry.isDirectory()) continue;
        const candidate = path.join(dir, entry.name);
        if (await exists(path.join(candidate, 'plugin.json')) || await exists(path.join(candidate, '.aos-plugin', 'plugin.json'))) {
          found.push(entry.name);
        }
      }
    }
    return found.sort();
  }

  private async resolve(nameOrPath: string): Promise<string> {
    if (await exists(nameOrPath)) return path.resolve(nameOrPath);
    for (const dir of this.searchDirs) {
      const candidate = path.join(dir, nameOrPath);
      if (await exists(candidate)) return candidate;
    }
    throw new Error(`Plugin not found: ${nameOrPath}`);
  }

  private async readManifest(pluginPath: string): Promise<AOSPluginManifest> {
    const candidates = [
      path.join(pluginPath, 'plugin.json'),
      path.join(pluginPath, '.aos-plugin', 'plugin.json'),
      path.join(pluginPath, 'package.json'),
    ];
    for (const candidate of candidates) {
      if (!await exists(candidate)) continue;
      const raw = await fs.readFile(candidate, 'utf-8');
      const parsed = JSON.parse(raw);
      return {
        name: parsed.name ?? path.basename(pluginPath),
        version: parsed.version,
        description: parsed.description,
        entry: parsed.entry ?? parsed.main,
        skills: parsed.skills,
        commands: parsed.commands,
        tools: parsed.tools,
      };
    }
    return { name: path.basename(pluginPath), entry: 'index.js' };
  }

  private async loadEntry(entryPath: string): Promise<Record<string, unknown>> {
    if (!await exists(entryPath)) return {};
    const mod = await import(pathToFileURL(entryPath).href);
    const exp = mod.default ?? mod;
    return typeof exp === 'function' ? await exp() : exp;
  }
}

function normalizeArray<T>(value: unknown): T[] {
  if (!value) return [];
  return Array.isArray(value) ? value as T[] : [value as T];
}

async function exists(target: string): Promise<boolean> {
  try {
    await fs.access(target);
    return true;
  } catch {
    return false;
  }
}
