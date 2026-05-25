import { z } from 'zod';
import type { Tool } from './Tool.js';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';

export interface ToolCatalogEntry {
  name: string;
  description: string;
  toolset: string;
  searchHint?: string;
  deferred?: boolean;
}

export class ToolCatalog {
  private factories = new Map<string, () => ToolSet | Promise<ToolSet>>();
  private loaded = new Map<string, ToolSet>();

  register(name: string, factory: () => ToolSet | Promise<ToolSet>, opts?: { preload?: boolean }): this {
    this.factories.set(name, factory);
    if (opts?.preload) {
      void this.load(name);
    }
    return this;
  }

  async load(name: string): Promise<ToolSet> {
    const loaded = this.loaded.get(name);
    if (loaded) return loaded;
    const factory = this.factories.get(name);
    if (!factory) throw new Error(`Toolset not found: ${name}`);
    const toolset = await factory();
    this.loaded.set(name, toolset);
    return toolset;
  }

  async list(): Promise<ToolCatalogEntry[]> {
    const entries: ToolCatalogEntry[] = [];
    for (const [name] of this.factories) {
      const loaded = this.loaded.get(name);
      if (!loaded) {
        entries.push({ name, description: `Deferred toolset: ${name}`, toolset: name, deferred: true });
        continue;
      }
      for (const tool of loaded.list()) {
        entries.push({
          name: tool.name,
          description: tool.description,
          toolset: name,
          searchHint: tool.searchHint,
          deferred: tool.shouldDefer,
        });
      }
    }
    return entries;
  }

  async search(query: string, limit = 20): Promise<ToolCatalogEntry[]> {
    const q = normalizeForSearch(query);
    const entries = await this.list();
    return entries
      .map((entry) => ({
        entry,
        score: scoreEntry(entry, q),
      }))
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map((item) => item.entry);
  }
}

export function toolSearchToolSet(catalog: ToolCatalog): ToolSet {
  return new ToolSet('tool_search', [
    defineTool<{ query: string; limit?: number }, { results: ToolCatalogEntry[] }>({
      name: 'search_tools',
      aliases: ['ToolSearch'],
      operation: 'read',
      description: 'Search available loaded and deferred tools by name, description, and hint.',
      parameters: z.object({
        query: z.string(),
        limit: z.number().int().positive().optional().default(20),
      }),
      isReadOnly: () => true,
      execute: async ({ query, limit }) => ({ results: await catalog.search(query, limit) }),
    }),
    defineTool<{ toolset: string }, { loaded: string; tools: string[] }>({
      name: 'load_toolset',
      operation: 'task',
      description: 'Load a deferred toolset into the catalog.',
      parameters: z.object({ toolset: z.string() }),
      isReadOnly: () => false,
      execute: async ({ toolset }) => {
        const loaded = await catalog.load(toolset);
        return { loaded: toolset, tools: loaded.list().map((tool) => tool.name) };
      },
    }),
  ]);
}

function scoreEntry(entry: ToolCatalogEntry, query: string): number {
  const name = normalizeForSearch(entry.name);
  const haystack = normalizeForSearch(`${entry.name} ${entry.description} ${entry.searchHint ?? ''}`);
  const tokens = query.split(' ').filter(Boolean);
  if (name === query) return 100;
  if (name.includes(query)) return 60;
  if (tokens.length > 1 && tokens.every((token) => haystack.includes(token))) return 45;
  if (haystack.includes(query)) return 20;
  return 0;
}

function normalizeForSearch(value: string): string {
  return value.toLowerCase().replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
}
