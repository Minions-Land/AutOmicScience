import { promises as fs } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import type { Skill, SkillLoader } from './Skill.js';

export interface SkillManifestEntry {
  name: string;
  description: string;
  path: string;
  source: 'builtin' | 'user' | 'project' | 'file';
  active?: boolean;
}

/**
 * Loads skills from one of two formats:
 *  - .md  — the body becomes `instructions`. Optional YAML-ish front-matter:
 *           ---
 *           name: my-skill
 *           description: does X
 *           ---
 *  - .ts/.js — must default-export a `Skill` (or an async factory returning one).
 */
export class FileSkillLoader implements SkillLoader {
  constructor(private readonly searchDirs: string[] = []) {}

  async load(pathOrName: string): Promise<Skill> {
    const resolved = await this.resolve(pathOrName);
    if (resolved.endsWith('.md')) return this.loadMarkdown(resolved);
    return this.loadModule(resolved);
  }

  async read(pathOrName: string): Promise<Skill & { path: string }> {
    const resolved = await this.resolve(pathOrName);
    const skill = resolved.endsWith('.md')
      ? await this.loadMarkdown(resolved)
      : await this.loadModule(resolved);
    return { ...skill, path: resolved };
  }

  async list(): Promise<SkillManifestEntry[]> {
    const seen = new Set<string>();
    const entries: SkillManifestEntry[] = [];
    for (const dir of this.searchDirs) {
      let children: string[];
      try {
        children = await fs.readdir(dir);
      } catch {
        continue;
      }
      for (const child of children) {
        const candidates = [
          path.join(dir, child),
          path.join(dir, child, 'SKILL.md'),
        ];
        for (const candidate of candidates) {
          const file = await resolveSkillFile(candidate);
          if (!file || seen.has(file)) continue;
          seen.add(file);
          entries.push(await this.describeFile(file, dir));
        }
      }
    }
    return entries.sort((a, b) => a.name.localeCompare(b.name));
  }

  private async resolve(pathOrName: string): Promise<string> {
    const direct = await resolveSkillFile(pathOrName);
    if (direct) return direct;
    for (const dir of this.searchDirs) {
      const candidates = [
        path.join(dir, pathOrName),
        path.join(dir, pathOrName, 'SKILL.md'),
        ...['.md', '.ts', '.js', '.mjs'].map((ext) => path.join(dir, `${pathOrName}${ext}`)),
      ];
      for (const candidate of candidates) {
        const file = await resolveSkillFile(candidate);
        if (file) return file;
      }
    }
    throw new Error(`Skill not found: ${pathOrName}`);
  }

  private async describeFile(file: string, searchDir: string): Promise<SkillManifestEntry> {
    if (file.endsWith('.md')) {
      const raw = await fs.readFile(file, 'utf8');
      const { meta } = parseFrontMatter(raw);
      const fallbackName = path.basename(file, path.extname(file)) === 'SKILL'
        ? path.basename(path.dirname(file))
        : path.basename(file, path.extname(file));
      return {
        name: meta.name ?? fallbackName,
        description: meta.description ?? `Skill loaded from ${path.basename(file)}`,
        path: file,
        source: classifySkillSource(searchDir),
      };
    }
    const skill = await this.loadModule(file);
    return {
      name: skill.name,
      description: skill.description,
      path: file,
      source: classifySkillSource(searchDir),
    };
  }

  private async loadMarkdown(file: string): Promise<Skill> {
    const raw = await fs.readFile(file, 'utf8');
    const { meta, body } = parseFrontMatter(raw);
    const fallbackName = path.basename(file, path.extname(file)) === 'SKILL'
      ? path.basename(path.dirname(file))
      : path.basename(file, path.extname(file));
    return {
      name: meta.name ?? fallbackName,
      description: meta.description ?? `Skill loaded from ${path.basename(file)}`,
      instructions: body.trim(),
    };
  }

  private async loadModule(file: string): Promise<Skill> {
    const mod = await import(pathToFileURL(file).href);
    const exp = mod.default ?? mod.skill ?? mod;
    const skill = typeof exp === 'function' ? await exp() : exp;
    if (!skill?.name || !skill?.instructions) {
      throw new Error(`Module at ${file} did not export a valid Skill`);
    }
    return skill as Skill;
  }
}

async function resolveSkillFile(candidate: string): Promise<string | null> {
  try {
    const stat = await fs.stat(candidate);
    if (stat.isDirectory()) {
      for (const name of ['SKILL.md', 'skill.md']) {
        const nested = path.join(candidate, name);
        try {
          await fs.access(nested);
          return path.resolve(nested);
        } catch {
          // Try next conventional skill filename.
        }
      }
      return null;
    }
    if (/\.(md|ts|js|mjs)$/i.test(candidate)) return path.resolve(candidate);
    return null;
  } catch {
    return null;
  }
}

function classifySkillSource(searchDir: string): SkillManifestEntry['source'] {
  const normalized = searchDir.replace(/\\/g, '/').toLowerCase();
  if (normalized.includes('/src/skill/builtin')) return 'builtin';
  if (normalized.includes('/.aos/skills')) return 'user';
  if (normalized.endsWith('/skills')) return 'project';
  return 'file';
}

function parseFrontMatter(raw: string): { meta: Record<string, string>; body: string } {
  const m = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/.exec(raw);
  if (!m) return { meta: {}, body: raw };
  const meta: Record<string, string> = {};
  for (const line of m[1].split('\n')) {
    const idx = line.indexOf(':');
    if (idx < 0) continue;
    meta[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
  }
  return { meta, body: m[2] };
}
