import { promises as fs } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import type { Skill, SkillLoader } from './Skill.js';

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

  private async resolve(pathOrName: string): Promise<string> {
    try {
      await fs.access(pathOrName);
      return path.resolve(pathOrName);
    } catch {
      // fall through to search dirs
    }
    for (const dir of this.searchDirs) {
      for (const ext of ['.md', '.ts', '.js', '.mjs']) {
        const candidate = path.join(dir, `${pathOrName}${ext}`);
        try {
          await fs.access(candidate);
          return candidate;
        } catch {
          // continue
        }
      }
    }
    throw new Error(`Skill not found: ${pathOrName}`);
  }

  private async loadMarkdown(file: string): Promise<Skill> {
    const raw = await fs.readFile(file, 'utf8');
    const { meta, body } = parseFrontMatter(raw);
    const fallbackName = path.basename(file, path.extname(file));
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
