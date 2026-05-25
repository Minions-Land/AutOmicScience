import { promises as fs } from 'node:fs';
import { existsSync, readdirSync, readFileSync, writeFileSync, mkdirSync, copyFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { parseFrontmatter, serializeFrontmatter } from '../utils/template.js';

export interface AgentTemplate {
  name: string;
  model: string | string[];
  systemPrompt?: string;
  toolsets?: string[];
  skills?: string[];
  mcp?: { name: string; command?: string; args?: string[]; url?: string }[];
  version?: string;
  description?: string;
}

export interface TeamTemplate {
  name: string;
  pattern: 'sequential' | 'swarm' | 'coordinator' | 'moa';
  members: string[];
  coordinator?: string;
  version?: string;
  description?: string;
}

export type TemplateCategory = 'agents' | 'teams' | 'skills' | 'prompts';

export interface TemplateFile {
  id: string;
  name: string;
  category: TemplateCategory;
  path: string;
  version?: string;
  description?: string;
}

export class TemplateManager {
  public readonly root: string;
  private readonly bundledDir: string;

  constructor(root?: string, bundledDir?: string) {
    this.root = root ?? path.join(os.homedir(), '.aos');
    this.bundledDir = bundledDir ?? path.join(this.root, '..', '.aos-bundled');
  }

  async init(): Promise<void> {
    for (const dir of ['agents', 'teams', 'skills', 'prompts']) {
      await fs.mkdir(path.join(this.root, dir), { recursive: true });
    }
  }

  agentPath(name: string): string { return path.join(this.root, 'agents', `${name}.md`); }
  teamPath(name: string): string { return path.join(this.root, 'teams', `${name}.md`); }
  skillPath(name: string): string { return path.join(this.root, 'skills', name, 'SKILL.md'); }
  promptPath(name: string): string { return path.join(this.root, 'prompts', `${name}.md`); }

  async saveAgent(t: AgentTemplate): Promise<void> {
    await this.init();
    const content = serializeFrontmatter(
      { name: t.name, model: Array.isArray(t.model) ? t.model.join(', ') : t.model, version: t.version ?? '1.0.0', description: t.description ?? '' },
      t.systemPrompt ?? '',
    );
    await fs.writeFile(this.agentPath(t.name), content, 'utf-8');
  }

  async loadAgent(name: string): Promise<AgentTemplate> {
    const content = await fs.readFile(this.agentPath(name), 'utf-8');
    const { meta, body } = parseFrontmatter(content);
    return { name: meta.name ?? name, model: meta.model ?? 'gpt-5.5', systemPrompt: body, version: meta.version, description: meta.description, toolsets: meta.toolsets, skills: meta.skills, mcp: meta.mcp };
  }

  async listAgents(): Promise<string[]> {
    return this.listCategory('agents');
  }

  async saveTeam(t: TeamTemplate): Promise<void> {
    await this.init();
    const content = serializeFrontmatter(
      { name: t.name, pattern: t.pattern, members: JSON.stringify(t.members), coordinator: t.coordinator, version: t.version ?? '1.0.0' },
      t.description ?? '',
    );
    await fs.writeFile(this.teamPath(t.name), content, 'utf-8');
  }

  async loadTeam(name: string): Promise<TeamTemplate> {
    const content = await fs.readFile(this.teamPath(name), 'utf-8');
    const { meta, body } = parseFrontmatter(content);
    const members = typeof meta.members === 'string' ? JSON.parse(meta.members) : meta.members ?? [];
    return { name: meta.name ?? name, pattern: meta.pattern ?? 'sequential', members, coordinator: meta.coordinator, version: meta.version, description: body };
  }

  async listTeams(): Promise<string[]> {
    return this.listCategory('teams');
  }

  async listSkills(): Promise<string[]> {
    const dir = path.join(this.root, 'skills');
    if (!existsSync(dir)) return [];
    return readdirSync(dir).filter((e) => {
      const skillFile = path.join(dir, e, 'SKILL.md');
      return existsSync(skillFile);
    });
  }

  async listTemplateFiles(category?: TemplateCategory): Promise<TemplateFile[]> {
    const categories = category ? [category] : ['agents', 'teams', 'skills', 'prompts'] as TemplateCategory[];
    const files: TemplateFile[] = [];
    for (const cat of categories) {
      const dir = path.join(this.root, cat);
      if (!existsSync(dir)) continue;
      for (const entry of readdirSync(dir)) {
        const fullPath = path.join(dir, entry);
        if (entry.endsWith('.md')) {
          const content = readFileSync(fullPath, 'utf-8');
          const { meta } = parseFrontmatter(content);
          files.push({ id: entry.replace(/\.md$/, ''), name: meta.name ?? entry.replace(/\.md$/, ''), category: cat, path: fullPath, version: meta.version, description: meta.description });
        } else if (cat === 'skills' && existsSync(path.join(fullPath, 'SKILL.md'))) {
          const content = readFileSync(path.join(fullPath, 'SKILL.md'), 'utf-8');
          const { meta } = parseFrontmatter(content);
          files.push({ id: entry, name: meta.name ?? entry, category: 'skills', path: path.join(fullPath, 'SKILL.md'), version: meta.version, description: meta.description });
        }
      }
    }
    return files;
  }

  syncTemplates(force = false): { synced: string[]; skipped: string[] } {
    const synced: string[] = [];
    const skipped: string[] = [];
    if (!existsSync(this.bundledDir)) return { synced, skipped };

    for (const cat of ['agents', 'teams', 'skills', 'prompts']) {
      const srcDir = path.join(this.bundledDir, cat);
      const destDir = path.join(this.root, cat);
      if (!existsSync(srcDir)) continue;
      if (!existsSync(destDir)) mkdirSync(destDir, { recursive: true });

      for (const entry of readdirSync(srcDir)) {
        const src = path.join(srcDir, entry);
        const dest = path.join(destDir, entry);
        if (!force && existsSync(dest)) {
          skipped.push(`${cat}/${entry}`);
          continue;
        }
        copyFileSync(src, dest);
        synced.push(`${cat}/${entry}`);
      }
    }
    return { synced, skipped };
  }

  async deleteTemplate(category: TemplateCategory, name: string): Promise<boolean> {
    const filePath = category === 'skills' ? this.skillPath(name) : path.join(this.root, category, `${name}.md`);
    try {
      await fs.unlink(filePath);
      return true;
    } catch {
      return false;
    }
  }

  private listCategory(category: string): string[] {
    const dir = path.join(this.root, category);
    if (!existsSync(dir)) return [];
    return readdirSync(dir).filter((e) => e.endsWith('.md')).map((e) => e.replace(/\.md$/, ''));
  }
}
