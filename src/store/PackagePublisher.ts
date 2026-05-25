import { readFileSync, existsSync } from 'fs';
import { join, basename, relative } from 'path';
import { glob } from 'fs';
import type { StoreEntry } from './StoreEntry.js';

export interface PublishablePackage {
  name: string;
  category: StoreEntry['category'];
  version: string;
  description: string;
  content: string;
  files?: Record<string, string>;
  tags?: string[];
}

export class PackagePublisher {
  buildFromPath(filePath: string, opts?: { version?: string; description?: string }): PublishablePackage {
    if (!existsSync(filePath)) throw new Error(`File not found: ${filePath}`);

    const content = readFileSync(filePath, 'utf-8');
    const meta = this.extractFrontmatter(content);
    const name = meta.name ?? meta.id ?? basename(filePath, '.md');
    const category = this.inferCategory(filePath, meta);
    const version = opts?.version ?? meta.version ?? '1.0.0';
    const description = opts?.description ?? meta.description ?? '';

    const pkg: PublishablePackage = { name, category, version, description, content };

    if (category === 'team' && meta.agents) {
      pkg.files = this.collectTeamFiles(filePath, meta);
    }
    if (category === 'skill') {
      pkg.files = this.collectSkillFiles(filePath);
    }

    if (meta.tags) pkg.tags = meta.tags;
    return pkg;
  }

  private extractFrontmatter(content: string): Record<string, any> {
    const match = content.match(/^---\n([\s\S]*?)\n---/);
    if (!match) return {};
    const lines = match[1].split('\n');
    const result: Record<string, any> = {};
    for (const line of lines) {
      const [key, ...rest] = line.split(':');
      if (key && rest.length) {
        const value = rest.join(':').trim();
        if (value.startsWith('[')) {
          try { result[key.trim()] = JSON.parse(value); } catch { result[key.trim()] = value; }
        } else {
          result[key.trim()] = value;
        }
      }
    }
    return result;
  }

  private inferCategory(filePath: string, meta: Record<string, any>): StoreEntry['category'] {
    if (meta.type) return meta.type;
    if (filePath.includes('/agents/') || filePath.includes('\\agents\\')) return 'agent';
    if (filePath.includes('/teams/') || filePath.includes('\\teams\\')) return 'team';
    if (filePath.includes('/skills/') || filePath.includes('\\skills\\')) return 'skill';
    if (filePath.includes('/tools/') || filePath.includes('\\tools\\')) return 'tool';
    return 'agent';
  }

  private collectTeamFiles(teamFilePath: string, meta: Record<string, any>): Record<string, string> {
    const files: Record<string, string> = {};
    const dir = join(teamFilePath, '..');
    const agents = meta.agents ?? [];
    for (const agentRef of agents) {
      const ref = typeof agentRef === 'string' ? agentRef : agentRef?.ref;
      if (!ref) continue;
      const agentPath = join(dir, '..', ref);
      if (existsSync(agentPath)) {
        files[ref] = readFileSync(agentPath, 'utf-8');
      }
    }
    return files;
  }

  private collectSkillFiles(skillFilePath: string): Record<string, string> {
    const files: Record<string, string> = {};
    const dir = join(skillFilePath, '..');
    if (!existsSync(dir)) return files;
    const { readdirSync, statSync } = require('fs');
    const walk = (d: string) => {
      for (const entry of readdirSync(d)) {
        const full = join(d, entry);
        if (statSync(full).isDirectory()) {
          walk(full);
        } else if (full !== skillFilePath) {
          const rel = relative(join(dir, '..', '..'), full);
          files[rel] = readFileSync(full, 'utf-8');
        }
      }
    };
    walk(dir);
    return files;
  }
}
