import { existsSync, mkdirSync, writeFileSync, unlinkSync, rmSync, readdirSync } from 'fs';
import { join } from 'path';
import { homedir } from 'os';

export type PackageType = 'agent' | 'team' | 'skill' | 'tool';

function medrixDir(): string {
  return join(homedir(), '.medrix');
}

export class PackageInstaller {
  private readonly baseDir: string;

  constructor(baseDir?: string) {
    this.baseDir = baseDir ?? medrixDir();
  }

  private dirFor(type: PackageType): string {
    switch (type) {
      case 'agent': return join(this.baseDir, 'agents');
      case 'team': return join(this.baseDir, 'teams');
      case 'skill': return join(this.baseDir, 'skills');
      case 'tool': return join(this.baseDir, 'tools');
    }
  }

  install(type: PackageType, name: string, content: string, files?: Record<string, string>): string[] {
    const written: string[] = [];

    if (type === 'agent') {
      const target = join(this.dirFor('agent'), `${name}.md`);
      mkdirSync(join(this.dirFor('agent')), { recursive: true });
      writeFileSync(target, content, 'utf-8');
      written.push(target);
    } else if (type === 'team') {
      const target = join(this.dirFor('team'), `${name}.md`);
      mkdirSync(join(this.dirFor('team')), { recursive: true });
      writeFileSync(target, content, 'utf-8');
      written.push(target);
      if (files) {
        for (const [relPath, fileContent] of Object.entries(files)) {
          const fileTarget = join(this.baseDir, relPath);
          mkdirSync(join(fileTarget, '..'), { recursive: true });
          writeFileSync(fileTarget, fileContent, 'utf-8');
          written.push(fileTarget);
        }
      }
    } else if (type === 'skill') {
      const skillDir = join(this.dirFor('skill'), name);
      const target = join(skillDir, 'SKILL.md');
      mkdirSync(skillDir, { recursive: true });
      writeFileSync(target, content, 'utf-8');
      written.push(target);
      if (files) {
        for (const [relPath, fileContent] of Object.entries(files)) {
          const fileTarget = join(this.baseDir, relPath);
          mkdirSync(join(fileTarget, '..'), { recursive: true });
          writeFileSync(fileTarget, fileContent, 'utf-8');
          written.push(fileTarget);
        }
      }
    } else if (type === 'tool') {
      const target = join(this.dirFor('tool'), `${name}.md`);
      mkdirSync(join(this.dirFor('tool')), { recursive: true });
      writeFileSync(target, content, 'utf-8');
      written.push(target);
    }

    return written;
  }

  uninstall(type: PackageType, name: string): string[] {
    const removed: string[] = [];

    if (type === 'agent' || type === 'team' || type === 'tool') {
      const target = join(this.dirFor(type), `${name}.md`);
      if (existsSync(target)) {
        unlinkSync(target);
        removed.push(target);
      }
    } else if (type === 'skill') {
      const skillDir = join(this.dirFor('skill'), name);
      if (existsSync(skillDir)) {
        rmSync(skillDir, { recursive: true, force: true });
        removed.push(skillDir);
      } else {
        const flat = join(this.dirFor('skill'), `${name}.md`);
        if (existsSync(flat)) {
          unlinkSync(flat);
          removed.push(flat);
        }
      }
    }

    return removed;
  }

  listInstalled(type: PackageType): string[] {
    const dir = this.dirFor(type);
    if (!existsSync(dir)) return [];
    return readdirSync(dir)
      .filter((f) => f.endsWith('.md') || (type === 'skill' && !f.includes('.')))
      .map((f) => f.replace(/\.md$/, ''));
  }
}
