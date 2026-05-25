import { promises as fs } from 'node:fs';
import path from 'node:path';

export interface ProjectInstructionFile {
  path: string;
  content: string;
}

export interface ProjectInstructionOptions {
  cwd?: string;
  filenames?: string[];
  maxBytes?: number;
}

const DEFAULT_FILES = ['AGENTS.md', 'AOS.md', 'AUTOMICSCIENCE.md', '.aos/instructions.md'];

export async function loadProjectInstructions(opts: ProjectInstructionOptions = {}): Promise<ProjectInstructionFile[]> {
  const cwd = opts.cwd ?? process.cwd();
  const filenames = opts.filenames ?? DEFAULT_FILES;
  const maxBytes = opts.maxBytes ?? 64 * 1024;
  const files: ProjectInstructionFile[] = [];

  for (const dir of ancestorDirs(cwd)) {
    for (const filename of filenames) {
      const candidate = path.join(dir, filename);
      try {
        const stat = await fs.stat(candidate);
        if (!stat.isFile() || stat.size > maxBytes) continue;
        files.push({
          path: candidate,
          content: await fs.readFile(candidate, 'utf-8'),
        });
      } catch {
        // ignore missing/unreadable instruction files
      }
    }
  }

  return dedupeByPath(files);
}

export function formatProjectInstructions(files: ProjectInstructionFile[]): string {
  if (files.length === 0) return '';
  const parts = ['## Project Instructions'];
  for (const file of files) {
    parts.push(`\n### ${file.path}\n${file.content.trim()}`);
  }
  return parts.join('\n');
}

function ancestorDirs(cwd: string): string[] {
  const dirs: string[] = [];
  let current = path.resolve(cwd);
  while (true) {
    dirs.unshift(current);
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return dirs;
}

function dedupeByPath(files: ProjectInstructionFile[]): ProjectInstructionFile[] {
  const seen = new Set<string>();
  const result: ProjectInstructionFile[] = [];
  for (const file of files) {
    if (seen.has(file.path)) continue;
    seen.add(file.path);
    result.push(file);
  }
  return result;
}
