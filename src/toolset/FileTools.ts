/**
 * FileTools — File CRUD, grep, glob, patch, and metadata operations.
 */

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';
import { PermissionManager } from '../permissions/index.js';
import type { PermissionManagerOptions } from '../permissions/index.js';

export interface FileToolsOptions {
  /** Root directory for resolving relative paths. */
  rootDir?: string;
  /** Maximum file size in bytes that can be read (default 10MB). */
  maxReadSize?: number;
  /** Optional shared permission manager for AutOmicScience tool permission checks. */
  permissionManager?: PermissionManager;
  permissions?: PermissionManagerOptions;
}

export function fileToolSet(opts: FileToolsOptions = {}): ToolSet {
  const rootDir = opts.rootDir ?? process.cwd();
  const maxReadSize = opts.maxReadSize ?? 10 * 1024 * 1024;
  const resolve = (p: string) => (path.isAbsolute(p) ? p : path.resolve(rootDir, p));

  return new ToolSet('file', [
    // -----------------------------------------------------------------------
    // read_file
    // -----------------------------------------------------------------------
    defineTool<
      { path: string; startLine?: number; endLine?: number },
      { content: string; totalLines: number; path: string }
    >({
      name: 'read_file',
      aliases: ['Read'],
      operation: 'read',
      maxResultSizeChars: 120_000,
      description:
        'Read a text file. Optionally specify a line range (1-indexed, inclusive).',
      parameters: z.object({
        path: z.string().describe('File path (relative to workspace or absolute)'),
        startLine: z.number().int().positive().optional().describe('Start line (1-indexed)'),
        endLine: z.number().int().positive().optional().describe('End line (1-indexed)'),
      }),
      getPath: ({ path: p }) => resolve(p),
      isReadOnly: () => true,
      isDestructive: () => false,
      execute: async ({ path: p, startLine, endLine }) => {
        const full = resolve(p);
        const stat = await fs.stat(full);
        if (stat.size > maxReadSize) {
          throw new Error(`File too large (${stat.size} bytes). Max: ${maxReadSize}`);
        }
        const content = await fs.readFile(full, 'utf8');
        const lines = content.split('\n');
        if (startLine || endLine) {
          const start = (startLine ?? 1) - 1;
          const end = endLine ?? lines.length;
          return {
            content: lines.slice(start, end).join('\n'),
            totalLines: lines.length,
            path: full,
          };
        }
        return { content, totalLines: lines.length, path: full };
      },
    }),

    // -----------------------------------------------------------------------
    // write_file
    // -----------------------------------------------------------------------
    defineTool<
      { path: string; content: string; createDirs?: boolean },
      { ok: boolean; path: string; bytesWritten: number }
    >({
      name: 'write_file',
      aliases: ['Write'],
      operation: 'write',
      description: 'Write content to a file. Creates parent directories if needed.',
      parameters: z.object({
        path: z.string().describe('File path'),
        content: z.string().describe('Content to write'),
        createDirs: z.boolean().optional().default(true).describe('Create parent dirs if missing'),
      }),
      getPath: ({ path: p }) => resolve(p),
      isReadOnly: () => false,
      isDestructive: () => true,
      execute: async ({ path: p, content, createDirs }) => {
        const full = resolve(p);
        if (createDirs) {
          await fs.mkdir(path.dirname(full), { recursive: true });
        }
        await fs.writeFile(full, content, 'utf8');
        return { ok: true, path: full, bytesWritten: Buffer.byteLength(content, 'utf8') };
      },
    }),

    // -----------------------------------------------------------------------
    // edit_file (find/replace)
    // -----------------------------------------------------------------------
    defineTool<
      { path: string; oldString: string; newString: string; replaceAll?: boolean },
      { ok: boolean; replacements: number; path: string }
    >({
      name: 'edit_file',
      aliases: ['Edit'],
      operation: 'write',
      description:
        'Edit a file by finding and replacing a string. ' +
        'By default replaces only the first occurrence; set replaceAll for all.',
      parameters: z.object({
        path: z.string().describe('File path'),
        oldString: z.string().describe('Exact string to find'),
        newString: z.string().describe('Replacement string'),
        replaceAll: z.boolean().optional().default(false).describe('Replace all occurrences'),
      }),
      getPath: ({ path: p }) => resolve(p),
      isReadOnly: () => false,
      isDestructive: () => true,
      execute: async ({ path: p, oldString, newString, replaceAll }) => {
        const full = resolve(p);
        let content = await fs.readFile(full, 'utf8');
        const count = content.split(oldString).length - 1;
        if (count === 0) {
          throw new Error('oldString not found in file');
        }
        if (replaceAll) {
          content = content.replaceAll(oldString, newString);
        } else {
          content = content.replace(oldString, newString);
        }
        await fs.writeFile(full, content, 'utf8');
        return { ok: true, replacements: replaceAll ? count : 1, path: full };
      },
    }),

    // -----------------------------------------------------------------------
    // list_directory (with glob support)
    // -----------------------------------------------------------------------
    defineTool<
      { path: string; pattern?: string; recursive?: boolean },
      { entries: { name: string; type: string; size: number }[] }
    >({
      name: 'list_directory',
      aliases: ['LS', 'Glob'],
      operation: 'read',
      maxResultSizeChars: 120_000,
      description:
        'List files and directories. Optionally filter by glob pattern and recurse.',
      parameters: z.object({
        path: z.string().default('.').describe('Directory path'),
        pattern: z.string().optional().describe('Glob pattern to filter (e.g. "*.ts")'),
        recursive: z.boolean().optional().default(false).describe('Recurse into subdirectories'),
      }),
      getPath: ({ path: p }) => resolve(p),
      isReadOnly: () => true,
      isDestructive: () => false,
      execute: async ({ path: p, pattern, recursive }) => {
        const dir = resolve(p);
        const entries: { name: string; type: string; size: number }[] = [];

        async function walk(d: string, prefix: string): Promise<void> {
          const items = await fs.readdir(d, { withFileTypes: true });
          for (const item of items) {
            const relName = prefix ? `${prefix}/${item.name}` : item.name;
            const fullPath = path.join(d, item.name);

            if (item.isDirectory()) {
              if (['node_modules', '.git'].includes(item.name)) continue;
              entries.push({ name: relName + '/', type: 'directory', size: 0 });
              if (recursive) await walk(fullPath, relName);
            } else {
              // Apply glob pattern filter (simple wildcard matching)
              if (pattern && !matchGlob(item.name, pattern)) continue;
              const stat = await fs.stat(fullPath).catch(() => null);
              entries.push({
                name: relName,
                type: 'file',
                size: stat?.size ?? 0,
              });
            }
          }
        }

        await walk(dir, '');
        return { entries };
      },
    }),

    // -----------------------------------------------------------------------
    // grep
    // -----------------------------------------------------------------------
    defineTool<
      { pattern: string; path?: string; extensions?: string[]; maxResults?: number },
      { matches: { file: string; line: number; content: string }[] }
    >({
      name: 'grep',
      aliases: ['Grep'],
      operation: 'read',
      maxResultSizeChars: 120_000,
      description:
        'Search file contents using a regex pattern. Returns matching lines with context.',
      parameters: z.object({
        pattern: z.string().describe('Regex pattern to search for'),
        path: z.string().optional().default('.').describe('Directory or file to search'),
        extensions: z.array(z.string()).optional().describe('File extensions to include'),
        maxResults: z.number().int().positive().optional().default(100).describe('Max results'),
      }),
      getPath: ({ path: p }) => resolve(p ?? '.'),
      isReadOnly: () => true,
      isDestructive: () => false,
      execute: async ({ pattern, path: p, extensions, maxResults }) => {
        const target = resolve(p ?? '.');
        const re = new RegExp(pattern, 'gi');
        const matches: { file: string; line: number; content: string }[] = [];
        const limit = maxResults ?? 100;

        async function searchFile(filePath: string): Promise<void> {
          if (matches.length >= limit) return;
          try {
            const content = await fs.readFile(filePath, 'utf8');
            const lines = content.split('\n');
            for (let i = 0; i < lines.length && matches.length < limit; i++) {
              re.lastIndex = 0;
              if (re.test(lines[i])) {
                matches.push({
                  file: filePath,
                  line: i + 1,
                  content: lines[i].trim().slice(0, 300),
                });
              }
            }
          } catch {
            // Skip binary/unreadable files
          }
        }

        async function walk(d: string): Promise<void> {
          if (matches.length >= limit) return;
          const items = await fs.readdir(d, { withFileTypes: true }).catch(() => []);
          for (const item of items) {
            if (matches.length >= limit) return;
            const full = path.join(d, item.name);
            if (item.isDirectory()) {
              if (['node_modules', '.git', 'dist', '__pycache__'].includes(item.name)) continue;
              await walk(full);
            } else if (item.isFile()) {
              if (extensions && extensions.length > 0) {
                const ext = path.extname(item.name).toLowerCase();
                if (!extensions.includes(ext)) continue;
              }
              await searchFile(full);
            }
          }
        }

        const stat = await fs.stat(target);
        if (stat.isFile()) {
          await searchFile(target);
        } else {
          await walk(target);
        }

        return { matches };
      },
    }),

    // -----------------------------------------------------------------------
    // apply_patch (unified diff)
    // -----------------------------------------------------------------------
    defineTool<
      { path: string; patch: string },
      { ok: boolean; path: string; hunksApplied: number }
    >({
      name: 'apply_patch',
      aliases: ['Patch'],
      operation: 'write',
      description:
        'Apply a unified diff patch to a file. The patch should be in standard unified diff format.',
      parameters: z.object({
        path: z.string().describe('File path to patch'),
        patch: z.string().describe('Unified diff content'),
      }),
      getPath: ({ path: p }) => resolve(p),
      isReadOnly: () => false,
      isDestructive: () => true,
      execute: async ({ path: p, patch }) => {
        const full = resolve(p);
        let content: string;
        try {
          content = await fs.readFile(full, 'utf8');
        } catch {
          content = '';
        }

        const result = applyUnifiedDiff(content, patch);
        await fs.writeFile(full, result.content, 'utf8');
        return { ok: true, path: full, hunksApplied: result.hunksApplied };
      },
    }),

    // -----------------------------------------------------------------------
    // file_info
    // -----------------------------------------------------------------------
    defineTool<
      { path: string },
      { path: string; size: number; modified: string; created: string; isFile: boolean; isDirectory: boolean; extension: string }
    >({
      name: 'file_info',
      operation: 'read',
      description: 'Get file metadata: size, modification time, type.',
      parameters: z.object({
        path: z.string().describe('File or directory path'),
      }),
      getPath: ({ path: p }) => resolve(p),
      isReadOnly: () => true,
      isDestructive: () => false,
      execute: async ({ path: p }) => {
        const full = resolve(p);
        const stat = await fs.stat(full);
        return {
          path: full,
          size: stat.size,
          modified: stat.mtime.toISOString(),
          created: stat.birthtime.toISOString(),
          isFile: stat.isFile(),
          isDirectory: stat.isDirectory(),
          extension: path.extname(full),
        };
      },
    }),
  ], {
    permissionManager: opts.permissionManager,
    permissions: opts.permissions,
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Simple glob matching (supports * and ?). */
function matchGlob(name: string, pattern: string): boolean {
  const re = new RegExp(
    '^' +
      pattern
        .replace(/[.+^${}()|[\]\\]/g, '\\$&')
        .replace(/\*/g, '.*')
        .replace(/\?/g, '.') +
      '$',
    'i',
  );
  return re.test(name);
}

/** Apply a unified diff to content. Minimal implementation. */
function applyUnifiedDiff(content: string, patch: string): { content: string; hunksApplied: number } {
  const lines = content.split('\n');
  const patchLines = patch.split('\n');
  let hunksApplied = 0;
  let offset = 0;

  for (let i = 0; i < patchLines.length; i++) {
    const hunkHeader = patchLines[i].match(/^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@/);
    if (!hunkHeader) continue;

    const origStart = parseInt(hunkHeader[1], 10) - 1 + offset;
    const removals: number[] = [];
    const additions: { line: number; text: string }[] = [];
    let pos = origStart;

    for (let j = i + 1; j < patchLines.length; j++) {
      const pl = patchLines[j];
      if (pl.startsWith('@@') || pl.startsWith('diff ') || pl.startsWith('---') || pl.startsWith('+++')) {
        break;
      }
      if (pl.startsWith('-')) {
        removals.push(pos);
        pos++;
      } else if (pl.startsWith('+')) {
        additions.push({ line: pos, text: pl.slice(1) });
      } else if (pl.startsWith(' ') || pl === '') {
        pos++;
      }
    }

    // Apply removals in reverse order
    for (const idx of removals.reverse()) {
      if (idx < lines.length) {
        lines.splice(idx, 1);
        offset--;
      }
    }

    // Apply additions
    const insertAt = origStart + offset - removals.length + removals.length;
    for (let a = 0; a < additions.length; a++) {
      lines.splice(insertAt + a, 0, additions[a].text);
      offset++;
    }

    hunksApplied++;
  }

  return { content: lines.join('\n'), hunksApplied };
}
