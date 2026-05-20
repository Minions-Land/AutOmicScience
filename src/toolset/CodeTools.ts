/**
 * CodeTools — Code analysis, symbol search, and structure extraction.
 * Uses regex-based parsing as a portable fallback (no native tree-sitter dependency).
 */

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';

// ---------------------------------------------------------------------------
// Language detection
// ---------------------------------------------------------------------------

const EXT_TO_LANG: Record<string, string> = {
  '.py': 'python',
  '.js': 'javascript',
  '.ts': 'typescript',
  '.tsx': 'typescript',
  '.jsx': 'javascript',
  '.rs': 'rust',
  '.go': 'go',
  '.java': 'java',
  '.rb': 'ruby',
  '.c': 'c',
  '.cpp': 'cpp',
  '.h': 'c',
  '.hpp': 'cpp',
  '.cs': 'csharp',
  '.swift': 'swift',
  '.kt': 'kotlin',
};

function detectLanguage(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase();
  return EXT_TO_LANG[ext] ?? 'unknown';
}

// ---------------------------------------------------------------------------
// Symbol extraction (regex-based, multi-language)
// ---------------------------------------------------------------------------

export interface SymbolInfo {
  name: string;
  kind: 'class' | 'function' | 'method' | 'interface' | 'type' | 'variable';
  startLine: number;
  endLine: number;
  signature: string;
  children?: SymbolInfo[];
}

/** Regex patterns per language for top-level symbols. */
const PATTERNS: Record<string, RegExp[]> = {
  python: [
    /^(?<indent>\s*)class\s+(?<name>\w+)(?:\(.*?\))?:/gm,
    /^(?<indent>\s*)(?:async\s+)?def\s+(?<name>\w+)\(.*?\).*?:/gm,
  ],
  typescript: [
    /^(?<indent>\s*)(?:export\s+)?(?:abstract\s+)?class\s+(?<name>\w+)/gm,
    /^(?<indent>\s*)(?:export\s+)?(?:async\s+)?function\s+(?<name>\w+)/gm,
    /^(?<indent>\s*)(?:export\s+)?interface\s+(?<name>\w+)/gm,
    /^(?<indent>\s*)(?:export\s+)?type\s+(?<name>\w+)/gm,
  ],
  javascript: [
    /^(?<indent>\s*)(?:export\s+)?class\s+(?<name>\w+)/gm,
    /^(?<indent>\s*)(?:export\s+)?(?:async\s+)?function\s+(?<name>\w+)/gm,
  ],
};

// Fallback: use typescript patterns for unknown languages
function getPatternsForLang(lang: string): RegExp[] {
  return PATTERNS[lang] ?? PATTERNS['typescript'] ?? [];
}

function extractSymbols(content: string, lang: string): SymbolInfo[] {
  const lines = content.split('\n');
  const symbols: SymbolInfo[] = [];
  const patterns = getPatternsForLang(lang);

  for (const pattern of patterns) {
    // Reset lastIndex for global regex
    const re = new RegExp(pattern.source, pattern.flags);
    let match: RegExpExecArray | null;

    while ((match = re.exec(content)) !== null) {
      const lineNum = content.slice(0, match.index).split('\n').length;
      const indent = (match.groups?.indent ?? '').length;
      const name = match.groups?.name ?? match[1] ?? 'unknown';
      const signature = match[0].trim();

      // Determine kind from the matched text
      let kind: SymbolInfo['kind'] = 'function';
      if (/class\s/.test(match[0])) kind = 'class';
      else if (/interface\s/.test(match[0])) kind = 'interface';
      else if (/type\s/.test(match[0])) kind = 'type';
      else if (indent > 0 && lang === 'python') kind = 'method';

      // Estimate end line: find next symbol at same or lower indent, or EOF
      let endLine = lines.length;
      for (let i = lineNum; i < lines.length; i++) {
        const line = lines[i];
        if (i === lineNum - 1) continue;
        if (line.trim() === '') continue;
        const currentIndent = line.length - line.trimStart().length;
        if (currentIndent <= indent && i > lineNum) {
          endLine = i;
          break;
        }
      }

      const symbol: SymbolInfo = {
        name,
        kind,
        startLine: lineNum,
        endLine,
        signature,
      };

      // Nest methods inside classes
      if (kind === 'method' && symbols.length > 0) {
        const parent = symbols[symbols.length - 1];
        if (parent.kind === 'class' && lineNum >= parent.startLine && lineNum <= parent.endLine) {
          if (!parent.children) parent.children = [];
          parent.children.push(symbol);
          continue;
        }
      }

      symbols.push(symbol);
    }
  }

  return symbols;
}

// ---------------------------------------------------------------------------
// Toolset factory
// ---------------------------------------------------------------------------

export interface CodeToolsOptions {
  /** Root directory for resolving relative paths. */
  rootDir?: string;
}

export function codeToolSet(opts: CodeToolsOptions = {}): ToolSet {
  const rootDir = opts.rootDir ?? process.cwd();
  const resolve = (p: string) => (path.isAbsolute(p) ? p : path.resolve(rootDir, p));

  return new ToolSet('code', [
    defineTool<
      { filePath: string },
      { file: string; language: string; totalLines: number; symbols: SymbolInfo[] }
    >({
      name: 'analyze_code',
      description:
        'Parse a source file and return its structure: classes, functions, interfaces, ' +
        'with line ranges and signatures. Useful for understanding large files without reading all code.',
      parameters: z.object({
        filePath: z.string().describe('Path to the source file'),
      }),
      execute: async ({ filePath }) => {
        const full = resolve(filePath);
        const content = await fs.readFile(full, 'utf8');
        const lang = detectLanguage(full);
        const symbols = extractSymbols(content, lang);
        const totalLines = content.split('\n').length;
        return { file: full, language: lang, totalLines, symbols };
      },
    }),

    defineTool<
      { query: string; directory?: string; extensions?: string[] },
      { matches: { file: string; name: string; kind: string; line: number; signature: string }[] }
    >({
      name: 'search_symbols',
      description:
        'Search for symbol definitions (functions, classes, interfaces) matching a query ' +
        'across files in a directory. Returns matching symbol names with file locations.',
      parameters: z.object({
        query: z.string().describe('Symbol name or substring to search for'),
        directory: z.string().optional().describe('Directory to search (defaults to project root)'),
        extensions: z
          .array(z.string())
          .optional()
          .describe('File extensions to include, e.g. [".ts", ".py"]'),
      }),
      execute: async ({ query, directory, extensions }) => {
        const dir = resolve(directory ?? '.');
        const exts = extensions ?? Object.keys(EXT_TO_LANG);
        const matches: { file: string; name: string; kind: string; line: number; signature: string }[] = [];

        async function walk(d: string): Promise<void> {
          let entries;
          try {
            entries = await fs.readdir(d, { withFileTypes: true });
          } catch {
            return;
          }
          for (const entry of entries) {
            const full = path.join(d, entry.name);
            if (entry.isDirectory()) {
              // Skip common non-source directories
              if (['node_modules', '.git', 'dist', '__pycache__', '.venv'].includes(entry.name)) continue;
              await walk(full);
            } else if (entry.isFile()) {
              const ext = path.extname(entry.name).toLowerCase();
              if (!exts.includes(ext)) continue;
              try {
                const content = await fs.readFile(full, 'utf8');
                const lang = detectLanguage(full);
                const symbols = extractSymbols(content, lang);
                for (const sym of symbols) {
                  if (sym.name.toLowerCase().includes(query.toLowerCase())) {
                    matches.push({
                      file: full,
                      name: sym.name,
                      kind: sym.kind,
                      line: sym.startLine,
                      signature: sym.signature,
                    });
                  }
                  // Also check children
                  if (sym.children) {
                    for (const child of sym.children) {
                      if (child.name.toLowerCase().includes(query.toLowerCase())) {
                        matches.push({
                          file: full,
                          name: `${sym.name}.${child.name}`,
                          kind: child.kind,
                          line: child.startLine,
                          signature: child.signature,
                        });
                      }
                    }
                  }
                }
              } catch {
                // Skip unreadable files
              }
            }
          }
        }

        await walk(dir);
        return { matches: matches.slice(0, 100) };
      },
    }),

    defineTool<
      { symbol: string; directory?: string; extensions?: string[] },
      { references: { file: string; line: number; context: string }[] }
    >({
      name: 'get_references',
      description:
        'Find all references (usages) of a symbol name across files. ' +
        'Searches for the exact symbol name as a word boundary match.',
      parameters: z.object({
        symbol: z.string().describe('Symbol name to find references for'),
        directory: z.string().optional().describe('Directory to search'),
        extensions: z.array(z.string()).optional().describe('File extensions to include'),
      }),
      execute: async ({ symbol, directory, extensions }) => {
        const dir = resolve(directory ?? '.');
        const exts = extensions ?? Object.keys(EXT_TO_LANG);
        const references: { file: string; line: number; context: string }[] = [];
        const re = new RegExp(`\\b${symbol.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'g');

        async function walk(d: string): Promise<void> {
          let entries;
          try {
            entries = await fs.readdir(d, { withFileTypes: true });
          } catch {
            return;
          }
          for (const entry of entries) {
            const full = path.join(d, entry.name);
            if (entry.isDirectory()) {
              if (['node_modules', '.git', 'dist', '__pycache__', '.venv'].includes(entry.name)) continue;
              await walk(full);
            } else if (entry.isFile()) {
              const ext = path.extname(entry.name).toLowerCase();
              if (!exts.includes(ext)) continue;
              try {
                const content = await fs.readFile(full, 'utf8');
                const lines = content.split('\n');
                for (let i = 0; i < lines.length; i++) {
                  if (re.test(lines[i])) {
                    references.push({
                      file: full,
                      line: i + 1,
                      context: lines[i].trim().slice(0, 200),
                    });
                  }
                  re.lastIndex = 0;
                }
              } catch {
                // Skip unreadable files
              }
            }
          }
        }

        await walk(dir);
        return { references: references.slice(0, 200) };
      },
    }),

    defineTool<
      { filePath: string; startLine?: number; endLine?: number },
      { file: string; language: string; code: string; startLine: number; endLine: number }
    >({
      name: 'view_code_item',
      description:
        'View source code of a specific line range in a file. ' +
        'Use after analyze_code to drill into specific symbols.',
      parameters: z.object({
        filePath: z.string().describe('Path to the source file'),
        startLine: z.number().int().positive().optional().describe('Start line (1-indexed, inclusive)'),
        endLine: z.number().int().positive().optional().describe('End line (1-indexed, inclusive)'),
      }),
      execute: async ({ filePath, startLine, endLine }) => {
        const full = resolve(filePath);
        const content = await fs.readFile(full, 'utf8');
        const lines = content.split('\n');
        const start = startLine ?? 1;
        const end = endLine ?? lines.length;
        const code = lines.slice(start - 1, end).join('\n');
        return {
          file: full,
          language: detectLanguage(full),
          code,
          startLine: start,
          endLine: end,
        };
      },
    }),
  ]);
}
