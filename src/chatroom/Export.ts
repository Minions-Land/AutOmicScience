import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  copyFileSync,
  statSync,
  readdirSync,
} from 'node:fs';
import { join, dirname, relative, extname, resolve } from 'node:path';
import type { Message } from '../types.js';

// --- Types ---

export interface ExportOptions {
  /** Include system messages in export. Default: false */
  includeSystem?: boolean;
  /** Filter messages after this date (ISO string or timestamp). */
  startDate?: string | number;
  /** Filter messages before this date (ISO string or timestamp). */
  endDate?: string | number;
  /** Compress output to .zip. Default: false */
  compress?: boolean;
  /** Maximum file size to include in bundle (bytes). Default: 100MB */
  sizeLimitBytes?: number;
}

export interface ExportResult {
  success: boolean;
  bundlePath?: string;
  message: string;
  stats?: {
    messages: number;
    filesCopied: number;
    filesSkipped: number;
  };
}

export interface ImportResult {
  success: boolean;
  chatId?: string;
  chatName?: string;
  message: string;
}

interface ManifestFile {
  original: string;
  local: string;
  size: number;
}

interface Manifest {
  version: string;
  chatId: string;
  chatName: string;
  exportedAt: string;
  files: ManifestFile[];
  skippedLargeFiles: string[];
  stats: {
    messages: number;
    filesCopied: number;
    filesSkipped: number;
  };
}

// --- Path Scanning ---

const SKIP_PREFIXES = [
  '/usr/', '/bin/', '/sbin/', '/opt/homebrew/', '/opt/local/',
  '/System/', '/Library/', '/Applications/',
  '/nix/', '/snap/',
];

const SKIP_EXTENSIONS = new Set([
  '', '.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe',
  '.o', '.a', '.ko', '.class',
]);

const ABS_PATH_RE = /(?<=["\s,:\[({])(\/(Users|home|tmp|var|opt)\/[^\s"'\\,\]})]{3,})/g;

function isExportable(filePath: string): boolean {
  if (SKIP_PREFIXES.some((p) => filePath.startsWith(p))) return false;
  const ext = extname(filePath).toLowerCase();
  if (SKIP_EXTENSIONS.has(ext)) return false;
  try {
    const stat = statSync(filePath);
    return stat.isFile();
  } catch {
    return false;
  }
}

function scanFilePaths(text: string): Set<string> {
  const paths = new Set<string>();
  const matches = text.matchAll(ABS_PATH_RE);
  for (const m of matches) {
    let raw = m[1];
    // Clean trailing punctuation
    raw = raw.replace(/['"\`,;:)\]}*\\]+$/, '');
    if (isExportable(raw)) {
      paths.add(raw);
    }
  }
  return paths;
}

function relativeFilesPath(absPath: string): string {
  return 'files/' + absPath.replace(/^\//, '');
}

// --- Export ---

/**
 * Export a chat and its referenced files into a portable bundle.
 *
 * Bundle structure:
 *   <bundle>/
 *     manifest.json
 *     chat.jsonl
 *     chat.meta.json
 *     files/
 */
export function exportChatBundle(
  memoryDir: string,
  chatId: string,
  outputDir: string,
  opts?: ExportOptions,
): ExportResult {
  const jsonlPath = join(memoryDir, `${chatId}.jsonl`);
  const metaPath = join(memoryDir, `${chatId}.meta.json`);

  if (!existsSync(jsonlPath)) {
    return { success: false, message: `Chat ${chatId} not found` };
  }

  let jsonlText = readFileSync(jsonlPath, 'utf-8');
  let metaText = existsSync(metaPath) ? readFileSync(metaPath, 'utf-8') : '{}';
  const meta = JSON.parse(metaText) as Record<string, unknown>;

  // Apply date filtering if specified
  if (opts?.startDate || opts?.endDate) {
    const lines = jsonlText.split('\n').filter((l) => l.trim());
    const startTs = opts.startDate
      ? typeof opts.startDate === 'string' ? new Date(opts.startDate).getTime() : opts.startDate
      : 0;
    const endTs = opts.endDate
      ? typeof opts.endDate === 'string' ? new Date(opts.endDate).getTime() : opts.endDate
      : Infinity;

    const filtered = lines.filter((line) => {
      try {
        const msg = JSON.parse(line) as Record<string, unknown>;
        const ts = (msg.timestamp as number) ?? 0;
        return ts >= startTs && ts <= endTs;
      } catch {
        return true; // Keep unparseable lines
      }
    });
    jsonlText = filtered.join('\n') + '\n';
  }

  // Filter system messages if requested
  if (opts?.includeSystem === false) {
    const lines = jsonlText.split('\n').filter((l) => l.trim());
    const filtered = lines.filter((line) => {
      try {
        const msg = JSON.parse(line) as Record<string, unknown>;
        return msg.role !== 'system';
      } catch {
        return true;
      }
    });
    jsonlText = filtered.join('\n') + '\n';
  }

  // Scan for file references
  const allPaths = new Set([
    ...scanFilePaths(jsonlText),
    ...scanFilePaths(metaText),
  ]);

  // Prepare output
  mkdirSync(outputDir, { recursive: true });
  const filesDir = join(outputDir, 'files');
  mkdirSync(filesDir, { recursive: true });

  const sizeLimitBytes = opts?.sizeLimitBytes ?? 100 * 1024 * 1024;
  const copiedFiles: ManifestFile[] = [];
  const skippedFiles: string[] = [];

  for (const absPath of Array.from(allPaths).sort()) {
    let fileSize: number;
    try {
      fileSize = statSync(absPath).size;
    } catch {
      skippedFiles.push(absPath);
      continue;
    }

    if (fileSize > sizeLimitBytes) {
      skippedFiles.push(absPath);
      continue;
    }

    const rel = relativeFilesPath(absPath);
    const dest = join(outputDir, rel);
    try {
      mkdirSync(dirname(dest), { recursive: true });
      copyFileSync(absPath, dest);
      copiedFiles.push({ original: absPath, local: rel, size: fileSize });
    } catch {
      skippedFiles.push(absPath);
    }
  }

  // Rewrite paths in jsonl/meta
  let rewrittenJsonl = jsonlText;
  let rewrittenMeta = metaText;
  for (const f of copiedFiles) {
    rewrittenJsonl = rewrittenJsonl.replaceAll(f.original, './' + f.local);
    rewrittenMeta = rewrittenMeta.replaceAll(f.original, './' + f.local);
  }

  writeFileSync(join(outputDir, 'chat.jsonl'), rewrittenJsonl, 'utf-8');
  writeFileSync(join(outputDir, 'chat.meta.json'), rewrittenMeta, 'utf-8');

  // Write manifest
  const manifest: Manifest = {
    version: '1.0',
    chatId,
    chatName: (meta.name as string) ?? '',
    exportedAt: new Date().toISOString(),
    files: copiedFiles,
    skippedLargeFiles: skippedFiles,
    stats: {
      messages: rewrittenJsonl.split('\n').filter((l) => l.trim()).length,
      filesCopied: copiedFiles.length,
      filesSkipped: skippedFiles.length,
    },
  };
  writeFileSync(join(outputDir, 'manifest.json'), JSON.stringify(manifest, null, 2), 'utf-8');

  return {
    success: true,
    bundlePath: outputDir,
    message: `Exported ${copiedFiles.length} files`,
    stats: manifest.stats,
  };
}

// --- Import ---

/**
 * Import a chat from an exported bundle.
 * Maps relative paths back to targetRoot.
 */
export function importChatBundle(
  memoryDir: string,
  bundlePath: string,
  targetRoot: string,
): ImportResult {
  const jsonlPath = join(bundlePath, 'chat.jsonl');
  const metaPath = join(bundlePath, 'chat.meta.json');
  const manifestPath = join(bundlePath, 'manifest.json');

  if (!existsSync(jsonlPath)) {
    return { success: false, message: 'Invalid bundle: chat.jsonl not found' };
  }

  const manifest: Partial<Manifest> = existsSync(manifestPath)
    ? JSON.parse(readFileSync(manifestPath, 'utf-8'))
    : {};

  let jsonlText = readFileSync(jsonlPath, 'utf-8');
  let metaText = existsSync(metaPath) ? readFileSync(metaPath, 'utf-8') : '{}';
  const meta = JSON.parse(metaText) as Record<string, unknown>;

  // Copy files and rewrite paths
  const filesDir = join(bundlePath, 'files');
  let filesCopied = 0;

  if (existsSync(filesDir)) {
    const walkFiles = (dir: string): string[] => {
      const results: string[] = [];
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const full = join(dir, entry.name);
        if (entry.isDirectory()) {
          results.push(...walkFiles(full));
        } else {
          results.push(full);
        }
      }
      return results;
    };

    for (const src of walkFiles(filesDir)) {
      const relToFiles = relative(filesDir, src);
      const dest = '/' + relToFiles;
      const destDir = dirname(dest);

      try {
        mkdirSync(destDir, { recursive: true });
        if (!existsSync(dest)) {
          copyFileSync(src, dest);
          filesCopied++;
        }
      } catch {
        // Skip files we can't copy
      }

      // Rewrite paths
      const bundleRel = `./files/${relToFiles}`;
      jsonlText = jsonlText.replaceAll(bundleRel, dest);
      metaText = metaText.replaceAll(bundleRel, dest);
    }
  }

  // Determine chat ID
  const originalId = manifest.chatId ?? (meta.id as string) ?? '';
  const originalName = (meta.name as string) ?? manifest.chatName ?? 'Imported Chat';

  // Check if already exists
  if (originalId && existsSync(join(memoryDir, `${originalId}.jsonl`))) {
    return {
      success: true,
      chatId: originalId,
      chatName: originalName,
      message: `Chat '${originalName}' already exists - skipped (files updated)`,
    };
  }

  const chatId = originalId || crypto.randomUUID();
  const updatedMeta = { ...JSON.parse(metaText), id: chatId };

  mkdirSync(memoryDir, { recursive: true });
  writeFileSync(join(memoryDir, `${chatId}.jsonl`), jsonlText, 'utf-8');
  writeFileSync(
    join(memoryDir, `${chatId}.meta.json`),
    JSON.stringify(updatedMeta, null, 2),
    'utf-8',
  );

  return {
    success: true,
    chatId,
    chatName: originalName,
    message: `Imported '${originalName}' with ${filesCopied} files`,
  };
}

// --- Markdown Export ---

/**
 * Export a chat to a readable Markdown file.
 */
export function exportChatToMarkdown(
  memoryDir: string,
  chatId: string,
  outputPath: string,
  opts?: ExportOptions,
): ExportResult {
  const jsonlPath = join(memoryDir, `${chatId}.jsonl`);
  const metaPath = join(memoryDir, `${chatId}.meta.json`);

  if (!existsSync(jsonlPath)) {
    return { success: false, message: `Chat ${chatId} not found` };
  }

  const jsonlText = readFileSync(jsonlPath, 'utf-8');
  const meta = existsSync(metaPath)
    ? JSON.parse(readFileSync(metaPath, 'utf-8')) as Record<string, unknown>
    : {};

  const lines = jsonlText.split('\n').filter((l) => l.trim());
  const messages: Message[] = [];

  for (const line of lines) {
    try {
      const msg = JSON.parse(line) as Message & { timestamp?: number };

      // Filter by date
      if (opts?.startDate || opts?.endDate) {
        const ts = (msg as unknown as Record<string, unknown>).timestamp as number ?? 0;
        const startTs = opts.startDate
          ? typeof opts.startDate === 'string' ? new Date(opts.startDate).getTime() : opts.startDate
          : 0;
        const endTs = opts.endDate
          ? typeof opts.endDate === 'string' ? new Date(opts.endDate).getTime() : opts.endDate
          : Infinity;
        if (ts < startTs || ts > endTs) continue;
      }

      // Filter system messages
      if (!opts?.includeSystem && msg.role === 'system') continue;

      messages.push(msg);
    } catch {
      // Skip malformed lines
    }
  }

  // Build markdown
  const parts: string[] = [];
  const chatName = (meta.name as string) ?? 'Untitled Chat';
  parts.push(`# ${chatName}\n\n`);
  parts.push(`> Exported: ${new Date().toISOString()}\n`);
  parts.push(`> Messages: ${messages.length}\n\n`);
  parts.push('---\n\n');

  for (const msg of messages) {
    const roleLabel = msg.role === 'user' ? 'User' : msg.role === 'assistant' ? 'Assistant' : msg.role;
    const content = typeof msg.content === 'string'
      ? msg.content
      : Array.isArray(msg.content)
        ? msg.content
            .filter((b): b is { type: 'text'; text: string } => b.type === 'text')
            .map((b) => b.text)
            .join('\n')
        : '';

    parts.push(`### ${roleLabel}\n\n`);
    parts.push(`${content}\n\n`);
  }

  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, parts.join(''), 'utf-8');

  return {
    success: true,
    bundlePath: outputPath,
    message: `Exported ${messages.length} messages to Markdown`,
    stats: { messages: messages.length, filesCopied: 0, filesSkipped: 0 },
  };
}

// --- JSON Export ---

/**
 * Export a chat to a structured JSON file.
 */
export function exportChatToJSON(
  memoryDir: string,
  chatId: string,
  outputPath: string,
  opts?: ExportOptions,
): ExportResult {
  const jsonlPath = join(memoryDir, `${chatId}.jsonl`);
  const metaPath = join(memoryDir, `${chatId}.meta.json`);

  if (!existsSync(jsonlPath)) {
    return { success: false, message: `Chat ${chatId} not found` };
  }

  const jsonlText = readFileSync(jsonlPath, 'utf-8');
  const meta = existsSync(metaPath)
    ? JSON.parse(readFileSync(metaPath, 'utf-8')) as Record<string, unknown>
    : {};

  const lines = jsonlText.split('\n').filter((l) => l.trim());
  const messages: unknown[] = [];

  for (const line of lines) {
    try {
      const msg = JSON.parse(line) as Record<string, unknown>;

      if (!opts?.includeSystem && msg.role === 'system') continue;

      if (opts?.startDate || opts?.endDate) {
        const ts = (msg.timestamp as number) ?? 0;
        const startTs = opts.startDate
          ? typeof opts.startDate === 'string' ? new Date(opts.startDate).getTime() : opts.startDate
          : 0;
        const endTs = opts.endDate
          ? typeof opts.endDate === 'string' ? new Date(opts.endDate).getTime() : opts.endDate
          : Infinity;
        if (ts < startTs || ts > endTs) continue;
      }

      messages.push(msg);
    } catch {
      // Skip malformed
    }
  }

  const output = {
    chatId,
    chatName: (meta.name as string) ?? 'Untitled Chat',
    exportedAt: new Date().toISOString(),
    messageCount: messages.length,
    metadata: meta,
    messages,
  };

  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, JSON.stringify(output, null, 2), 'utf-8');

  return {
    success: true,
    bundlePath: outputPath,
    message: `Exported ${messages.length} messages to JSON`,
    stats: { messages: messages.length, filesCopied: 0, filesSkipped: 0 },
  };
}
