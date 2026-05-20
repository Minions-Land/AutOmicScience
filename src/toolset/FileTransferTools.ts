/**
 * FileTransferTools — Send and receive files between agents via shared storage.
 *
 * The shared-file registry is persisted as JSON at
 *   ~/.medrix/file-transfer/registry.json
 * (override with env MEDRIX_FILE_TRANSFER_REGISTRY).
 *
 * Writes go through a temp-file + rename to make concurrent writers safe on
 * POSIX filesystems. This makes cross-process file transfer work without
 * needing NATS — every tool invocation reads the registry fresh from disk.
 */

import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';

// ---------------------------------------------------------------------------
// Shared file registry (persisted JSON)
// ---------------------------------------------------------------------------

interface SharedFile {
  id: string;
  filename: string;
  sourcePath: string;
  senderAgent: string;
  recipientAgent?: string;
  size: number;
  checksum: string;
  sharedAt: string;
  chunks?: number;
  metadata: Record<string, unknown>;
}

interface RegistryShape {
  version: 1;
  files: SharedFile[];
}

function defaultRegistryPath(): string {
  return (
    process.env.MEDRIX_FILE_TRANSFER_REGISTRY ??
    path.join(os.homedir(), '.medrix', 'file-transfer', 'registry.json')
  );
}

function defaultSharedDir(): string {
  return path.resolve(
    process.env.MEDRIX_SHARED_DIR ??
      path.join(os.homedir(), '.medrix', 'file-transfer', 'shared'),
  );
}

async function ensureDir(dir: string): Promise<void> {
  await fs.mkdir(dir, { recursive: true });
}

/** Read the registry. Returns an empty registry if the file doesn't exist or is corrupt. */
async function readRegistry(registryPath: string): Promise<RegistryShape> {
  try {
    const raw = await fs.readFile(registryPath, 'utf8');
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && Array.isArray((parsed as RegistryShape).files)) {
      return parsed as RegistryShape;
    }
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'ENOENT') {
      // Treat corrupt registry as empty rather than failing every transfer.
    }
  }
  return { version: 1, files: [] };
}

/**
 * Atomic-ish write: write to <path>.<rand>.tmp then rename to final.
 * On POSIX, rename is atomic within the same filesystem. We also serialise
 * writes from this process via a simple in-memory mutex so the read-modify-
 * write cycle is consistent within a process.
 */
let writeChain: Promise<void> = Promise.resolve();

async function withRegistryLock<T>(fn: () => Promise<T>): Promise<T> {
  const next = writeChain.then(fn, fn);
  // Keep the chain alive but don't propagate errors.
  writeChain = next.then(
    () => undefined,
    () => undefined,
  );
  return next;
}

async function writeRegistry(registryPath: string, reg: RegistryShape): Promise<void> {
  await ensureDir(path.dirname(registryPath));
  const tmp = `${registryPath}.${process.pid}.${crypto.randomBytes(4).toString('hex')}.tmp`;
  await fs.writeFile(tmp, JSON.stringify(reg, null, 2), 'utf8');
  await fs.rename(tmp, registryPath);
}

function computeChecksum(buffer: Buffer): string {
  return crypto.createHash('sha256').update(buffer).digest('hex').slice(0, 16);
}

// ---------------------------------------------------------------------------
// Toolset factory
// ---------------------------------------------------------------------------

export interface FileTransferToolsOptions {
  /** Agent name for this toolset instance. */
  agentName?: string;
  /** Shared directory for transferred file payloads. */
  sharedDir?: string;
  /** Path to the JSON registry file. */
  registryPath?: string;
  /** Maximum file size for transfer in bytes (default 100MB). */
  maxFileSize?: number;
  /** Chunk size for large file transfers (default 1MB). */
  chunkSize?: number;
}

export function fileTransferToolSet(opts: FileTransferToolsOptions = {}): ToolSet {
  const agentName = opts.agentName ?? 'unknown';
  const maxFileSize = opts.maxFileSize ?? 100 * 1024 * 1024;
  const chunkSize = opts.chunkSize ?? 1024 * 1024;
  const sharedDir = path.resolve(opts.sharedDir ?? defaultSharedDir());
  const registryPath = opts.registryPath ?? defaultRegistryPath();

  return new ToolSet('file_transfer', [
    // -----------------------------------------------------------------------
    // send_file
    // -----------------------------------------------------------------------
    defineTool<
      { filePath: string; recipientAgent?: string; metadata?: Record<string, unknown> },
      { ok: boolean; fileId: string; filename: string; size: number; checksum: string }
    >({
      name: 'send_file',
      description:
        'Share a file with another agent (or all agents in the room). ' +
        'The file is copied to shared storage and registered for retrieval.',
      parameters: z.object({
        filePath: z.string().describe('Path to the file to share'),
        recipientAgent: z.string().optional().describe('Target agent name (omit for broadcast)'),
        metadata: z.record(z.unknown()).optional().describe('Metadata to attach to the shared file'),
      }),
      execute: async ({ filePath, recipientAgent, metadata }) => {
        const resolvedPath = path.resolve(filePath);
        const stat = await fs.stat(resolvedPath);

        if (!stat.isFile()) throw new Error('Path is not a file');
        if (stat.size > maxFileSize) {
          throw new Error(`File too large (${stat.size} bytes). Max: ${maxFileSize}`);
        }

        await ensureDir(sharedDir);
        const fileId = `ft_${Date.now().toString(36)}_${crypto.randomBytes(4).toString('hex')}`;
        const filename = path.basename(resolvedPath);
        const destPath = path.join(sharedDir, `${fileId}_${filename}`);

        const buffer = await fs.readFile(resolvedPath);
        const checksum = computeChecksum(buffer);

        const numChunks = Math.ceil(buffer.length / chunkSize);
        if (numChunks > 1) {
          for (let i = 0; i < numChunks; i++) {
            const chunkBuf = buffer.subarray(i * chunkSize, (i + 1) * chunkSize);
            await fs.writeFile(`${destPath}.chunk${i}`, chunkBuf);
          }
          await fs.writeFile(
            `${destPath}.manifest`,
            JSON.stringify({ chunks: numChunks, totalSize: buffer.length, checksum }),
            'utf8',
          );
        } else {
          await fs.writeFile(destPath, buffer);
        }

        const shared: SharedFile = {
          id: fileId,
          filename,
          sourcePath: destPath,
          senderAgent: agentName,
          recipientAgent,
          size: stat.size,
          checksum,
          sharedAt: new Date().toISOString(),
          chunks: numChunks > 1 ? numChunks : undefined,
          metadata: metadata ?? {},
        };

        await withRegistryLock(async () => {
          const reg = await readRegistry(registryPath);
          reg.files.push(shared);
          await writeRegistry(registryPath, reg);
        });

        return { ok: true, fileId, filename, size: stat.size, checksum };
      },
    }),

    // -----------------------------------------------------------------------
    // receive_file
    // -----------------------------------------------------------------------
    defineTool<
      { fileId: string; outputPath?: string },
      { ok: boolean; path: string; size: number; checksum: string; sender: string }
    >({
      name: 'receive_file',
      description:
        'Receive a shared file by its ID. Downloads from shared storage to a local path.',
      parameters: z.object({
        fileId: z.string().describe('File ID from send_file or list_shared_files'),
        outputPath: z.string().optional().describe('Local path to save the file'),
      }),
      execute: async ({ fileId, outputPath }) => {
        const reg = await readRegistry(registryPath);
        const shared = reg.files.find((f) => f.id === fileId);
        if (!shared) throw new Error(`File '${fileId}' not found in shared storage`);

        if (shared.recipientAgent && shared.recipientAgent !== agentName) {
          throw new Error(`File '${fileId}' is not addressed to this agent`);
        }

        const destPath = outputPath
          ? path.resolve(outputPath)
          : path.resolve(process.cwd(), shared.filename);

        await ensureDir(path.dirname(destPath));

        if (shared.chunks) {
          const chunks: Buffer[] = [];
          for (let i = 0; i < shared.chunks; i++) {
            const chunkBuf = await fs.readFile(`${shared.sourcePath}.chunk${i}`);
            chunks.push(chunkBuf);
          }
          const assembled = Buffer.concat(chunks);
          await fs.writeFile(destPath, assembled);
        } else {
          await fs.copyFile(shared.sourcePath, destPath);
        }

        return {
          ok: true,
          path: destPath,
          size: shared.size,
          checksum: shared.checksum,
          sender: shared.senderAgent,
        };
      },
    }),

    // -----------------------------------------------------------------------
    // list_shared_files
    // -----------------------------------------------------------------------
    defineTool<
      { senderAgent?: string },
      { files: { id: string; filename: string; sender: string; size: number; sharedAt: string; recipient?: string }[] }
    >({
      name: 'list_shared_files',
      description:
        'List files available in shared storage. Shows files addressed to this agent or broadcast.',
      parameters: z.object({
        senderAgent: z.string().optional().describe('Filter by sender agent name'),
      }),
      execute: async ({ senderAgent }) => {
        const reg = await readRegistry(registryPath);
        const files = reg.files
          .filter((f) => {
            if (f.recipientAgent && f.recipientAgent !== agentName) return false;
            if (senderAgent && f.senderAgent !== senderAgent) return false;
            return true;
          })
          .map((f) => ({
            id: f.id,
            filename: f.filename,
            sender: f.senderAgent,
            size: f.size,
            sharedAt: f.sharedAt,
            recipient: f.recipientAgent,
          }));

        return { files };
      },
    }),
  ]);
}
