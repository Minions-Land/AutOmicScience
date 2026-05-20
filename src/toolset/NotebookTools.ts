import { z } from 'zod';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';

interface NotebookCell {
  cell_type: 'code' | 'markdown';
  source: string[];
  metadata: Record<string, unknown>;
  outputs?: unknown[];
  execution_count?: number | null;
}

interface NotebookJson {
  nbformat: number;
  nbformat_minor: number;
  metadata: Record<string, unknown>;
  cells: NotebookCell[];
}

function emptyNotebook(): NotebookJson {
  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: { kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' } },
    cells: [],
  };
}

const CreateArgs = z.object({ filePath: z.string(), kernel: z.string().default('python3') });
type CreateArgs = z.infer<typeof CreateArgs>;

const ReadArgs = z.object({ filePath: z.string() });
type ReadArgs = z.infer<typeof ReadArgs>;

const AddCellArgs = z.object({
  filePath: z.string(),
  cellType: z.enum(['code', 'markdown']).default('code'),
  source: z.string(),
});
type AddCellArgs = z.infer<typeof AddCellArgs>;

const ExecuteCellArgs = z.object({
  filePath: z.string(),
  cellIndex: z.number().int().nonnegative(),
});
type ExecuteCellArgs = z.infer<typeof ExecuteCellArgs>;

/** Notebook toolset with real .ipynb file operations. */
export function notebookToolSet(): ToolSet {
  return new ToolSet('notebook', [
    defineTool<CreateArgs, { created: string }>({
      name: 'notebook_create',
      description: 'Create a new empty .ipynb notebook file.',
      parameters: CreateArgs,
      execute: async ({ filePath, kernel }) => {
        const nb = emptyNotebook();
        nb.metadata.kernelspec = { display_name: kernel, language: 'python', name: kernel };
        await fs.mkdir(path.dirname(filePath), { recursive: true });
        await fs.writeFile(filePath, JSON.stringify(nb, null, 2), 'utf8');
        return { created: filePath };
      },
    }),
    defineTool<ReadArgs, { cells: { index: number; type: string; source: string }[] }>({
      name: 'notebook_read',
      description: 'Read a .ipynb notebook and return its cells.',
      parameters: ReadArgs,
      execute: async ({ filePath }) => {
        const raw = await fs.readFile(filePath, 'utf8');
        const nb: NotebookJson = JSON.parse(raw);
        return {
          cells: nb.cells.map((c, i) => ({
            index: i,
            type: c.cell_type,
            source: c.source.join(''),
          })),
        };
      },
    }),
    defineTool<AddCellArgs, { cellIndex: number }>({
      name: 'notebook_add_cell',
      description: 'Append a cell to an existing .ipynb notebook.',
      parameters: AddCellArgs,
      execute: async ({ filePath, cellType, source }) => {
        const raw = await fs.readFile(filePath, 'utf8');
        const nb: NotebookJson = JSON.parse(raw);
        const cell: NotebookCell = {
          cell_type: cellType,
          source: source.split('\n').map((l, i, arr) => (i < arr.length - 1 ? l + '\n' : l)),
          metadata: {},
          ...(cellType === 'code' ? { outputs: [], execution_count: null } : {}),
        };
        nb.cells.push(cell);
        await fs.writeFile(filePath, JSON.stringify(nb, null, 2), 'utf8');
        return { cellIndex: nb.cells.length - 1 };
      },
    }),
    defineTool<ExecuteCellArgs, { stdout: string; stderr: string; exitCode: number }>({
      name: 'notebook_execute_cell',
      description: 'Execute a code cell from a notebook by running it with Python.',
      parameters: ExecuteCellArgs,
      execute: async ({ filePath, cellIndex }) => {
        const raw = await fs.readFile(filePath, 'utf8');
        const nb: NotebookJson = JSON.parse(raw);
        const cell = nb.cells[cellIndex];
        if (!cell) throw new Error(`Cell index ${cellIndex} out of range`);
        if (cell.cell_type !== 'code') throw new Error(`Cell ${cellIndex} is not a code cell`);

        const code = cell.source.join('');
        return new Promise((resolve) => {
          const child = spawn('python', ['-c', code], { stdio: ['ignore', 'pipe', 'pipe'] });
          let stdout = '';
          let stderr = '';
          child.stdout.on('data', (b) => (stdout += b.toString()));
          child.stderr.on('data', (b) => (stderr += b.toString()));
          child.on('close', (exitCode) => resolve({ stdout, stderr, exitCode: exitCode ?? -1 }));
          child.on('error', (err) => resolve({ stdout, stderr: String(err), exitCode: -1 }));
        });
      },
    }),
  ]);
}
