/**
 * NotebookTools — Jupyter notebook management with persistent Python kernels.
 *
 * Variables, imports, and module state persist across `run_cell` invocations
 * within the same notebook because each notebook is backed by a long-lived
 * `python -i -u` subprocess (a JupyterKernel). The kernel sends a unique
 * sentinel after each cell so we can demarcate stdout/stderr per cell.
 *
 * `kernel_restart` kills and respawns the Python process, clearing all state.
 *
 * Tool API surface is unchanged from the previous fresh-process implementation:
 *   create_notebook, add_cell, run_cell, get_output, list_notebooks,
 *   kernel_restart, read_notebook.
 */

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import crypto from 'node:crypto';
import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';
import { resolvePythonBin } from '../bridge/PythonBridge.js';

// ---------------------------------------------------------------------------
// Notebook JSON types
// ---------------------------------------------------------------------------

interface NotebookCell {
  cell_type: 'code' | 'markdown' | 'raw';
  source: string[];
  metadata: Record<string, unknown>;
  outputs?: CellOutput[];
  execution_count?: number | null;
}

interface CellOutput {
  output_type: 'stream' | 'execute_result' | 'display_data' | 'error';
  text?: string[];
  data?: Record<string, unknown>;
  name?: string;
  ename?: string;
  evalue?: string;
  traceback?: string[];
}

interface NotebookJson {
  nbformat: number;
  nbformat_minor: number;
  metadata: Record<string, unknown>;
  cells: NotebookCell[];
}

// ---------------------------------------------------------------------------
// Persistent kernel
// ---------------------------------------------------------------------------

/**
 * JupyterKernel — a thin wrapper over a long-lived `python -i -u` subprocess.
 *
 * Each `run(code)` call:
 *  1. wraps the user code in a try/except so a failure prints a traceback and
 *     prints a unique sentinel on stdout (and another on stderr) so we know
 *     when the cell finished.
 *  2. writes that block to the kernel's stdin.
 *  3. reads stdout and stderr until both sentinels arrive.
 *
 * Cells preserve global state because all writes go through the same
 * interactive session. To clear state, call `restart()`.
 */
class JupyterKernel {
  private proc: ChildProcessWithoutNullStreams | null = null;
  private stdoutBuf = '';
  private stderrBuf = '';
  private busy: Promise<unknown> = Promise.resolve();
  executionCount = 0;

  constructor(
    private readonly pythonBin: string,
    private readonly cwd: string,
  ) {}

  private spawnProc(): ChildProcessWithoutNullStreams {
    const proc = spawn(this.pythonBin, ['-i', '-u', '-q'], {
      cwd: this.cwd,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
    });
    proc.stdout.setEncoding('utf8');
    proc.stderr.setEncoding('utf8');
    proc.stdout.on('data', (chunk: string) => {
      this.stdoutBuf += chunk;
    });
    proc.stderr.on('data', (chunk: string) => {
      this.stderrBuf += chunk;
    });
    proc.on('exit', () => {
      this.proc = null;
    });
    return proc;
  }

  private async ensureRunning(): Promise<ChildProcessWithoutNullStreams> {
    if (this.proc && !this.proc.killed && this.proc.exitCode === null) {
      return this.proc;
    }
    this.proc = this.spawnProc();
    this.stdoutBuf = '';
    this.stderrBuf = '';
    this.proc.stdin.write(
      [
        'import sys as _sys',
        'import os as _os',
        '_sys.ps1 = ""',
        '_sys.ps2 = ""',
        '',
      ].join('\n') + '\n',
    );
    await new Promise((r) => setTimeout(r, 50));
    this.stdoutBuf = '';
    this.stderrBuf = '';
    return this.proc;
  }

  /**
   * Run a block of code and return its captured stdout/stderr.
   * Calls are serialised — concurrent run() invocations are queued.
   */
  run(code: string, timeoutMs: number): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    const next = this.busy.then(() => this.runImpl(code, timeoutMs));
    this.busy = next.catch(() => undefined);
    return next;
  }

  private async runImpl(
    code: string,
    timeoutMs: number,
  ): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    const proc = await this.ensureRunning();

    const sentinel = `__AOS_CELL_${crypto.randomBytes(8).toString('hex')}__`;
    const stdoutMark = `${sentinel}_OUT`;
    const stderrMark = `${sentinel}_ERR`;

    this.stdoutBuf = '';
    this.stderrBuf = '';

    const indentedCode = code
      .split('\n')
      .map((l) => '    ' + l)
      .join('\n');

    const wrapper = [
      'import sys as _sys, traceback as _tb',
      '_exit_code = 0',
      'try:',
      indentedCode || '    pass',
      'except SystemExit as _e:',
      '    _exit_code = int(_e.code) if isinstance(_e.code, int) else 1',
      '    _tb.print_exc()',
      'except BaseException:',
      '    _exit_code = 1',
      '    _tb.print_exc()',
      `print(${JSON.stringify(stdoutMark)}, flush=True)`,
      `print(${JSON.stringify(stderrMark)}, file=_sys.stderr, flush=True)`,
      '',
    ].join('\n');

    // Send the wrapper as a single exec() call so the multi-line block is
    // treated as one statement by the interactive interpreter.
    const payload = `exec(compile(${JSON.stringify(wrapper)}, '<cell>', 'exec'))\n`;
    proc.stdin.write(payload);

    const start = Date.now();
    const interval = 25;
    let stdoutSawMark = false;
    let stderrSawMark = false;

    while (Date.now() - start < timeoutMs) {
      if (!stdoutSawMark && this.stdoutBuf.includes(stdoutMark)) stdoutSawMark = true;
      if (!stderrSawMark && this.stderrBuf.includes(stderrMark)) stderrSawMark = true;
      if (stdoutSawMark && stderrSawMark) break;
      if (this.proc !== proc) break;
      await new Promise((r) => setTimeout(r, interval));
    }

    if (!stdoutSawMark || !stderrSawMark) {
      // Timed out — kill the kernel so subsequent calls get a fresh interpreter.
      this.kill();
      const stdout = trimAtMark(this.stdoutBuf, stdoutMark) + '\n[Execution timed out]';
      const stderr = trimAtMark(this.stderrBuf, stderrMark);
      this.stdoutBuf = '';
      this.stderrBuf = '';
      return { stdout, stderr, exitCode: -1 };
    }

    const stdout = trimAtMark(this.stdoutBuf, stdoutMark);
    const stderr = trimAtMark(this.stderrBuf, stderrMark);
    this.stdoutBuf = this.stdoutBuf.split(stdoutMark).slice(1).join(stdoutMark);
    this.stderrBuf = this.stderrBuf.split(stderrMark).slice(1).join(stderrMark);

    const exitCode = stderr.includes('Traceback (most recent call last)') ? 1 : 0;
    return { stdout, stderr, exitCode };
  }

  /** Restart: kill the current process and reset counters. */
  async restart(): Promise<void> {
    this.kill();
    this.executionCount = 0;
    await new Promise((r) => setTimeout(r, 25));
  }

  kill(): void {
    if (this.proc && this.proc.exitCode === null) {
      try {
        this.proc.kill('SIGKILL');
      } catch {
        // ignore
      }
    }
    this.proc = null;
    this.stdoutBuf = '';
    this.stderrBuf = '';
  }
}

/** Strip trailing-newline noise from the captured buffer up to the sentinel. */
function trimAtMark(buf: string, mark: string): string {
  const idx = buf.indexOf(mark);
  if (idx < 0) return buf;
  let s = buf.slice(0, idx);
  if (s.endsWith('\n')) s = s.slice(0, -1);
  return s;
}

// ---------------------------------------------------------------------------
// Kernel registry (one kernel per notebook path)
// ---------------------------------------------------------------------------

const kernels = new Map<string, JupyterKernel>();

function getKernel(notebookPath: string, pythonBin: string): JupyterKernel {
  let k = kernels.get(notebookPath);
  if (!k) {
    k = new JupyterKernel(pythonBin, path.dirname(notebookPath));
    kernels.set(notebookPath, k);
  }
  return k;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function emptyNotebook(kernel = 'python3'): NotebookJson {
  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: { display_name: 'Python 3', language: 'python', name: kernel },
      language_info: { name: 'python', version: '3.10.0' },
    },
    cells: [],
  };
}

async function readNotebook(filePath: string): Promise<NotebookJson> {
  const raw = await fs.readFile(filePath, 'utf8');
  return JSON.parse(raw) as NotebookJson;
}

async function writeNotebook(filePath: string, nb: NotebookJson): Promise<void> {
  await fs.writeFile(filePath, JSON.stringify(nb, null, 2), 'utf8');
}

function sourceToLines(source: string): string[] {
  return source.split('\n').map((l, i, arr) => (i < arr.length - 1 ? l + '\n' : l));
}

// ---------------------------------------------------------------------------
// Toolset factory
// ---------------------------------------------------------------------------

export interface NotebookToolsOptions {
  /** Python binary for kernel execution. */
  pythonBin?: string;
  /** Default directory for notebooks. */
  notebookDir?: string;
  /** Timeout for cell execution in ms (default 120s). */
  cellTimeoutMs?: number;
}

export function notebookToolSet(opts: NotebookToolsOptions = {}): ToolSet {
  const pythonBin = opts.pythonBin ?? resolvePythonBin();
  const notebookDir = opts.notebookDir ?? process.cwd();
  const cellTimeout = opts.cellTimeoutMs ?? 120_000;
  const resolve = (p: string) => (path.isAbsolute(p) ? p : path.resolve(notebookDir, p));

  return new ToolSet('notebook', [
    // -----------------------------------------------------------------------
    // create_notebook
    // -----------------------------------------------------------------------
    defineTool<
      { filePath: string; kernel?: string; title?: string },
      { created: string; kernel: string }
    >({
      name: 'create_notebook',
      description: 'Create a new empty Jupyter notebook (.ipynb) file.',
      parameters: z.object({
        filePath: z.string().describe('Path for the new notebook'),
        kernel: z.string().optional().default('python3').describe('Kernel name'),
        title: z.string().optional().describe('Optional title (added as first markdown cell)'),
      }),
      execute: async ({ filePath, kernel, title }) => {
        const full = resolve(filePath);
        await fs.mkdir(path.dirname(full), { recursive: true });
        const nb = emptyNotebook(kernel ?? 'python3');
        if (title) {
          nb.cells.push({
            cell_type: 'markdown',
            source: sourceToLines(`# ${title}`),
            metadata: {},
          });
        }
        await writeNotebook(full, nb);
        return { created: full, kernel: kernel ?? 'python3' };
      },
    }),

    // -----------------------------------------------------------------------
    // add_cell
    // -----------------------------------------------------------------------
    defineTool<
      { filePath: string; cellType?: string; source: string; position?: number },
      { cellIndex: number; totalCells: number }
    >({
      name: 'add_cell',
      description: 'Add a code or markdown cell to an existing notebook.',
      parameters: z.object({
        filePath: z.string().describe('Notebook file path'),
        cellType: z.enum(['code', 'markdown', 'raw']).optional().default('code'),
        source: z.string().describe('Cell content'),
        position: z.number().int().nonnegative().optional().describe('Insert position (default: append)'),
      }),
      execute: async ({ filePath, cellType, source, position }) => {
        const full = resolve(filePath);
        const nb = await readNotebook(full);
        const cell: NotebookCell = {
          cell_type: (cellType ?? 'code') as NotebookCell['cell_type'],
          source: sourceToLines(source),
          metadata: {},
          ...(cellType === 'code' || !cellType ? { outputs: [], execution_count: null } : {}),
        };
        if (position !== undefined && position < nb.cells.length) {
          nb.cells.splice(position, 0, cell);
        } else {
          nb.cells.push(cell);
        }
        await writeNotebook(full, nb);
        return { cellIndex: position ?? nb.cells.length - 1, totalCells: nb.cells.length };
      },
    }),

    // -----------------------------------------------------------------------
    // run_cell
    // -----------------------------------------------------------------------
    defineTool<
      { filePath: string; cellIndex: number; timeoutMs?: number },
      { stdout: string; stderr: string; exitCode: number; outputs: CellOutput[] }
    >({
      name: 'run_cell',
      description:
        'Execute a code cell from a notebook in a long-lived Python kernel. ' +
        'Variables, imports, and module state persist across cells in the same notebook. ' +
        'Use kernel_restart to clear state.',
      parameters: z.object({
        filePath: z.string().describe('Notebook file path'),
        cellIndex: z.number().int().nonnegative().describe('Cell index to execute'),
        timeoutMs: z.number().int().positive().optional().describe('Execution timeout in ms'),
      }),
      execute: async ({ filePath, cellIndex, timeoutMs }) => {
        const full = resolve(filePath);
        const nb = await readNotebook(full);
        const cell = nb.cells[cellIndex];
        if (!cell) throw new Error(`Cell index ${cellIndex} out of range (${nb.cells.length} cells)`);
        if (cell.cell_type !== 'code') throw new Error(`Cell ${cellIndex} is not a code cell`);

        const code = cell.source.join('');
        const timeout = timeoutMs ?? cellTimeout;
        const kernel = getKernel(full, pythonBin);
        const result = await kernel.run(code, timeout);

        const outputs: CellOutput[] = [];
        if (result.stdout) {
          outputs.push({
            output_type: 'stream',
            name: 'stdout',
            text: result.stdout.split('\n').map((l) => l + '\n'),
          });
        }
        if (result.stderr && result.exitCode !== 0) {
          outputs.push({
            output_type: 'error',
            ename: 'ExecutionError',
            evalue: result.stderr.split('\n').filter(Boolean).pop() ?? '',
            traceback: result.stderr.split('\n'),
          });
        } else if (result.stderr) {
          outputs.push({
            output_type: 'stream',
            name: 'stderr',
            text: result.stderr.split('\n').map((l) => l + '\n'),
          });
        }

        kernel.executionCount++;
        cell.outputs = outputs;
        cell.execution_count = kernel.executionCount;
        await writeNotebook(full, nb);

        return { ...result, outputs };
      },
    }),

    // -----------------------------------------------------------------------
    // get_output
    // -----------------------------------------------------------------------
    defineTool<
      { filePath: string; cellIndex: number },
      { cellIndex: number; executionCount: number | null; outputs: CellOutput[] }
    >({
      name: 'get_output',
      description: 'Get the outputs of a previously executed cell.',
      parameters: z.object({
        filePath: z.string().describe('Notebook file path'),
        cellIndex: z.number().int().nonnegative().describe('Cell index'),
      }),
      execute: async ({ filePath, cellIndex }) => {
        const full = resolve(filePath);
        const nb = await readNotebook(full);
        const cell = nb.cells[cellIndex];
        if (!cell) throw new Error(`Cell index ${cellIndex} out of range`);
        return {
          cellIndex,
          executionCount: cell.execution_count ?? null,
          outputs: cell.outputs ?? [],
        };
      },
    }),

    // -----------------------------------------------------------------------
    // list_notebooks
    // -----------------------------------------------------------------------
    defineTool<
      { directory?: string },
      { notebooks: { path: string; cells: number; kernel: string }[] }
    >({
      name: 'list_notebooks',
      description: 'List Jupyter notebooks in a directory.',
      parameters: z.object({
        directory: z.string().optional().describe('Directory to search (defaults to notebook dir)'),
      }),
      execute: async ({ directory }) => {
        const dir = resolve(directory ?? '.');
        const notebooks: { path: string; cells: number; kernel: string }[] = [];

        async function walk(d: string): Promise<void> {
          const entries = await fs.readdir(d, { withFileTypes: true }).catch(() => []);
          for (const entry of entries) {
            const full = path.join(d, entry.name);
            if (entry.isDirectory()) {
              if (['node_modules', '.git', '.ipynb_checkpoints'].includes(entry.name)) continue;
              await walk(full);
            } else if (entry.name.endsWith('.ipynb')) {
              try {
                const nb = await readNotebook(full);
                const kernel = (nb.metadata?.kernelspec as Record<string, string>)?.name ?? 'unknown';
                notebooks.push({ path: full, cells: nb.cells.length, kernel });
              } catch {
                notebooks.push({ path: full, cells: -1, kernel: 'error' });
              }
            }
          }
        }

        await walk(dir);
        return { notebooks };
      },
    }),

    // -----------------------------------------------------------------------
    // kernel_restart
    // -----------------------------------------------------------------------
    defineTool<
      { filePath: string },
      { ok: boolean; message: string }
    >({
      name: 'kernel_restart',
      description: 'Kill and respawn the Python kernel for a notebook, clearing all variables and imports.',
      parameters: z.object({
        filePath: z.string().describe('Notebook file path'),
      }),
      execute: async ({ filePath }) => {
        const full = resolve(filePath);
        const kernel = kernels.get(full);
        if (kernel) {
          await kernel.restart();
        }
        const nb = await readNotebook(full);
        for (const cell of nb.cells) {
          if (cell.cell_type === 'code') {
            cell.outputs = [];
            cell.execution_count = null;
          }
        }
        await writeNotebook(full, nb);
        return { ok: true, message: `Kernel restarted for ${path.basename(full)}` };
      },
    }),

    // -----------------------------------------------------------------------
    // read_notebook (summary view)
    // -----------------------------------------------------------------------
    defineTool<
      { filePath: string },
      { cells: { index: number; type: string; source: string; hasOutput: boolean; executionCount: number | null }[] }
    >({
      name: 'read_notebook',
      description: 'Read a notebook and return a summary of all cells.',
      parameters: z.object({
        filePath: z.string().describe('Notebook file path'),
      }),
      execute: async ({ filePath }) => {
        const full = resolve(filePath);
        const nb = await readNotebook(full);
        return {
          cells: nb.cells.map((c, i) => ({
            index: i,
            type: c.cell_type,
            source: c.source.join(''),
            hasOutput: (c.outputs?.length ?? 0) > 0,
            executionCount: c.execution_count ?? null,
          })),
        };
      },
    }),
  ]);
}

/** Cleanly kill all live kernels — call from process shutdown hooks. */
export function shutdownAllNotebookKernels(): void {
  for (const k of kernels.values()) k.kill();
  kernels.clear();
}
