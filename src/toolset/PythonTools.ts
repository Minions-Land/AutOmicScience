/**
 * PythonTools — Execute Python code via PythonBridge subprocess,
 * install packages, and capture output.
 */

import { spawn } from 'node:child_process';
import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';
import { runPython, resolvePythonBin, type BridgeOptions } from '../bridge/PythonBridge.js';

export interface PythonToolsOptions {
  /** Python binary path. Default: env MEDRIX_PYTHON_BIN or 'python'. */
  pythonBin?: string;
  /** Working directory for Python execution. */
  cwd?: string;
  /** Timeout in ms for code execution (default 60s). */
  timeoutMs?: number;
}

export function pythonToolSet(opts: PythonToolsOptions = {}): ToolSet {
  const pythonBin = opts.pythonBin ?? resolvePythonBin();
  const cwd = opts.cwd ?? process.cwd();
  const defaultTimeout = opts.timeoutMs ?? 60_000;

  return new ToolSet('python', [
    defineTool<
      { code: string; timeoutMs?: number },
      { stdout: string; stderr: string; exitCode: number }
    >({
      name: 'run_python',
      description:
        'Execute Python code in a subprocess and return stdout/stderr/exitCode. ' +
        'The code is passed via stdin to the Python interpreter.',
      parameters: z.object({
        code: z.string().describe('Python code to execute'),
        timeoutMs: z
          .number()
          .int()
          .positive()
          .optional()
          .describe('Timeout in milliseconds (default 60000)'),
      }),
      execute: async ({ code, timeoutMs }) => {
        const timeout = timeoutMs ?? defaultTimeout;

        return new Promise((resolve) => {
          const child = spawn(pythonBin, ['-c', code], {
            cwd,
            stdio: ['ignore', 'pipe', 'pipe'],
            env: process.env,
          });

          let stdout = '';
          let stderr = '';
          let killed = false;

          const timer = setTimeout(() => {
            killed = true;
            child.kill('SIGKILL');
          }, timeout);

          child.stdout.on('data', (b) => (stdout += b.toString()));
          child.stderr.on('data', (b) => (stderr += b.toString()));

          child.on('close', (exitCode) => {
            clearTimeout(timer);
            if (killed) {
              stderr += `\n[Process killed: timeout after ${timeout}ms]`;
            }
            resolve({ stdout, stderr, exitCode: exitCode ?? -1 });
          });

          child.on('error', (err) => {
            clearTimeout(timer);
            resolve({ stdout, stderr: String(err), exitCode: -1 });
          });
        });
      },
    }),

    defineTool<
      { packages: string[]; upgrade?: boolean },
      { stdout: string; stderr: string; exitCode: number; installed: string[] }
    >({
      name: 'install_package',
      description: 'Install Python packages using pip.',
      parameters: z.object({
        packages: z.array(z.string()).min(1).describe('Package names to install'),
        upgrade: z.boolean().optional().default(false).describe('Upgrade if already installed'),
      }),
      execute: async ({ packages, upgrade }) => {
        const args = ['-m', 'pip', 'install', ...(upgrade ? ['--upgrade'] : []), ...packages];

        return new Promise((resolve) => {
          const child = spawn(pythonBin, args, {
            cwd,
            stdio: ['ignore', 'pipe', 'pipe'],
            env: process.env,
          });

          let stdout = '';
          let stderr = '';

          const timer = setTimeout(() => child.kill('SIGKILL'), 120_000);

          child.stdout.on('data', (b) => (stdout += b.toString()));
          child.stderr.on('data', (b) => (stderr += b.toString()));

          child.on('close', (exitCode) => {
            clearTimeout(timer);
            resolve({
              stdout,
              stderr,
              exitCode: exitCode ?? -1,
              installed: exitCode === 0 ? packages : [],
            });
          });

          child.on('error', (err) => {
            clearTimeout(timer);
            resolve({ stdout, stderr: String(err), exitCode: -1, installed: [] });
          });
        });
      },
    }),

    defineTool<
      { expression: string },
      { result: string; error?: string }
    >({
      name: 'eval_python',
      description:
        'Evaluate a Python expression and return its repr. ' +
        'Useful for quick computations without full script execution.',
      parameters: z.object({
        expression: z.string().describe('Python expression to evaluate'),
      }),
      execute: async ({ expression }) => {
        const code = `import sys; sys.stdout.write(repr(${expression}))`;
        return new Promise((resolve) => {
          const child = spawn(pythonBin, ['-c', code], {
            cwd,
            stdio: ['ignore', 'pipe', 'pipe'],
            env: process.env,
          });

          let stdout = '';
          let stderr = '';

          const timer = setTimeout(() => child.kill('SIGKILL'), 10_000);

          child.stdout.on('data', (b) => (stdout += b.toString()));
          child.stderr.on('data', (b) => (stderr += b.toString()));

          child.on('close', (exitCode) => {
            clearTimeout(timer);
            if (exitCode === 0) {
              resolve({ result: stdout });
            } else {
              resolve({ result: '', error: stderr || `Exit code: ${exitCode}` });
            }
          });

          child.on('error', (err) => {
            clearTimeout(timer);
            resolve({ result: '', error: String(err) });
          });
        });
      },
    }),
  ]);
}
