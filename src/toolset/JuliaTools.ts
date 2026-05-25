/**
 * JuliaTools — Execute Julia code via subprocess, install packages.
 */

import { spawn } from 'node:child_process';
import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';

export interface JuliaToolsOptions {
  /** Path to Julia binary. Default: 'julia'. */
  juliaBin?: string;
  /** Working directory. */
  cwd?: string;
  /** Timeout in ms (default 120s). */
  timeoutMs?: number;
  /** Number of threads for Julia (default: auto). */
  threads?: number;
}

export function juliaToolSet(opts: JuliaToolsOptions = {}): ToolSet {
  const juliaBin = opts.juliaBin ?? 'julia';
  const cwd = opts.cwd ?? process.cwd();
  const defaultTimeout = opts.timeoutMs ?? 120_000;
  const threads = opts.threads;

  function buildEnv(): NodeJS.ProcessEnv {
    const env = { ...process.env };
    if (threads) {
      env.JULIA_NUM_THREADS = String(threads);
    }
    return env;
  }

  return new ToolSet('julia', [
    // -----------------------------------------------------------------------
    // run_julia
    // -----------------------------------------------------------------------
    defineTool<
      { code: string; timeoutMs?: number },
      { stdout: string; stderr: string; exitCode: number }
    >({
      name: 'run_julia',
      description:
        'Execute Julia code in a subprocess and return stdout/stderr/exitCode.',
      parameters: z.object({
        code: z.string().describe('Julia code to execute'),
        timeoutMs: z.number().int().positive().optional().describe('Timeout in ms'),
      }),
      execute: async ({ code, timeoutMs }) => {
        const timeout = timeoutMs ?? defaultTimeout;

        return new Promise((resolve) => {
          const child = spawn(juliaBin, ['-e', code], {
            cwd,
            stdio: ['ignore', 'pipe', 'pipe'],
            env: buildEnv(),
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
            if (killed) stderr += '\n[Process killed: timeout]';
            resolve({ stdout, stderr, exitCode: exitCode ?? -1 });
          });

          child.on('error', (err) => {
            clearTimeout(timer);
            resolve({ stdout, stderr: String(err), exitCode: -1 });
          });
        });
      },
    }),

    // -----------------------------------------------------------------------
    // install_julia_package
    // -----------------------------------------------------------------------
    defineTool<
      { packages: string[] },
      { stdout: string; stderr: string; exitCode: number; installed: string[] }
    >({
      name: 'install_julia_package',
      description: 'Install Julia packages using Pkg.',
      parameters: z.object({
        packages: z.array(z.string()).min(1).describe('Package names to install'),
      }),
      execute: async ({ packages }) => {
        const pkgList = packages.map((p) => `"${p}"`).join(', ');
        const code = `using Pkg; Pkg.add([${pkgList}])`;

        return new Promise((resolve) => {
          const child = spawn(juliaBin, ['-e', code], {
            cwd,
            stdio: ['ignore', 'pipe', 'pipe'],
            env: buildEnv(),
          });

          let stdout = '';
          let stderr = '';

          // Julia package installs can be slow (precompilation)
          const timer = setTimeout(() => child.kill('SIGKILL'), 600_000); // 10 min

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
  ]);
}
