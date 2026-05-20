/**
 * RTools — Execute R code via PythonBridge (rpy2) or direct R subprocess.
 */

import { spawn } from 'node:child_process';
import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';
import { runPython, resolvePythonBin, type BridgeOptions } from '../bridge/PythonBridge.js';

export interface RToolsOptions {
  /** Use rpy2 via Python (true) or direct Rscript subprocess (false). Default: false. */
  useRpy2?: boolean;
  /** Path to Rscript binary. Default: 'Rscript'. */
  rscriptBin?: string;
  /** PythonBridge options (for rpy2 mode). */
  bridgeOptions?: BridgeOptions;
  /** Working directory. */
  cwd?: string;
  /** Timeout in ms (default 120s). */
  timeoutMs?: number;
}

export function rToolSet(opts: RToolsOptions = {}): ToolSet {
  const useRpy2 = opts.useRpy2 ?? false;
  const rscriptBin = opts.rscriptBin ?? 'Rscript';
  const bridgeOpts = opts.bridgeOptions ?? {};
  const cwd = opts.cwd ?? process.cwd();
  const defaultTimeout = opts.timeoutMs ?? 120_000;

  return new ToolSet('r', [
    // -----------------------------------------------------------------------
    // run_r
    // -----------------------------------------------------------------------
    defineTool<
      { code: string; timeoutMs?: number },
      { stdout: string; stderr: string; exitCode: number }
    >({
      name: 'run_r',
      description:
        'Execute R code and return stdout/stderr. Uses Rscript subprocess by default, ' +
        'or rpy2 via Python if configured.',
      parameters: z.object({
        code: z.string().describe('R code to execute'),
        timeoutMs: z.number().int().positive().optional().describe('Timeout in ms'),
      }),
      execute: async ({ code, timeoutMs }) => {
        const timeout = timeoutMs ?? defaultTimeout;

        if (useRpy2) {
          // Execute via Python + rpy2
          const pythonCode = `
import rpy2.robjects as ro
from rpy2.robjects import r
import sys

try:
    result = r('''${code.replace(/'/g, "\\'")}''')
    if result is not None:
        print(str(result))
except Exception as e:
    print(str(e), file=sys.stderr)
    sys.exit(1)
`;
          const result = await runPython(
            'eval',
            [['--code', pythonCode]],
            { ...bridgeOpts, timeoutMs: timeout },
          );
          return { stdout: result.stdout, stderr: result.stderr, exitCode: result.exitCode };
        }

        // Direct Rscript execution
        return new Promise((resolve) => {
          const child = spawn(rscriptBin, ['-e', code], {
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
    // install_r_package
    // -----------------------------------------------------------------------
    defineTool<
      { packages: string[]; repos?: string },
      { stdout: string; stderr: string; exitCode: number; installed: string[] }
    >({
      name: 'install_r_package',
      description: 'Install R packages from CRAN or Bioconductor.',
      parameters: z.object({
        packages: z.array(z.string()).min(1).describe('Package names to install'),
        repos: z.string().optional().describe('Repository URL (default: CRAN)'),
      }),
      execute: async ({ packages, repos }) => {
        const repoUrl = repos ?? 'https://cloud.r-project.org';
        const pkgList = packages.map((p) => `"${p}"`).join(', ');
        const code = `install.packages(c(${pkgList}), repos="${repoUrl}", quiet=TRUE)`;

        return new Promise((resolve) => {
          const child = spawn(rscriptBin, ['-e', code], {
            cwd,
            stdio: ['ignore', 'pipe', 'pipe'],
            env: process.env,
          });

          let stdout = '';
          let stderr = '';

          const timer = setTimeout(() => child.kill('SIGKILL'), 300_000); // 5 min for installs

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
