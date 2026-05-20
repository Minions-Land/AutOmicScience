/**
 * ShellTools — Execute shell commands with timeout, working directory,
 * environment variables, and security controls.
 */

import { spawn, execFile } from 'node:child_process';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';

/** Commands that are never allowed regardless of configuration. */
const BLOCKED_PATTERNS = [
  /rm\s+(-rf?|--recursive)\s+\/\s*$/,
  /rm\s+-rf?\s+\/\s*$/,
  /mkfs\./,
  /dd\s+.*of=\/dev\/sd/,
  /:(){ :\|:& };:/,
  />\s*\/dev\/sd/,
  /chmod\s+-R\s+777\s+\//,
];

function isBlocked(command: string): boolean {
  return BLOCKED_PATTERNS.some((re) => re.test(command));
}

export interface ShellToolsOptions {
  /** Default working directory for commands. */
  cwd?: string;
  /** Default environment variables merged with process.env. */
  env?: Record<string, string>;
  /** Maximum allowed timeout in ms (cap). Default: 120000. */
  maxTimeoutMs?: number;
  /** Optional allowlist of command prefixes. If set, only these are permitted. */
  allowlist?: string[];
  /** Optional blocklist of command prefixes (checked after allowlist). */
  blocklist?: string[];
}

export function shellToolSet(opts: ShellToolsOptions = {}): ToolSet {
  const defaultCwd = opts.cwd ?? process.cwd();
  const defaultEnv = opts.env ?? {};
  const maxTimeout = opts.maxTimeoutMs ?? 120_000;

  function checkPermission(command: string): string | null {
    if (isBlocked(command)) {
      return 'Command is blocked for safety reasons.';
    }
    if (opts.blocklist) {
      const cmd = command.trimStart();
      for (const prefix of opts.blocklist) {
        if (cmd.startsWith(prefix)) return `Command prefix '${prefix}' is blocked.`;
      }
    }
    if (opts.allowlist && opts.allowlist.length > 0) {
      const cmd = command.trimStart();
      const allowed = opts.allowlist.some((prefix) => cmd.startsWith(prefix));
      if (!allowed) return 'Command not in allowlist.';
    }
    return null;
  }

  return new ToolSet('shell', [
    defineTool<
      { command: string; cwd?: string; timeoutMs?: number; env?: Record<string, string> },
      { stdout: string; stderr: string; exitCode: number }
    >({
      name: 'execute_command',
      description:
        'Execute a shell command. Returns stdout, stderr, and exit code. ' +
        'Supports timeout, working directory, and environment variables.',
      parameters: z.object({
        command: z.string().describe('Shell command to execute'),
        cwd: z.string().optional().describe('Working directory (defaults to project root)'),
        timeoutMs: z
          .number()
          .int()
          .positive()
          .max(120_000)
          .optional()
          .describe('Timeout in milliseconds (default 30000)'),
        env: z
          .record(z.string())
          .optional()
          .describe('Additional environment variables'),
      }),
      execute: async ({ command, cwd, timeoutMs, env }) => {
        const denial = checkPermission(command);
        if (denial) {
          return { stdout: '', stderr: denial, exitCode: 126 };
        }

        const effectiveCwd = cwd ? path.resolve(defaultCwd, cwd) : defaultCwd;
        const effectiveTimeout = Math.min(timeoutMs ?? 30_000, maxTimeout);
        const effectiveEnv = { ...process.env, ...defaultEnv, ...env };

        return new Promise((resolve) => {
          const child = spawn(command, {
            shell: true,
            cwd: effectiveCwd,
            env: effectiveEnv,
            stdio: ['ignore', 'pipe', 'pipe'],
          });

          let stdout = '';
          let stderr = '';
          let killed = false;

          const timer = setTimeout(() => {
            killed = true;
            child.kill('SIGKILL');
          }, effectiveTimeout);

          child.stdout.on('data', (b) => (stdout += b.toString()));
          child.stderr.on('data', (b) => (stderr += b.toString()));

          child.on('close', (code) => {
            clearTimeout(timer);
            if (killed) {
              stderr += `\n[Process killed: timeout after ${effectiveTimeout}ms]`;
            }
            resolve({ stdout, stderr, exitCode: code ?? -1 });
          });

          child.on('error', (err) => {
            clearTimeout(timer);
            resolve({ stdout, stderr: stderr + '\n' + String(err), exitCode: -1 });
          });
        });
      },
    }),

    defineTool<
      { scriptPath: string; args?: string[]; cwd?: string; timeoutMs?: number },
      { stdout: string; stderr: string; exitCode: number }
    >({
      name: 'execute_script',
      description: 'Execute a script file (bash, python, etc.) with optional arguments.',
      parameters: z.object({
        scriptPath: z.string().describe('Path to the script file'),
        args: z.array(z.string()).optional().describe('Arguments to pass to the script'),
        cwd: z.string().optional().describe('Working directory'),
        timeoutMs: z
          .number()
          .int()
          .positive()
          .max(120_000)
          .optional()
          .describe('Timeout in milliseconds'),
      }),
      execute: async ({ scriptPath, args, cwd, timeoutMs }) => {
        const resolvedScript = path.resolve(defaultCwd, scriptPath);
        const effectiveCwd = cwd ? path.resolve(defaultCwd, cwd) : defaultCwd;
        const effectiveTimeout = Math.min(timeoutMs ?? 60_000, maxTimeout);

        // Check script exists
        try {
          await fs.access(resolvedScript);
        } catch {
          return { stdout: '', stderr: `Script not found: ${resolvedScript}`, exitCode: 127 };
        }

        return new Promise((resolve) => {
          const child = execFile(resolvedScript, args ?? [], {
            cwd: effectiveCwd,
            env: { ...process.env, ...defaultEnv },
            timeout: effectiveTimeout,
            maxBuffer: 10 * 1024 * 1024,
          }, (error, stdout, stderr) => {
            if (error && 'killed' in error && error.killed) {
              resolve({
                stdout: stdout ?? '',
                stderr: (stderr ?? '') + `\n[Process killed: timeout]`,
                exitCode: -1,
              });
            } else {
              resolve({
                stdout: stdout ?? '',
                stderr: stderr ?? '',
                exitCode: error ? (error as NodeJS.ErrnoException & { code?: number }).code as unknown as number ?? 1 : 0,
              });
            }
          });
        });
      },
    }),
  ]);
}
