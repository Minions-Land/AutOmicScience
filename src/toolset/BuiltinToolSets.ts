import { promises as fs } from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';

export function fileToolSet(rootDir: string = process.cwd()): ToolSet {
  const resolve = (p: string) => path.resolve(rootDir, p);

  return new ToolSet('file', [
    defineTool<{ path: string }, string>({
      name: 'read_file',
      description: 'Read a UTF-8 text file from the workspace.',
      parameters: z.object({ path: z.string() }),
      execute: async ({ path: p }) => fs.readFile(resolve(p), 'utf8'),
    }),
    defineTool<{ path: string; content: string }, { ok: boolean; path: string }>({
      name: 'write_file',
      description: 'Write a UTF-8 text file to the workspace (overwrites).',
      parameters: z.object({ path: z.string(), content: z.string() }),
      execute: async ({ path: p, content }) => {
        const full = resolve(p);
        await fs.mkdir(path.dirname(full), { recursive: true });
        await fs.writeFile(full, content, 'utf8');
        return { ok: true, path: full };
      },
    }),
    defineTool<{ path: string }, string[]>({
      name: 'list_dir',
      description: 'List entries of a directory.',
      parameters: z.object({ path: z.string().default('.') }),
      execute: async ({ path: p }) => fs.readdir(resolve(p)),
    }),
  ]);
}

export function shellToolSet(): ToolSet {
  return new ToolSet('shell', [
    defineTool<
      { command: string; cwd?: string; timeoutMs: number },
      { stdout: string; stderr: string; exitCode: number }
    >({
      name: 'shell_exec',
      description: 'Run a shell command and return stdout/stderr/exitCode.',
      parameters: z.object({
        command: z.string(),
        cwd: z.string().optional(),
        timeoutMs: z.number().int().positive().max(120_000).default(30_000),
      }),
      execute: async ({ command, cwd, timeoutMs }) =>
        new Promise((resolve) => {
          const child = spawn(command, { shell: true, cwd });
          let stdout = '';
          let stderr = '';
          const timer = setTimeout(() => child.kill('SIGKILL'), timeoutMs);
          child.stdout.on('data', (b) => (stdout += b.toString()));
          child.stderr.on('data', (b) => (stderr += b.toString()));
          child.on('close', (code) => {
            clearTimeout(timer);
            resolve({ stdout, stderr, exitCode: code ?? -1 });
          });
        }),
    }),
  ]);
}

export function webToolSet(): ToolSet {
  return new ToolSet('web', [
    defineTool<{ url: string }, { status: number; body: string }>({
      name: 'http_get',
      description: 'HTTP GET a URL and return text content.',
      parameters: z.object({ url: z.string().url() }),
      execute: async ({ url }) => {
        const res = await fetch(url);
        return { status: res.status, body: await res.text() };
      },
    }),
  ]);
}

export function codeToolSet(): ToolSet {
  return new ToolSet('code', [
    defineTool<{ code: string }, { result: unknown }>({
      name: 'eval_js',
      description: 'Evaluate a JavaScript expression in a sandboxed function (NOT secure).',
      parameters: z.object({ code: z.string() }),
      execute: async ({ code }) => {
        // Minimal stub. Replace with a real sandbox (vm2, isolated-vm) for production.
        // eslint-disable-next-line no-new-func
        const fn = new Function(`return (async () => { return (${code}); })();`);
        const result = await (fn() as Promise<unknown>);
        return { result };
      },
    }),
  ]);
}
