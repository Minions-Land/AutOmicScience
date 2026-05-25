import { spawn, type ChildProcess } from 'child_process';
import { createConnection } from 'net';

export interface ProcessResult {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  timedOut: boolean;
}

export interface SpawnOpts {
  cwd?: string;
  env?: Record<string, string>;
  timeout?: number;
  stdin?: string;
}

export async function spawnWithTimeout(cmd: string, args: string[], opts: SpawnOpts = {}): Promise<ProcessResult> {
  return new Promise((resolve) => {
    const timeout = opts.timeout ?? 30000;
    let stdout = '';
    let stderr = '';
    let timedOut = false;

    const proc = spawn(cmd, args, {
      cwd: opts.cwd,
      env: opts.env ? { ...process.env, ...opts.env } : undefined,
      timeout,
    });

    if (opts.stdin) {
      proc.stdin.write(opts.stdin);
      proc.stdin.end();
    }

    proc.stdout.on('data', (d) => { stdout += d; });
    proc.stderr.on('data', (d) => { stderr += d; });

    const timer = setTimeout(() => {
      timedOut = true;
      killProcessTree(proc.pid!);
    }, timeout);

    proc.on('close', (code) => {
      clearTimeout(timer);
      resolve({ stdout, stderr, exitCode: code, timedOut });
    });

    proc.on('error', (err) => {
      clearTimeout(timer);
      resolve({ stdout, stderr: err.message, exitCode: -1, timedOut: false });
    });
  });
}

export function killProcessTree(pid: number): void {
  try {
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', String(pid), '/T', '/F']);
    } else {
      process.kill(-pid, 'SIGKILL');
    }
  } catch {
    try { process.kill(pid, 'SIGKILL'); } catch { /* ignore */ }
  }
}

export function isPortAvailable(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const conn = createConnection({ port, host: '127.0.0.1' });
    conn.on('connect', () => { conn.destroy(); resolve(false); });
    conn.on('error', () => { resolve(true); });
    setTimeout(() => { conn.destroy(); resolve(true); }, 1000);
  });
}

export async function findAvailablePort(start = 4222, max = 100): Promise<number> {
  for (let port = start; port < start + max; port++) {
    if (await isPortAvailable(port)) return port;
  }
  throw new Error(`No available port found in range ${start}-${start + max}`);
}
