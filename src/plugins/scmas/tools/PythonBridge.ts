import { spawn } from 'node:child_process';
import path from 'node:path';

export interface BridgeOptions {
  /** Python executable; defaults to env SCMAS_PYTHON_BIN or `python`. */
  pythonBin?: string;
  /** Working dir; defaults to env SCMAS_ROOT or the vendored Python source. */
  cwd?: string;
  /** Extra env vars merged on top of process.env. */
  env?: NodeJS.ProcessEnv;
  /** Hard timeout in ms; default 10 minutes. */
  timeoutMs?: number;
}

export interface BridgeResult {
  exitCode: number;
  stdout: string;
  stderr: string;
  /** Last JSON object parsed from stdout, if any. */
  parsedJson?: unknown;
}

const DEFAULT_TIMEOUT_MS = 10 * 60 * 1000;
const VENDORED_DIR = path.resolve(
  new URL('../../../../_import_scMAS/CanChen_MAS', import.meta.url).pathname,
);

export function resolveScmasRoot(opt?: BridgeOptions): string {
  return opt?.cwd ?? process.env.SCMAS_ROOT ?? VENDORED_DIR;
}

export function resolvePythonBin(opt?: BridgeOptions): string {
  return opt?.pythonBin ?? process.env.SCMAS_PYTHON_BIN ?? 'python';
}

/**
 * Build the argv for `python -m scmas <subcommand> [...flags]`.
 * Each entry in flags is either a positional value, a `--flag` switch (boolean),
 * or a `[flag, value]` tuple. `null`/`undefined` values are skipped.
 */
export function buildScmasArgv(
  subcommand: string,
  flags: ReadonlyArray<string | [string, string | number | boolean | null | undefined]>,
): string[] {
  const argv: string[] = ['-m', 'scmas', subcommand];
  for (const f of flags) {
    if (typeof f === 'string') {
      argv.push(f);
      continue;
    }
    const [flag, value] = f;
    if (value === null || value === undefined || value === '') continue;
    if (typeof value === 'boolean') {
      if (value) argv.push(flag);
      continue;
    }
    argv.push(flag, String(value));
  }
  return argv;
}

/**
 * Spawn `python -m scmas <subcommand> ...` and resolve when the process exits.
 * Always returns; non-zero exit codes do NOT throw — callers decide how strict to be.
 */
export function runScmas(
  subcommand: string,
  flags: ReadonlyArray<string | [string, string | number | boolean | null | undefined]>,
  opt: BridgeOptions = {},
): Promise<BridgeResult> {
  return new Promise((resolve) => {
    const argv = buildScmasArgv(subcommand, flags);
    const child = spawn(resolvePythonBin(opt), argv, {
      cwd: resolveScmasRoot(opt),
      env: { ...process.env, ...opt.env },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => child.kill('SIGKILL'), opt.timeoutMs ?? DEFAULT_TIMEOUT_MS);
    child.stdout.on('data', (b) => (stdout += b.toString()));
    child.stderr.on('data', (b) => (stderr += b.toString()));
    child.on('close', (code) => {
      clearTimeout(timer);
      const parsedJson = extractTrailingJson(stdout);
      resolve({ exitCode: code ?? -1, stdout, stderr, parsedJson });
    });
    child.on('error', (err) => {
      clearTimeout(timer);
      resolve({ exitCode: -1, stdout, stderr: stderr + String(err) });
    });
  });
}

function extractTrailingJson(text: string): unknown | undefined {
  const trimmed = text.trim();
  if (!trimmed) return undefined;
  const lastOpen = trimmed.lastIndexOf('{');
  const lastClose = trimmed.lastIndexOf('}');
  if (lastOpen < 0 || lastClose < lastOpen) return undefined;
  const candidate = trimmed.slice(lastOpen, lastClose + 1);
  try {
    return JSON.parse(candidate);
  } catch {
    return undefined;
  }
}
