import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

/**
 * Generic Python subprocess bridge.
 *
 * Novaeve-Agent's tools (Bio/Synthetic/Benchmark/AnnotationStage) call into
 * the bundled Python runtime that owns the heavy scientific compute
 * (anndata, scanpy, sklearn, torch, R/scDesign3). This file is the single
 * process boundary; everything above it is typed Tool/Agent surface.
 *
 * Swap the runtime for a Rust binary, MCP server, or pure-TS port
 * later by reimplementing `runPython()` against a different transport.
 */

export interface BridgeOptions {
  /** Python executable. Default: env NOVAEVE_PYTHON_BIN or `python`. */
  pythonBin?: string;
  /** Runtime root. Default: env NOVAEVE_PYTHON_RUNTIME or the bundled runtime/. */
  cwd?: string;
  /** Module name to invoke with `python -m <module>`. Default: `novaeve_bio`. */
  moduleName?: string;
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

export type CliFlag =
  | string
  | [string, string | number | boolean | null | undefined];

const DEFAULT_TIMEOUT_MS = 10 * 60 * 1000;
const HERE = path.dirname(fileURLToPath(import.meta.url));
// src/bridge/ → src/bridge/runtime (the absorbed Python runtime)
const VENDORED_DIR = path.resolve(HERE, 'runtime');

// Internal Python module name for the biological compute runtime.
const DEFAULT_PYTHON_MODULE = 'novaeve_bio';

export function resolveVendorRoot(opt?: BridgeOptions): string {
  return opt?.cwd ?? process.env.NOVAEVE_PYTHON_RUNTIME ?? VENDORED_DIR;
}

export function resolvePythonBin(opt?: BridgeOptions): string {
  return opt?.pythonBin ?? process.env.NOVAEVE_PYTHON_BIN ?? 'python';
}

/**
 * Build the argv for `python -m <module> <subcommand> [...flags]`.
 * Entries are either a positional, a `[flag, value]` tuple, or a `[flag, true]`
 * boolean switch. null/undefined/empty values are skipped.
 */
export function buildPythonArgv(
  module: string,
  subcommand: string,
  flags: ReadonlyArray<CliFlag>,
): string[] {
  const argv: string[] = ['-m', module, subcommand];
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
 * Spawn `python -m <module> <subcommand> ...` and resolve when the process exits.
 * Never throws on non-zero exit codes — callers decide how strict to be.
 */
export function runPython(
  subcommand: string,
  flags: ReadonlyArray<CliFlag>,
  opt: BridgeOptions = {},
): Promise<BridgeResult> {
  const moduleName = opt.moduleName ?? DEFAULT_PYTHON_MODULE;
  return new Promise((resolve) => {
    const argv = buildPythonArgv(moduleName, subcommand, flags);
    const child = spawn(resolvePythonBin(opt), argv, {
      cwd: resolveVendorRoot(opt),
      env: { ...process.env, ...opt.env },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(
      () => child.kill('SIGKILL'),
      opt.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    );
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
  try {
    return JSON.parse(trimmed.slice(lastOpen, lastClose + 1));
  } catch {
    return undefined;
  }
}
