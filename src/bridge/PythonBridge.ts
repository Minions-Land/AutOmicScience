import { existsSync, readFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

/**
 * Generic Python subprocess bridge.
 *
 * AutOmicScience's tools (Bio/Synthetic/Benchmark/AnnotationStage) call into
 * the bundled Python runtime that owns the heavy scientific compute
 * (anndata, scanpy, sklearn, torch, R/scDesign3). This file is the single
 * process boundary; everything above it is typed Tool/Agent surface.
 *
 * Swap the runtime for a Rust binary, MCP server, or pure-TS port
 * later by reimplementing `runPython()` against a different transport.
 */

export interface BridgeOptions {
  /** Python executable. Default: env AOS_PYTHON_BIN or `python`. */
  pythonBin?: string;
  /** Runtime root. Default: env AOS_PYTHON_RUNTIME or the bundled runtime/. */
  cwd?: string;
  /** Module name to invoke with `python -m <module>`. Default: `aos_agent`. */
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
  /** JSON report read from a path printed by the Python CLI, when available. */
  reportJson?: unknown;
  /** Report file path printed by the Python CLI, when available. */
  reportPath?: string;
}

export type CliFlag =
  | string
  | [string, string | number | boolean | null | undefined];

const DEFAULT_TIMEOUT_MS = 10 * 60 * 1000;
const HERE = path.dirname(fileURLToPath(import.meta.url));
// src/bridge/ → src/bridge/runtime (the absorbed Python runtime)
const DIST_RUNTIME_DIR = path.resolve(HERE, 'runtime');
const SOURCE_RUNTIME_DIR = path.resolve(HERE, '..', '..', 'src', 'bridge', 'runtime');
// In dev/tsx this is src/bridge/runtime. Built dist/ does not contain the
// Python tree, so fall back to the repository source runtime.
const VENDORED_DIR = existsSync(DIST_RUNTIME_DIR) ? DIST_RUNTIME_DIR : SOURCE_RUNTIME_DIR;

// Internal Python module name for the biological compute runtime.
const DEFAULT_PYTHON_MODULE = 'aos_agent';

export function resolveVendorRoot(opt?: BridgeOptions): string {
  return opt?.cwd ?? process.env.AOS_PYTHON_RUNTIME ?? VENDORED_DIR;
}

export function resolvePythonBin(opt?: BridgeOptions): string {
  return opt?.pythonBin ?? process.env.AOS_PYTHON_BIN ?? 'python';
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
      env: {
        ...process.env,
        PYTHONIOENCODING: process.env.PYTHONIOENCODING ?? 'utf-8',
        PYTHONUTF8: process.env.PYTHONUTF8 ?? '1',
        ...opt.env,
      },
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
      const report = readJsonReportFromStdout(stdout);
      resolve({ exitCode: code ?? -1, stdout, stderr, parsedJson, ...report });
    });
    child.on('error', (err) => {
      clearTimeout(timer);
      resolve({ exitCode: -1, stdout, stderr: stderr + String(err) });
    });
  });
}

function readJsonReportFromStdout(stdout: string): { reportJson?: unknown; reportPath?: string } {
  const pathText = stdout.trim().split(/\r?\n/).filter(Boolean).at(-1);
  if (!pathText || !pathText.toLowerCase().endsWith('.json')) return {};
  try {
    const parsed = JSON.parse(readFileSync(pathText, 'utf-8'));
    return { reportJson: parsed, reportPath: pathText };
  } catch {
    return { reportPath: pathText };
  }
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
