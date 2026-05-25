/* eslint-disable no-console */
import { promises as fsp } from 'node:fs';
import * as fsSync from 'node:fs';
import path from 'node:path';
import os from 'node:os';

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LEVEL_ORDER: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 };

const ANSI = {
  reset: '\x1b[0m',
  gray: '\x1b[90m',
  blue: '\x1b[34m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
};

const LEVEL_COLOR: Record<LogLevel, string> = {
  debug: ANSI.gray,
  info: ANSI.blue,
  warn: ANSI.yellow,
  error: ANSI.red,
};

const DEFAULT_LOG_FILE = path.join(os.homedir(), '.aos', 'logs', 'aos.log');
const MAX_BYTES = 10 * 1024 * 1024; // 10MB
const MAX_ROTATIONS = 5;

let currentLevel: LogLevel = ((): LogLevel => {
  const env = (process.env.AOS_LOG_LEVEL ?? '').toLowerCase();
  if (env === 'debug' || env === 'info' || env === 'warn' || env === 'error') return env;
  return 'info';
})();

const fileDisabled = process.env.AOS_LOG_FILE_DISABLE === '1' || process.env.AOS_LOG_FILE_DISABLE === 'true';
const filePath = process.env.AOS_LOG_FILE || DEFAULT_LOG_FILE;

let dirEnsured = false;
function ensureDirSync(p: string): void {
  if (dirEnsured) return;
  try {
    fsSync.mkdirSync(path.dirname(p), { recursive: true });
    dirEnsured = true;
  } catch {
    // Best-effort; if mkdir fails, file writes will simply fail and be swallowed.
  }
}

/** Serialize append queue so concurrent rotations don't interleave. */
let writeChain: Promise<void> = Promise.resolve();

function rotateIfNeeded(p: string): void {
  let size = 0;
  try {
    size = fsSync.statSync(p).size;
  } catch {
    return; // No file yet.
  }
  if (size < MAX_BYTES) return;

  // Shift aos.log.(N-1) -> aos.log.N, dropping the oldest.
  const oldest = `${p}.${MAX_ROTATIONS}`;
  try { fsSync.rmSync(oldest, { force: true }); } catch { /* ignore */ }
  for (let i = MAX_ROTATIONS - 1; i >= 1; i--) {
    const src = `${p}.${i}`;
    const dst = `${p}.${i + 1}`;
    try {
      if (fsSync.existsSync(src)) fsSync.renameSync(src, dst);
    } catch {
      // Best effort.
    }
  }
  try {
    fsSync.renameSync(p, `${p}.1`);
  } catch {
    // If rename fails, fall back to truncating so the file doesn't grow unbounded.
    try { fsSync.truncateSync(p, 0); } catch { /* ignore */ }
  }
}

function writeFileLine(line: string): void {
  if (fileDisabled) return;
  ensureDirSync(filePath);
  writeChain = writeChain
    .then(async () => {
      try {
        rotateIfNeeded(filePath);
        await fsp.appendFile(filePath, line + '\n', 'utf8');
      } catch {
        // Logging must never throw.
      }
    })
    .catch(() => { /* swallow */ });
}

function shouldLog(level: LogLevel): boolean {
  return LEVEL_ORDER[level] >= LEVEL_ORDER[currentLevel];
}

function consoleStreamFor(level: LogLevel): NodeJS.WriteStream {
  return level === 'error' || level === 'warn' ? process.stderr : process.stdout;
}

function formatConsole(level: LogLevel, message: string, args: unknown[]): string {
  const stream = consoleStreamFor(level);
  const useColor = !!stream.isTTY && process.env.NO_COLOR === undefined;
  const tag = `[${level}]`;
  const head = useColor ? `${LEVEL_COLOR[level]}${tag}${ANSI.reset}` : tag;
  if (args.length === 0) return `${head} ${message}`;
  const tail = args
    .map((a) => {
      if (a instanceof Error) return a.stack ?? a.message;
      if (typeof a === 'string') return a;
      try { return JSON.stringify(a); } catch { return String(a); }
    })
    .join(' ');
  return `${head} ${message} ${tail}`;
}

function formatJsonLine(level: LogLevel, message: string, args: unknown[]): string {
  const data = args.length > 0
    ? args.map((a) => {
        if (a instanceof Error) return { name: a.name, message: a.message, stack: a.stack };
        return a;
      })
    : undefined;
  const record: Record<string, unknown> = {
    ts: new Date().toISOString(),
    level,
    msg: message,
  };
  if (data && data.length > 0) record.data = data.length === 1 ? data[0] : data;
  try {
    return JSON.stringify(record);
  } catch {
    // Fall back to a safe representation.
    return JSON.stringify({ ts: record.ts, level, msg: message, data: '[unserializable]' });
  }
}

function emit(level: LogLevel, message: string, args: unknown[]): void {
  if (!shouldLog(level)) return;
  const line = formatConsole(level, message, args);
  consoleStreamFor(level).write(line + '\n');
  writeFileLine(formatJsonLine(level, message, args));
}

export function setLogLevel(level: LogLevel): void {
  currentLevel = level;
}

export function getLogLevel(): LogLevel {
  return currentLevel;
}

export const logger = {
  debug: (message: string, ...args: unknown[]) => emit('debug', message, args),
  info: (message: string, ...args: unknown[]) => emit('info', message, args),
  warn: (message: string, ...args: unknown[]) => emit('warn', message, args),
  error: (message: string, ...args: unknown[]) => emit('error', message, args),
};
