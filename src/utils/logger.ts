/* eslint-disable no-console */
export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LEVEL_ORDER: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 };

let currentLevel: LogLevel = (process.env.PANTHEON_LOG_LEVEL as LogLevel) || 'info';

export function setLogLevel(level: LogLevel): void {
  currentLevel = level;
}

function shouldLog(level: LogLevel): boolean {
  return LEVEL_ORDER[level] >= LEVEL_ORDER[currentLevel];
}

export const logger = {
  debug: (...args: unknown[]) => shouldLog('debug') && console.debug('[debug]', ...args),
  info: (...args: unknown[]) => shouldLog('info') && console.log('[info]', ...args),
  warn: (...args: unknown[]) => shouldLog('warn') && console.warn('[warn]', ...args),
  error: (...args: unknown[]) => shouldLog('error') && console.error('[error]', ...args),
};
