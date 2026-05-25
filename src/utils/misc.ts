import { randomBytes } from 'node:crypto';

export function uid(prefix = 'id'): string {
  return `${prefix}_${randomBytes(8).toString('hex')}`;
}

export function safeJsonParse<T = unknown>(s: string, fallback: T): T {
  try {
    return JSON.parse(s) as T;
  } catch {
    return fallback;
  }
}

export function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
