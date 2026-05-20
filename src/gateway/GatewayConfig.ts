import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { homedir } from 'os';
import type { ChannelType } from './Channel.js';
import { ALL_CHANNELS } from './Channel.js';

const SENSITIVE_FIELDS: [string, string][] = [
  ['slack', 'appToken'],
  ['slack', 'botToken'],
  ['telegram', 'token'],
  ['discord', 'token'],
  ['wechat', 'token'],
  ['feishu', 'appSecret'],
  ['feishu', 'verificationToken'],
  ['feishu', 'encryptKey'],
  ['qq', 'clientSecret'],
];

const DEFAULT_CONFIG: Record<string, unknown> = {
  channel: null,
  autoStart: [],
  images: { enabled: true, maxSizeBytes: 10 * 1024 * 1024, maxDimension: 1568 },
  slack: { appToken: null, botToken: null },
  telegram: { token: null, allowedUsers: [] },
  discord: { token: null },
  wechat: { token: null, baseUrl: 'https://ilinkai.weixin.qq.com', allowFrom: [] },
  feishu: { appId: null, appSecret: null, connectionMode: 'websocket' },
  qq: { appId: null, clientSecret: null, markdown: false },
  imessage: { cliPath: 'imsg', dbPath: '~/Library/Messages/chat.db' },
};

function deepMerge(base: Record<string, any>, override: Record<string, any>): Record<string, any> {
  const merged = JSON.parse(JSON.stringify(base));
  for (const [key, value] of Object.entries(override ?? {})) {
    if (value && typeof value === 'object' && !Array.isArray(value) && merged[key] && typeof merged[key] === 'object') {
      merged[key] = deepMerge(merged[key], value);
    } else {
      merged[key] = value;
    }
  }
  return merged;
}

function maskSecret(value: unknown): string {
  if (!value) return '';
  const text = String(value);
  if (text.length <= 4) return '*'.repeat(text.length);
  return `${text.slice(0, 2)}${'*'.repeat(text.length - 4)}${text.slice(-2)}`;
}

function defaultConfigPath(): string {
  return join(homedir(), '.medrix', 'gateway', 'config.json');
}

export function channelConfigured(channel: ChannelType, cfg: Record<string, any>): boolean {
  const section = cfg[channel] ?? {};
  switch (channel) {
    case 'slack': return !!(section.appToken && section.botToken);
    case 'telegram': return !!section.token;
    case 'discord': return !!section.token;
    case 'wechat': return !!section.token;
    case 'feishu': return !!(section.appId && section.appSecret);
    case 'qq': return !!(section.appId && section.clientSecret);
    case 'imessage': return !!(section.cliPath || section.dbPath);
    default: return false;
  }
}

export class GatewayConfigStore {
  private readonly path: string;

  constructor(path?: string) {
    this.path = path ?? defaultConfigPath();
  }

  load(): Record<string, any> {
    const base = JSON.parse(JSON.stringify(DEFAULT_CONFIG));
    if (!existsSync(this.path)) return base;
    try {
      const raw = JSON.parse(readFileSync(this.path, 'utf-8'));
      return deepMerge(base, raw);
    } catch {
      return base;
    }
  }

  save(config: Record<string, any>): Record<string, any> {
    const existing = this.load();
    const merged = deepMerge(JSON.parse(JSON.stringify(DEFAULT_CONFIG)), config ?? {});
    for (const [section, field] of SENSITIVE_FIELDS) {
      const incoming = merged[section]?.[field];
      if (!incoming || looksMasked(incoming)) {
        const oldValue = existing[section]?.[field];
        if (oldValue) {
          if (!merged[section]) merged[section] = {};
          merged[section][field] = oldValue;
        }
      }
    }
    const dir = dirname(this.path);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    writeFileSync(this.path, JSON.stringify(merged, null, 2), 'utf-8');
    return merged;
  }

  loadMasked(): Record<string, any> {
    const cfg = this.load();
    const masked = JSON.parse(JSON.stringify(cfg));
    for (const [section, field] of SENSITIVE_FIELDS) {
      if (masked[section] && typeof masked[section] === 'object') {
        masked[section][field] = maskSecret(masked[section][field]);
      }
    }
    return masked;
  }
}

function looksMasked(value: unknown): boolean {
  if (typeof value !== 'string') return false;
  const stripped = value.trim();
  if (!stripped) return false;
  if ([...stripped].every((c) => c === '*' || c === '•')) return true;
  if (stripped.length >= 5 && [...stripped.slice(2, -2)].every((c) => c === '*')) return true;
  return false;
}
