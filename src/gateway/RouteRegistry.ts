import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { homedir } from 'os';
import type { ConversationRoute } from './Route.js';
import { routeKey } from './Route.js';

export interface RouteEntry {
  channel: string;
  scopeType: string;
  scopeId: string;
  threadId?: string;
  senderId?: string;
  routeKey: string;
  chatId: string;
  chatName: string;
  updatedAt: string;
}

function defaultRegistryPath(): string {
  return join(homedir(), '.aos', 'gateway', 'routes.json');
}

export class RouteRegistry {
  private readonly path: string;
  private entries: Map<string, RouteEntry> = new Map();

  constructor(path?: string) {
    this.path = path ?? defaultRegistryPath();
    this.load();
  }

  private load(): void {
    if (!existsSync(this.path)) return;
    try {
      const data = JSON.parse(readFileSync(this.path, 'utf-8'));
      if (data && typeof data === 'object') {
        for (const [key, value] of Object.entries(data)) {
          this.entries.set(key, value as RouteEntry);
        }
      }
    } catch {
      // ignore corrupt file
    }
  }

  private save(): void {
    const dir = dirname(this.path);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    const obj: Record<string, RouteEntry> = {};
    for (const [key, entry] of this.entries) {
      obj[key] = entry;
    }
    writeFileSync(this.path, JSON.stringify(obj, null, 2), 'utf-8');
  }

  get(route: ConversationRoute): RouteEntry | undefined {
    return this.entries.get(routeKey(route));
  }

  set(route: ConversationRoute, chatId: string, chatName: string): RouteEntry {
    const entry: RouteEntry = {
      channel: route.channel,
      scopeType: route.scopeType,
      scopeId: route.scopeId,
      threadId: route.threadId,
      senderId: route.senderId,
      routeKey: routeKey(route),
      chatId,
      chatName,
      updatedAt: new Date().toISOString(),
    };
    this.entries.set(routeKey(route), entry);
    this.save();
    return entry;
  }

  touch(route: ConversationRoute): void {
    const entry = this.entries.get(routeKey(route));
    if (!entry) return;
    entry.updatedAt = new Date().toISOString();
    this.save();
  }

  remove(route: ConversationRoute): RouteEntry | undefined {
    const key = routeKey(route);
    const entry = this.entries.get(key);
    if (!entry) return undefined;
    this.entries.delete(key);
    this.save();
    return entry;
  }

  listEntries(): RouteEntry[] {
    return [...this.entries.values()].sort(
      (a, b) => (b.updatedAt ?? '').localeCompare(a.updatedAt ?? ''),
    );
  }
}
