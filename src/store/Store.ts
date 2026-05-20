import type { StoreEntry } from './StoreEntry.js';

export interface Store {
  search(query: string, opts?: { type?: string; limit?: number }): Promise<StoreEntry[]>;
  install(id: string, version?: string): Promise<string[]>;
  uninstall(id: string): Promise<string[]>;
  publish(entry: StoreEntry & { content: string }): Promise<void>;
  list(category?: StoreEntry['category']): Promise<StoreEntry[]>;
  getPackage(id: string): Promise<StoreEntry | undefined>;
}
