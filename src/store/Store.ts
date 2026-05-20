import type { StoreEntry } from './StoreEntry.js';

/**
 * Interface for the Novaeve package store — search, install, publish,
 * and list agents, skills, tools, and teams.
 */
export interface Store {
  /** Search entries by query string. */
  search(query: string): Promise<StoreEntry[]>;
  /** Install an entry by id into the local environment. */
  install(id: string): Promise<void>;
  /** Publish a new entry to the store. */
  publish(entry: StoreEntry): Promise<void>;
  /** List all entries, optionally filtered by category. */
  list(category?: StoreEntry['category']): Promise<StoreEntry[]>;
}
