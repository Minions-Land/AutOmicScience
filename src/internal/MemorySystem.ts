import { readFile, writeFile, mkdir, readdir, stat, unlink } from 'node:fs/promises';
import { join } from 'node:path';
import { homedir } from 'node:os';
import type { Message } from '../types.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export type MemoryType = 'user' | 'feedback' | 'project' | 'reference' | 'workflow' | 'session_note';

export interface MemoryEntry {
  id: string;
  title: string;
  summary: string;
  type: MemoryType;
  content: string;
  createdAt: number;
  updatedAt: number;
  accessCount: number;
  tags: string[];
}

export interface RetrievalResult {
  entry: MemoryEntry;
  relevanceScore: number;
  freshnessScore: number;
  combinedScore: number;
}

// ── Freshness Scoring ─────────────────────────────────────────────────────────

export function memoryAgeDays(updatedAt: number): number {
  return Math.max(0, Math.floor((Date.now() - updatedAt) / 86_400_000));
}

export function memoryAgeText(updatedAt: number): string {
  const days = memoryAgeDays(updatedAt);
  if (days === 0) return 'today';
  if (days === 1) return 'yesterday';
  return `${days} days ago`;
}

export function freshnessScore(updatedAt: number): number {
  const days = memoryAgeDays(updatedAt);
  // Exponential decay: half-life of 7 days
  return Math.exp(-0.099 * days);
}

export function stalenessWarning(updatedAt: number): string | null {
  const days = memoryAgeDays(updatedAt);
  if (days <= 1) return null;
  return (
    `This memory is ${days} days old. ` +
    'Memories are point-in-time observations — ' +
    'claims may be outdated. Verify against current state.'
  );
}

// ── Store ─────────────────────────────────────────────────────────────────────

export class MemoryStore {
  private storePath: string;
  private cache: Map<string, MemoryEntry> = new Map();
  private loaded = false;

  constructor(storePath?: string) {
    this.storePath = storePath ?? join(homedir(), '.medrix', 'memory');
  }

  private async ensureDir(): Promise<void> {
    await mkdir(this.storePath, { recursive: true });
  }

  async load(): Promise<void> {
    await this.ensureDir();
    try {
      const files = await readdir(this.storePath);
      for (const file of files) {
        if (!file.endsWith('.json')) continue;
        try {
          const content = await readFile(join(this.storePath, file), 'utf-8');
          const entry = JSON.parse(content) as MemoryEntry;
          this.cache.set(entry.id, entry);
        } catch {
          // Skip corrupted files
        }
      }
    } catch {
      // Directory might not exist yet
    }
    this.loaded = true;
  }

  async save(entry: MemoryEntry): Promise<void> {
    await this.ensureDir();
    const filename = `${entry.id}.json`;
    await writeFile(
      join(this.storePath, filename),
      JSON.stringify(entry, null, 2),
      'utf-8',
    );
    this.cache.set(entry.id, entry);
  }

  async delete(id: string): Promise<boolean> {
    try {
      await unlink(join(this.storePath, `${id}.json`));
      this.cache.delete(id);
      return true;
    } catch {
      return false;
    }
  }

  getAll(): MemoryEntry[] {
    return [...this.cache.values()];
  }

  getById(id: string): MemoryEntry | undefined {
    return this.cache.get(id);
  }

  getByType(type: MemoryType): MemoryEntry[] {
    return this.getAll().filter((e) => e.type === type);
  }

  isLoaded(): boolean {
    return this.loaded;
  }
}

// ── Extractor ─────────────────────────────────────────────────────────────────

export class MemoryExtractor {
  private extractionPatterns: Array<{ pattern: RegExp; type: MemoryType }> = [
    { pattern: /(?:my name is|I am|I'm)\s+(\w+)/i, type: 'user' },
    { pattern: /(?:the project|this project|our project)\s+(?:is|uses|requires)\s+(.+)/i, type: 'project' },
    { pattern: /(?:remember|note|important):\s*(.+)/i, type: 'reference' },
    { pattern: /(?:I prefer|I like|I want|please always)\s+(.+)/i, type: 'user' },
    { pattern: /(?:the workflow|the process|the steps)\s+(?:is|are|involves)\s+(.+)/i, type: 'workflow' },
  ];

  /**
   * Extract memorable facts/decisions from a conversation.
   */
  extract(messages: Message[]): MemoryEntry[] {
    const entries: MemoryEntry[] = [];
    const seen = new Set<string>();

    for (const msg of messages) {
      if (msg.role !== 'user' && msg.role !== 'assistant') continue;
      const content = typeof msg.content === 'string' ? msg.content : '';
      if (content.length < 10) continue;

      for (const { pattern, type } of this.extractionPatterns) {
        const match = content.match(pattern);
        if (match && match[1] && match[1].length > 5) {
          const key = match[1].slice(0, 50).toLowerCase();
          if (seen.has(key)) continue;
          seen.add(key);

          entries.push({
            id: `mem-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            title: match[1].slice(0, 80),
            summary: match[1].slice(0, 200),
            type,
            content: `Source: [${msg.role}] ${content.slice(0, 500)}`,
            createdAt: Date.now(),
            updatedAt: Date.now(),
            accessCount: 0,
            tags: [type],
          });
        }
      }

      // Extract decisions (assistant messages with decision language)
      if (msg.role === 'assistant') {
        const decisionMatch = content.match(
          /(?:I'll|I will|Let's|We should|The best approach is)\s+(.{20,200})/i,
        );
        if (decisionMatch && decisionMatch[1]) {
          const key = decisionMatch[1].slice(0, 50).toLowerCase();
          if (!seen.has(key)) {
            seen.add(key);
            entries.push({
              id: `mem-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
              title: `Decision: ${decisionMatch[1].slice(0, 60)}`,
              summary: decisionMatch[1].slice(0, 200),
              type: 'workflow',
              content: decisionMatch[1],
              createdAt: Date.now(),
              updatedAt: Date.now(),
              accessCount: 0,
              tags: ['decision'],
            });
          }
        }
      }
    }

    return entries;
  }
}

// ── Retrieval ─────────────────────────────────────────────────────────────────

export class MemoryRetriever {
  private store: MemoryStore;
  private maxResults: number;

  constructor(store: MemoryStore, maxResults = 5) {
    this.store = store;
    this.maxResults = maxResults;
  }

  /**
   * Retrieve memories relevant to a query.
   * Scores by keyword relevance and freshness.
   */
  retrieve(query: string, alreadyShown?: Set<string>): RetrievalResult[] {
    const entries = this.store.getAll();
    const queryLower = query.toLowerCase();
    const queryWords = queryLower.split(/\s+/).filter((w) => w.length > 2);

    const results: RetrievalResult[] = [];

    for (const entry of entries) {
      if (alreadyShown?.has(entry.id)) continue;

      // Compute relevance score based on keyword overlap
      const entryText = `${entry.title} ${entry.summary} ${entry.content} ${entry.tags.join(' ')}`.toLowerCase();
      let relevance = 0;
      for (const word of queryWords) {
        if (entryText.includes(word)) relevance += 1;
      }
      // Boost for title matches
      if (entry.title.toLowerCase().includes(queryLower)) relevance += 3;

      if (relevance === 0) continue;

      const fresh = freshnessScore(entry.updatedAt);
      const combined = relevance * 0.7 + fresh * 0.3 + (entry.accessCount * 0.05);

      results.push({
        entry,
        relevanceScore: relevance,
        freshnessScore: fresh,
        combinedScore: combined,
      });
    }

    return results
      .sort((a, b) => b.combinedScore - a.combinedScore)
      .slice(0, this.maxResults);
  }
}

// ── Consolidation (Dream) ─────────────────────────────────────────────────────

export class MemoryConsolidator {
  private store: MemoryStore;
  private sessionCount = 0;
  private consolidationThreshold: number;

  constructor(store: MemoryStore, threshold = 10) {
    this.store = store;
    this.consolidationThreshold = threshold;
  }

  incrementSession(): void {
    this.sessionCount++;
  }

  shouldConsolidate(): boolean {
    return this.sessionCount >= this.consolidationThreshold;
  }

  /**
   * Consolidate memories: merge duplicates, remove stale entries,
   * and organize by type.
   */
  async consolidate(): Promise<{ merged: number; removed: number }> {
    const entries = this.store.getAll();
    let merged = 0;
    let removed = 0;

    // Find near-duplicates by title similarity
    const titleMap = new Map<string, MemoryEntry[]>();
    for (const entry of entries) {
      const key = entry.title.toLowerCase().replace(/[^a-z0-9]/g, '').slice(0, 30);
      const group = titleMap.get(key) ?? [];
      group.push(entry);
      titleMap.set(key, group);
    }

    for (const [, group] of titleMap) {
      if (group.length <= 1) continue;

      // Keep the most recently updated, merge content
      group.sort((a, b) => b.updatedAt - a.updatedAt);
      const keeper = group[0];
      for (let i = 1; i < group.length; i++) {
        const dup = group[i];
        // Merge content if different
        if (!keeper.content.includes(dup.content.slice(0, 50))) {
          keeper.content += `\n---\n${dup.content}`;
        }
        keeper.accessCount += dup.accessCount;
        keeper.updatedAt = Math.max(keeper.updatedAt, dup.updatedAt);
        await this.store.delete(dup.id);
        merged++;
      }
      await this.store.save(keeper);
    }

    // Remove very stale entries (> 90 days, never accessed)
    for (const entry of this.store.getAll()) {
      if (memoryAgeDays(entry.updatedAt) > 90 && entry.accessCount === 0) {
        await this.store.delete(entry.id);
        removed++;
      }
    }

    this.sessionCount = 0;
    return { merged, removed };
  }
}

// ── Runtime ───────────────────────────────────────────────────────────────────

export interface MemorySystemConfig {
  enabled?: boolean;
  storePath?: string;
  maxRetrievalResults?: number;
  consolidationThreshold?: number;
  extractEnabled?: boolean;
}

/**
 * MemoryRuntime manages the full memory lifecycle:
 * - Extracts key facts/decisions from conversations
 * - Persists to ~/.medrix/memory/
 * - Retrieves relevant memories for new conversations
 * - Scores by recency and relevance
 * - Consolidates memories periodically
 */
export class MemoryRuntime {
  private config: MemorySystemConfig;
  private store: MemoryStore;
  private extractor: MemoryExtractor;
  private retriever: MemoryRetriever;
  private consolidator: MemoryConsolidator;
  private shownMemories: Map<string, Set<string>> = new Map();
  private initialized = false;

  constructor(config: MemorySystemConfig = {}) {
    this.config = { enabled: true, extractEnabled: true, ...config };
    this.store = new MemoryStore(config.storePath);
    this.extractor = new MemoryExtractor();
    this.retriever = new MemoryRetriever(this.store, config.maxRetrievalResults ?? 5);
    this.consolidator = new MemoryConsolidator(this.store, config.consolidationThreshold ?? 10);
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;
    await this.store.load();
    this.initialized = true;
  }

  /** Retrieve memories relevant to a query for a session. */
  async retrieveRelevant(query: string, sessionId: string): Promise<RetrievalResult[]> {
    await this.initialize();
    const shown = this.shownMemories.get(sessionId) ?? new Set();
    const results = this.retriever.retrieve(query, shown);

    // Track shown memories
    for (const r of results) {
      shown.add(r.entry.id);
      r.entry.accessCount++;
      await this.store.save(r.entry);
    }
    this.shownMemories.set(sessionId, shown);

    return results;
  }

  /** Extract and store memories from a conversation turn. */
  async onRunEnd(sessionId: string, messages: Message[]): Promise<MemoryEntry[]> {
    if (!this.config.extractEnabled) return [];
    await this.initialize();

    const entries = this.extractor.extract(messages);
    for (const entry of entries) {
      await this.store.save(entry);
    }

    // Increment session counter for consolidation
    this.consolidator.incrementSession();

    return entries;
  }

  /** Run consolidation if threshold is met. */
  async maybeConsolidate(): Promise<{ merged: number; removed: number } | null> {
    if (!this.consolidator.shouldConsolidate()) return null;
    return this.consolidator.consolidate();
  }

  /** Write a memory entry directly. */
  async writeMemory(entry: MemoryEntry): Promise<void> {
    await this.initialize();
    await this.store.save(entry);
  }

  /** Build a bootstrap memory block for system prompt injection. */
  async buildBootstrapMemory(sessionId: string, query?: string): Promise<string> {
    await this.initialize();
    if (!query) {
      const entries = this.store.getAll()
        .sort((a, b) => b.accessCount - a.accessCount)
        .slice(0, 5);
      if (entries.length === 0) return '';
      return this.formatMemories(entries);
    }
    const results = await this.retrieveRelevant(query, sessionId);
    if (results.length === 0) return '';
    return this.formatMemories(results.map((r) => r.entry));
  }

  private formatMemories(entries: MemoryEntry[]): string {
    const lines = ['## Relevant Memories\n'];
    for (const entry of entries) {
      const age = memoryAgeText(entry.updatedAt);
      lines.push(`- [${entry.type}] ${entry.title} (${age}): ${entry.summary}`);
    }
    return lines.join('\n');
  }

  /** Get the underlying store. */
  getStore(): MemoryStore {
    return this.store;
  }

  isInitialized(): boolean {
    return this.initialized;
  }
}
