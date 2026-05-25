import { readFile, writeFile, mkdir, readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { homedir } from 'node:os';
import type { Message } from '../types.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export type PatternCategory = 'tool_usage' | 'error_recovery' | 'domain_knowledge' | 'workflow' | 'preference';

export interface LearningEntry {
  id: string;
  title: string;
  description: string;
  category: PatternCategory;
  content: string;
  tags: string[];
  createdAt: number;
  updatedAt: number;
  useCount: number;
  agentScope?: string[];
}

export interface ExtractionResult {
  entries: LearningEntry[];
  source: string;
}

// ── Store ─────────────────────────────────────────────────────────────────────

export class LearningStore {
  private storePath: string;
  private cache: Map<string, LearningEntry> = new Map();
  private loaded = false;

  constructor(storePath?: string) {
    this.storePath = storePath ?? join(homedir(), '.aos', 'learning');
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
          const entry = JSON.parse(content) as LearningEntry;
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

  async save(entry: LearningEntry): Promise<void> {
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
    const filename = `${id}.json`;
    try {
      const { unlink } = await import('node:fs/promises');
      await unlink(join(this.storePath, filename));
      this.cache.delete(id);
      return true;
    } catch {
      return false;
    }
  }

  getAll(): LearningEntry[] {
    return [...this.cache.values()];
  }

  getByCategory(category: PatternCategory): LearningEntry[] {
    return this.getAll().filter((e) => e.category === category);
  }

  getById(id: string): LearningEntry | undefined {
    return this.cache.get(id);
  }

  search(query: string, limit = 10): LearningEntry[] {
    const lower = query.toLowerCase();
    const scored = this.getAll().map((entry) => {
      let score = 0;
      if (entry.title.toLowerCase().includes(lower)) score += 3;
      if (entry.description.toLowerCase().includes(lower)) score += 2;
      if (entry.content.toLowerCase().includes(lower)) score += 1;
      for (const tag of entry.tags) {
        if (tag.toLowerCase().includes(lower)) score += 2;
      }
      return { entry, score };
    });
    return scored
      .filter((s) => s.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map((s) => s.entry);
  }

  isLoaded(): boolean {
    return this.loaded;
  }
}

// ── Extractor ─────────────────────────────────────────────────────────────────

export class LearningExtractor {
  private patterns: RegExp[] = [
    /(?:learned|discovered|found out|realized|noted)\s+that\s+(.+)/i,
    /(?:the correct|the right|the proper)\s+(?:way|approach|method)\s+(?:is|to)\s+(.+)/i,
    /(?:always|never|remember to)\s+(.+)/i,
    /(?:tip|note|important):\s*(.+)/i,
  ];

  /**
   * Extract learning patterns from a conversation.
   * Looks for tool usage patterns, error recovery sequences, and explicit lessons.
   */
  extract(messages: Message[]): LearningEntry[] {
    const entries: LearningEntry[] = [];

    // Extract tool usage patterns
    entries.push(...this.extractToolPatterns(messages));

    // Extract error recovery patterns
    entries.push(...this.extractErrorRecovery(messages));

    // Extract explicit lessons from assistant messages
    entries.push(...this.extractExplicitLessons(messages));

    return entries;
  }

  private extractToolPatterns(messages: Message[]): LearningEntry[] {
    const entries: LearningEntry[] = [];
    const toolSequences: Array<{ name: string; success: boolean }> = [];

    for (const msg of messages) {
      if (msg.role === 'assistant' && msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          toolSequences.push({ name: tc.name, success: true });
        }
      }
      if (msg.role === 'tool' && typeof msg.content === 'string') {
        const last = toolSequences[toolSequences.length - 1];
        if (last && msg.content.toLowerCase().includes('error')) {
          last.success = false;
        }
      }
    }

    // Find repeated successful tool sequences (patterns)
    if (toolSequences.length >= 3) {
      const successfulTools = toolSequences
        .filter((t) => t.success)
        .map((t) => t.name);
      if (successfulTools.length >= 2) {
        const pattern = successfulTools.slice(0, 5).join(' -> ');
        entries.push({
          id: `tool-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          title: `Tool sequence: ${successfulTools[0]}`,
          description: `Successful tool usage pattern: ${pattern}`,
          category: 'tool_usage',
          content: `When performing this type of task, the following tool sequence works well:\n${pattern}`,
          tags: successfulTools.slice(0, 3),
          createdAt: Date.now(),
          updatedAt: Date.now(),
          useCount: 0,
        });
      }
    }

    return entries;
  }

  private extractErrorRecovery(messages: Message[]): LearningEntry[] {
    const entries: LearningEntry[] = [];
    let errorContext = '';
    let recoveryFound = false;

    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      const content = typeof msg.content === 'string' ? msg.content : '';

      if (content.toLowerCase().includes('error') || content.toLowerCase().includes('failed')) {
        errorContext = content.slice(0, 200);
        recoveryFound = false;
      } else if (errorContext && msg.role === 'assistant' && !content.toLowerCase().includes('error')) {
        recoveryFound = true;
        entries.push({
          id: `recovery-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          title: `Error recovery: ${errorContext.slice(0, 50)}`,
          description: `Recovery pattern after error`,
          category: 'error_recovery',
          content: `When encountering: "${errorContext.slice(0, 100)}"\nRecovery approach: ${content.slice(0, 300)}`,
          tags: ['error', 'recovery'],
          createdAt: Date.now(),
          updatedAt: Date.now(),
          useCount: 0,
        });
        errorContext = '';
      }
    }

    return entries;
  }

  private extractExplicitLessons(messages: Message[]): LearningEntry[] {
    const entries: LearningEntry[] = [];

    for (const msg of messages) {
      if (msg.role !== 'assistant') continue;
      const content = typeof msg.content === 'string' ? msg.content : '';

      for (const pattern of this.patterns) {
        const match = content.match(pattern);
        if (match && match[1] && match[1].length > 10) {
          entries.push({
            id: `lesson-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            title: match[1].slice(0, 80),
            description: match[1].slice(0, 200),
            category: 'domain_knowledge',
            content: match[1],
            tags: ['lesson'],
            createdAt: Date.now(),
            updatedAt: Date.now(),
            useCount: 0,
          });
          break; // One lesson per message
        }
      }
    }

    return entries;
  }
}

// ── Injector ──────────────────────────────────────────────────────────────────

export class LearningInjector {
  private store: LearningStore;
  private maxInjectedEntries: number;
  private maxInjectedTokens: number;

  constructor(store: LearningStore, opts?: { maxEntries?: number; maxTokens?: number }) {
    this.store = store;
    this.maxInjectedEntries = opts?.maxEntries ?? 10;
    this.maxInjectedTokens = opts?.maxTokens ?? 2000;
  }

  /**
   * Build a guidance block to inject into an agent's system prompt.
   * Selects relevant patterns based on the query context.
   */
  buildGuidance(query?: string, agentName?: string): string {
    let entries = this.store.getAll();

    // Filter by agent scope
    if (agentName) {
      entries = entries.filter(
        (e) => !e.agentScope || e.agentScope.includes(agentName),
      );
    }

    // If query provided, search for relevant entries
    if (query) {
      entries = this.store.search(query, this.maxInjectedEntries);
    } else {
      // Sort by useCount and recency
      entries = entries
        .sort((a, b) => {
          const scoreA = a.useCount * 2 + (a.updatedAt / 1e10);
          const scoreB = b.useCount * 2 + (b.updatedAt / 1e10);
          return scoreB - scoreA;
        })
        .slice(0, this.maxInjectedEntries);
    }

    if (entries.length === 0) return '';

    // Build guidance text within token budget
    const lines: string[] = ['## Learned Patterns\n'];
    let charCount = lines[0].length;
    const maxChars = this.maxInjectedTokens * 4;

    for (const entry of entries) {
      const line = `- [${entry.category}] ${entry.title}: ${entry.description}`;
      if (charCount + line.length > maxChars) break;
      lines.push(line);
      charCount += line.length;
    }

    return lines.join('\n');
  }

  /** Mark an entry as used (increments useCount). */
  async markUsed(id: string): Promise<void> {
    const entry = this.store.getById(id);
    if (entry) {
      entry.useCount++;
      entry.updatedAt = Date.now();
      await this.store.save(entry);
    }
  }
}

// ── Runtime ───────────────────────────────────────────────────────────────────

export interface LearningConfig {
  enabled?: boolean;
  storePath?: string;
  extractEnabled?: boolean;
  extractInterval?: number;
  maxEntries?: number;
  maxInjectedTokens?: number;
}

/**
 * LearningRuntime manages the full learning loop:
 * - Watches agent interactions
 * - Extracts patterns via LearningExtractor
 * - Stores them via LearningStore
 * - Injects relevant patterns via LearningInjector
 */
export class LearningRuntime {
  private config: LearningConfig;
  private store: LearningStore;
  private extractor: LearningExtractor;
  private injector: LearningInjector;
  private runCounter: Map<string, number> = new Map();
  private initialized = false;

  constructor(config: LearningConfig = {}) {
    this.config = { enabled: true, extractEnabled: true, extractInterval: 5, ...config };
    this.store = new LearningStore(config.storePath);
    this.extractor = new LearningExtractor();
    this.injector = new LearningInjector(this.store, {
      maxEntries: config.maxEntries,
      maxTokens: config.maxInjectedTokens,
    });
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;
    await this.store.load();
    this.initialized = true;
  }

  /** Increment run counter and maybe extract patterns. */
  async onRunEnd(sessionId: string, messages: Message[]): Promise<LearningEntry[]> {
    if (!this.config.extractEnabled) return [];

    const count = (this.runCounter.get(sessionId) ?? 0) + 1;
    this.runCounter.set(sessionId, count);

    if (count < (this.config.extractInterval ?? 5)) return [];

    // Reset counter and extract
    this.runCounter.set(sessionId, 0);
    return this.extractAndStore(messages);
  }

  /** Force extraction from messages. */
  async extractAndStore(messages: Message[]): Promise<LearningEntry[]> {
    await this.initialize();
    const entries = this.extractor.extract(messages);

    for (const entry of entries) {
      await this.store.save(entry);
    }

    return entries;
  }

  /** Build guidance text for injection into agent system prompt. */
  async buildGuidance(query?: string, agentName?: string): Promise<string> {
    await this.initialize();
    return this.injector.buildGuidance(query, agentName);
  }

  /** Get the underlying store for direct access. */
  getStore(): LearningStore {
    return this.store;
  }

  /** Check if the runtime is initialized. */
  isInitialized(): boolean {
    return this.initialized;
  }
}
