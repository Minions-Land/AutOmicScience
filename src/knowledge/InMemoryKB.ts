import type { Document } from './Document.js';
import type { SearchResult } from './SearchResult.js';
import type { KnowledgeBase } from './KnowledgeBase.js';

/**
 * In-memory knowledge base using simple substring matching.
 * Placeholder for a real vector DB implementation (Qdrant, Pinecone, etc.).
 */
export class InMemoryKB implements KnowledgeBase {
  private docs: Document[] = [];

  /** Ingest documents into the in-memory store. */
  async ingest(docs: Document[]): Promise<void> {
    this.docs.push(...docs);
  }

  /**
   * Query using case-insensitive substring matching.
   * Returns results sorted by a simple relevance heuristic.
   * @param q - Query string.
   * @param topK - Maximum number of results (default 10).
   */
  async query(q: string, topK = 10): Promise<SearchResult[]> {
    const lower = q.toLowerCase();
    const scored: SearchResult[] = [];
    for (const doc of this.docs) {
      const content = doc.content.toLowerCase();
      if (content.includes(lower)) {
        // Simple score: shorter documents that match rank higher
        const score = lower.length / content.length;
        scored.push({ document: doc, score });
      }
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  /** Clear all documents. */
  async clear(): Promise<void> {
    this.docs = [];
  }
}
