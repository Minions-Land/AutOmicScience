import type { Document } from './Document.js';
import type { SearchResult } from './SearchResult.js';

/**
 * Interface for a knowledge base that supports document ingestion
 * and semantic/keyword search (RAG pattern).
 */
export interface KnowledgeBase {
  /** Ingest documents into the knowledge base. */
  ingest(docs: Document[]): Promise<void>;
  /** Query the knowledge base and return the top-K most relevant results. */
  query(q: string, topK?: number): Promise<SearchResult[]>;
  /** Clear all documents from the knowledge base. */
  clear(): Promise<void>;
}
