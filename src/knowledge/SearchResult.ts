import type { Document } from './Document.js';

/**
 * A search result returned from the knowledge base.
 */
export interface SearchResult {
  /** The matched document. */
  document: Document;
  /** Relevance score (higher is more relevant). */
  score: number;
}
