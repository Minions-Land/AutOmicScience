/**
 * A document that can be ingested into the knowledge base.
 */
export interface Document {
  /** Unique document identifier. */
  id: string;
  /** The text content of the document. */
  content: string;
  /** Optional metadata (source, author, tags, etc.). */
  metadata?: Record<string, unknown>;
}
