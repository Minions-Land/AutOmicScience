import type { Document } from './Document.js';
import type { SearchResult } from './SearchResult.js';
import type { KnowledgeBase } from './KnowledgeBase.js';

/**
 * A term-frequency vector for TF-IDF-like similarity computation.
 */
interface TermVector {
  terms: Map<string, number>;
  magnitude: number;
}

/**
 * In-memory knowledge base using TF-IDF cosine similarity for vector search.
 * No external dependencies required — implements a working RAG retrieval layer.
 */
export class InMemoryKB implements KnowledgeBase {
  private docs: Document[] = [];
  private vectors: TermVector[] = [];
  /** Document frequency: how many documents contain each term. */
  private df = new Map<string, number>();

  /** Ingest documents into the in-memory store and compute TF-IDF vectors. */
  async ingest(docs: Document[]): Promise<void> {
    for (const doc of docs) {
      this.docs.push(doc);
      const tf = this.computeTF(doc.content);
      // Update document frequencies
      for (const term of tf.keys()) {
        this.df.set(term, (this.df.get(term) ?? 0) + 1);
      }
      // Placeholder vector — will be recomputed on query
      this.vectors.push({ terms: tf, magnitude: 0 });
    }
    // Recompute all vectors with updated IDF
    this.recomputeVectors();
  }

  /**
   * Query using TF-IDF cosine similarity.
   * Computes a query vector and returns top-K documents by cosine similarity.
   * @param q - Query string.
   * @param topK - Maximum number of results (default 10).
   */
  async query(q: string, topK = 10): Promise<SearchResult[]> {
    if (this.docs.length === 0) return [];

    const queryTF = this.computeTF(q);
    const queryVector = this.toTFIDF(queryTF);
    const queryMag = this.magnitude(queryVector);

    if (queryMag === 0) return [];

    const scored: SearchResult[] = [];
    for (let i = 0; i < this.docs.length; i++) {
      const docVec = this.vectors[i];
      const sim = this.cosineSimilarity(queryVector, queryMag, docVec.terms, docVec.magnitude);
      if (sim > 0) {
        scored.push({ document: this.docs[i], score: sim });
      }
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  /** Clear all documents and reset the index. */
  async clear(): Promise<void> {
    this.docs = [];
    this.vectors = [];
    this.df.clear();
  }

  /** Tokenize text into lowercase terms. */
  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter((t) => t.length > 1);
  }

  /** Compute raw term frequencies for a text. */
  private computeTF(text: string): Map<string, number> {
    const tokens = this.tokenize(text);
    const tf = new Map<string, number>();
    for (const token of tokens) {
      tf.set(token, (tf.get(token) ?? 0) + 1);
    }
    // Normalize by max frequency
    const maxFreq = Math.max(...tf.values(), 1);
    for (const [term, count] of tf) {
      tf.set(term, count / maxFreq);
    }
    return tf;
  }

  /** Convert a TF map to TF-IDF weighted map. */
  private toTFIDF(tf: Map<string, number>): Map<string, number> {
    const N = this.docs.length || 1;
    const tfidf = new Map<string, number>();
    for (const [term, freq] of tf) {
      const docFreq = this.df.get(term) ?? 0;
      const idf = docFreq > 0 ? Math.log(N / docFreq) + 1 : 1;
      tfidf.set(term, freq * idf);
    }
    return tfidf;
  }

  /** Compute the magnitude (L2 norm) of a vector. */
  private magnitude(vec: Map<string, number>): number {
    let sum = 0;
    for (const v of vec.values()) {
      sum += v * v;
    }
    return Math.sqrt(sum);
  }

  /** Compute cosine similarity between two sparse vectors. */
  private cosineSimilarity(
    vecA: Map<string, number>,
    magA: number,
    vecB: Map<string, number>,
    magB: number,
  ): number {
    if (magA === 0 || magB === 0) return 0;
    let dot = 0;
    // Iterate over the smaller vector for efficiency
    const [smaller, larger] = vecA.size <= vecB.size ? [vecA, vecB] : [vecB, vecA];
    for (const [term, val] of smaller) {
      const otherVal = larger.get(term);
      if (otherVal !== undefined) {
        dot += val * otherVal;
      }
    }
    return dot / (magA * magB);
  }

  /** Recompute all document TF-IDF vectors after ingestion. */
  private recomputeVectors(): void {
    for (let i = 0; i < this.vectors.length; i++) {
      const tfidf = this.toTFIDF(this.vectors[i].terms);
      this.vectors[i] = { terms: tfidf, magnitude: this.magnitude(tfidf) };
    }
  }
}
