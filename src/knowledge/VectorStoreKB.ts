import type { Document } from './Document.js';
import type { SearchResult } from './SearchResult.js';
import type { KnowledgeBase } from './KnowledgeBase.js';

/**
 * Embedding-based knowledge base that uses an LLM provider for embeddings.
 * Falls back to InMemoryKB (TF-IDF) if no embedding provider is configured.
 */
export interface EmbeddingProvider {
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
}

export class OpenAIEmbeddingProvider implements EmbeddingProvider {
  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseUrl: string;

  constructor(opts?: { apiKey?: string; model?: string; baseUrl?: string }) {
    this.apiKey = opts?.apiKey ?? process.env.OPENAI_API_KEY ?? '';
    this.model = opts?.model ?? 'text-embedding-3-small';
    this.baseUrl = (opts?.baseUrl ?? process.env.OPENAI_API_BASE ?? 'https://api.openai.com/v1').replace(/\/$/, '');
    if (!this.apiKey) throw new Error('OpenAI embedding requires OPENAI_API_KEY');
  }

  async embed(text: string): Promise<number[]> {
    const [vec] = await this.embedBatch([text]);
    return vec;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const resp = await fetch(`${this.baseUrl}/embeddings`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.model, input: texts }),
    });
    if (!resp.ok) throw new Error(`Embedding failed: ${await resp.text()}`);
    const data = await resp.json() as any;
    return data.data.map((d: any) => d.embedding);
  }
}

export class GeminiEmbeddingProvider implements EmbeddingProvider {
  private readonly apiKey: string;
  private readonly model: string;

  constructor(opts?: { apiKey?: string; model?: string }) {
    this.apiKey = opts?.apiKey ?? process.env.GOOGLE_API_KEY ?? process.env.GEMINI_API_KEY ?? '';
    this.model = opts?.model ?? 'text-embedding-004';
    if (!this.apiKey) throw new Error('Gemini embedding requires GOOGLE_API_KEY or GEMINI_API_KEY');
  }

  async embed(text: string): Promise<number[]> {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${this.model}:embedContent?key=${this.apiKey}`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: { parts: [{ text }] } }),
    });
    if (!resp.ok) throw new Error(`Gemini embedding failed: ${await resp.text()}`);
    const data = await resp.json() as any;
    return data.embedding?.values ?? [];
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map((t) => this.embed(t)));
  }
}

export class VectorStoreKB implements KnowledgeBase {
  private docs: Document[] = [];
  private vectors: number[][] = [];
  private readonly embedder: EmbeddingProvider;

  constructor(embedder?: EmbeddingProvider) {
    this.embedder = embedder ?? this.createDefaultEmbedder();
  }

  private createDefaultEmbedder(): EmbeddingProvider {
    if (process.env.OPENAI_API_KEY) return new OpenAIEmbeddingProvider();
    if (process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY) return new GeminiEmbeddingProvider();
    throw new Error('No embedding API key configured. Set OPENAI_API_KEY or GOOGLE_API_KEY.');
  }

  async ingest(docs: Document[]): Promise<void> {
    if (docs.length === 0) return;
    const texts = docs.map((d) => d.content);
    const embeddings = await this.embedder.embedBatch(texts);
    this.docs.push(...docs);
    this.vectors.push(...embeddings);
  }

  async query(q: string, topK = 10): Promise<SearchResult[]> {
    if (this.docs.length === 0) return [];
    const queryVec = await this.embedder.embed(q);
    const scored: SearchResult[] = [];
    for (let i = 0; i < this.docs.length; i++) {
      const sim = cosineSimilarity(queryVec, this.vectors[i]);
      if (sim > 0) scored.push({ document: this.docs[i], score: sim });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  async clear(): Promise<void> {
    this.docs = [];
    this.vectors = [];
  }

  size(): number {
    return this.docs.length;
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, magA = 0, magB = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}
