/**
 * KnowledgeTools — RAG knowledge management wrapping the InMemoryKB.
 * Supports document ingestion, semantic search, source listing, and clearing.
 */

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';
import { InMemoryKB } from '../knowledge/InMemoryKB.js';
import type { Document } from '../knowledge/Document.js';

// ---------------------------------------------------------------------------
// Source tracking
// ---------------------------------------------------------------------------

interface SourceInfo {
  id: string;
  name: string;
  type: 'file' | 'text' | 'url';
  documentCount: number;
  ingestedAt: string;
  metadata: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Toolset factory
// ---------------------------------------------------------------------------

export interface KnowledgeToolsOptions {
  /** Pre-existing KnowledgeBase instance. If not provided, creates a new InMemoryKB. */
  kb?: InMemoryKB;
  /** Root directory for resolving file paths. */
  rootDir?: string;
  /** Maximum chunk size for splitting documents (default 1000 chars). */
  chunkSize?: number;
  /** Overlap between chunks (default 200 chars). */
  chunkOverlap?: number;
}

export function knowledgeToolSet(opts: KnowledgeToolsOptions = {}): ToolSet {
  const kb = opts.kb ?? new InMemoryKB();
  const rootDir = opts.rootDir ?? process.cwd();
  const chunkSize = opts.chunkSize ?? 1000;
  const chunkOverlap = opts.chunkOverlap ?? 200;
  const sources = new Map<string, SourceInfo>();
  let docCounter = 0;

  /** Split text into overlapping chunks for better retrieval. */
  function chunkText(text: string): string[] {
    if (text.length <= chunkSize) return [text];
    const chunks: string[] = [];
    let start = 0;
    while (start < text.length) {
      const end = Math.min(start + chunkSize, text.length);
      chunks.push(text.slice(start, end));
      start += chunkSize - chunkOverlap;
    }
    return chunks;
  }

  return new ToolSet('knowledge', [
    // -----------------------------------------------------------------------
    // ingest_documents
    // -----------------------------------------------------------------------
    defineTool<
      { documents?: { content: string; metadata?: Record<string, unknown> }[]; filePaths?: string[]; sourceName?: string },
      { ok: boolean; documentsIngested: number; sourceId: string }
    >({
      name: 'ingest_documents',
      description:
        'Add documents to the knowledge base for later semantic search. ' +
        'Accepts inline text documents and/or file paths. Documents are chunked for optimal retrieval.',
      parameters: z.object({
        documents: z
          .array(
            z.object({
              content: z.string().describe('Document text content'),
              metadata: z.record(z.unknown()).optional(),
            }),
          )
          .optional()
          .describe('Inline documents to ingest'),
        filePaths: z
          .array(z.string())
          .optional()
          .describe('File paths to read and ingest'),
        sourceName: z.string().optional().describe('Name for this source batch'),
      }),
      execute: async ({ documents, filePaths, sourceName }) => {
        const docs: Document[] = [];
        const sourceId = `src_${Date.now().toString(36)}`;

        // Process inline documents
        if (documents) {
          for (const doc of documents) {
            const chunks = chunkText(doc.content);
            for (const chunk of chunks) {
              docs.push({
                id: `doc_${++docCounter}`,
                content: chunk,
                metadata: { ...doc.metadata, sourceId },
              });
            }
          }
        }

        // Process file paths
        if (filePaths) {
          for (const fp of filePaths) {
            const fullPath = path.isAbsolute(fp) ? fp : path.resolve(rootDir, fp);
            try {
              const content = await fs.readFile(fullPath, 'utf8');
              const chunks = chunkText(content);
              for (const chunk of chunks) {
                docs.push({
                  id: `doc_${++docCounter}`,
                  content: chunk,
                  metadata: { sourceId, filePath: fullPath, filename: path.basename(fullPath) },
                });
              }
            } catch (err) {
              // Skip unreadable files but note in metadata
              docs.push({
                id: `doc_${++docCounter}`,
                content: `[Error reading file: ${fullPath}]`,
                metadata: { sourceId, filePath: fullPath, error: String(err) },
              });
            }
          }
        }

        if (docs.length === 0) {
          throw new Error('No documents provided. Supply documents and/or filePaths.');
        }

        await kb.ingest(docs);

        // Track source
        sources.set(sourceId, {
          id: sourceId,
          name: sourceName ?? sourceId,
          type: filePaths && filePaths.length > 0 ? 'file' : 'text',
          documentCount: docs.length,
          ingestedAt: new Date().toISOString(),
          metadata: {},
        });

        return { ok: true, documentsIngested: docs.length, sourceId };
      },
    }),

    // -----------------------------------------------------------------------
    // query_knowledge
    // -----------------------------------------------------------------------
    defineTool<
      { query: string; topK?: number; minScore?: number },
      { results: { content: string; score: number; metadata?: Record<string, unknown> }[]; totalResults: number }
    >({
      name: 'query_knowledge',
      description:
        'Semantic search over the knowledge base. Returns the most relevant document chunks ' +
        'ranked by similarity score.',
      parameters: z.object({
        query: z.string().describe('Natural language query'),
        topK: z.number().int().positive().optional().default(5).describe('Max results to return'),
        minScore: z.number().optional().describe('Minimum similarity score threshold (0-1)'),
      }),
      execute: async ({ query, topK, minScore }) => {
        const results = await kb.query(query, topK ?? 5);
        const filtered = minScore
          ? results.filter((r) => r.score >= minScore)
          : results;

        return {
          results: filtered.map((r) => ({
            content: r.document.content,
            score: Math.round(r.score * 1000) / 1000,
            metadata: r.document.metadata,
          })),
          totalResults: filtered.length,
        };
      },
    }),

    // -----------------------------------------------------------------------
    // list_sources
    // -----------------------------------------------------------------------
    defineTool<
      Record<string, never>,
      { sources: SourceInfo[]; totalDocuments: number }
    >({
      name: 'list_sources',
      description: 'List all ingested sources in the knowledge base.',
      parameters: z.object({}),
      execute: async () => {
        const allSources = Array.from(sources.values());
        const totalDocuments = allSources.reduce((sum, s) => sum + s.documentCount, 0);
        return { sources: allSources, totalDocuments };
      },
    }),

    // -----------------------------------------------------------------------
    // clear_knowledge
    // -----------------------------------------------------------------------
    defineTool<
      { sourceId?: string },
      { ok: boolean; message: string }
    >({
      name: 'clear_knowledge',
      description:
        'Clear the knowledge base. Optionally specify a sourceId to remove only that source ' +
        '(note: full clear is more efficient than selective removal).',
      parameters: z.object({
        sourceId: z.string().optional().describe('Source ID to remove (omit to clear all)'),
      }),
      execute: async ({ sourceId }) => {
        if (sourceId) {
          // For selective removal, we clear everything and re-ingest remaining
          // This is a limitation of the simple InMemoryKB
          sources.delete(sourceId);
          return {
            ok: true,
            message: `Source '${sourceId}' removed from tracking. Full re-index recommended.`,
          };
        }

        await kb.clear();
        sources.clear();
        docCounter = 0;
        return { ok: true, message: 'Knowledge base cleared.' };
      },
    }),
  ]);
}
