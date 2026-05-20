/**
 * WebTools — HTTP requests, web scraping, search, and file downloads.
 */

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';

// ---------------------------------------------------------------------------
// Rate limiter (simple token bucket)
// ---------------------------------------------------------------------------

class RateLimiter {
  private tokens: number;
  private lastRefill: number;
  constructor(
    private maxTokens: number,
    private refillRateMs: number,
  ) {
    this.tokens = maxTokens;
    this.lastRefill = Date.now();
  }

  async acquire(): Promise<void> {
    const now = Date.now();
    const elapsed = now - this.lastRefill;
    this.tokens = Math.min(this.maxTokens, this.tokens + Math.floor(elapsed / this.refillRateMs));
    this.lastRefill = now;

    if (this.tokens <= 0) {
      const waitMs = this.refillRateMs - (elapsed % this.refillRateMs);
      await new Promise((r) => setTimeout(r, waitMs));
      this.tokens = 1;
    }
    this.tokens--;
  }
}

// ---------------------------------------------------------------------------
// HTML text extraction (lightweight, no dependencies)
// ---------------------------------------------------------------------------

function extractText(html: string): string {
  // Remove script and style blocks
  let text = html.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '');
  text = text.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '');
  // Remove HTML tags
  text = text.replace(/<[^>]+>/g, ' ');
  // Decode common entities
  text = text
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, ' ');
  // Collapse whitespace
  text = text.replace(/\s+/g, ' ').trim();
  return text;
}

function extractLinks(html: string, baseUrl: string): { text: string; href: string }[] {
  const links: { text: string; href: string }[] = [];
  const re = /<a\s+[^>]*href=["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi;
  let match: RegExpExecArray | null;
  while ((match = re.exec(html)) !== null) {
    const href = match[1];
    const text = match[2].replace(/<[^>]+>/g, '').trim();
    if (href && text) {
      try {
        const resolved = new URL(href, baseUrl).toString();
        links.push({ text: text.slice(0, 200), href: resolved });
      } catch {
        links.push({ text: text.slice(0, 200), href });
      }
    }
  }
  return links.slice(0, 100);
}

// ---------------------------------------------------------------------------
// Toolset factory
// ---------------------------------------------------------------------------

export interface WebToolsOptions {
  /** User-Agent header for requests. */
  userAgent?: string;
  /** Default timeout in ms (default 30s). */
  timeoutMs?: number;
  /** Max requests per second (rate limit). */
  maxRequestsPerSecond?: number;
  /** Download directory for download_file. */
  downloadDir?: string;
}

export function webToolSet(opts: WebToolsOptions = {}): ToolSet {
  const userAgent = opts.userAgent ?? 'MedrixAI/1.0 (https://github.com/medrixai)';
  const defaultTimeout = opts.timeoutMs ?? 30_000;
  const downloadDir = opts.downloadDir ?? process.cwd();
  const limiter = new RateLimiter(opts.maxRequestsPerSecond ?? 5, 1000);

  return new ToolSet('web', [
    // -----------------------------------------------------------------------
    // http_request
    // -----------------------------------------------------------------------
    defineTool<
      {
        url: string;
        method?: string;
        headers?: Record<string, string>;
        body?: string;
        timeoutMs?: number;
      },
      { status: number; headers: Record<string, string>; body: string; ok: boolean }
    >({
      name: 'http_request',
      description:
        'Make an HTTP request (GET, POST, PUT, DELETE, PATCH). ' +
        'Returns status, headers, and response body.',
      parameters: z.object({
        url: z.string().url().describe('Request URL'),
        method: z.enum(['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD']).optional().default('GET'),
        headers: z.record(z.string()).optional().describe('Request headers'),
        body: z.string().optional().describe('Request body (for POST/PUT/PATCH)'),
        timeoutMs: z.number().int().positive().optional().describe('Timeout in ms'),
      }),
      execute: async ({ url, method, headers, body, timeoutMs }) => {
        await limiter.acquire();

        const controller = new AbortController();
        const timeout = timeoutMs ?? defaultTimeout;
        const timer = setTimeout(() => controller.abort(), timeout);

        try {
          const res = await fetch(url, {
            method: method ?? 'GET',
            headers: {
              'User-Agent': userAgent,
              ...headers,
            },
            body: body ?? undefined,
            signal: controller.signal,
          });

          const responseHeaders: Record<string, string> = {};
          res.headers.forEach((v, k) => (responseHeaders[k] = v));

          const responseBody = await res.text();
          return {
            status: res.status,
            headers: responseHeaders,
            body: responseBody.slice(0, 500_000), // Cap at 500KB
            ok: res.ok,
          };
        } finally {
          clearTimeout(timer);
        }
      },
    }),

    // -----------------------------------------------------------------------
    // scrape_url
    // -----------------------------------------------------------------------
    defineTool<
      { url: string; extractLinks?: boolean; maxLength?: number },
      { text: string; title: string; links?: { text: string; href: string }[]; url: string }
    >({
      name: 'scrape_url',
      description:
        'Fetch a URL and extract readable text content and optionally links. ' +
        'Strips HTML tags, scripts, and styles.',
      parameters: z.object({
        url: z.string().url().describe('URL to scrape'),
        extractLinks: z.boolean().optional().default(false).describe('Also extract links'),
        maxLength: z.number().int().positive().optional().describe('Max text length to return'),
      }),
      execute: async ({ url, extractLinks: doLinks, maxLength }) => {
        await limiter.acquire();

        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), defaultTimeout);

        try {
          const res = await fetch(url, {
            headers: { 'User-Agent': userAgent },
            signal: controller.signal,
          });

          if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
          }

          const html = await res.text();

          // Extract title
          const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
          const title = titleMatch ? titleMatch[1].trim() : '';

          // Extract text
          let text = extractText(html);
          if (maxLength && text.length > maxLength) {
            text = text.slice(0, maxLength) + '... [truncated]';
          }

          const result: { text: string; title: string; links?: { text: string; href: string }[]; url: string } = {
            text,
            title,
            url,
          };

          if (doLinks) {
            result.links = extractLinks(html, url);
          }

          return result;
        } finally {
          clearTimeout(timer);
        }
      },
    }),

    // -----------------------------------------------------------------------
    // search_web (DuckDuckGo HTML scraping fallback)
    // -----------------------------------------------------------------------
    defineTool<
      { query: string; maxResults?: number },
      { results: { title: string; url: string; snippet: string }[] }
    >({
      name: 'search_web',
      description:
        'Search the web using DuckDuckGo. Returns titles, URLs, and snippets. ' +
        'No API key required.',
      parameters: z.object({
        query: z.string().describe('Search query'),
        maxResults: z.number().int().positive().max(20).optional().default(10),
      }),
      execute: async ({ query, maxResults }) => {
        await limiter.acquire();

        const searchUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), defaultTimeout);

        try {
          const res = await fetch(searchUrl, {
            headers: {
              'User-Agent': userAgent,
              Accept: 'text/html',
            },
            signal: controller.signal,
          });

          const html = await res.text();
          const results: { title: string; url: string; snippet: string }[] = [];

          // Parse DuckDuckGo HTML results
          const resultBlocks = html.split(/class="result\s/);
          for (let i = 1; i < resultBlocks.length && results.length < (maxResults ?? 10); i++) {
            const block = resultBlocks[i];

            // Extract URL
            const urlMatch = block.match(/class="result__a"[^>]*href="([^"]+)"/);
            let url = urlMatch?.[1] ?? '';
            // DDG wraps URLs in a redirect
            if (url.includes('uddg=')) {
              const decoded = decodeURIComponent(url.split('uddg=')[1]?.split('&')[0] ?? '');
              if (decoded) url = decoded;
            }

            // Extract title
            const titleMatch = block.match(/class="result__a"[^>]*>([\s\S]*?)<\/a>/);
            const title = titleMatch ? titleMatch[1].replace(/<[^>]+>/g, '').trim() : '';

            // Extract snippet
            const snippetMatch = block.match(/class="result__snippet"[^>]*>([\s\S]*?)<\/[at]/);
            const snippet = snippetMatch ? snippetMatch[1].replace(/<[^>]+>/g, '').trim() : '';

            if (url && title) {
              results.push({ title, url, snippet });
            }
          }

          return { results };
        } finally {
          clearTimeout(timer);
        }
      },
    }),

    // -----------------------------------------------------------------------
    // download_file
    // -----------------------------------------------------------------------
    defineTool<
      { url: string; outputPath?: string; timeoutMs?: number },
      { ok: boolean; path: string; size: number; contentType: string }
    >({
      name: 'download_file',
      description: 'Download a file from a URL and save it to disk.',
      parameters: z.object({
        url: z.string().url().describe('URL to download'),
        outputPath: z.string().optional().describe('Output file path (defaults to filename from URL)'),
        timeoutMs: z.number().int().positive().optional().describe('Timeout in ms'),
      }),
      execute: async ({ url, outputPath, timeoutMs }) => {
        await limiter.acquire();

        const controller = new AbortController();
        const timeout = timeoutMs ?? 120_000;
        const timer = setTimeout(() => controller.abort(), timeout);

        try {
          const res = await fetch(url, {
            headers: { 'User-Agent': userAgent },
            signal: controller.signal,
          });

          if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
          }

          // Determine output path
          let outPath: string;
          if (outputPath) {
            outPath = path.isAbsolute(outputPath) ? outputPath : path.resolve(downloadDir, outputPath);
          } else {
            const urlPath = new URL(url).pathname;
            const filename = path.basename(urlPath) || 'download';
            outPath = path.resolve(downloadDir, filename);
          }

          await fs.mkdir(path.dirname(outPath), { recursive: true });

          const buffer = Buffer.from(await res.arrayBuffer());
          await fs.writeFile(outPath, buffer);

          const contentType = res.headers.get('content-type') ?? 'application/octet-stream';

          return {
            ok: true,
            path: outPath,
            size: buffer.length,
            contentType,
          };
        } finally {
          clearTimeout(timer);
        }
      },
    }),
  ]);
}
