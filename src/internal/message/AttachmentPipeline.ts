import { promises as fs } from 'node:fs';
import path from 'node:path';
import type { Message, TextContent, ImageContent } from '../../types.js';

/**
 * Attachment detection pipeline.
 *
 * Inspect a free-form user text and surface structured "attachments" — image
 * URLs, local file paths, http URLs, fenced code blocks — so callers can
 * upgrade plain-text messages into multi-modal messages or expand inline
 * references before sending to the LLM.
 *
 * Use:
 *   const p = new AttachmentPipeline();
 *   p.addDetector(new ImageDetector());
 *   const expanded = await p.expand({ role: 'user', content: text });
 */

export interface Attachment {
  /** Logical kind of the attachment. */
  type: 'image' | 'file' | 'url' | 'code';
  /** The matched substring from the source text. */
  content: string;
  /** Detector-specific extras. */
  metadata?: Record<string, unknown>;
}

export interface AttachmentDetector {
  name: string;
  detect(text: string): Attachment[];
}

// ---------------------------------------------------------------------------
// Built-in detectors
// ---------------------------------------------------------------------------

const IMAGE_EXT = /\.(png|jpe?g|gif|webp|bmp|svg)\b/i;

/** Detects image URLs (http(s)://...png), data: URIs, and local paths to images. */
export class ImageDetector implements AttachmentDetector {
  readonly name = 'image';

  detect(text: string): Attachment[] {
    const found: Attachment[] = [];

    // data: URIs (image/*)
    const dataUriRe = /data:image\/[a-zA-Z0-9+.\-]+;base64,[A-Za-z0-9+/=]+/g;
    for (const m of text.matchAll(dataUriRe)) {
      found.push({
        type: 'image',
        content: m[0],
        metadata: { source: 'data-uri' },
      });
    }

    // http(s) URLs ending in image extension
    const httpImageRe = /https?:\/\/[^\s)>"']+\.(?:png|jpe?g|gif|webp|bmp|svg)\b/gi;
    for (const m of text.matchAll(httpImageRe)) {
      found.push({
        type: 'image',
        content: m[0],
        metadata: { source: 'url' },
      });
    }

    // Local-looking paths ending in image extension. We accept absolute paths,
    // home-relative (~/), and dotted relative (./, ../). Bare relative names
    // can be ambiguous, so we keep this conservative.
    const localImageRe = /(?:^|[\s"'(<])((?:~\/|\.{0,2}\/|\/)[^\s"')>]+?\.(?:png|jpe?g|gif|webp|bmp|svg))\b/gi;
    for (const m of text.matchAll(localImageRe)) {
      const p = m[1];
      // Skip if it's actually a URL we already captured.
      if (/^https?:/i.test(p)) continue;
      found.push({
        type: 'image',
        content: p,
        metadata: { source: 'path' },
      });
    }

    return dedupe(found);
  }
}

/** Detects local file path references (existing files only). */
export class FilePathDetector implements AttachmentDetector {
  readonly name = 'file';

  detect(text: string): Attachment[] {
    const found: Attachment[] = [];
    // Absolute, home, or dotted-relative paths. Skip image extensions
    // (those are handled by ImageDetector).
    const re = /(?:^|[\s"'(<])((?:~\/|\.{0,2}\/|\/)[^\s"')>]+)/g;
    for (const m of text.matchAll(re)) {
      const p = m[1];
      if (/^https?:/i.test(p)) continue;
      if (IMAGE_EXT.test(p)) continue;
      found.push({
        type: 'file',
        content: p,
        metadata: { source: 'path' },
      });
    }
    return dedupe(found);
  }
}

/** Detects http/https URLs (excluding image URLs handled by ImageDetector). */
export class UrlDetector implements AttachmentDetector {
  readonly name = 'url';

  detect(text: string): Attachment[] {
    const found: Attachment[] = [];
    const re = /https?:\/\/[^\s)>"']+/g;
    for (const m of text.matchAll(re)) {
      const url = m[0];
      if (IMAGE_EXT.test(url)) continue;
      found.push({
        type: 'url',
        content: url,
        metadata: { source: 'http' },
      });
    }
    return dedupe(found);
  }
}

/** Detects fenced code blocks: ```lang\n...\n``` */
export class CodeDetector implements AttachmentDetector {
  readonly name = 'code';

  detect(text: string): Attachment[] {
    const found: Attachment[] = [];
    const re = /```([a-zA-Z0-9_+-]*)\s*\n([\s\S]*?)```/g;
    for (const m of text.matchAll(re)) {
      const lang = m[1] || '';
      const body = m[2];
      found.push({
        type: 'code',
        content: m[0],
        metadata: { language: lang || undefined, body },
      });
    }
    return found;
  }
}

function dedupe(items: Attachment[]): Attachment[] {
  const seen = new Set<string>();
  const out: Attachment[] = [];
  for (const a of items) {
    const key = `${a.type}:${a.content}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(a);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

export interface AttachmentPipelineOptions {
  /** Maximum bytes to inline-read for small text files in expand(). Default 32 KB. */
  maxInlineFileBytes?: number;
  /** If false, expand() will not read file contents (only convert paths to file:// URIs). */
  inlineSmallFiles?: boolean;
}

const DEFAULT_INLINE_BYTES = 32 * 1024;

export class AttachmentPipeline {
  private detectors: AttachmentDetector[] = [];
  private readonly maxInlineFileBytes: number;
  private readonly inlineSmallFiles: boolean;

  constructor(opts: AttachmentPipelineOptions = {}) {
    this.maxInlineFileBytes = opts.maxInlineFileBytes ?? DEFAULT_INLINE_BYTES;
    this.inlineSmallFiles = opts.inlineSmallFiles ?? true;
  }

  /** Add a detector. Returns this for chaining. */
  addDetector(detector: AttachmentDetector): this {
    this.detectors.push(detector);
    return this;
  }

  /** Run all detectors and return a flat, deduped attachment list. */
  detect(text: string): Attachment[] {
    const all: Attachment[] = [];
    for (const d of this.detectors) {
      try {
        all.push(...d.detect(text));
      } catch {
        // A misbehaving detector must not break the pipeline.
      }
    }
    // Final dedupe across detectors (e.g. a URL flagged as both image and url).
    return dedupe(all);
  }

  /**
   * Expand a message in place:
   *  - For images detected as local paths: convert to file:// URI image content blocks.
   *  - For images already URLs / data URIs: add them as image content blocks.
   *  - For small text files (file detector): inline their contents as a text block
   *    with a header (only if inlineSmallFiles is enabled).
   *
   * Original text is preserved as the first text block. Non-text input (already
   * an array of content blocks) is returned unchanged.
   */
  async expand(message: Message): Promise<Message> {
    if (typeof message.content !== 'string') return message;
    const text = message.content;
    const attachments = this.detect(text);
    if (attachments.length === 0) return message;

    const blocks: (TextContent | ImageContent)[] = [{ type: 'text', text }];

    for (const att of attachments) {
      if (att.type === 'image') {
        const block = await this.imageToBlock(att);
        if (block) blocks.push(block);
      } else if (att.type === 'file' && this.inlineSmallFiles) {
        const inlined = await this.maybeInlineFile(att);
        if (inlined) blocks.push(inlined);
      }
    }

    if (blocks.length === 1) return message;
    return { ...message, content: blocks };
  }

  private async imageToBlock(att: Attachment): Promise<ImageContent | null> {
    const c = att.content;
    if (c.startsWith('data:')) {
      const mediaMatch = c.match(/^data:([^;]+);/);
      return { type: 'image', source: c, mediaType: mediaMatch?.[1] };
    }
    if (/^https?:\/\//i.test(c)) {
      return { type: 'image', source: c, mediaType: guessMediaType(c) };
    }
    // Local path: resolve to file:// URI. Verify it exists; otherwise drop silently.
    const resolved = await resolveLocalPath(c);
    if (!resolved) return null;
    return {
      type: 'image',
      source: pathToFileUri(resolved),
      mediaType: guessMediaType(resolved),
    };
  }

  private async maybeInlineFile(att: Attachment): Promise<TextContent | null> {
    const resolved = await resolveLocalPath(att.content);
    if (!resolved) return null;
    try {
      const stat = await fs.stat(resolved);
      if (!stat.isFile()) return null;
      if (stat.size > this.maxInlineFileBytes) return null;
      const body = await fs.readFile(resolved, 'utf8');
      // Skip clearly binary content (heuristic: NUL byte present).
      if (body.includes(' ')) return null;
      return {
        type: 'text',
        text: `\n[file: ${resolved}]\n${body}\n[/file]\n`,
      };
    } catch {
      return null;
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function resolveLocalPath(p: string): Promise<string | null> {
  let resolved = p;
  if (resolved.startsWith('~/')) {
    const home = process.env.HOME || process.env.USERPROFILE || '';
    if (!home) return null;
    resolved = path.join(home, resolved.slice(2));
  }
  resolved = path.resolve(resolved);
  try {
    await fs.access(resolved);
    return resolved;
  } catch {
    return null;
  }
}

function pathToFileUri(p: string): string {
  // path.resolve already gives us an absolute path
  const normalized = p.replace(/\\/g, '/');
  if (normalized.startsWith('/')) return `file://${normalized}`;
  return `file:///${normalized}`;
}

function guessMediaType(p: string): string | undefined {
  const m = p.toLowerCase().match(/\.(png|jpe?g|gif|webp|bmp|svg)\b/);
  if (!m) return undefined;
  const ext = m[1] === 'jpg' ? 'jpeg' : m[1];
  return `image/${ext}`;
}

/** Convenience: a pipeline pre-loaded with all built-in detectors. */
export function defaultAttachmentPipeline(opts?: AttachmentPipelineOptions): AttachmentPipeline {
  return new AttachmentPipeline(opts)
    .addDetector(new ImageDetector())
    .addDetector(new FilePathDetector())
    .addDetector(new UrlDetector())
    .addDetector(new CodeDetector());
}
