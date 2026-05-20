/**
 * Token counter with optional tiktoken-rs / gpt-tokenizer support.
 *
 * Tries to load `gpt-tokenizer` (npm package) for accurate counts; falls back
 * to a heuristic that's much better than chars/4:
 * - Splits on word boundaries
 * - Counts most words as 1 token, long words as multiple
 * - Adds overhead for punctuation
 */

let _tokenizer: { encode(text: string): number[] } | null = null;
let _tokenizerLoaded = false;

async function loadTokenizer(): Promise<{ encode(text: string): number[] } | null> {
  if (_tokenizerLoaded) return _tokenizer;
  _tokenizerLoaded = true;
  try {
    const mod = await import('gpt-tokenizer' as any);
    _tokenizer = mod;
    return mod;
  } catch {
    return null;
  }
}

/** BPE-aware heuristic: better than chars/4 by ~20-30%. */
function heuristicTokenCount(text: string): number {
  if (!text) return 0;
  // Split on whitespace and punctuation
  const tokens = text.split(/[\s,.!?;:()\[\]{}<>'"`~@#$%^&*+=|\\/\-]+/).filter((t) => t.length > 0);
  let count = 0;
  for (const word of tokens) {
    // Most English words: ~1 token. Long words split into BPE pieces.
    if (word.length <= 4) count += 1;
    else if (word.length <= 8) count += 2;
    else count += Math.ceil(word.length / 4);
  }
  // Add overhead for punctuation/whitespace (about 10% of word count)
  count += Math.ceil(count * 0.1);
  return Math.max(count, 1);
}

export function approxTokenCount(text: string): number {
  if (!text) return 0;
  if (_tokenizer) {
    try {
      return _tokenizer.encode(text).length;
    } catch {
      return heuristicTokenCount(text);
    }
  }
  return heuristicTokenCount(text);
}

export function approxMessagesTokenCount(messages: Array<{ content: string | unknown[] }>): number {
  let total = 0;
  for (const m of messages) {
    if (typeof m.content === 'string') {
      total += approxTokenCount(m.content);
    } else if (Array.isArray(m.content)) {
      for (const part of m.content) {
        if ((part as any).type === 'text') total += approxTokenCount((part as any).text ?? '');
        else if ((part as any).type === 'image') total += 85; // vision token estimate
      }
    }
    total += 4; // role/structure overhead per message
  }
  return total;
}

/** Pre-load the tokenizer so subsequent calls are synchronous and accurate. */
export async function preloadTokenizer(): Promise<boolean> {
  return (await loadTokenizer()) !== null;
}
