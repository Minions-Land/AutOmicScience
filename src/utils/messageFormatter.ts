import type { Message, TextContent, ImageContent } from '../types.js';

/**
 * Format messages for a specific LLM provider API.
 * Converts the internal Message format to the provider's expected shape.
 */
export function formatForProvider(messages: Message[], provider: string): any[] {
  switch (provider.toLowerCase()) {
    case 'openai':
      return messages.map(formatForOpenAI);
    case 'anthropic':
      return messages.map(formatForAnthropic);
    case 'gemini':
      return messages.map(formatForGemini);
    default:
      return messages.map(formatForOpenAI);
  }
}

/** Extract plain text content from a potentially multimodal message. */
export function extractTextContent(message: Message): string {
  if (typeof message.content === 'string') {
    return message.content;
  }
  if (Array.isArray(message.content)) {
    return message.content
      .filter((block): block is TextContent => block.type === 'text')
      .map((block) => block.text)
      .join('\n');
  }
  return '';
}

/** Build a multimodal message containing text and image references. */
export function buildMultimodalMessage(text: string, images: string[]): Message {
  const content: (TextContent | ImageContent)[] = [{ type: 'text', text }];
  for (const source of images) {
    content.push({ type: 'image', source });
  }
  return { role: 'user', content };
}

/**
 * Truncate a message array to fit within a token budget.
 * Removes oldest non-system messages first.
 */
export function truncateMessages(messages: Message[], maxTokens: number): Message[] {
  const estimateTokens = (msg: Message): number => {
    const text = extractTextContent(msg);
    return Math.ceil(text.length / 4);
  };

  let total = messages.reduce((sum, m) => sum + estimateTokens(m), 0);
  if (total <= maxTokens) return messages;

  const system = messages.filter((m) => m.role === 'system');
  const nonSystem = messages.filter((m) => m.role !== 'system');

  const result = [...nonSystem];
  while (total > maxTokens && result.length > 1) {
    const removed = result.shift()!;
    total -= estimateTokens(removed);
  }

  return [...system, ...result];
}

// --- Provider-specific formatters ---

function formatForOpenAI(m: Message): any {
  if (m.role === 'tool') {
    return { role: 'tool', tool_call_id: m.tool_call_id, content: extractTextContent(m) };
  }
  if (typeof m.content === 'string') {
    const base: any = { role: m.role, content: m.content };
    if (m.name) base.name = m.name;
    if (m.tool_calls?.length) {
      base.tool_calls = m.tool_calls.map((tc) => ({
        id: tc.id,
        type: 'function',
        function: { name: tc.name, arguments: JSON.stringify(tc.arguments ?? {}) },
      }));
    }
    return base;
  }
  // Multimodal
  const content = (m.content as (TextContent | ImageContent)[]).map((block) => {
    if (block.type === 'text') return { type: 'text', text: block.text };
    return { type: 'image_url', image_url: { url: (block as ImageContent).source } };
  });
  return { role: m.role, content };
}

function formatForAnthropic(m: Message): any {
  if (m.role === 'system') {
    return { role: 'user', content: extractTextContent(m) };
  }
  if (typeof m.content === 'string') {
    return { role: m.role === 'tool' ? 'user' : m.role, content: m.content };
  }
  const content = (m.content as (TextContent | ImageContent)[]).map((block) => {
    if (block.type === 'text') return { type: 'text', text: block.text };
    const img = block as ImageContent;
    return {
      type: 'image',
      source: { type: 'base64', media_type: img.mediaType ?? 'image/png', data: img.source },
    };
  });
  return { role: m.role === 'tool' ? 'user' : m.role, content };
}

function formatForGemini(m: Message): any {
  const role = m.role === 'assistant' ? 'model' : 'user';
  if (typeof m.content === 'string') {
    return { role, parts: [{ text: m.content }] };
  }
  const parts = (m.content as (TextContent | ImageContent)[]).map((block) => {
    if (block.type === 'text') return { text: block.text };
    return { inlineData: { mimeType: (block as ImageContent).mediaType ?? 'image/png', data: (block as ImageContent).source } };
  });
  return { role, parts };
}
