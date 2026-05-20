// Approximate token counter: ~4 chars per token. Replace with tiktoken/anthropic
// tokenizers as needed.
export function approxTokenCount(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

export function approxMessagesTokenCount(messages: Array<{ content: string }>): number {
  return messages.reduce((sum, m) => sum + approxTokenCount(m.content ?? ''), 0);
}
