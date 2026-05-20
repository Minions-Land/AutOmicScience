import type { LLMProvider } from './Provider.js';
import { OpenAIProvider } from './OpenAIProvider.js';
import { AnthropicProvider } from './AnthropicProvider.js';
import { GeminiProvider } from './GeminiProvider.js';
import { parseModelString } from './Provider.js';

export function providerForModel(model: string): LLMProvider {
  const { base } = parseModelString(model);
  const lower = base.toLowerCase();
  if (lower.startsWith('gpt-') || lower.startsWith('o1') || lower.startsWith('o3') || lower.startsWith('o4')) {
    return new OpenAIProvider();
  }
  if (lower.startsWith('claude')) return new AnthropicProvider();
  if (lower.startsWith('gemini')) return new GeminiProvider();
  // Default: try whichever key is set.
  if (process.env.OPENAI_API_KEY) return new OpenAIProvider();
  if (process.env.ANTHROPIC_API_KEY) return new AnthropicProvider();
  if (process.env.GOOGLE_API_KEY) return new GeminiProvider();
  throw new Error(`Cannot determine provider for model '${model}' and no API keys are set.`);
}

export function defaultModel(): string {
  return process.env.NOVAEVE_MODEL || 'gpt-4o-mini';
}
