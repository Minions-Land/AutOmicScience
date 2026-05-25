import type { LLMProvider } from '../provider/Provider.js';
import { detectProvider } from './modelDiscovery.js';

type ProviderFactory = () => LLMProvider;

const registry = new Map<string, ProviderFactory>();

export function registerProvider(name: string, factory: ProviderFactory): void {
  registry.set(name, factory);
}

export function getProvider(name: string): LLMProvider {
  const factory = registry.get(name);
  if (!factory) throw new Error(`Provider not found: ${name}. Available: ${[...registry.keys()].join(', ')}`);
  return factory();
}

export function getProviderForModel(model: string): LLMProvider {
  const providerName = detectProvider(model);
  return getProvider(providerName);
}

export function listRegisteredProviders(): string[] {
  return [...registry.keys()];
}

export function hasProvider(name: string): boolean {
  return registry.has(name);
}

export function autoRegisterProviders(): void {
  if (process.env.OPENAI_API_KEY && !registry.has('openai')) {
    registerProvider('openai', () => {
      const { OpenAIProvider } = require('../provider/OpenAIProvider.js');
      return new OpenAIProvider();
    });
  }
  if (process.env.ANTHROPIC_API_KEY && !registry.has('anthropic')) {
    registerProvider('anthropic', () => {
      const { AnthropicProvider } = require('../provider/AnthropicProvider.js');
      return new AnthropicProvider();
    });
  }
  if ((process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY) && !registry.has('gemini')) {
    registerProvider('gemini', () => {
      const { GeminiProvider } = require('../provider/GeminiProvider.js');
      return new GeminiProvider();
    });
  }
}
