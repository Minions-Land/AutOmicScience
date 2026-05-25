export interface ModelInfo {
  name: string;
  provider: string;
  contextWindow: number;
  supportsVision: boolean;
  supportsTools: boolean;
  supportsStreaming: boolean;
}

const KNOWN_MODELS: ModelInfo[] = [
  { name: 'gpt-4o', provider: 'openai', contextWindow: 128000, supportsVision: true, supportsTools: true, supportsStreaming: true },
  { name: 'gpt-4o-mini', provider: 'openai', contextWindow: 128000, supportsVision: true, supportsTools: true, supportsStreaming: true },
  { name: 'gpt-4-turbo', provider: 'openai', contextWindow: 128000, supportsVision: true, supportsTools: true, supportsStreaming: true },
  { name: 'o1', provider: 'openai', contextWindow: 200000, supportsVision: true, supportsTools: true, supportsStreaming: true },
  { name: 'o3', provider: 'openai', contextWindow: 200000, supportsVision: true, supportsTools: true, supportsStreaming: true },
  { name: 'gemini-2.5-pro', provider: 'gemini', contextWindow: 1000000, supportsVision: true, supportsTools: true, supportsStreaming: true },
  { name: 'gemini-2.5-flash', provider: 'gemini', contextWindow: 1000000, supportsVision: true, supportsTools: true, supportsStreaming: true },
  { name: 'gemini-2.0-flash', provider: 'gemini', contextWindow: 1000000, supportsVision: true, supportsTools: true, supportsStreaming: true },
];

export function getModelInfo(model: string): ModelInfo | undefined {
  const base = model.replace(/\+think(:\w+)?$/, '');
  const provider = detectProvider(base);
  if (provider === 'anthropic') {
    return {
      name: base,
      provider,
      contextWindow: 200000,
      supportsVision: true,
      supportsTools: true,
      supportsStreaming: true,
    };
  }
  return KNOWN_MODELS.find((m) => m.name === base || base.startsWith(m.name));
}

export function detectProvider(model: string): string {
  const base = model.replace(/\+think(:\w+)?$/, '');
  if (base.startsWith('gpt-') || base.startsWith('o1') || base.startsWith('o3') || base.startsWith('o4')) return 'openai';
  if (base.startsWith('anthropic/')) return 'anthropic';
  if (base.startsWith('gemini-')) return 'gemini';
  const info = KNOWN_MODELS.find((m) => m.name === base);
  return info?.provider ?? 'openai';
}

export function getDefaultModel(): string {
  if (process.env.AOS_MODEL) return process.env.AOS_MODEL;
  if (process.env.AOS_OPENAI_API_KEY || process.env.OPENAI_API_KEY) return 'gpt-5.5';
  if (process.env.ANTHROPIC_API_KEY && process.env.ANTHROPIC_MODEL) return `anthropic/${process.env.ANTHROPIC_MODEL}`;
  if (process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY) return 'gemini-2.5-pro';
  return 'gpt-5.5';
}

export function listAvailableProviders(): string[] {
  const available: string[] = [];
  if (process.env.AOS_OPENAI_API_KEY || process.env.OPENAI_API_KEY) available.push('openai');
  if (process.env.ANTHROPIC_API_KEY) available.push('anthropic');
  if (process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY) available.push('gemini');
  return available;
}

export function getContextWindow(model: string): number {
  const info = getModelInfo(model);
  return info?.contextWindow ?? 128000;
}

export function supportsVision(model: string): boolean {
  const info = getModelInfo(model);
  return info?.supportsVision ?? false;
}

export const KNOWN_MODELS_BY_PROVIDER: Record<string, string[]> = {
  openai: KNOWN_MODELS.filter((m) => m.provider === 'openai').map((m) => m.name),
  anthropic: process.env.ANTHROPIC_MODEL ? [`anthropic/${process.env.ANTHROPIC_MODEL}`] : [],
  gemini: KNOWN_MODELS.filter((m) => m.provider === 'gemini').map((m) => m.name),
};

export function listKnownModels(): string[] {
  return [
    ...KNOWN_MODELS.map((m) => m.name),
    ...KNOWN_MODELS_BY_PROVIDER.anthropic,
  ];
}
