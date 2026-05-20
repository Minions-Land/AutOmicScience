import { describe, it, expect } from 'vitest';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { LLMProvider, ProviderStreamChunk } from '../../src/provider/Provider.js';
import type { Message, ChatOptions } from '../../src/types.js';
import { ToolSet } from '../../src/toolset/ToolSet.js';
import { createScmasPipeline } from '../../src/plugins/scmas/team/ScmasPipeline.js';
import { loadScmasAnnotationSkill } from '../../src/plugins/scmas/skills/index.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));

function tagProvider(tag: string): LLMProvider {
  return {
    name: `mock-${tag}`,
    supportsTools: false,
    async *chat(messages: Message[], _options: ChatOptions): AsyncGenerator<ProviderStreamChunk> {
      const last = messages[messages.length - 1]?.content ?? '';
      yield { type: 'text', text: `${last}|${tag}` };
      yield { type: 'done', finishReason: 'stop' };
    },
  };
}

describe('scMAS pipeline team factory', () => {
  it('builds a Sequential of Stage-2 -> Stage-3 -> Stage-4 agents', async () => {
    const team = await createScmasPipeline({
      model: 'mock',
      provider: tagProvider('S2'),
      toolset: new ToolSet('empty'),
    });
    expect(team.name).toBe('scmas-pipeline');
  });

  it('pipes input through three agents in order', async () => {
    const team = await createScmasPipeline({
      model: 'mock',
      provider: tagProvider('A'),
      toolset: new ToolSet('empty'),
    });
    let final = '';
    for await (const ev of team.run('hello')) {
      if (ev.type === 'done') final = String(ev.data ?? '');
    }
    // The provider stamps |A on each hop, three hops total.
    expect(final.endsWith('|A|A|A')).toBe(true);
  });
});

describe('scMAS skill', () => {
  it('loads the scmas-annotation markdown skill', async () => {
    // Resolve relative to repo root since skill files live under src/.
    process.chdir(path.resolve(HERE, '..', '..'));
    const skill = await loadScmasAnnotationSkill();
    expect(skill.name).toBe('scmas-annotation');
    expect(skill.instructions.length).toBeGreaterThan(100);
    expect(skill.instructions).toContain('Stage 1');
  });
});
