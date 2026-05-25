import { describe, it, expect } from 'vitest';
import type { LLMProvider, ProviderStreamChunk } from '../src/provider/Provider.js';
import type { Message, ChatOptions } from '../src/types.js';
import { ToolSet } from '../src/toolset/ToolSet.js';
import { createAnnotationPipeline } from '../src/team/AnnotationPipeline.js';
import { loadAnnotationPipelineSkill } from '../src/skill/BuiltinSkills.js';

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

describe('annotation pipeline team factory', () => {
  it('builds a Sequential of Selector -> Adapter -> Adjudicator', async () => {
    const team = await createAnnotationPipeline({
      model: 'mock',
      provider: tagProvider('S'),
      toolset: new ToolSet('empty'),
    });
    expect(team.name).toBe('annotation-pipeline');
  });

  it('pipes input through three agents in order', async () => {
    const team = await createAnnotationPipeline({
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

describe('annotation pipeline skill', () => {
  it('loads the annotation-pipeline markdown skill', async () => {
    const skill = await loadAnnotationPipelineSkill();
    expect(skill.name).toBe('annotation-pipeline');
    expect(skill.instructions.length).toBeGreaterThan(100);
    expect(skill.instructions).toContain('Selector');
  });
});
