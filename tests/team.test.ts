import { describe, it, expect } from 'vitest';
import { Sequential } from '../src/team/index.js';
import { Agent } from '../src/agent/index.js';
import type { LLMProvider, ProviderStreamChunk } from '../src/provider/Provider.js';
import type { Message, ChatOptions } from '../src/types.js';

function makeEchoProvider(suffix: string): LLMProvider {
  return {
    name: `mock-${suffix}`,
    supportsTools: false,
    async *chat(messages: Message[], _options: ChatOptions): AsyncGenerator<ProviderStreamChunk> {
      const last = messages[messages.length - 1]?.content ?? '';
      yield { type: 'text', text: `${last}|${suffix}` };
      yield { type: 'done', finishReason: 'stop' };
    },
  };
}

describe('Sequential team', () => {
  it('pipes output from one agent to the next', async () => {
    const a = new Agent({ name: 'A', model: 'm', provider: makeEchoProvider('A') });
    const b = new Agent({ name: 'B', model: 'm', provider: makeEchoProvider('B') });
    const team = new Sequential([a, b]);
    let final = '';
    for await (const ev of team.run('hello')) {
      if (ev.type === 'done') final = String(ev.data ?? '');
    }
    expect(final).toBe('hello|A|B');
  });
});
