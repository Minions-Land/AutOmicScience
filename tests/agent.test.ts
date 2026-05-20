import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import { Agent } from '../src/agent/index.js';
import { ToolSet, defineTool } from '../src/toolset/index.js';
import type { LLMProvider, ProviderStreamChunk } from '../src/provider/Provider.js';
import type { Message, ChatOptions } from '../src/types.js';

class MockProvider implements LLMProvider {
  name = 'mock';
  supportsTools = true;
  private call = 0;
  async *chat(_messages: Message[], _options: ChatOptions): AsyncGenerator<ProviderStreamChunk> {
    this.call++;
    if (this.call === 1) {
      yield {
        type: 'tool_call',
        toolCall: { id: 'c1', name: 'echo', arguments: { msg: 'hi' } },
      };
      yield { type: 'done', finishReason: 'tool_calls' };
      return;
    }
    yield { type: 'text', text: 'final answer' };
    yield { type: 'done', finishReason: 'stop' };
  }
}

describe('Agent', () => {
  it('runs the tool-call loop until completion', async () => {
    const toolset = new ToolSet('t', [
      defineTool({
        name: 'echo',
        description: 'echo',
        parameters: z.object({ msg: z.string() }),
        execute: async ({ msg }) => msg,
      }),
    ]);
    const agent = new Agent({
      name: 'tester',
      model: 'mock-model',
      provider: new MockProvider(),
      toolset,
    });
    const events: { type: string; data: unknown }[] = [];
    for await (const ev of agent.run('hello')) events.push(ev);
    const types = events.map((e) => e.type);
    expect(types).toContain('tool_call');
    expect(types).toContain('tool_result');
    expect(types).toContain('done');
    const finalEv = events.find((e) => e.type === 'done');
    expect(finalEv?.data).toBe('final answer');
  });
});
