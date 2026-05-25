import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import { Agent } from '../src/agent/index.js';
import { ToolSet, defineTool } from '../src/toolset/index.js';
import { HookManager } from '../src/hooks/index.js';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
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

class SkillToolProvider implements LLMProvider {
  name = 'skill-tool';
  supportsTools = true;
  private call = 0;
  async *chat(_messages: Message[], _options: ChatOptions): AsyncGenerator<ProviderStreamChunk> {
    this.call++;
    if (this.call === 1) {
      yield {
        type: 'tool_call',
        toolCall: { id: 'skill-call', name: 'list_available_skills', arguments: {} },
      };
      yield { type: 'done', finishReason: 'tool_calls' };
      return;
    }
    yield { type: 'text', text: 'skills listed' };
    yield { type: 'done', finishReason: 'stop' };
  }
}

class InspectingProvider implements LLMProvider {
  name = 'inspect';
  supportsTools = true;
  seenMessages: Message[] = [];
  async *chat(messages: Message[], _options: ChatOptions): AsyncGenerator<ProviderStreamChunk> {
    this.seenMessages = messages;
    yield { type: 'text', text: 'ok' };
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

  it('exposes history, model, tool, skill, and memory controls', async () => {
    const agent = new Agent({
      name: 'tester',
      model: 'mock-model',
      provider: new MockProvider(),
      toolset: new ToolSet('empty'),
      skills: [{ name: 's1', description: 'skill one', instructions: 'be precise' }],
    });

    agent.setModel(['model-a', 'model-b']);
    expect(agent.modelsList).toEqual(['model-a', 'model-b']);

    agent.addSkill({ name: 's2', description: 'skill two', instructions: 'be concise' });
    expect(agent.listSkills().map((s) => s.name)).toEqual(['s1', 's2']);
    expect(agent.removeSkill('s1')).toBe(true);

    agent.addTool(
      defineTool({
        name: 'noop',
        description: 'no op',
        parameters: z.object({}),
        execute: async () => 'ok',
      }),
    );
    expect(agent.listTools().map((t) => t.name)).toContain('noop');
    expect(agent.removeTool('noop')).toBe(true);

    await agent.setHistory([{ role: 'user', content: 'hello' }]);
    expect(await agent.getHistory()).toHaveLength(1);
    await agent.remember('k', 'v');
    expect(await agent.recall('k')).toBe('v');

    const snapshot = await agent.snapshot();
    expect(snapshot.models).toEqual(['model-a', 'model-b']);
    expect(snapshot.skillCount).toBe(1);
    expect(snapshot.messageCount).toBe(1);
  });

  it('registers first-class skill tools and can inspect built-in skills', async () => {
    const agent = new Agent({
      name: 'tester',
      model: 'mock-model',
      provider: new SkillToolProvider(),
      toolset: new ToolSet('empty'),
    });
    expect(agent.listTools().map((tool) => tool.name)).toContain('list_available_skills');

    const listed = await agent.executeTool('list_available_skills', {});
    const listPayload = JSON.parse(listed.content);
    expect(listPayload.skills.some((skill: any) => skill.name === 'annotation-pipeline')).toBe(true);

    const loaded = await agent.executeTool('load_skill', { name: 'annotation-pipeline' });
    expect(JSON.parse(loaded.content).loaded.name).toBe('annotation-pipeline');
    expect(agent.listSkills().map((skill) => skill.name)).toContain('annotation-pipeline');

    const read = await agent.executeTool('read_skill', { name: 'annotation-pipeline' });
    expect(JSON.parse(read.content).instructions).toContain('Annotation Pipeline');

    const events: string[] = [];
    for await (const event of agent.run('skill呢')) {
      if (event.type === 'tool_call') events.push((event.data as any[])[0].name);
    }
    expect(events).toContain('list_available_skills');
  });

  it('compacts long histories with a summary message', async () => {
    const agent = new Agent({
      name: 'tester',
      model: 'mock-model',
      provider: new MockProvider(),
      maxHistoryTokens: 20,
    });

    await agent.setHistory([
      { role: 'user', content: 'first '.repeat(100) },
      { role: 'assistant', content: 'second '.repeat(100) },
      { role: 'user', content: 'third '.repeat(100) },
      { role: 'assistant', content: 'fourth '.repeat(100) },
      { role: 'user', content: 'recent question' },
    ]);

    const compacted = await agent.compactHistory(40);
    expect(compacted.some((m) => String(m.content).includes('summary'))).toBe(true);
    expect(await agent.getHistory()).toEqual(compacted);
  });

  it('injects project instructions into the system prompt', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'aos-project-'));
    await writeFile(path.join(dir, 'AGENTS.md'), 'Always answer in Chinese.', 'utf-8');
    const provider = new InspectingProvider();
    const agent = new Agent({
      name: 'tester',
      model: 'mock-model',
      provider,
      projectInstructions: { cwd: dir },
    });

    await agent.runToText('hello');
    expect(String(provider.seenMessages[0].content)).toContain('Always answer in Chinese.');
    await rm(dir, { recursive: true, force: true });
  });

  it('emits lifecycle hooks for agent and tool execution', async () => {
    const events: string[] = [];
    const hooks = new HookManager()
      .on('agent:beforeRun', () => events.push('before'))
      .on('tool:beforeCall', () => events.push('tool-before'))
      .on('tool:afterCall', () => events.push('tool-after'))
      .on('agent:afterRun', () => events.push('after'));
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
      hooks,
    });

    await agent.runToText('hello');
    expect(events).toEqual(['before', 'tool-before', 'tool-after', 'after']);
  });
});
