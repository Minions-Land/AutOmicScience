import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import { ToolSet, defineTool, tool } from '../src/toolset/index.js';

describe('ToolSet', () => {
  it('registers and executes a tool', async () => {
    const ts = new ToolSet('t');
    ts.register(
      defineTool({
        name: 'add',
        description: 'add two numbers',
        parameters: z.object({ a: z.number(), b: z.number() }),
        execute: async ({ a, b }) => a + b,
      }),
    );
    expect(ts.size()).toBe(1);
    const res = await ts.execute('add', { a: 2, b: 3 }, { agentName: 't' });
    expect(res.content).toBe('5');
  });

  it('toOpenAITools() returns function-typed defs', () => {
    const ts = new ToolSet('t', [
      defineTool({
        name: 'noop',
        description: 'no op',
        parameters: z.object({}),
        execute: async () => 'ok',
      }),
    ]);
    const defs = ts.toOpenAITools();
    expect(defs).toHaveLength(1);
    expect(defs[0].type).toBe('function');
    expect(defs[0].function.name).toBe('noop');
  });

  it('builds a ToolSet from a class with @tool decorators', async () => {
    class Calc {
      @tool('add', z.object({ a: z.number(), b: z.number() }))
      add(args: { a: number; b: number }) {
        return args.a + args.b;
      }
    }
    const ts = ToolSet.fromClass(new Calc());
    expect(ts.has('add')).toBe(true);
    const res = await ts.execute('add', { a: 1, b: 2 }, { agentName: 'x' });
    expect(res.content).toBe('3');
  });
});
