import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import { McpClient } from '../src/mcp/index.js';
import type { Tool } from '../src/toolset/Tool.js';

describe('McpClient', () => {
  it('throws if getTools() is called before connect()', async () => {
    const c = new McpClient('test', { kind: 'stdio', command: 'no-such-binary' });
    await expect(c.getTools()).rejects.toThrow(/not connected/);
  });

  it('exposes connected MCP tools as Tool[] (mocked client)', async () => {
    const c = new McpClient('mock', { kind: 'stdio', command: 'noop' });
    // Inject a fake client + transport bypassing connect().
    (c as any).connected = true;
    (c as any).client = {
      listTools: async () => ({
        tools: [
          { name: 'add', description: 'adds', inputSchema: { type: 'object' } },
        ],
      }),
      callTool: async ({ name, arguments: _args }: any) => ({
        content: [{ type: 'text', text: `called:${name}` }],
      }),
    };
    const tools: Tool[] = await c.getTools();
    expect(tools).toHaveLength(1);
    expect(tools[0].name).toBe('mock__add');
    const result = await tools[0].execute({}, { agentName: 't' });
    expect(String(result)).toContain('called:add');
    // Schema should be a zod object/record.
    expect(typeof tools[0].parameters.parse).toBe('function');
    void z;
  });
});
