import { z, ZodTypeAny } from 'zod';
import type { Tool } from '../toolset/Tool.js';
import type { McpPlugin } from './McpPlugin.js';

export type McpTransport =
  | { kind: 'stdio'; command: string; args?: string[]; env?: Record<string, string> }
  | { kind: 'sse'; url: string; headers?: Record<string, string> };

/**
 * Wraps an MCP server (stdio or SSE) and exposes its tools as AutOmicScience `Tool`s.
 * Uses `@modelcontextprotocol/sdk` under the hood. Imports are lazy so the SDK
 * is optional at install time.
 */
export class McpClient implements McpPlugin {
  public readonly name: string;
  private connected = false;
  // Loosely typed because we lazy-import the SDK.
  private client: any = null;
  private transportInst: any = null;

  constructor(name: string, private readonly transport: McpTransport) {
    this.name = name;
  }

  async connect(): Promise<void> {
    if (this.connected) return;
    const { Client } = await import('@modelcontextprotocol/sdk/client/index.js');
    if (this.transport.kind === 'stdio') {
      const { StdioClientTransport } = await import(
        '@modelcontextprotocol/sdk/client/stdio.js'
      );
      this.transportInst = new StdioClientTransport({
        command: this.transport.command,
        args: this.transport.args ?? [],
        env: this.transport.env,
      });
    } else {
      const { SSEClientTransport } = await import(
        '@modelcontextprotocol/sdk/client/sse.js'
      );
      this.transportInst = new SSEClientTransport(new URL(this.transport.url), {
        requestInit: { headers: this.transport.headers },
      } as any);
    }
    this.client = new Client(
      { name: `aos-ai:${this.name}`, version: '0.1.0' },
      { capabilities: {} },
    );
    await this.client.connect(this.transportInst);
    this.connected = true;
  }

  async disconnect(): Promise<void> {
    if (!this.connected) return;
    try {
      await this.client?.close?.();
    } catch {
      // ignore
    }
    this.connected = false;
    this.client = null;
    this.transportInst = null;
  }

  async getTools(): Promise<Tool[]> {
    if (!this.connected) {
      throw new Error(`McpClient '${this.name}' is not connected. Call connect() first.`);
    }
    const list = await this.client.listTools();
    const tools: Tool[] = [];
    for (const t of list.tools ?? []) {
      tools.push({
        name: `${this.name}__${t.name}`,
        description: t.description ?? `MCP tool ${t.name}`,
        parameters: jsonSchemaToZod(t.inputSchema),
        execute: async (args) => {
          const res = await this.client.callTool({ name: t.name, arguments: args ?? {} });
          // MCP returns content as an array of { type, text } parts.
          const content = Array.isArray(res?.content)
            ? res.content
                .map((c: any) => (typeof c?.text === 'string' ? c.text : JSON.stringify(c)))
                .join('\n')
            : JSON.stringify(res);
          return content;
        },
      });
    }
    return tools;
  }
}

/**
 * Minimal JSON-Schema -> Zod fallback. We only need a permissive schema; the
 * MCP server enforces validation server-side.
 */
function jsonSchemaToZod(_schema: unknown): ZodTypeAny {
  return z.record(z.unknown()).optional() as unknown as ZodTypeAny;
}
