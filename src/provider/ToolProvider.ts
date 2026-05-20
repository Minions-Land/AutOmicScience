import type { Tool } from '../toolset/Tool.js';

export interface ToolProvider {
  readonly name: string;
  listTools(): Promise<Tool[]>;
  callTool(name: string, args: Record<string, unknown>): Promise<unknown>;
  close?(): Promise<void>;
}

export class LocalProvider implements ToolProvider {
  readonly name: string;
  private readonly tools: Tool[];

  constructor(name: string, tools: Tool[]) {
    this.name = name;
    this.tools = tools;
  }

  async listTools(): Promise<Tool[]> {
    return this.tools;
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<unknown> {
    const tool = this.tools.find((t) => t.name === name);
    if (!tool) throw new Error(`Tool not found: ${name}`);
    return tool.execute(args, {} as any);
  }
}

export class RemoteToolSetProvider implements ToolProvider {
  readonly name: string;
  private readonly baseUrl: string;
  private toolsCache: Tool[] | null = null;

  constructor(name: string, baseUrl: string) {
    this.name = name;
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  async listTools(): Promise<Tool[]> {
    if (this.toolsCache) return this.toolsCache;
    const resp = await fetch(`${this.baseUrl}/tools`);
    if (!resp.ok) throw new Error(`Failed to list tools from ${this.baseUrl}`);
    const data = await resp.json() as any[];
    this.toolsCache = data.map((t) => ({
      name: `${this.name}__${t.name}`,
      description: t.description ?? '',
      parameters: {} as any,
      execute: async (args: any) => this.callTool(t.name, args),
    }));
    return this.toolsCache;
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<unknown> {
    const toolName = name.startsWith(`${this.name}__`) ? name.slice(this.name.length + 2) : name;
    const resp = await fetch(`${this.baseUrl}/tools/${toolName}/call`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(args),
    });
    if (!resp.ok) throw new Error(`Tool call failed: ${resp.statusText}`);
    return resp.json();
  }

  invalidateCache(): void {
    this.toolsCache = null;
  }
}

export class MCPToolProvider implements ToolProvider {
  readonly name: string;
  private readonly uri: string;
  private readonly filterPrefix?: string;
  private client: any = null;
  private toolsCache: Tool[] | null = null;

  private static instances = new Map<string, MCPToolProvider>();

  static getInstance(uri: string, filterPrefix?: string): MCPToolProvider {
    const key = `${uri}:${filterPrefix ?? ''}`;
    if (!MCPToolProvider.instances.has(key)) {
      MCPToolProvider.instances.set(key, new MCPToolProvider(uri, filterPrefix));
    }
    return MCPToolProvider.instances.get(key)!;
  }

  private constructor(uri: string, filterPrefix?: string) {
    this.uri = uri;
    this.filterPrefix = filterPrefix;
    this.name = `mcp:${uri}`;
  }

  async listTools(): Promise<Tool[]> {
    if (this.toolsCache) return this.toolsCache;
    await this.ensureConnected();
    const list = await this.client.listTools();
    let tools = list.tools ?? [];
    if (this.filterPrefix) {
      tools = tools.filter((t: any) => t.name.startsWith(this.filterPrefix!));
    }
    this.toolsCache = tools.map((t: any) => ({
      name: t.name,
      description: t.description ?? '',
      parameters: {} as any,
      execute: async (args: any) => this.callTool(t.name, args),
    }));
    return this.toolsCache!;
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<unknown> {
    await this.ensureConnected();
    const res = await this.client.callTool({ name, arguments: args });
    if (Array.isArray(res?.content)) {
      return res.content.map((c: any) => c.text ?? JSON.stringify(c)).join('\n');
    }
    return res;
  }

  async close(): Promise<void> {
    if (this.client) {
      try { await this.client.close(); } catch { /* ignore */ }
      this.client = null;
    }
    this.toolsCache = null;
  }

  private async ensureConnected(): Promise<void> {
    if (this.client) return;
    const { Client } = await import('@modelcontextprotocol/sdk/client/index.js');
    const { SSEClientTransport } = await import('@modelcontextprotocol/sdk/client/sse.js');
    const transport = new SSEClientTransport(new URL(this.uri));
    this.client = new Client({ name: 'medrix-mcp', version: '1.0.0' }, { capabilities: {} });
    await this.client.connect(transport);
  }

  invalidateCache(): void {
    this.toolsCache = null;
  }
}
