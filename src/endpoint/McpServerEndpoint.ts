import type { Agent } from '../agent/Agent.js';
import type { ToolSet } from '../toolset/ToolSet.js';
import { logger } from '../utils/logger.js';

/**
 * Exposes MedrixAI agents and toolsets AS an MCP server that external
 * clients (Claude Desktop, other MCP consumers) can connect to.
 */
export interface McpServerEndpoint {
  /**
   * Start serving the given agents and toolsets over the MCP protocol.
   * @param agents - Agents to expose as MCP tools.
   * @param toolsets - ToolSets to expose as MCP tools.
   */
  expose(agents: Agent[], toolsets: ToolSet[]): Promise<void>;
  /** Close the MCP server and release resources. */
  close(): Promise<void>;
}

/**
 * MCP server endpoint using @modelcontextprotocol/sdk (lazy-imported).
 * Exposes each agent as a callable tool over stdio transport.
 */
export class StubMcpServerEndpoint implements McpServerEndpoint {
  private running = false;
  private server: any = null;

  /**
   * Start serving agents and toolsets over MCP stdio transport.
   * Each agent becomes a tool named after the agent.
   */
  async expose(agents: Agent[], toolsets: ToolSet[]): Promise<void> {
    const serverMod: any = await import('@modelcontextprotocol/sdk/server');
    const { Server } = serverMod;
    const { StdioServerTransport } = serverMod;

    this.server = new Server(
      { name: 'medrix-mcp', version: '0.1.0' },
      { capabilities: { tools: {} } },
    );

    // Build tool definitions
    const toolDefs: Array<{ name: string; description: string; inputSchema: Record<string, unknown> }> = [];

    for (const agent of agents) {
      toolDefs.push({
        name: agent.name,
        description: `Invoke agent: ${agent.name}`,
        inputSchema: {
          type: 'object',
          properties: {
            input: { type: 'string', description: 'The input message for the agent.' },
          },
          required: ['input'],
        },
      });
    }

    for (const ts of toolsets) {
      for (const t of ts.list()) {
        toolDefs.push({
          name: t.name,
          description: t.description,
          inputSchema: {
            type: 'object',
            properties: {
              args: { type: 'object', description: 'Tool arguments.' },
            },
          },
        });
      }
    }

    this.server.setRequestHandler('tools/list' as any, async () => ({
      tools: toolDefs,
    }));

    this.server.setRequestHandler('tools/call' as any, async (request: any) => {
      const { name, arguments: args } = request.params;

      const agent = agents.find((a) => a.name === name);
      if (agent) {
        const text = await agent.runToText(args?.input ?? '');
        return { content: [{ type: 'text', text }] };
      }

      for (const ts of toolsets) {
        if (ts.has(name)) {
          const result = await ts.execute(name, args?.args ?? args ?? {}, { agentName: 'mcp' });
          return { content: [{ type: 'text', text: result.content }] };
        }
      }

      return { content: [{ type: 'text', text: `Tool '${name}' not found.` }], isError: true };
    });

    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    this.running = true;
    logger.info('MCP server endpoint started on stdio');
  }

  /** Close the MCP server. */
  async close(): Promise<void> {
    if (this.server) {
      try {
        await this.server.close();
      } catch {
        // ignore
      }
      this.server = null;
    }
    this.running = false;
  }

  /** Whether the endpoint is currently running. */
  get isRunning(): boolean {
    return this.running;
  }
}
