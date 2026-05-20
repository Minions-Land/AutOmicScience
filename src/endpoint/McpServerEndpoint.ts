import type { Agent } from '../agent/Agent.js';
import type { ToolSet } from '../toolset/ToolSet.js';
import { logger } from '../utils/logger.js';

/**
 * Exposes MedrixAI agents and toolsets AS an MCP server that external
 * clients (Claude Desktop, other MCP consumers) can connect to.
 *
 * Implements:
 * - tools/list + tools/call  — each agent and toolset tool becomes an MCP tool
 * - resources/list + resources/read — agent conversation history as resources
 * - prompts/list + prompts/get — agent system prompts
 */
export interface McpServerEndpoint {
  expose(agents: Agent[], toolsets: ToolSet[]): Promise<void>;
  close(): Promise<void>;
}

export interface McpServerEndpointOptions {
  /** Server name reported to MCP clients. Default: 'medrix-mcp' */
  name?: string;
  /** Server version. Default: '0.1.0' */
  version?: string;
  /** Transport to use: 'stdio' (default) or 'sse' */
  transport?: 'stdio' | 'sse';
  /** Port for SSE transport. Default: 3000 */
  port?: number;
}

/** One conversation turn stored per-agent for resource exposure. */
interface ConversationEntry {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

/**
 * Full MCP server endpoint using @modelcontextprotocol/sdk.
 *
 * Each registered agent is exposed as:
 *   - An MCP tool   (tools/list + tools/call)
 *   - A resource    (resources/list + resources/read) — conversation history
 *   - A prompt      (prompts/list + prompts/get) — agent system prompt
 *
 * ToolSet tools are also exposed as MCP tools.
 */
export class McpServerEndpointImpl implements McpServerEndpoint {
  private running = false;
  private server: any = null;
  private opts: Required<McpServerEndpointOptions>;

  /** In-memory conversation history per agent (agentName -> entries). */
  private history: Map<string, ConversationEntry[]> = new Map();

  constructor(opts: McpServerEndpointOptions = {}) {
    this.opts = {
      name: opts.name ?? 'medrix-mcp',
      version: opts.version ?? '0.1.0',
      transport: opts.transport ?? 'stdio',
      port: opts.port ?? 3000,
    };
  }

  /**
   * Start serving agents and toolsets over MCP.
   * Each agent becomes a tool, resource, and prompt entry.
   */
  async expose(agents: Agent[], toolsets: ToolSet[]): Promise<void> {
    const { Server } = await import('@modelcontextprotocol/sdk/server/index.js');
    const { ListToolsRequestSchema, CallToolRequestSchema } = await import(
      '@modelcontextprotocol/sdk/types.js'
    );
    const { ListResourcesRequestSchema, ReadResourceRequestSchema } = await import(
      '@modelcontextprotocol/sdk/types.js'
    );
    const { ListPromptsRequestSchema, GetPromptRequestSchema } = await import(
      '@modelcontextprotocol/sdk/types.js'
    );

    this.server = new Server(
      { name: this.opts.name, version: this.opts.version },
      { capabilities: { tools: {}, resources: {}, prompts: {} } },
    );

    // Initialise history buckets
    for (const agent of agents) {
      if (!this.history.has(agent.name)) {
        this.history.set(agent.name, []);
      }
    }

    // ── tools/list ────────────────────────────────────────────────────────────
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      const tools: any[] = [];

      for (const agent of agents) {
        tools.push({
          name: `agent__${agent.name}`,
          description: `Invoke MedrixAI agent: ${agent.name}`,
          inputSchema: {
            type: 'object',
            properties: {
              input: { type: 'string', description: 'The message to send to the agent.' },
            },
            required: ['input'],
          },
        });
      }

      for (const ts of toolsets) {
        for (const t of ts.list()) {
          tools.push({
            name: t.name,
            description: t.description,
            inputSchema: {
              type: 'object',
              properties: {
                args: { type: 'object', description: 'Arguments for the tool.' },
              },
            },
          });
        }
      }

      return { tools };
    });

    // ── tools/call ────────────────────────────────────────────────────────────
    this.server.setRequestHandler(CallToolRequestSchema, async (request: any) => {
      const { name, arguments: args } = request.params as { name: string; arguments: any };

      // Agent tool?
      if (name.startsWith('agent__')) {
        const agentName = name.slice('agent__'.length);
        const agent = agents.find((a) => a.name === agentName);
        if (!agent) {
          return {
            content: [{ type: 'text', text: `Agent '${agentName}' not found.` }],
            isError: true,
          };
        }

        const input = args?.input ?? '';
        // Record user turn
        this.history.get(agentName)!.push({
          role: 'user',
          content: input,
          timestamp: new Date().toISOString(),
        });

        const text = await agent.runToText(input);

        // Record assistant turn
        this.history.get(agentName)!.push({
          role: 'assistant',
          content: text,
          timestamp: new Date().toISOString(),
        });

        return { content: [{ type: 'text', text }] };
      }

      // ToolSet tool?
      for (const ts of toolsets) {
        if (ts.has(name)) {
          const result = await ts.execute(name, args?.args ?? args ?? {}, { agentName: 'mcp' });
          return { content: [{ type: 'text', text: result.content }] };
        }
      }

      return {
        content: [{ type: 'text', text: `Tool '${name}' not found.` }],
        isError: true,
      };
    });

    // ── resources/list ────────────────────────────────────────────────────────
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => {
      const resources = agents.map((agent) => ({
        uri: `medrix://agent/${agent.name}/history`,
        name: `${agent.name} conversation history`,
        description: `Conversation history for agent ${agent.name}`,
        mimeType: 'application/json',
      }));
      return { resources };
    });

    // ── resources/read ────────────────────────────────────────────────────────
    this.server.setRequestHandler(ReadResourceRequestSchema, async (request: any) => {
      const uri: string = request.params.uri;
      const match = uri.match(/^medrix:\/\/agent\/([^\/]+)\/history$/);
      if (!match) {
        throw new Error(`Unknown resource URI: ${uri}`);
      }
      const agentName = match[1];
      if (!this.history.has(agentName)) {
        throw new Error(`Agent '${agentName}' not found.`);
      }
      const entries = this.history.get(agentName)!;
      return {
        contents: [
          {
            uri,
            mimeType: 'application/json',
            text: JSON.stringify(entries, null, 2),
          },
        ],
      };
    });

    // ── prompts/list ──────────────────────────────────────────────────────────
    this.server.setRequestHandler(ListPromptsRequestSchema, async () => {
      const prompts = agents.map((agent) => ({
        name: `${agent.name}__system_prompt`,
        description: `System prompt used by agent ${agent.name}`,
      }));
      return { prompts };
    });

    // ── prompts/get ───────────────────────────────────────────────────────────
    this.server.setRequestHandler(GetPromptRequestSchema, async (request: any) => {
      const { name } = request.params as { name: string };
      const suffix = '__system_prompt';
      if (!name.endsWith(suffix)) {
        throw new Error(`Unknown prompt: ${name}`);
      }
      const agentName = name.slice(0, -suffix.length);
      const agent = agents.find((a) => a.name === agentName);
      if (!agent) {
        throw new Error(`Agent '${agentName}' not found.`);
      }
      // Access system prompt via public getter if available, else fallback
      const systemPrompt: string =
        typeof (agent as any).systemPrompt === 'string'
          ? (agent as any).systemPrompt
          : `You are ${agentName}, a helpful AI assistant.`;

      return {
        description: `System prompt for ${agentName}`,
        messages: [
          {
            role: 'user',
            content: { type: 'text', text: systemPrompt },
          },
        ],
      };
    });

    // ── Connect transport ──────────────────────────────────────────────────────
    if (this.opts.transport === 'sse') {
      const { SSEServerTransport } = await import('@modelcontextprotocol/sdk/server/sse.js');
      const { createServer } = await import('node:http');
      const httpServer = createServer();
      // Map session id -> transport for SSE
      const sessions = new Map<string, any>();

      httpServer.on('request', async (req: any, res: any) => {
        if (req.method === 'GET' && req.url === '/sse') {
          const transport = new SSEServerTransport('/message', res);
          sessions.set(transport.sessionId, transport);
          await this.server.connect(transport);
          res.on('close', () => sessions.delete(transport.sessionId));
        } else if (req.method === 'POST' && req.url?.startsWith('/message')) {
          const url = new URL(req.url, `http://localhost`);
          const sessionId = url.searchParams.get('sessionId') ?? '';
          const t = sessions.get(sessionId);
          if (t) await t.handlePostMessage(req, res);
          else { res.writeHead(404); res.end(); }
        } else {
          res.writeHead(404);
          res.end();
        }
      });

      await new Promise<void>((resolve) => httpServer.listen(this.opts.port, resolve));
      logger.info(`MCP SSE server listening on port ${this.opts.port}`);
    } else {
      const { StdioServerTransport } = await import(
        '@modelcontextprotocol/sdk/server/stdio.js'
      );
      const transport = new StdioServerTransport();
      await this.server.connect(transport);
      logger.info('MCP stdio server endpoint started');
    }

    this.running = true;
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

  get isRunning(): boolean {
    return this.running;
  }
}
