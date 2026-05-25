import { createServer, type Server as NodeHttpServer, type IncomingMessage, type ServerResponse } from 'node:http';
import type { Agent } from '../agent/Agent.js';
import type { ToolSet } from '../toolset/ToolSet.js';
import { logger } from '../utils/logger.js';

/**
 * REST/SSE HTTP endpoint for invoking AutOmicScience agents over HTTP.
 *
 * Routes:
 *   GET  /health              — health check
 *   GET  /api/agents          — list agents with metadata
 *   POST /api/chat            — SSE streaming chat with an agent
 *   GET  /api/tools           — list all tools across all toolsets
 *   POST /api/tools/:name     — call a specific tool (used by ToolsetProxy)
 */
export class HttpEndpoint {
  private server: NodeHttpServer | null = null;
  private agents: Map<string, Agent>;
  private toolsets: ToolSet[];

  constructor(agents: Agent[] = [], toolsets: ToolSet[] = []) {
    this.agents = new Map(agents.map((a) => [a.name, a]));
    this.toolsets = toolsets;
  }

  /** Register an agent after construction. */
  addAgent(agent: Agent): void {
    this.agents.set(agent.name, agent);
  }

  /** Register a toolset after construction. */
  addToolset(ts: ToolSet): void {
    this.toolsets.push(ts);
  }

  /** Start listening on the given port. */
  async listen(port: number): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
        try {
          await this.handleRequest(req, res);
        } catch (err) {
          logger.warn('HttpEndpoint request error:', (err as Error).message);
          if (!res.headersSent) {
            res.writeHead(500, { 'Content-Type': 'application/json' });
          }
          res.end(JSON.stringify({ error: (err as Error).message }));
        }
      });
      this.server.on('error', reject);
      this.server.listen(port, () => {
        logger.info(`HttpEndpoint listening on port ${port}`);
        resolve();
      });
    });
  }

  /** Close the HTTP server. */
  async close(): Promise<void> {
    return new Promise((resolve) => {
      if (!this.server) { resolve(); return; }
      this.server.close(() => resolve());
    });
  }

  private async handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);

    // CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

    // GET /health
    if (req.method === 'GET' && url.pathname === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok', timestamp: new Date().toISOString() }));
      return;
    }

    // GET /api/agents
    if (req.method === 'GET' && url.pathname === '/api/agents') {
      const agents = Array.from(this.agents.values()).map((a) => ({
        name: a.name,
      }));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ agents }));
      return;
    }

    // GET /api/tools
    if (req.method === 'GET' && url.pathname === '/api/tools') {
      const tools: Array<{ name: string; description: string; toolset: string }> = [];
      for (const ts of this.toolsets) {
        for (const t of ts.list()) {
          tools.push({ name: t.name, description: t.description, toolset: ts.name });
        }
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ tools }));
      return;
    }

    // POST /api/tools/:name
    const toolMatch = url.pathname.match(/^\/api\/tools\/(.+)$/);
    if (req.method === 'POST' && toolMatch) {
      const toolName = decodeURIComponent(toolMatch[1]);
      const body = await this.readBody(req);
      const args = body ? (JSON.parse(body) as Record<string, unknown>) : {};

      for (const ts of this.toolsets) {
        if (ts.has(toolName)) {
          const result = await ts.execute(toolName, args, { agentName: 'http' });
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ result: result.content }));
          return;
        }
      }

      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: `Tool '${toolName}' not found` }));
      return;
    }

    // POST /api/chat  — SSE streaming
    if (req.method === 'POST' && url.pathname === '/api/chat') {
      const body = await this.readBody(req);
      const { agent: agentName, message } = JSON.parse(body) as { agent: string; message: string };

      const agent = this.agents.get(agentName);
      if (!agent) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: `Agent '${agentName}' not found` }));
        return;
      }

      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });

      for await (const ev of agent.run(message)) {
        const data = JSON.stringify(ev);
        res.write(`event: ${ev.type}\ndata: ${data}\n\n`);
      }

      res.write('event: close\ndata: {}\n\n');
      res.end();
      return;
    }

    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
  }

  private readBody(req: IncomingMessage): Promise<string> {
    return new Promise((resolve, reject) => {
      let data = '';
      req.on('data', (chunk) => (data += chunk.toString()));
      req.on('end', () => resolve(data));
      req.on('error', reject);
    });
  }
}
