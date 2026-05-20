import { createServer, type Server as NodeHttpServer, type IncomingMessage, type ServerResponse } from 'node:http';
import type { Agent } from '../agent/Agent.js';
import { logger } from '../utils/logger.js';

/**
 * REST/SSE HTTP endpoint for invoking MedrixAI agents over HTTP.
 */
export interface HttpEndpoint {
  /** Start listening on the given port. */
  listen(port: number): Promise<void>;
  /** Close the HTTP server. */
  close(): Promise<void>;
}

/**
 * Real HTTP endpoint implementation with SSE streaming.
 *
 * Routes:
 * - POST /api/chat — body `{ agent: string, message: string }` -> SSE stream of AgentEvents
 * - GET /api/agents — JSON list of agent names
 */
export class StubHttpEndpoint implements HttpEndpoint {
  private server: NodeHttpServer | null = null;
  private agents: Map<string, Agent>;

  constructor(agents: Agent[] = []) {
    this.agents = new Map(agents.map((a) => [a.name, a]));
  }

  /** Register an agent after construction. */
  addAgent(agent: Agent): void {
    this.agents.set(agent.name, agent);
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
      if (!this.server) {
        resolve();
        return;
      }
      this.server.close(() => resolve());
    });
  }

  private async handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);

    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    if (req.method === 'GET' && url.pathname === '/api/agents') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ agents: Array.from(this.agents.keys()) }));
      return;
    }

    if (req.method === 'POST' && url.pathname === '/api/chat') {
      const body = await this.readBody(req);
      const { agent: agentName, message } = JSON.parse(body) as { agent: string; message: string };

      const agent = this.agents.get(agentName);
      if (!agent) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: `Agent '${agentName}' not found` }));
        return;
      }

      // SSE streaming
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
