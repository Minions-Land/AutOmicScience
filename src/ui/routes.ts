/**
 * Stub route definitions for the Novaeve API.
 * These define the shape of the HTTP API surface. In production,
 * wire these into Express, Hono, or Fastify.
 */

export interface RouteHandler {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  description: string;
  handler: (req: RouteRequest) => Promise<RouteResponse>;
}

export interface RouteRequest {
  params: Record<string, string>;
  query: Record<string, string>;
  body: unknown;
}

export interface RouteResponse {
  status: number;
  headers?: Record<string, string>;
  body: unknown;
}

/**
 * POST /api/chat — Send a message and receive an SSE stream of AgentEvents.
 */
export const chatRoute: RouteHandler = {
  method: 'POST',
  path: '/api/chat',
  description: 'Send a message to the active agent and stream back AgentEvents via SSE.',
  async handler(_req: RouteRequest): Promise<RouteResponse> {
    return { status: 501, body: { status: 'not_implemented' } };
  },
};

/**
 * GET /api/agents — List all registered agents.
 */
export const listAgentsRoute: RouteHandler = {
  method: 'GET',
  path: '/api/agents',
  description: 'List all registered agents with their names and capabilities.',
  async handler(_req: RouteRequest): Promise<RouteResponse> {
    return { status: 501, body: { status: 'not_implemented' } };
  },
};

/**
 * POST /api/agents/:name/run — Run a specific agent by name.
 */
export const runAgentRoute: RouteHandler = {
  method: 'POST',
  path: '/api/agents/:name/run',
  description: 'Invoke a specific agent by name with the given input.',
  async handler(_req: RouteRequest): Promise<RouteResponse> {
    return { status: 501, body: { status: 'not_implemented' } };
  },
};

/** All defined routes. */
export const routes: RouteHandler[] = [chatRoute, listAgentsRoute, runAgentRoute];
