import type { Agent } from '../agent/Agent.js';
import type { ToolSet } from '../toolset/ToolSet.js';
import { logger } from '../utils/logger.js';
import { McpServerEndpointImpl } from './McpServerEndpoint.js';
import { HttpEndpoint } from './HttpEndpoint.js';

export interface EndpointInfo {
  name: string;
  type: 'mcp' | 'http';
  /** Agents hosted by this endpoint. */
  agentNames: string[];
  running: boolean;
}

type AnyEndpoint = McpServerEndpointImpl | HttpEndpoint;

interface HubEntry {
  name: string;
  type: 'mcp' | 'http';
  endpoint: AnyEndpoint;
  agents: Agent[];
  toolsets: ToolSet[];
}

/**
 * Hub — registry that holds multiple McpServerEndpoints and HttpEndpoints.
 *
 * Methods:
 *   register(name, endpoint, agents?, toolsets?) — add endpoint to registry
 *   start()  — start all registered endpoints
 *   stop()   — stop all registered endpoints
 *   list()   — return info about all endpoints
 *   route(agentName) — find which endpoint hosts a given agent
 */
export class Hub {
  private entries: Map<string, HubEntry> = new Map();

  /**
   * Register an endpoint.
   *
   * @param name     - Unique name for this endpoint within the hub.
   * @param endpoint - McpServerEndpointImpl or HttpEndpoint instance.
   * @param agents   - Agents to associate with this endpoint.
   * @param toolsets - ToolSets to associate with this endpoint.
   */
  register(
    name: string,
    endpoint: McpServerEndpointImpl | HttpEndpoint,
    agents: Agent[] = [],
    toolsets: ToolSet[] = [],
  ): this {
    if (this.entries.has(name)) {
      throw new Error(`Endpoint '${name}' is already registered in Hub.`);
    }
    const type: 'mcp' | 'http' =
      endpoint instanceof McpServerEndpointImpl ? 'mcp' : 'http';
    this.entries.set(name, { name, type, endpoint, agents, toolsets });
    logger.info(`Hub: registered endpoint '${name}' (${type})`);
    return this;
  }

  /**
   * Start all registered endpoints.
   * MCP endpoints call expose(); HTTP endpoints call listen() on port 3001+index.
   */
  async start(): Promise<void> {
    let httpPort = 3001;
    const tasks: Promise<void>[] = [];

    for (const entry of this.entries.values()) {
      if (entry.endpoint instanceof McpServerEndpointImpl) {
        tasks.push(
          entry.endpoint.expose(entry.agents, entry.toolsets).then(() => {
            logger.info(`Hub: MCP endpoint '${entry.name}' started`);
          }),
        );
      } else {
        const port = httpPort++;
        tasks.push(
          (entry.endpoint as HttpEndpoint).listen(port).then(() => {
            logger.info(`Hub: HTTP endpoint '${entry.name}' started on port ${port}`);
          }),
        );
      }
    }

    await Promise.all(tasks);
  }

  /** Stop all registered endpoints. */
  async stop(): Promise<void> {
    const tasks: Promise<void>[] = [];
    for (const entry of this.entries.values()) {
      tasks.push(
        entry.endpoint.close().then(() => {
          logger.info(`Hub: endpoint '${entry.name}' stopped`);
        }),
      );
    }
    await Promise.all(tasks);
  }

  /** List all registered endpoints with basic metadata. */
  list(): EndpointInfo[] {
    return Array.from(this.entries.values()).map((e) => ({
      name: e.name,
      type: e.type,
      agentNames: e.agents.map((a) => a.name),
      running: true, // Hub tracks started state externally
    }));
  }

  /**
   * Find which endpoint hosts the given agent.
   * Returns the endpoint name and instance, or undefined if not found.
   */
  route(agentName: string): { name: string; endpoint: AnyEndpoint } | undefined {
    for (const entry of this.entries.values()) {
      if (entry.agents.some((a) => a.name === agentName)) {
        return { name: entry.name, endpoint: entry.endpoint };
      }
    }
    return undefined;
  }

  /** Number of registered endpoints. */
  get size(): number {
    return this.entries.size;
  }
}
