import { logger } from '../utils/logger.js';
import type { OpenAIToolDef } from '../types.js';

/**
 * A ToolSet-compatible proxy that forwards tool calls to a remote HTTP endpoint.
 *
 * Mirrors PantheonOS toolset_proxy.py — proxies getTools() and execute() via HTTP.
 */
export interface ToolProxy {
  name: string;
  description: string;
  /** Execute the tool with the given arguments. */
  execute(args: Record<string, unknown>): Promise<unknown>;
}

export interface ToolsetProxyOptions {
  /** Base URL of the remote MedrixAI HttpEndpoint (e.g. http://localhost:4000). */
  baseUrl: string;
  /** Name of the toolset on the remote server. */
  toolsetName: string;
  /** Fetch timeout in ms. Default: 30_000. */
  timeoutMs?: number;
}

/**
 * Proxies tool calls to a remote HttpEndpoint over HTTP.
 *
 * Usage:
 *   const proxy = new ToolsetProxy({ baseUrl: 'http://localhost:4000', toolsetName: 'bio' });
 *   const tools = await proxy.getTools();
 *   const result = await tools[0].execute({ query: 'BRCA1' });
 */
export class ToolsetProxy {
  private readonly baseUrl: string;
  private readonly toolsetName: string;
  private readonly timeoutMs: number;
  private _tools: ToolProxy[] | null = null;

  constructor(opts: ToolsetProxyOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/$/, '');
    this.toolsetName = opts.toolsetName;
    this.timeoutMs = opts.timeoutMs ?? 30_000;
  }

  /**
   * Fetch the list of tools from GET {baseUrl}/api/tools.
   * Results are cached; call invalidate() to refresh.
   */
  async getTools(): Promise<ToolProxy[]> {
    if (this._tools) return this._tools;

    const url = `${this.baseUrl}/api/tools`;
    logger.debug(`ToolsetProxy: fetching tools from ${url}`);

    const res = await this._fetch(url, { method: 'GET' });
    if (!res.ok) {
      throw new Error(`ToolsetProxy.getTools failed: ${res.status} ${res.statusText}`);
    }

    const body = (await res.json()) as { tools?: Array<{ name: string; description: string }> };
    const rawTools = body.tools ?? [];

    this._tools = rawTools.map((t) => this._makeProxy(t.name, t.description));
    logger.debug(`ToolsetProxy: loaded ${this._tools.length} tools from ${this.baseUrl}`);
    return this._tools;
  }

  /** Clear the cached tool list so the next getTools() call re-fetches. */
  invalidate(): void {
    this._tools = null;
  }

  /**
   * Directly execute a named tool.
   * Equivalent to POST {baseUrl}/api/tools/{name} with body { args }.
   */
  async execute(name: string, args: Record<string, unknown> = {}): Promise<unknown> {
    const url = `${this.baseUrl}/api/tools/${encodeURIComponent(name)}`;
    logger.debug(`ToolsetProxy: executing ${name} at ${url}`);

    const res = await this._fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ args }),
    });

    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`ToolsetProxy.execute('${name}') failed: ${res.status} ${text}`);
    }

    return res.json();
  }

  /** Build an OpenAI-style tool definition list (for passing to an Agent). */
  async toOpenAITools(): Promise<OpenAIToolDef[]> {
    const tools = await this.getTools();
    return tools.map((t) => ({
      type: 'function' as const,
      function: {
        name: t.name,
        description: t.description,
        parameters: { type: 'object', properties: {}, additionalProperties: true },
      },
    }));
  }

  // ── private ───────────────────────────────────────────────────────────────

  private _makeProxy(name: string, description: string): ToolProxy {
    const self = this;
    return {
      name,
      description,
      async execute(args: Record<string, unknown>) {
        return self.execute(name, args);
      },
    };
  }

  private _fetch(url: string, init: RequestInit): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    return fetch(url, { ...init, signal: controller.signal }).finally(() =>
      clearTimeout(timer),
    );
  }
}
