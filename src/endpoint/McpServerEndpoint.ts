/**
 * Exposes Novaeve agents and toolsets AS an MCP server that external
 * clients (Claude Desktop, other MCP consumers) can connect to.
 */
export interface McpServerEndpoint {
  /**
   * Start serving the given agents and toolsets over the MCP protocol.
   * @param agents - Agents to expose as MCP resources/tools.
   * @param toolsets - ToolSets to expose as MCP tools.
   */
  expose(agents: Array<{ name: string }>, toolsets: Array<{ name: string }>): Promise<void>;
  /** Close the MCP server and release resources. */
  close(): Promise<void>;
}

/**
 * Stub MCP server endpoint. Does not actually start an MCP server.
 */
export class StubMcpServerEndpoint implements McpServerEndpoint {
  private running = false;

  /** Start serving (stub — sets running flag). */
  async expose(
    _agents: Array<{ name: string }>,
    _toolsets: Array<{ name: string }>,
  ): Promise<void> {
    this.running = true;
  }

  /** Close the MCP server. */
  async close(): Promise<void> {
    this.running = false;
  }

  /** Whether the endpoint is currently running. */
  get isRunning(): boolean {
    return this.running;
  }
}
