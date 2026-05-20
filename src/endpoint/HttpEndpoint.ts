import { createServer, type Server as NodeHttpServer, type IncomingMessage, type ServerResponse } from 'node:http';

/**
 * REST/SSE HTTP endpoint for invoking Novaeve agents over HTTP.
 */
export interface HttpEndpoint {
  /** Start listening on the given port. */
  listen(port: number): Promise<void>;
  /** Close the HTTP server. */
  close(): Promise<void>;
}

/**
 * Stub HTTP endpoint implementation.
 * Returns 501 for all requests until wired to real agent dispatch.
 */
export class StubHttpEndpoint implements HttpEndpoint {
  private server: NodeHttpServer | null = null;

  /** Start listening on the given port. */
  async listen(port: number): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server = createServer((_req: IncomingMessage, res: ServerResponse) => {
        res.writeHead(501, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'not_implemented' }));
      });
      this.server.on('error', reject);
      this.server.listen(port, () => resolve());
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
}
