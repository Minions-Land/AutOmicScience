import { createServer, type Server as HttpServer, type IncomingMessage, type ServerResponse } from 'node:http';

/**
 * Interface for a development/production UI server that serves
 * the Novaeve web interface and API endpoints.
 */
export interface UIServer {
  /** Start listening on the given port. */
  start(port: number): Promise<void>;
  /** Gracefully stop the server. */
  stop(): Promise<void>;
}

/**
 * Minimal development server that serves a placeholder HTML page.
 * Replace with a full Vite/Next.js build for production.
 */
export class DevServer implements UIServer {
  private server: HttpServer | null = null;

  /** Start listening on the given port. */
  async start(port: number): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server = createServer((_req: IncomingMessage, res: ServerResponse) => {
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        res.end(PLACEHOLDER_HTML);
      });
      this.server.on('error', reject);
      this.server.listen(port, () => resolve());
    });
  }

  /** Gracefully stop the server. */
  async stop(): Promise<void> {
    return new Promise((resolve) => {
      if (!this.server) {
        resolve();
        return;
      }
      this.server.close(() => resolve());
    });
  }
}

const PLACEHOLDER_HTML = `<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Novaeve</title></head>
<body>
<h1>Novaeve Agent</h1>
<p>UI server running. Connect a frontend or use the API endpoints.</p>
</body>
</html>`;
