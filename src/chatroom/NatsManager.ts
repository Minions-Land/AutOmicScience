import { spawn, type ChildProcess } from 'node:child_process';
import { createConnection } from 'node:net';
import { existsSync, mkdirSync, readFileSync, writeFileSync, unlinkSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { homedir } from 'node:os';
import { which } from '../utils/which.js';

// --- Types ---

export interface NatsServerInfo {
  tcpUrl: string;
  wsUrl: string;
  httpUrl: string;
  configFile: string | null;
  logFile: string | null;
  pid: number | null;
  reused?: boolean;
  external?: boolean;
}

export interface NatsManagerOptions {
  tcpPort?: number;
  wsPort?: number;
  httpPort?: number;
  workDir?: string;
  dataDir?: string;
}

interface InstanceData {
  pid: number;
  tcpPort: number;
  wsPort: number;
  httpPort: number;
  configFile?: string;
  logFile?: string;
  tcpUrl: string;
  wsUrl: string;
  httpUrl: string;
}

// --- Helpers ---

function isPortInUse(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const conn = createConnection({ port, host: '127.0.0.1' });
    conn.on('connect', () => {
      conn.destroy();
      resolve(true);
    });
    conn.on('error', () => {
      resolve(false);
    });
    conn.setTimeout(500, () => {
      conn.destroy();
      resolve(false);
    });
  });
}

async function findAvailablePort(startPort: number, maxAttempts = 100): Promise<number> {
  for (let port = startPort; port < startPort + maxAttempts; port++) {
    const inUse = await isPortInUse(port);
    if (!inUse) return port;
  }
  throw new Error(`No available port found in range ${startPort}-${startPort + maxAttempts}`);
}

async function httpGet(url: string, timeoutMs = 2000): Promise<{ status: number; body: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { signal: controller.signal });
    const body = await resp.text();
    return { status: resp.status, body };
  } finally {
    clearTimeout(timer);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// --- NatsManager ---

/**
 * Manages the lifecycle of a local NATS server subprocess.
 *
 * Responsibilities:
 * - Detect nats-server binary (PATH or bundled)
 * - Check port availability and auto-allocate alternatives
 * - Generate config from template with runtime substitution
 * - Start/stop subprocess with proper cleanup
 * - Health monitoring via HTTP endpoint
 * - Instance tracking for multi-instance isolation
 * - Detect and reuse existing NATS servers
 */
export class NatsManager {
  private tcpPort: number;
  private wsPort: number;
  private httpPort: number;
  private workDir: string;
  private dataDir: string;
  private process: ChildProcess | null = null;
  private configFile: string | null = null;
  private instanceFile: string;

  constructor(opts?: NatsManagerOptions) {
    this.tcpPort = opts?.tcpPort ?? 4222;
    this.wsPort = opts?.wsPort ?? 8080;
    this.httpPort = opts?.httpPort ?? 8222;
    this.workDir = opts?.workDir ?? process.cwd();
    this.dataDir = opts?.dataDir ?? join(homedir(), '.medrix');
    this.instanceFile = join(this.dataDir, '.nats-instance.json');
  }

  // --- Binary Detection ---

  checkBinaryAvailable(): { available: boolean; path: string } {
    const binary = which('nats-server');
    if (binary) {
      return { available: true, path: binary };
    }
    return {
      available: false,
      path:
        'nats-server binary not found.\n\n' +
        'Installation options:\n' +
        '1. Via Go: go install github.com/nats-io/nats-server/v2@latest\n' +
        '2. Via Homebrew (macOS): brew install nats-server\n' +
        '3. Via Docker: docker run -p 4222:4222 -p 8080:8080 nats:alpine',
    };
  }

  // --- Port Management ---

  async checkPortsAvailable(): Promise<{ allFree: boolean; occupied: number[] }> {
    const occupied: number[] = [];
    for (const port of [this.tcpPort, this.wsPort, this.httpPort]) {
      if (await isPortInUse(port)) {
        occupied.push(port);
      }
    }
    return { allFree: occupied.length === 0, occupied };
  }

  async autoConfigurePorts(): Promise<void> {
    const { allFree, occupied } = await this.checkPortsAvailable();
    if (allFree) return;

    if (occupied.includes(this.tcpPort)) {
      this.tcpPort = await findAvailablePort(4222);
    }
    if (occupied.includes(this.wsPort)) {
      this.wsPort = await findAvailablePort(8080);
    }
    if (occupied.includes(this.httpPort)) {
      this.httpPort = await findAvailablePort(8222);
    }
  }

  // --- Config Generation ---

  private generateConfig(): string {
    const jetstreamDir = join(this.workDir, '.nats-jetstream');
    mkdirSync(jetstreamDir, { recursive: true });

    const config = `
# MedrixAI NATS Server Configuration (auto-generated)
server_name: medrix-nats-local

listen: 0.0.0.0:${this.tcpPort}

http_port: ${this.httpPort}

websocket {
  port: ${this.wsPort}
  no_tls: true
}

jetstream {
  store_dir: "${jetstreamDir.replace(/\\/g, '/')}"
  max_mem: 256MB
  max_file: 1GB
}

# Logging
debug: false
trace: false
logtime: true
`;

    const configPath = join(this.workDir, '.nats-config.conf');
    writeFileSync(configPath, config, 'utf-8');
    this.configFile = configPath;
    return configPath;
  }

  // --- Detect Existing Instance ---

  async detectExisting(): Promise<NatsServerInfo | null> {
    if (!existsSync(this.instanceFile)) return null;

    try {
      const raw = readFileSync(this.instanceFile, 'utf-8');
      const data: InstanceData = JSON.parse(raw);

      if (!data.pid || !data.tcpPort || !data.wsPort || !data.httpPort) {
        return null;
      }

      // Check if process is alive
      try {
        process.kill(data.pid, 0);
      } catch {
        // Process not alive, clean up stale file
        try { unlinkSync(this.instanceFile); } catch { /* ignore */ }
        return null;
      }

      // Check HTTP healthz
      try {
        const { status } = await httpGet(`http://localhost:${data.httpPort}/healthz`);
        if (status !== 200) return null;
      } catch {
        return null;
      }

      // Check TCP connectivity
      const tcpOk = await isPortInUse(data.tcpPort);
      if (!tcpOk) return null;

      // All checks passed - reuse
      this.tcpPort = data.tcpPort;
      this.wsPort = data.wsPort;
      this.httpPort = data.httpPort;

      return {
        tcpUrl: `nats://localhost:${data.tcpPort}`,
        wsUrl: `ws://127.0.0.1:${data.wsPort}`,
        httpUrl: `http://localhost:${data.httpPort}`,
        configFile: data.configFile ?? null,
        logFile: data.logFile ?? null,
        pid: data.pid,
        reused: true,
      };
    } catch {
      try { unlinkSync(this.instanceFile); } catch { /* ignore */ }
      return null;
    }
  }

  // --- Detect External NATS ---

  async detectExternal(): Promise<NatsServerInfo | null> {
    const TCP_DEFAULT = 4222;
    const HTTP_DEFAULT = 8222;
    const WS_CANDIDATES = [8080, 9222, 4223];

    // Check HTTP healthz
    try {
      const { status } = await httpGet(`http://localhost:${HTTP_DEFAULT}/healthz`, 1500);
      if (status !== 200) return null;
    } catch {
      return null;
    }

    // Check TCP
    const tcpOk = await isPortInUse(TCP_DEFAULT);
    if (!tcpOk) return null;

    // Find working WS port
    let chosenWs: number | null = null;
    for (const port of WS_CANDIDATES) {
      if (await isPortInUse(port)) {
        chosenWs = port;
        break;
      }
    }

    if (chosenWs === null) return null;

    this.tcpPort = TCP_DEFAULT;
    this.wsPort = chosenWs;
    this.httpPort = HTTP_DEFAULT;

    return {
      tcpUrl: `nats://localhost:${TCP_DEFAULT}`,
      wsUrl: `ws://127.0.0.1:${chosenWs}`,
      httpUrl: `http://localhost:${HTTP_DEFAULT}`,
      configFile: null,
      logFile: null,
      pid: null,
      reused: true,
      external: true,
    };
  }

  // --- Start ---

  async start(): Promise<NatsServerInfo> {
    // 1. Try to reuse existing instance
    const existing = await this.detectExisting();
    if (existing) return existing;

    // 2. Try to detect external NATS
    const external = await this.detectExternal();
    if (external) return external;

    // 3. Check binary
    const { available, path: binaryPath } = this.checkBinaryAvailable();
    if (!available) {
      throw new Error(`Cannot start NATS: ${binaryPath}`);
    }

    // 4. Auto-configure ports
    await this.autoConfigurePorts();

    // 5. Generate config
    const configPath = this.generateConfig();

    // 6. Create log file path
    const logFile = join(this.workDir, '.nats-server.log');

    // 7. Start subprocess
    const { openSync, closeSync } = await import('node:fs');
    const logFd = openSync(logFile, 'w');

    this.process = spawn(binaryPath, ['-c', configPath], {
      stdio: ['ignore', logFd, logFd],
      detached: false,
    });

    closeSync(logFd);

    const pid = this.process.pid ?? 0;

    // Handle unexpected exit
    this.process.on('exit', (code) => {
      if (code !== null && code !== 0) {
        console.error(`[NATS] Server exited with code ${code}`);
      }
      this.process = null;
    });

    // 8. Wait for ready
    const ready = await this.waitForReady(30000);
    if (!ready) {
      this.process?.kill();
      this.process = null;
      throw new Error(
        `NATS server failed to start within timeout.\nLog file: ${logFile}`,
      );
    }

    // 9. Build server info
    const serverInfo: NatsServerInfo = {
      tcpUrl: `nats://localhost:${this.tcpPort}`,
      wsUrl: `ws://127.0.0.1:${this.wsPort}`,
      httpUrl: `http://localhost:${this.httpPort}`,
      configFile: configPath,
      logFile,
      pid,
    };

    // 10. Write instance file
    this.writeInstanceFile(serverInfo);

    return serverInfo;
  }

  // --- Health Check ---

  async waitForReady(timeoutMs = 30000): Promise<boolean> {
    const start = Date.now();
    let httpReady = false;
    let tcpReady = false;

    while (Date.now() - start < timeoutMs) {
      // Check if process died
      if (this.process && this.process.exitCode !== null) {
        return false;
      }

      // Check HTTP healthz
      if (!httpReady) {
        try {
          const { status } = await httpGet(
            `http://localhost:${this.httpPort}/healthz`,
            1000,
          );
          if (status === 200) httpReady = true;
        } catch {
          // Not ready yet
        }
      }

      // Check TCP
      if (httpReady && !tcpReady) {
        tcpReady = await isPortInUse(this.tcpPort);
      }

      if (httpReady && tcpReady) return true;

      await sleep(500);
    }

    return false;
  }

  // --- Stop ---

  async stop(): Promise<void> {
    if (!this.process) return;

    if (this.process.exitCode === null) {
      // Send SIGTERM
      this.process.kill('SIGTERM');

      // Wait up to 5 seconds
      const exited = await Promise.race([
        new Promise<boolean>((resolve) => {
          this.process!.on('exit', () => resolve(true));
        }),
        sleep(5000).then(() => false),
      ]);

      if (!exited && this.process.exitCode === null) {
        this.process.kill('SIGKILL');
      }
    }

    this.process = null;

    // Clean up config file
    if (this.configFile && existsSync(this.configFile)) {
      try { unlinkSync(this.configFile); } catch { /* ignore */ }
    }

    // Remove instance file
    if (existsSync(this.instanceFile)) {
      try { unlinkSync(this.instanceFile); } catch { /* ignore */ }
    }
  }

  // --- Instance Tracking ---

  private writeInstanceFile(serverInfo: NatsServerInfo): void {
    try {
      mkdirSync(this.dataDir, { recursive: true });
      const data: InstanceData = {
        pid: serverInfo.pid ?? 0,
        tcpPort: this.tcpPort,
        wsPort: this.wsPort,
        httpPort: this.httpPort,
        configFile: serverInfo.configFile ?? undefined,
        logFile: serverInfo.logFile ?? undefined,
        tcpUrl: serverInfo.tcpUrl,
        wsUrl: serverInfo.wsUrl,
        httpUrl: serverInfo.httpUrl,
      };
      writeFileSync(this.instanceFile, JSON.stringify(data, null, 2), 'utf-8');
    } catch {
      /* non-critical */
    }
  }

  cleanupInstanceFile(): void {
    if (existsSync(this.instanceFile)) {
      try { unlinkSync(this.instanceFile); } catch { /* ignore */ }
    }
  }

  // --- Getters ---

  get ports(): { tcp: number; ws: number; http: number } {
    return { tcp: this.tcpPort, ws: this.wsPort, http: this.httpPort };
  }

  get isRunning(): boolean {
    return this.process !== null && this.process.exitCode === null;
  }
}
