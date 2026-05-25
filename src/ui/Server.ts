import { createServer, type Server as HttpServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Agent } from '../agent/index.js';
import { CommandRegistry } from '../commands/index.js';
import { HookManager } from '../hooks/index.js';
import { FilePermissionStore, PermissionManager, parsePermissionRule } from '../permissions/index.js';
import type { PermissionMode } from '../permissions/index.js';
import { AOSPluginRegistry, PluginLoader } from '../plugin/index.js';
import { defaultModel } from '../provider/ModelSelector.js';
import { loadProjectInstructions } from '../project/index.js';
import { FileSessionStore } from '../session/index.js';
import { createDefaultToolSet } from '../toolset/BuiltinToolSets.js';
import { FileTaskManager } from '../task/index.js';
import type { AgentEvent } from '../types.js';
import { listKnownModels } from '../utils/modelDiscovery.js';
import { AOS_SYSTEM_PROMPT } from '../agent/prompts/AOSSystemPrompt.js';
import { APP_HTML } from './AppHtml.js';
import { AOSCompat } from './AOSCompat.js';

const UI_DIR = path.dirname(fileURLToPath(import.meta.url));

export interface UIServer {
  start(port: number): Promise<void>;
  stop(): Promise<void>;
}

export interface DevServerOptions {
  agent?: Agent;
  rootDir?: string;
  sessionsDir?: string;
  permissionsFile?: string;
  tasksFile?: string;
  pluginsFile?: string;
  enableAOSCompat?: boolean;
  aosCompatDataDir?: string;
  aosServiceIdHash?: string;
}

export class DevServer implements UIServer {
  private server: HttpServer | null = null;
  private readonly rootDir: string;
  private readonly permissionManager: PermissionManager;
  private readonly permissionStore: FilePermissionStore;
  private readonly taskManager: FileTaskManager;
  private readonly sessionStore: FileSessionStore;
  private readonly pluginLoader: PluginLoader;
  private readonly pluginRegistry: AOSPluginRegistry;
  private readonly commands: CommandRegistry;
  private readonly hooks: HookManager;
  private readonly agent: Agent;
  private readonly aosCompat: AOSCompat;
  private readonly modelCandidates: string[];
  private loadedPlugins: string[] = [];
  private hookEvents: { event: string; at: string; data: unknown }[] = [];

  constructor(opts: DevServerOptions = {}) {
    this.rootDir = opts.rootDir ?? process.cwd();
    this.permissionManager = new PermissionManager({ mode: 'default', askFallback: 'deny' });
    this.permissionStore = new FilePermissionStore(opts.permissionsFile ?? path.join(os.homedir(), '.aos', 'permissions.json'));
    this.taskManager = new FileTaskManager(opts.tasksFile ?? path.join(os.homedir(), '.aos', 'tasks.json'));
    this.sessionStore = new FileSessionStore(opts.sessionsDir ?? path.join(os.homedir(), '.aos', 'sessions'));
    this.pluginLoader = new PluginLoader([
      path.join(os.homedir(), '.aos', 'plugins'),
      path.join(this.rootDir, 'plugins'),
    ]);
    this.pluginRegistry = new AOSPluginRegistry(opts.pluginsFile ?? path.join(os.homedir(), '.aos', 'plugins.json'));
    this.commands = new CommandRegistry();
    this.hooks = new HookManager();
    this.modelCandidates = unique([
      defaultModel(),
      'normal',
      'gpt-5.5',
      'gpt-5.4',
      'gpt-5.2',
      ...listKnownModels(),
    ]);
    this.installHookRecorder();

    const toolset = createDefaultToolSet({
      rootDir: this.rootDir,
      permissionManager: this.permissionManager,
      taskManager: this.taskManager,
    });

    this.agent = opts.agent ?? new Agent({
      name: 'aos-ui',
      model: defaultModel(),
      toolset,
      hooks: this.hooks,
      projectInstructions: { cwd: this.rootDir },
      systemPrompt: AOS_SYSTEM_PROMPT,
    });
    this.aosCompat = new AOSCompat({
      rootDir: this.rootDir,
      agent: this.agent,
      dataDir: opts.aosCompatDataDir,
      serviceIdHash: opts.aosServiceIdHash,
      enableNats: opts.enableAOSCompat ?? process.env.AOS_AOS_COMPAT === '1',
      models: () => [...this.modelCandidates],
    });
  }

  async start(port: number): Promise<void> {
    const stored = await this.permissionStore.load();
    this.permissionManager.setMode(stored.mode);
    for (const rule of stored.rules) this.permissionManager.addRule(rule);
    for (const entry of await this.pluginRegistry.list()) {
      if (!entry.enabled) continue;
      await this.loadPluginTarget(entry.target, false).catch(() => {});
    }
    await this.aosCompat.start();
    return new Promise((resolve, reject) => {
      this.server = createServer((req, res) => {
        this.handle(req, res).catch((err) => {
          this.sendJson(res, 500, { error: err instanceof Error ? err.message : String(err) });
        });
      });
      this.server.on('error', reject);
      this.server.listen(port, () => resolve());
    });
  }

  async stop(): Promise<void> {
    await this.aosCompat.stop();
    return new Promise((resolve) => {
      if (!this.server) {
        resolve();
        return;
      }
      this.server.close(() => resolve());
    });
  }

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url ?? '/', 'http://localhost');
    if (url.pathname === '/' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(APP_HTML);
      return;
    }
    if (await this.serveAOSStatic(req, res, url)) return;
    if (!url.pathname.startsWith('/api/')) {
      this.sendJson(res, 404, { error: 'not_found' });
      return;
    }

    if (url.pathname === '/api/state' && req.method === 'GET') return this.getState(res);
    if (await this.aosCompat.handleHttp(req, res, url)) return;
    if (url.pathname === '/api/chat' && req.method === 'POST') return this.postChat(req, res);
    if (url.pathname === '/api/model' && req.method === 'POST') return this.setModel(req, res);
    if (url.pathname === '/api/permissions/mode' && req.method === 'POST') return this.setPermissionMode(req, res);
    if (url.pathname === '/api/permissions/rules' && req.method === 'POST') return this.addPermissionRule(req, res);
    if (url.pathname === '/api/plugins/load' && req.method === 'POST') return this.loadPlugin(req, res);
    if (url.pathname === '/api/tasks' && req.method === 'GET') return this.getTasks(res);
    if (url.pathname === '/api/sessions' && req.method === 'GET') return this.getSessions(res);
    if (url.pathname === '/api/session/save' && req.method === 'POST') return this.saveSession(req, res);
    if (url.pathname === '/api/project-instructions' && req.method === 'GET') return this.getProjectInstructions(res);

    this.sendJson(res, 404, { error: 'not_found' });
  }

  private async getState(res: ServerResponse): Promise<void> {
    const snapshot = await this.agent.snapshot();
    this.sendJson(res, 200, {
      agent: snapshot,
      models: this.modelCandidates,
      permissions: {
        mode: this.permissionManager.getMode(),
        rules: this.permissionManager.listRules().map((rule) => ({
          ...rule,
          tool: rule.tool instanceof RegExp ? rule.tool.source : rule.tool,
        })),
      },
      plugins: this.loadedPlugins,
      pluginRegistry: await this.pluginRegistry.list(),
      commands: this.commands.list().map((cmd) => ({
        name: cmd.name,
        description: cmd.description,
        kind: cmd.kind,
        source: cmd.source,
      })),
      tasks: await this.taskManager.list(),
      sessions: await this.sessionStore.list(),
      hooks: this.hookEvents.slice(-30),
      aosCompat: this.aosCompat.status(),
    });
  }

  private async postChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await readJson<{ input?: string }>(req);
    const input = body.input?.trim();
    if (!input) return this.sendJson(res, 400, { error: 'input_required' });
    const events: AgentEvent[] = [];
    for await (const event of this.agent.run(input)) {
      events.push(event);
    }
    return this.sendJson(res, 200, { events });
  }

  private async setModel(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await readJson<{ model?: string; models?: string[] }>(req);
    const requested = body.models?.length
      ? body.models
      : typeof body.model === 'string'
        ? body.model.split('->')
        : [];
    const models = requested.map((model) => model.trim()).filter(Boolean);
    if (models.length === 0) return this.sendJson(res, 400, { error: 'model_required' });
    this.agent.setModel(models.length === 1 ? models[0] : models);
    for (const model of models) {
      if (!this.modelCandidates.includes(model)) this.modelCandidates.push(model);
    }
    return this.sendJson(res, 200, { models: this.agent.modelsList });
  }

  private async setPermissionMode(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await readJson<{ mode?: PermissionMode }>(req);
    if (!body.mode) return this.sendJson(res, 400, { error: 'mode_required' });
    this.permissionManager.setMode(body.mode);
    await this.permissionStore.persistManager(this.permissionManager);
    return this.sendJson(res, 200, { mode: this.permissionManager.getMode() });
  }

  private async addPermissionRule(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await readJson<{ rule?: string }>(req);
    if (!body.rule) return this.sendJson(res, 400, { error: 'rule_required' });
    const rule = parsePermissionRule(body.rule);
    this.permissionManager.addRule(rule);
    await this.permissionStore.persistManager(this.permissionManager);
    return this.sendJson(res, 200, { ok: true });
  }

  private async loadPlugin(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await readJson<{ target?: string }>(req);
    if (!body.target) return this.sendJson(res, 400, { error: 'target_required' });
    const plugin = await this.loadPluginTarget(body.target, true);
    return this.sendJson(res, 200, {
      name: plugin.manifest.name,
      skills: plugin.skills.length,
      tools: plugin.tools.length,
      commands: plugin.commands.length,
    });
  }

  private async loadPluginTarget(target: string, persist: boolean) {
    const plugin = await this.pluginLoader.load(target);
    this.agent.addPlugin(plugin);
    for (const command of plugin.commands) this.commands.register({ ...command, source: plugin.manifest.name });
    if (!this.loadedPlugins.includes(plugin.manifest.name)) this.loadedPlugins.push(plugin.manifest.name);
    if (persist) {
      await this.pluginRegistry.add({
        target,
        name: plugin.manifest.name,
        enabled: true,
        loadedAt: new Date().toISOString(),
      });
    }
    return plugin;
  }

  private async getTasks(res: ServerResponse): Promise<void> {
    this.sendJson(res, 200, { tasks: await this.taskManager.list() });
  }

  private async getSessions(res: ServerResponse): Promise<void> {
    this.sendJson(res, 200, { sessions: await this.sessionStore.list() });
  }

  private async saveSession(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await readJson<{ name?: string }>(req);
    const name = body.name?.trim() || `ui_${Date.now()}`;
    await this.sessionStore.save(name, {
      messages: await this.agent.getHistory(),
      savedAt: new Date().toISOString(),
      metadata: { source: 'ui', agent: this.agent.name, models: this.agent.modelsList },
    });
    this.sendJson(res, 200, { name });
  }

  private async getProjectInstructions(res: ServerResponse): Promise<void> {
    const files = await loadProjectInstructions({ cwd: this.rootDir });
    this.sendJson(res, 200, { files });
  }

  private async serveAOSStatic(req: IncomingMessage, res: ServerResponse, url: URL): Promise<boolean> {
    if (url.pathname === '/aos') {
      res.writeHead(302, { Location: '/aos/' });
      res.end();
      return true;
    }
    const isAOSAsset = url.pathname.startsWith('/aos/')
      || url.pathname.startsWith('/assets/');
    if (!isAOSAsset) return false;
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      this.sendJson(res, 405, { error: 'method_not_allowed' });
      return true;
    }

    const root = await this.resolveAOSRoot();
    if (!root) {
      this.sendJson(res, 404, { error: 'aos_frontend_not_found' });
      return true;
    }

    let relativeUrl: string;
    if (url.pathname === '/aos/' || url.pathname === '/aos/index.html') {
      relativeUrl = 'index.html';
    } else if (url.pathname.startsWith('/assets/')) {
      relativeUrl = path.join('assets', decodeURIComponent(url.pathname.slice('/assets/'.length)));
    } else {
      relativeUrl = decodeURIComponent(url.pathname.slice('/aos/'.length));
    }
    const target = path.resolve(root, relativeUrl);
    const rel = path.relative(root, target);
    if (rel.startsWith('..') || path.isAbsolute(rel)) {
      this.sendJson(res, 403, { error: 'forbidden' });
      return true;
    }

    try {
      const stat = await fs.stat(target);
      if (!stat.isFile()) throw new Error('not_file');
      const body = req.method === 'HEAD'
        ? undefined
        : target.endsWith('index.html')
          ? Buffer.from(await this.loadAOSIndexHtml(target, url), 'utf-8')
          : await fs.readFile(target);
      const type = contentTypeFor(target);
      const contentLength = body?.byteLength ?? stat.size;
      res.writeHead(200, {
        'Content-Type': type,
        'Cache-Control': 'no-cache',
        'Content-Length': contentLength,
      });
      res.end(body);
    } catch {
      this.sendJson(res, 404, { error: 'not_found' });
    }
    return true;
  }

  private async resolveAOSRoot(): Promise<string | null> {
    const candidates = [
      path.resolve(UI_DIR, 'aos'),
      path.resolve(this.rootDir, 'dist', 'ui', 'aos'),
      path.resolve(this.rootDir, 'src', 'ui', 'aos'),
    ];
    for (const candidate of candidates) {
      try {
        await fs.access(path.join(candidate, 'index.html'));
        return candidate;
      } catch {
        // Try the next packaged/source asset location.
      }
    }
    return null;
  }

  private async loadAOSIndexHtml(indexPath: string, url: URL): Promise<string> {
    const html = addAOSAssetVersion(await fs.readFile(indexPath, 'utf-8'));
    const ready = {
      service_id: this.aosCompat.status().serviceId,
      ws_url: this.aosCompat.status().nats.wsUrl,
    };
    const bootstrap = buildAOSBootstrapScript(ready, url);
    return html.includes('</body>')
      ? html.replace('</body>', `${bootstrap}\n</body>`)
      : `${html}\n${bootstrap}\n`;
  }

  private sendJson(res: ServerResponse, status: number, body: unknown): void {
    res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify(body));
  }

  private installHookRecorder(): void {
    const record = (event: string, data: unknown) => {
      this.hookEvents.push({ event, at: new Date().toISOString(), data });
      this.hookEvents = this.hookEvents.slice(-100);
    };
    this.hooks
      .on('agent:beforeRun', (payload) => record('agent:beforeRun', payload))
      .on('agent:afterRun', (payload) => record('agent:afterRun', payload))
      .on('agent:error', (payload) => record('agent:error', { error: payload.error.message }))
      .on('tool:beforeCall', (payload) => record('tool:beforeCall', payload))
      .on('tool:afterCall', (payload) => record('tool:afterCall', payload));
  }
}

async function readJson<T>(req: IncomingMessage): Promise<T> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  const raw = Buffer.concat(chunks).toString('utf-8').trim();
  return raw ? JSON.parse(raw) as T : {} as T;
}

function unique(values: string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter(Boolean))];
}

function addAOSAssetVersion(html: string): string {
  return html.replace(
    /(src|href)="([^"]+\/assets\/[^"?]+\.(?:js|css))"/g,
    (_match, attr: string, assetPath: string) => `${attr}="${assetPath}?v=aos-compat-3"`,
  );
}

function contentTypeFor(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase();
  const types: Record<string, string> = {
    '.css': 'text/css; charset=utf-8',
    '.html': 'text/html; charset=utf-8',
    '.ico': 'image/x-icon',
    '.js': 'application/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.png': 'image/png',
    '.svg': 'image/svg+xml; charset=utf-8',
    '.ttf': 'font/ttf',
    '.txt': 'text/plain; charset=utf-8',
    '.wasm': 'application/wasm',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
  };
  return types[ext] ?? 'application/octet-stream';
}

function buildAOSBootstrapScript(
  ready: { service_id: string; ws_url?: string },
  url: URL,
): string {
  const payload = JSON.stringify(ready);
  const current = JSON.stringify(url.toString());
  return `<script>
(() => {
  const ready = ${payload};
  const remoteApiOrigin = 'https://aos.local';
  const rewriteApiUrl = (input) => {
    try {
      const value = typeof input === 'string' || input instanceof URL ? input.toString() : input && input.url;
      if (!value) return input;
      const target = new URL(value, location.href);
      if (target.origin !== remoteApiOrigin || !target.pathname.startsWith('/api/')) return input;
      const local = location.origin + target.pathname + target.search + target.hash;
      if (typeof input === 'string') return local;
      if (input instanceof URL) return new URL(local);
      if (typeof Request !== 'undefined' && input instanceof Request) return new Request(local, input);
      return local;
    } catch {
      return input;
    }
  };
  const originalFetch = window.fetch;
  if (originalFetch) {
    window.fetch = (input, init) => originalFetch.call(window, rewriteApiUrl(input), init);
  }
  const originalOpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function(method, url, ...rest) {
    return originalOpen.call(this, method, rewriteApiUrl(url), ...rest);
  };
  if (!ready.service_id || !ready.ws_url) return;
  const current = new URL(${current});
  const hashRaw = current.hash.startsWith('#') ? current.hash.slice(1) : current.hash;
  const hashUrl = new URL(hashRaw || '/', current.origin);
  if (hashUrl.searchParams.get('service') && hashUrl.searchParams.get('nats')) return;
  hashUrl.searchParams.set('service', ready.service_id);
  hashUrl.searchParams.set('nats', ready.ws_url);
  hashUrl.searchParams.set('auto', 'true');
  const nextHash = '#' + hashUrl.pathname + hashUrl.search;
  if (current.hash !== nextHash) {
    location.replace(current.pathname + nextHash);
  }
})();
</script>`;
}
