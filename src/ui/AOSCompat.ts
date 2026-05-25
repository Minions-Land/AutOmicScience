import type { IncomingMessage, ServerResponse } from 'node:http';
import { createHash, randomUUID } from 'node:crypto';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Agent } from '../agent/index.js';
import type { AgentEvent, Message, ToolCall } from '../types.js';
import { LocalStore, PackageInstaller, type StoreEntry } from '../store/index.js';
import { NatsManager, type NatsServerInfo } from '../chatroom/index.js';

interface CompatChat {
  id: string;
  name: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
  running: boolean;
  project: Record<string, unknown> | null;
  workspaceMode: string;
  workspacePath: string | null;
  chatConfig: Record<string, unknown> | null;
  template: Record<string, unknown> | null;
}

interface InstalledPackage {
  package_id: string;
  package_name: string;
  package_type: string;
  version: string;
  installed_at: string;
}

interface CompatFileHandle {
  mode: 'read' | 'write';
  filePath: string;
  offset: number;
  data?: Buffer;
  chunks: Buffer[];
}

type TemplateKind = 'teams' | 'agents' | 'skills';

export interface AOSCompatOptions {
  rootDir: string;
  agent: Agent;
  enableNats?: boolean;
  dataDir?: string;
  serviceIdHash?: string;
  models?: () => string[];
}

export interface AOSCompatStatus {
  enabled: boolean;
  serviceId: string;
  serviceSubject: string;
  nats: {
    running: boolean;
    tcpUrl?: string;
    wsUrl?: string;
    httpUrl?: string;
    error?: string;
  };
}

export class AOSCompat {
  private readonly rootDir: string;
  private readonly baseAgent: Agent;
  private readonly dataDir: string;
  private readonly chatsDir: string;
  private readonly templatesDir: string;
  private readonly projectAOSDir: string;
  private readonly virtualHomeDir: string;
  private readonly installedFile: string;
  private readonly serviceId: string;
  private readonly serviceSubject: string;
  private readonly enableNats: boolean;
  private readonly modelsProvider?: () => string[];
  private readonly store = new LocalStore();
  private readonly installer: PackageInstaller;
  private readonly chatAgents = new Map<string, Agent>();
  private readonly runningChats = new Map<string, AbortController>();
  private readonly streamMessageIds = new Map<string, string>();
  private readonly streamChunkIndexes = new Map<string, number>();
  private readonly streamToolNames = new Map<string, string>();
  private readonly fileHandles = new Map<string, CompatFileHandle>();
  private savedModels: Record<string, string[]> = {};
  private natsManager: NatsManager | null = null;
  private natsInfo: NatsServerInfo | null = null;
  private nc: any = null;
  private codec: any = null;
  private subscription: any = null;
  private natsError: string | undefined;

  constructor(opts: AOSCompatOptions) {
    this.rootDir = opts.rootDir;
    this.baseAgent = opts.agent;
    this.enableNats = opts.enableNats ?? false;
    this.modelsProvider = opts.models;
    this.dataDir = opts.dataDir ?? path.join(os.homedir(), '.aos', 'aos-compat');
    this.chatsDir = path.join(this.dataDir, 'chats');
    this.templatesDir = path.join(this.dataDir, 'templates');
    this.projectAOSDir = path.join(this.rootDir, '.aos');
    this.virtualHomeDir = path.join(this.dataDir, 'home');
    this.installedFile = path.join(this.dataDir, 'installed.json');
    this.installer = new PackageInstaller(path.join(this.dataDir, 'installed-packages'));
    const hashSource = opts.serviceIdHash ?? process.env.AOS_AOS_ID_HASH ?? 'automic-science';
    this.serviceId = createHash('sha256').update(String(hashSource)).digest('hex');
    this.serviceSubject = `aos.service.${this.serviceId}`;
  }

  async start(): Promise<void> {
    await fs.mkdir(this.chatsDir, { recursive: true });
    await this.ensureTemplateStorage();
    if (!this.enableNats) return;

    try {
      const natsDir = path.join(this.dataDir, 'nats');
      await fs.mkdir(natsDir, { recursive: true });
      this.natsManager = new NatsManager({ workDir: natsDir, dataDir: this.dataDir });
      this.natsInfo = await this.natsManager.start();
      const mod: any = await import('nats');
      const { connect, JSONCodec } = mod;
      this.nc = await connect({ servers: this.natsInfo.tcpUrl });
      this.codec = JSONCodec();
      this.subscription = this.nc.subscribe(this.serviceSubject);
      this.consumeNatsRequests();
      this.natsError = undefined;
    } catch (err) {
      this.natsError = err instanceof Error ? err.message : String(err);
      this.natsInfo = null;
      this.nc = null;
      this.codec = null;
    }
  }

  async stop(): Promise<void> {
    if (this.subscription) {
      try {
        await this.subscription.unsubscribe();
      } catch {
        // Ignore shutdown races.
      }
      this.subscription = null;
    }
    if (this.nc) {
      try {
        await this.nc.drain();
      } catch {
        // Ignore shutdown races.
      }
      this.nc = null;
    }
    if (this.natsManager) {
      await this.natsManager.stop();
      this.natsManager = null;
    }
  }

  status(): AOSCompatStatus {
    return {
      enabled: this.enableNats,
      serviceId: this.serviceId,
      serviceSubject: this.serviceSubject,
      nats: {
        running: !!this.nc && !!this.natsInfo,
        tcpUrl: this.natsInfo?.tcpUrl,
        wsUrl: this.natsInfo?.wsUrl,
        httpUrl: this.natsInfo?.httpUrl,
        error: this.natsError,
      },
    };
  }

  async handleHttp(req: IncomingMessage, res: ServerResponse, url: URL): Promise<boolean> {
    if (url.pathname === '/api/aos/ready' && req.method === 'GET') {
      return this.json(res, 200, {
        success: true,
        compatibility: 'aos_frontend',
        service_id: this.serviceId,
        service_subject: this.serviceSubject,
        nats: this.status().nats,
        frontend_url: '/aos/',
      });
    }

    if (url.pathname === '/api/aos/rpc' && req.method === 'POST') {
      const body = await readBody(req);
      const method = String(body.method ?? '');
      if (!method) return this.json(res, 400, { error: 'method_required' });
      const result = await this.invoke(method, asRecord(body.parameters));
      return this.json(res, 200, { result });
    }

    if (url.pathname === '/api/auth/login' && req.method === 'POST') {
      const body = await readBody(req);
      const username = String(body.username ?? 'local');
      return this.json(res, 200, {
        access_token: 'aos-local-token',
        token_type: 'bearer',
        user: { id: 'local', username },
      });
    }

    if (url.pathname.startsWith('/api/store/')) {
      await this.handleStore(req, res, url);
      return true;
    }

    if (url.pathname.startsWith('/api/chatroom/') && req.method === 'POST') {
      const method = url.pathname.replace('/api/chatroom/', '').replaceAll('/', '_');
      const result = await this.invoke(method, asRecord(await readBody(req)));
      return this.json(res, 200, result);
    }

    if (url.pathname === '/api/toolsets' && req.method === 'GET') {
      return this.json(res, 200, await this.invoke('get_toolsets', {}));
    }

    if (url.pathname === '/api/agents' && req.method === 'GET') {
      return this.json(res, 200, await this.invoke('get_agents', {}));
    }

    if (url.pathname.match(/^\/api\/agents\/[^/]+\/run$/) && req.method === 'POST') {
      const body = await readBody(req);
      const text = String(body.input ?? body.message ?? '');
      const chat = await this.createChat({ chat_name: `api-${Date.now()}` });
      const result = await this.chat({ chat_id: chat.chat_id, message: [{ role: 'user', content: text }] });
      return this.json(res, 200, result);
    }

    return false;
  }

  async invoke(method: string, parameters: Record<string, unknown>): Promise<unknown> {
    switch (method) {
      case '_ping':
      case 'ping':
        return {
          status: 'ok',
          service_id: this.serviceId,
          service_name: 'aos-chatroom',
          version: '0.1.0',
          running_chats: this.runningChats.size,
        };
      case 'get_endpoint':
        return { success: true, service_name: 'aos-endpoint', service_id: this.serviceId };
      case 'set_endpoint':
        return { success: true, message: 'AutOmicScience uses the built-in local endpoint.' };
      case 'create_chat':
        return this.createChat(parameters);
      case 'delete_chat':
        return this.deleteChat(String(parameters.chat_id ?? parameters.id ?? ''));
      case 'list_chats':
        return this.listChats(optionalString(parameters.project_name));
      case 'get_chat_messages':
        return this.getChatMessages(String(parameters.chat_id ?? ''));
      case 'update_chat_name':
        return this.updateChatName(String(parameters.chat_id ?? ''), String(parameters.chat_name ?? parameters.name ?? 'Untitled'));
      case 'get_chat_context':
        return this.getChatContext(String(parameters.chat_id ?? ''));
      case 'update_chat_context':
        return this.updateChatContext(String(parameters.chat_id ?? ''), parameters.context ?? parameters.context_data ?? parameters.chat_context ?? parameters);
      case 'chat':
        return this.chat(parameters);
      case 'stop_chat':
        return this.stopChat(String(parameters.chat_id ?? ''));
      case 'setup_team_for_chat':
        return this.setupTeamForChat(String(parameters.chat_id ?? ''), asRecord(parameters.template_obj ?? parameters.template ?? parameters.team_template));
      case 'get_agents':
        return this.getAgents(optionalString(parameters.chat_id));
      case 'get_active_agent':
        return { success: true, agent: 'AOS' };
      case 'set_active_agent':
        return { success: true, message: 'Active agent set to AOS.' };
      case 'get_toolsets':
        return this.getToolsets();
      case 'proxy_toolset':
        return this.proxyToolset(parameters);
      case 'list_background_tasks':
        return { success: true, tasks: [] };
      case 'get_background_task_detail':
        return { success: false, message: 'Task not found' };
      case 'cancel_background_task':
      case 'remove_background_task':
        return { success: true, message: 'No running background task matched the request.' };
      case 'get_suggestions':
      case 'refresh_suggestions':
        return { success: true, suggestions: [] };
      case 'export_chat':
        return this.exportChat(String(parameters.chat_id ?? ''), optionalString(parameters.output_path), parameters.compress !== false);
      case 'import_chat':
        return this.importChat(String(parameters.bundle_path ?? ''));
      case 'revert_to_message':
        return this.revertToMessage(String(parameters.chat_id ?? ''), String(parameters.message_id ?? ''));
      case 'get_chat_template':
        return this.getChatTemplate(String(parameters.chat_id ?? ''));
      case 'validate_template':
        return this.validateTemplate(asRecord(parameters.template ?? parameters.template_obj ?? parameters.team_template ?? parameters));
      case 'list_template_files':
        return this.listTemplateFiles(String(parameters.file_type ?? parameters.kind ?? 'teams'));
      case 'read_template_file':
        return this.readTemplateFile(String(parameters.file_path ?? parameters.path ?? ''), parameters.resolve_refs === true);
      case 'write_template_file':
        return this.writeTemplateFile(
          String(parameters.file_path ?? parameters.path ?? ''),
          asRecord(parameters.content ?? parameters.template ?? parameters.template_obj ?? parameters.agent ?? {}),
        );
      case 'delete_template_file':
        return this.deleteTemplateFile(String(parameters.file_path ?? parameters.path ?? ''), parameters.force === true);
      case 'saved_models':
        return this.savedModelsMethod(parameters);
      case 'discover_provider_models':
      case 'list_available_models':
        return { success: true, models: this.models(), saved_models: this.savedModelsView() };
      case 'set_agent_model':
        return this.setAgentModel(parameters);
      case 'get_token_stats':
        return this.getTokenStats(String(parameters.chat_id ?? ''));
      case 'compress_chat':
        return this.compressChat(String(parameters.chat_id ?? ''));
      case 'install_store_package':
        return this.installStorePackage(parameters);
      case 'get_installed_store_packages':
        return this.getInstalledStorePackages();
      case 'reload_settings':
        return { success: true, message: 'Settings reloaded.' };
      case 'check_api_keys':
        return this.checkApiKeys();
      case 'get_gateway_channel_config':
        return { success: true, config: { enabled: false, channels: [] } };
      case 'save_gateway_channel_config':
        return { success: true, message: 'Gateway channel config accepted by AutOmicScience compatibility mode.', config: parameters.config ?? {} };
      case 'list_gateway_channels':
        return { success: true, channels: [] };
      case 'start_gateway_channel':
      case 'stop_gateway_channel':
        return { success: true, channel: parameters.channel ?? null, status: 'not_configured' };
      case 'get_gateway_channel_logs':
        return { success: true, channel: parameters.channel ?? null, logs: [] };
      case 'list_gateway_sessions':
        return { success: true, sessions: [] };
      case 'wechat_login_qr':
        return { success: false, message: 'WeChat gateway is not configured in this AutOmicScience local server.' };
      case 'wechat_login_status':
        return { success: false, qrcode_id: parameters.qrcode_id ?? null, status: 'not_configured' };
      case 'oauth_status':
        return { success: true, providers: {}, authenticated: false };
      case 'oauth_login':
      case 'oauth_start':
      case 'oauth_wait':
      case 'oauth_complete':
      case 'oauth_cancel':
      case 'oauth_import':
        return { success: false, message: 'OAuth is not configured in this local AutOmicScience compatibility server.' };
      case 'ollama_status':
        return this.ollamaStatus(String(parameters.url ?? 'http://localhost:11434'));
      case 'get_project_settings':
        return { success: true, root_dir: this.rootDir, workspace_path: this.rootDir };
      case 'list_projects':
        return { success: true, projects: [{ name: path.basename(this.rootDir), path: this.rootDir, is_active: true, exists: true }] };
      case 'get_active_project':
        return { success: true, active: { name: path.basename(this.rootDir), path: this.rootDir }, project: { name: path.basename(this.rootDir), path: this.rootDir } };
      case 'register_project':
      case 'switch_project':
        return { success: true, project: { name: path.basename(String(parameters.path ?? this.rootDir)), path: String(parameters.path ?? this.rootDir) } };
      case 'remove_project':
        return { success: true, message: 'Project removed from compatibility registry.' };
      case 'set_chat_project':
        return this.setChatProject(parameters);
      case 'set_chat_workspace_mode':
        return this.setChatWorkspaceMode(String(parameters.chat_id ?? ''), String(parameters.workspace_mode ?? 'project'));
      case 'change_template_scope':
        return { success: true, message: 'Template scope unchanged in AutOmicScience local compatibility mode.' };
      case 'speech_to_text':
        return { success: false, message: 'Speech-to-text is not configured in this local server.' };
      default:
        return { success: false, message: `Unsupported AOS-compatible method: ${method}` };
    }
  }

  private consumeNatsRequests(): void {
    const sub = this.subscription;
    if (!sub) return;
    (async () => {
      for await (const msg of sub) {
        try {
          const request = this.decodeNatsRequest(msg.data);
          const method = String(request.method ?? '');
          const parameters = asRecord(request.parameters);
          if (!method) {
            msg.respond(this.codec.encode({ error: 'method_required' }));
            continue;
          }
          if (method === 'chat') {
            const chatId = String(parameters.chat_id ?? '');
            this.runNatsChat(parameters, chatId);
            msg.respond(this.codec.encode({
              result: { success: true, message: 'Chat started', chat_id: chatId },
            }));
            continue;
          }
          const result = await this.invoke(method, parameters);
          msg.respond(this.codec.encode({ result }));
        } catch (err) {
          const error = err instanceof Error ? err.message : String(err);
          try {
            msg.respond(this.codec.encode({ error }));
          } catch {
            // Ignore failed replies.
          }
        }
      }
    })();
  }

  private runNatsChat(parameters: Record<string, unknown>, chatId: string): void {
    this.chat(parameters).then(async (result) => {
      if (asRecord(result).success === false) {
        await this.publishStream(chatId, {
          type: 'chat_finished',
          status: 'error',
          metadata: { message: String(asRecord(result).message ?? 'Chat failed') },
        });
      }
    }).catch(async (err) => {
      await this.publishStream(chatId, {
        type: 'chat_finished',
        status: 'error',
        metadata: { message: err instanceof Error ? err.message : String(err) },
      });
    });
  }

  private decodeNatsRequest(data: Uint8Array): Record<string, unknown> {
    try {
      return asRecord(this.codec.decode(data));
    } catch {
      const text = Buffer.from(data).toString('utf-8');
      return asRecord(JSON.parse(text));
    }
  }

  private async publishStream(chatId: string, data: Record<string, unknown>): Promise<void> {
    if (!this.nc || !this.codec) return;
    const eventData = this.prepareStreamData(chatId, data);
    const payload = {
      type: 'chat',
      session_id: `chat_${chatId}`,
      timestamp: Date.now() / 1000,
      data: { ...eventData, chat_id: chatId },
    };
    this.nc.publish(`aos.stream.chat_${chatId}`, this.codec.encode(payload));
  }

  private prepareStreamData(chatId: string, data: Record<string, unknown>): Record<string, unknown> {
    if (data.type === 'chat_finished') {
      this.streamMessageIds.delete(chatId);
      this.streamChunkIndexes.delete(chatId);
      return data;
    }

    if (data.type !== 'chunk') return data;

    const messageId = this.streamMessageIds.get(chatId) ?? randomUUID();
    const chunkIndex = (this.streamChunkIndexes.get(chatId) ?? 0) + 1;
    this.streamMessageIds.set(chatId, messageId);
    this.streamChunkIndexes.set(chatId, chunkIndex);

    const chunk = asRecord(data.chunk);
    return {
      ...data,
      chunk: {
        ...chunk,
        content: String(chunk.content ?? ''),
        message_id: optionalString(chunk.message_id) ?? messageId,
        chunk_index: Number(chunk.chunk_index ?? chunkIndex),
      },
    };
  }

  private async handleStore(req: IncomingMessage, res: ServerResponse, url: URL): Promise<void> {
    const parts = url.pathname.split('/').filter(Boolean);

    if (url.pathname === '/api/store/my/published' && req.method === 'GET') {
      const packages = await this.allPackages(url);
      this.json(res, 200, { packages });
      return;
    }

    if (url.pathname === '/api/store/my/installed' && req.method === 'GET') {
      const installs = await this.installedPackages();
      this.json(res, 200, { installs, packages: installs });
      return;
    }

    if (url.pathname === '/api/store/my/installed' && req.method === 'POST') {
      const body = await readBody(req);
      const install = await this.recordInstall(String(body.package_id ?? body.packageId ?? ''), String(body.version ?? 'latest'));
      this.json(res, 200, install);
      return;
    }

    if (parts[0] === 'api' && parts[1] === 'store' && parts[2] === 'my' && parts[3] === 'installed' && parts[4] && req.method === 'DELETE') {
      await this.recordUninstall(parts[4]);
      this.json(res, 200, { success: true, message: 'Install record removed' });
      return;
    }

    if (url.pathname === '/api/store/packages/stats' && req.method === 'GET') {
      const packages = await this.allPackages(url);
      const categoryCounts = countBy(packages, 'category');
      this.json(res, 200, {
        total: packages.length,
        by_type: countBy(packages, 'type'),
        by_category: countEntries(categoryCounts),
        by_category_map: categoryCounts,
        by_source: countBy(packages, 'source'),
      });
      return;
    }

    if (url.pathname === '/api/store/packages' && req.method === 'GET') {
      const packages = await this.allPackages(url);
      const offset = Number(url.searchParams.get('offset') ?? 0);
      const limit = Number(url.searchParams.get('limit') ?? 20);
      this.json(res, 200, {
        packages: packages.slice(offset, offset + limit),
        total: packages.length,
        limit,
        offset,
      });
      return;
    }

    if (url.pathname === '/api/store/packages' && req.method === 'POST') {
      const body = await readBody(req);
      const entry = bodyToStoreEntry(body);
      await this.store.publish(entry);
      this.json(res, 201, toAOSPackage(entry));
      return;
    }

    if (parts[0] === 'api' && parts[1] === 'store' && parts[2] === 'packages' && parts[3]) {
      const id = decodeURIComponent(parts[3]);
      const pkg = await this.findPackage(id);
      if (!pkg && req.method !== 'DELETE') {
        this.json(res, 404, { detail: 'Package not found' });
        return;
      }

      if (parts[4] === 'versions' && req.method === 'GET') {
        this.json(res, 200, { versions: [versionForPackage(pkg!)] });
        return;
      }

      if (parts[4] === 'versions' && req.method === 'POST') {
        const body = await readBody(req);
        const merged = bodyToStoreEntry({
          ...pkg,
          ...body,
          id: pkg!.id,
          name: pkg!.name,
          type: pkg!.type,
          category: pkg!.category,
          version: body.version ?? pkg!.latest_version,
        });
        await this.store.publish(merged);
        this.json(res, 201, versionForPackage(toAOSPackage(merged)));
        return;
      }

      if (parts[4] === 'download' && req.method === 'GET') {
        this.json(res, 200, {
          id: pkg!.id,
          name: pkg!.name,
          type: pkg!.type,
          version: parts[5] ?? pkg!.latest_version,
          content: pkg!.content ?? `# ${pkg!.display_name}\n\n${pkg!.description}`,
          files: pkg!.files ?? {},
        });
        return;
      }

      if (!parts[4] && req.method === 'GET') {
        this.json(res, 200, pkg);
        return;
      }

      if (!parts[4] && req.method === 'PUT') {
        const body = await readBody(req);
        const merged = bodyToStoreEntry({ ...pkg, ...body, id: pkg!.id, name: body.name ?? pkg!.name });
        await this.store.publish(merged);
        this.json(res, 200, toAOSPackage(merged));
        return;
      }

      if (!parts[4] && req.method === 'DELETE') {
        const deleted = await this.store.deletePackage(id);
        this.json(res, deleted ? 200 : 404, { success: deleted, message: deleted ? 'Package deleted' : 'Package not found' });
        return;
      }
    }

    this.json(res, 404, { error: 'not_found' });
  }

  private async allPackages(url?: URL): Promise<any[]> {
    const entries = (await this.store.list()).map(toAOSPackage);
    const builtins = builtinPackages();
    const merged = [...builtins.filter((b) => !entries.some((e) => e.id === b.id || e.name === b.name)), ...entries];
    const q = url?.searchParams.get('q')?.toLowerCase();
    const type = url?.searchParams.get('type');
    const category = url?.searchParams.get('category');
    return merged.filter((pkg) => {
      if (q) {
        const haystack = [pkg.id, pkg.name, pkg.display_name, pkg.description, ...(pkg.tags ?? [])].join(' ').toLowerCase();
        if (!haystack.includes(q)) return false;
      }
      if (type && pkg.type !== type) return false;
      if (category && pkg.category !== category) return false;
      return true;
    });
  }

  private async findPackage(idOrName: string): Promise<any | undefined> {
    return (await this.allPackages()).find((pkg) => pkg.id === idOrName || pkg.name === idOrName);
  }

  private async createChat(parameters: Record<string, unknown>): Promise<any> {
    const now = new Date().toISOString();
    const id = optionalString(parameters.chat_id) ?? optionalString(parameters.id) ?? randomUUID();
    const project: Record<string, unknown> = asRecord(parameters.project_metadata);
    const projectName = optionalString(parameters.project_name);
    if (projectName) project.name = projectName;
    const workspacePath = optionalString(parameters.workspace_path) ?? null;
    const workspaceMode = workspacePath ? 'isolated' : String(parameters.workspace_mode ?? 'project');
    if (workspacePath) {
      project.workspace_path = workspacePath;
      project.workspace_mode = workspaceMode;
      await fs.mkdir(workspacePath, { recursive: true }).catch(() => {});
    }
    const chat: CompatChat = {
      id,
      name: String(parameters.chat_name ?? parameters.name ?? 'New Chat'),
      messages: [],
      createdAt: now,
      updatedAt: now,
      running: false,
      project: Object.keys(project).length ? project : null,
      workspaceMode,
      workspacePath,
      chatConfig: asNullableRecord(parameters.chat_config),
      template: await this.defaultTemplate(),
    };
    await this.saveChat(chat);
    return {
      success: true,
      message: 'Chat created successfully',
      chat_name: chat.name,
      chat_id: chat.id,
      workspace_mode: chat.workspaceMode,
      workspace_path: chat.workspacePath,
      project: chat.project,
      chat_config: chat.chatConfig,
      template: chat.template,
    };
  }

  private async ensureChat(chatId: string, parameters: Record<string, unknown> = {}): Promise<CompatChat | null> {
    if (!chatId) return null;
    const existing = await this.readChat(chatId);
    if (existing) return existing;
    const now = new Date().toISOString();
    const workspacePath = optionalString(parameters.workspace_path) ?? null;
    const workspaceMode = workspacePath ? 'isolated' : String(parameters.workspace_mode ?? 'project');
    const project: Record<string, unknown> = asRecord(parameters.project_metadata);
    const projectName = optionalString(parameters.project_name);
    if (projectName) project.name = projectName;
    if (workspacePath) {
      project.workspace_path = workspacePath;
      project.workspace_mode = workspaceMode;
      await fs.mkdir(workspacePath, { recursive: true }).catch(() => {});
    }
    const chat: CompatChat = {
      id: chatId,
      name: String(parameters.chat_name ?? parameters.name ?? 'New Chat'),
      messages: [],
      createdAt: now,
      updatedAt: now,
      running: false,
      project: Object.keys(project).length ? project : null,
      workspaceMode,
      workspacePath,
      chatConfig: asNullableRecord(parameters.chat_config),
      template: await this.defaultTemplate(),
    };
    await this.saveChat(chat);
    return chat;
  }

  private async deleteChat(chatId: string): Promise<Record<string, unknown>> {
    if (!chatId) return { success: false, message: 'chat_id is required' };
    this.chatAgents.delete(chatId);
    const target = this.chatFile(chatId);
    try {
      await fs.unlink(target);
      return { success: true, message: 'Chat deleted successfully' };
    } catch {
      return { success: false, message: `Chat '${chatId}' not found` };
    }
  }

  private async listChats(projectName?: string): Promise<Record<string, unknown>> {
    await fs.mkdir(this.chatsDir, { recursive: true });
    const files = await fs.readdir(this.chatsDir).catch(() => []);
    const chats: any[] = [];
    for (const file of files.filter((f) => f.endsWith('.json'))) {
      const chat = await this.readChat(file.replace(/\.json$/, ''));
      if (!chat) continue;
      if (projectName && chat.project?.name !== projectName) continue;
      chats.push(chatInfo(chat));
    }
    chats.sort((a, b) => String(b.last_activity_date ?? '').localeCompare(String(a.last_activity_date ?? '')));
    return { success: true, chats };
  }

  private async getChatMessages(chatId: string): Promise<Record<string, unknown>> {
    const chat = await this.ensureChat(chatId);
    if (!chat) return { success: false, message: `Chat '${chatId}' not found`, messages: [] };
    return { success: true, messages: chat.messages.map(toAOSMessage) };
  }

  private async updateChatName(chatId: string, chatName: string): Promise<Record<string, unknown>> {
    const chat = await this.ensureChat(chatId);
    if (!chat) return { success: false, message: `Chat '${chatId}' not found` };
    chat.name = chatName;
    chat.updatedAt = new Date().toISOString();
    await this.saveChat(chat);
    return { success: true, message: 'Chat name updated successfully' };
  }

  private async getChatContext(chatId: string): Promise<Record<string, unknown>> {
    const chat = await this.ensureChat(chatId);
    if (!chat) return { success: false, message: `Chat '${chatId}' not found`, context: {} };
    return {
      success: true,
      context: asRecord((chat as CompatChat & Record<string, unknown>).context),
      chat_config: chat.chatConfig,
      project: chat.project,
      workspace_mode: chat.workspaceMode,
      workspace_path: chat.workspacePath,
    };
  }

  private async updateChatContext(chatId: string, context: unknown): Promise<Record<string, unknown>> {
    const chat = await this.ensureChat(chatId);
    if (!chat) return { success: false, message: `Chat '${chatId}' not found` };
    (chat as CompatChat & Record<string, unknown>).context = asRecord(context);
    chat.updatedAt = new Date().toISOString();
    await this.saveChat(chat);
    return { success: true, message: 'Chat context updated', context: (chat as CompatChat & Record<string, unknown>).context };
  }

  private async getChatTemplate(chatId: string): Promise<Record<string, unknown>> {
    const chat = await this.ensureChat(chatId);
    const template = asNullableRecord(chat?.template) ?? await this.defaultTemplate();
    return { success: true, template };
  }

  private async validateTemplate(template: Record<string, unknown>): Promise<Record<string, unknown>> {
    const normalized = normalizeTeamTemplate(template, optionalString(template.id) ?? slugify(String(template.name ?? 'team')));
    const agents = Array.isArray(normalized.agents) ? normalized.agents.map(asRecord) : [];
    const requiredToolsets = uniqueSorted(agents.flatMap((agent) => stringArray(agent.toolsets)));
    const requiredMcpServers = uniqueSorted(agents.flatMap((agent) => stringArray(agent.mcp_servers ?? agent.mcp)));
    const validationErrors: string[] = [];
    if (!optionalString(normalized.id)) validationErrors.push('id is required');
    if (!optionalString(normalized.name)) validationErrors.push('name is required');
    if (agents.length === 0) validationErrors.push('at least one agent is recommended');
    return {
      success: validationErrors.length === 0,
      compatible: validationErrors.length === 0,
      validation_errors: validationErrors,
      required_toolsets: requiredToolsets,
      required_mcp_servers: requiredMcpServers,
      agents: Object.fromEntries(agents.map((agent) => [String(agent.id ?? agent.name ?? randomUUID()), agent])),
      template: normalized,
      message: validationErrors.length ? validationErrors.join('; ') : 'Template is compatible with AutOmicScience.',
    };
  }

  private async listTemplateFiles(fileType: string): Promise<Record<string, unknown>> {
    await this.ensureTemplateStorage();
    const kinds = templateKindsFor(fileType);
    if (kinds.length === 0) return { success: false, error: `Unknown file_type: ${fileType}` };
    const files: Record<string, unknown>[] = [];
    for (const kind of kinds) {
      const entries = await this.listTemplateKind(kind);
      files.push(...entries);
    }
    files.sort((a, b) => String(a.path).localeCompare(String(b.path)));
    return { success: true, file_type: fileType || 'teams', files, total: files.length };
  }

  private async readTemplateFile(filePath: string, resolveRefs = false): Promise<Record<string, unknown>> {
    if (!filePath) return { success: false, error: 'file_path is required' };
    await this.ensureTemplateStorage();
    if (isSkillMarkdownPath(filePath)) return this.readSkillMarkdownFile(filePath);
    const parsed = parseTemplatePath(filePath);
    if (!parsed) return { success: false, error: `Unsupported template path: ${filePath}` };
    const target = this.templateFilePath(parsed.kind, parsed.id);
    try {
      const raw = await fs.readFile(target, 'utf-8');
      const content = JSON.parse(raw) as Record<string, unknown>;
      const normalized = normalizeTemplateContent(parsed.kind, content, parsed.id);
      if (resolveRefs && parsed.kind === 'teams') {
        normalized.agents = await this.resolveTeamAgents(arrayOfRecords(normalized.agents));
      }
      return {
        success: true,
        file_path: templateRelPath(parsed.kind, parsed.id),
        type: singularTemplateKind(parsed.kind),
        content: normalized,
      };
    } catch (err: any) {
      if (err?.code === 'ENOENT') return { success: false, error: `Template file '${filePath}' not found` };
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  private async writeTemplateFile(filePath: string, content: Record<string, unknown>): Promise<Record<string, unknown>> {
    await this.ensureTemplateStorage();
    if (isSkillMarkdownPath(filePath)) {
      const id = skillIdFromMarkdownPath(filePath);
      return this.writeTemplateFile(`skills/${id}.md`, {
        id,
        name: String(content.name ?? id),
        description: String(content.description ?? ''),
        content: String(content.content ?? content.body ?? ''),
        tags: stringArray(content.tags),
      });
    }
    const inferredKind = inferTemplateKind(content);
    const parsed = parseTemplatePath(filePath || `${inferredKind}/${slugify(String(content.id ?? content.name ?? Date.now()))}.md`);
    if (!parsed) return { success: false, error: `Unsupported template path: ${filePath}` };
    const kind = parsed.kind;
    const id = optionalString(content.id) ?? parsed.id;
    const normalized = normalizeTemplateContent(kind, { ...content, id }, id);
    const target = this.templateFilePath(kind, id);
    const existed = await pathExists(target);
    try {
      await fs.mkdir(path.dirname(target), { recursive: true });
      await fs.writeFile(target, JSON.stringify(normalized, null, 2), 'utf-8');
      if (kind === 'skills') await this.syncSkillMarkdown(id, normalized);
      return {
        success: true,
        operation: existed ? 'update' : 'create',
        file_path: templateRelPath(kind, id),
        type: singularTemplateKind(kind),
        id,
        content: normalized,
      };
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  private async deleteTemplateFile(filePath: string, _force = false): Promise<Record<string, unknown>> {
    if (!filePath) return { success: false, error: 'file_path is required' };
    if (isSkillMarkdownPath(filePath)) filePath = `skills/${skillIdFromMarkdownPath(filePath)}.md`;
    const parsed = parseTemplatePath(filePath);
    if (!parsed) return { success: false, error: `Unsupported template path: ${filePath}` };
    try {
      await fs.rm(this.templateFilePath(parsed.kind, parsed.id), { force: false });
      if (parsed.kind === 'skills') await fs.rm(path.join(this.projectAOSDir, 'skills', parsed.id), { recursive: true, force: true });
      return {
        success: true,
        operation: 'delete',
        file_path: templateRelPath(parsed.kind, parsed.id),
        type: singularTemplateKind(parsed.kind),
        id: parsed.id,
      };
    } catch (err: any) {
      if (err?.code === 'ENOENT') return { success: false, error: `Template file '${filePath}' not found` };
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  private async setupTeamForChat(chatId: string, template: Record<string, unknown>): Promise<Record<string, unknown>> {
    const chat = await this.ensureChat(chatId);
    if (!chat) return { success: false, message: `Chat '${chatId}' not found` };
    chat.template = Object.keys(template).length ? normalizeTeamTemplate(template, optionalString(template.id) ?? 'aos-bio-mas') : await this.defaultTemplate();
    chat.updatedAt = new Date().toISOString();
    await this.saveChat(chat);
    return {
      success: true,
      message: 'AOS Bio MAS is active for this chat.',
      template: chat.template,
      team: {
        name: String(chat.template.name ?? 'AOS Bio MAS'),
        agents: arrayOfRecords(chat.template.agents).map((agent) => String(agent.name ?? agent.id ?? 'AOS')),
      },
    };
  }

  private async exportChat(chatId: string, outputPath: string | undefined, compress: boolean): Promise<Record<string, unknown>> {
    const chat = await this.readChat(chatId);
    if (!chat) return { success: false, message: `Chat '${chatId}' not found` };
    const exportDir = outputPath ?? path.join(this.dataDir, 'exports', chatId);
    await fs.mkdir(exportDir, { recursive: true });
    const bundlePath = path.join(exportDir, 'chat.json');
    await fs.writeFile(bundlePath, JSON.stringify(chat, null, 2), 'utf-8');
    return {
      success: true,
      chat_id: chatId,
      output_path: exportDir,
      bundle_path: bundlePath,
      compressed: false,
      compress_requested: compress,
    };
  }

  private async importChat(bundlePath: string): Promise<Record<string, unknown>> {
    if (!bundlePath) return { success: false, message: 'bundle_path is required' };
    try {
      const raw = await fs.readFile(bundlePath, 'utf-8');
      const parsed = JSON.parse(raw) as Partial<CompatChat>;
      const now = new Date().toISOString();
      const chat: CompatChat = {
        id: parsed.id ?? randomUUID(),
        name: parsed.name ?? 'Imported Chat',
        messages: Array.isArray(parsed.messages) ? parsed.messages : [],
        createdAt: parsed.createdAt ?? now,
        updatedAt: now,
        running: false,
        project: parsed.project ?? null,
        workspaceMode: parsed.workspaceMode ?? 'project',
        workspacePath: parsed.workspacePath ?? null,
        chatConfig: parsed.chatConfig ?? null,
        template: parsed.template ?? await this.defaultTemplate(),
      };
      await this.saveChat(chat);
      return { success: true, chat_id: chat.id, chat: chatInfo(chat) };
    } catch (err) {
      return { success: false, message: err instanceof Error ? err.message : String(err) };
    }
  }

  private async revertToMessage(chatId: string, messageId: string): Promise<Record<string, unknown>> {
    const chat = await this.readChat(chatId);
    if (!chat) return { success: false, message: `Chat '${chatId}' not found` };
    const messages = chat.messages.map(toAOSMessage);
    const index = messages.findIndex((message) => message.id === messageId);
    if (index < 0) return { success: false, message: `Message '${messageId}' not found` };
    const reverted = messages[index];
    chat.messages = chat.messages.slice(0, index);
    chat.updatedAt = new Date().toISOString();
    await this.saveChat(chat);
    return { success: true, message: 'Chat reverted', reverted_content: reverted };
  }

  private async setChatProject(parameters: Record<string, unknown>): Promise<Record<string, unknown>> {
    const chat = await this.ensureChat(String(parameters.chat_id ?? ''), parameters);
    if (!chat) return { success: false, message: `Chat '${parameters.chat_id ?? ''}' not found` };
    const projectName = optionalString(parameters.project_name);
    const workspacePath = optionalString(parameters.workspace_path);
    if (!projectName && !workspacePath && !parameters.workspace_mode) {
      chat.project = null;
    } else {
      const project = asRecord(chat.project);
      if (projectName) project.name = projectName;
      if (workspacePath) project.workspace_path = workspacePath;
      if (parameters.workspace_mode) project.workspace_mode = String(parameters.workspace_mode);
      chat.project = project;
    }
    chat.workspacePath = workspacePath ?? chat.workspacePath;
    if (parameters.workspace_mode) chat.workspaceMode = String(parameters.workspace_mode);
    chat.updatedAt = new Date().toISOString();
    await this.saveChat(chat);
    return { success: true, message: 'Project metadata updated', project: chat.project };
  }

  private async setChatWorkspaceMode(chatId: string, workspaceMode: string): Promise<Record<string, unknown>> {
    if (workspaceMode !== 'project' && workspaceMode !== 'isolated') {
      return { success: false, message: "workspace_mode must be 'project' or 'isolated'" };
    }
    const chat = await this.ensureChat(chatId);
    if (!chat) return { success: false, message: `Chat '${chatId}' not found` };
    chat.workspaceMode = workspaceMode;
    if (workspaceMode === 'isolated' && !chat.workspacePath) {
      chat.workspacePath = path.join(this.dataDir, 'workspaces', chatId);
      await fs.mkdir(chat.workspacePath, { recursive: true });
    }
    if (workspaceMode === 'project') chat.workspacePath = null;
    chat.updatedAt = new Date().toISOString();
    await this.saveChat(chat);
    return { success: true, workspace_mode: chat.workspaceMode, workspace_path: chat.workspacePath };
  }

  private async chat(parameters: Record<string, unknown>): Promise<Record<string, unknown>> {
    const chatId = String(parameters.chat_id ?? '');
    const chat = await this.ensureChat(chatId, parameters);
    if (!chat) return { success: false, message: `Chat '${chatId}' not found` };
    if (this.runningChats.has(chatId)) return { success: false, message: 'Chat is already running' };
    const incoming = Array.isArray(parameters.message) ? parameters.message.map(normalizeMessage) : [];
    if (incoming.length === 0) return { success: false, message: 'message is required' };

    chat.running = true;
    chat.updatedAt = new Date().toISOString();
    await this.saveChat(chat);
    const controller = new AbortController();
    this.runningChats.set(chatId, controller);

    try {
      const slash = await this.trySlashCommand(chat, incoming);
      if (slash !== null) {
        const assistantMessage: Message = { role: 'assistant', content: slash, name: 'AOS' };
        chat.messages.push(...incoming, assistantMessage);
        chat.running = false;
        chat.updatedAt = new Date().toISOString();
        await this.saveChat(chat);
        await this.publishStream(chatId, { type: 'chunk', chunk: { content: slash } });
        await this.publishFinalAssistantMessage(chatId, assistantMessage, chat.messages.length - 1);
        await this.publishStream(chatId, { type: 'chat_finished' });
        return { success: true, response: slash, chat_id: chatId };
      }

      const agent = await this.agentForChat(chat);
      let finalText = '';
      for await (const event of agent.run(incoming, { signal: controller.signal })) {
        await this.publishAgentEvent(chatId, event);
        if (event.type === 'done') finalText = String(event.data ?? '');
      }
      chat.messages = await agent.getHistory();
      const finalMessageIndex = findLastAssistantMessageIndex(chat.messages);
      if (finalMessageIndex >= 0) {
        await this.publishFinalAssistantMessage(chatId, chat.messages[finalMessageIndex], finalMessageIndex);
      }
      await this.publishStream(chatId, { type: 'chat_finished' });
      return { success: true, response: finalText, chat_id: chatId };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.publishStream(chatId, {
        type: 'chat_finished',
        status: 'error',
        metadata: { message },
      });
      return { success: false, message, chat_id: chatId };
    } finally {
      this.runningChats.delete(chatId);
      const latest = await this.readChat(chatId);
      const toSave = latest ?? chat;
      toSave.running = false;
      toSave.updatedAt = new Date().toISOString();
      const agent = this.chatAgents.get(chatId);
      if (agent) toSave.messages = await agent.getHistory();
      else if (chat.messages.length > toSave.messages.length) toSave.messages = chat.messages;
      await this.saveChat(toSave);
    }
  }

  private async stopChat(chatId: string): Promise<Record<string, unknown>> {
    const controller = this.runningChats.get(chatId);
    if (!controller) return { success: true, message: 'Chat already stopped' };
    controller.abort();
    this.runningChats.delete(chatId);
    return { success: true, message: 'Chat stopped successfully' };
  }

  private async getAgents(_chatId?: string): Promise<Record<string, unknown>> {
    const snapshot = await this.baseAgent.snapshot();
    return {
      success: true,
      agents: [{
        name: 'AOS',
        instructions: 'AOS is a bioinformatics intelligent agent developed by AutOmicScience.',
        tools: snapshot.tools.map((tool) => tool.name),
        toolsets: ['aos_default'],
        icon: 'aos',
        not_loaded_toolsets: [],
        model: snapshot.model,
        models: snapshot.models,
      }],
      can_switch_agents: false,
      has_transfer: false,
    };
  }

  private async getToolsets(): Promise<Record<string, unknown>> {
    const snapshot = await this.baseAgent.snapshot();
    return {
      success: true,
      services: [{
        service_id: 'aos_default_toolset',
        name: 'aos_default',
        service_name: 'aos_default',
        service_type: 'toolset',
        status: 'running',
        tools: snapshot.tools,
        methods: snapshot.tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
        })),
      }],
    };
  }

  private async proxyToolset(parameters: Record<string, unknown>): Promise<unknown> {
    const methodName = String(parameters.method_name ?? parameters.tool ?? '');
    if (!methodName) return { success: false, error: 'method_name is required' };
    const args = asRecord(parameters.args);
    const compatResult = await this.proxyCompatTool(methodName, args);
    if (compatResult !== null) return compatResult;
    const result = await this.baseAgent.executeTool(methodName, args, { source: 'aos_compat' });
    try {
      return JSON.parse(result.content);
    } catch {
      return { success: true, output: result.content, metadata: result.metadata };
    }
  }

  private async proxyCompatTool(methodName: string, args: Record<string, unknown>): Promise<Record<string, unknown> | null> {
    switch (methodName) {
      case 'get_cwd':
        return { success: true, cwd: this.rootDir };
      case 'list_files':
        return this.compatListFiles(args);
      case 'read_file':
        return this.compatReadFile(args);
      case 'write_file':
        return this.compatWriteFile(args);
      case 'create_file':
        return this.compatWriteFile({ ...args, overwrite: false });
      case 'create_directory':
        return this.compatCreateDirectory(args);
      case 'delete_path':
      case 'delete_file':
      case 'delete_directory':
        return this.compatDeletePath(methodName, args);
      case 'move_file':
      case 'rename_file':
        return this.compatMoveFile(args);
      case 'fetch_image_base64':
        return this.compatFetchImageBase64(args);
      case 'fetch_resources_batch':
        return this.compatFetchResourcesBatch(args);
      case 'open_file_for_read':
        return this.compatOpenFileForRead(args);
      case 'read_chunk':
        return this.compatReadChunk(args);
      case 'open_file_for_write':
        return this.compatOpenFileForWrite(args);
      case 'write_chunk':
        return this.compatWriteChunk(args);
      case 'close_file':
        return this.compatCloseFile(args);
      case 'manage_service':
        return this.compatManageService(args);
      case 'list_collections':
        return { success: true, collections: [] };
      case 'get_chat_knowledge':
        return { success: true, config: { active_collection_ids: [], auto_search: false } };
      case 'create_collection':
        return {
          success: true,
          collection: {
            id: randomUUID(),
            name: String(args.name ?? 'Collection'),
            description: String(args.description ?? ''),
            status: 'active',
            total_docs: 0,
          },
        };
      case 'delete_collection':
      case 'remove_source':
      case 'enable_collection':
      case 'disable_collection':
      case 'set_auto_search':
        return { success: true, config: { active_collection_ids: [], auto_search: !!args.enabled } };
      case 'add_sources':
        return { success: true, source_ids: [] };
      case 'list_sources':
        return { success: true, sources: [] };
      case 'read_notebook':
      case 'create_notebook':
      case 'add_cell':
      case 'update_cell':
      case 'delete_cell':
      case 'move_cell':
      case 'execute_cell':
      case 'manage_kernel':
      case 'complete_request':
      case 'inspect_request':
      case 'notebook_edit':
      case 'notebook_execute':
      case 'notebook_read':
        return this.compatNotebook(methodName, args);
      default:
        return null;
    }
  }

  private async compatListFiles(args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const target = this.resolveWorkspacePath(optionalString(args.sub_dir) ?? optionalString(args.path) ?? '');
    try {
      const stat = await fs.stat(target);
      if (!stat.isDirectory()) return { success: false, error: 'Path is not a directory' };
      const names = await fs.readdir(target);
      const files: Record<string, unknown>[] = [];
      for (const name of names) {
        if (name === '.git' || name === 'node_modules' || name === 'dist' || name === '.endpoint-logs' || name === '.executor') continue;
        const entryPath = path.join(target, name);
        try {
          const entryStat = await fs.lstat(entryPath);
          files.push({
            name,
            path: this.displayWorkspacePath(entryPath),
            type: entryStat.isSymbolicLink() ? 'symlink' : entryStat.isDirectory() ? 'directory' : entryStat.isFile() ? 'file' : 'other',
            size: entryStat.isDirectory() ? 0 : entryStat.size,
            last_modified: formatLocalTimestamp(entryStat.mtime),
          });
        } catch {
          // Entries can disappear while the UI is refreshing.
        }
      }
      files.sort((a, b) => {
        const at = String(a.type);
        const bt = String(b.type);
        if (at !== bt) return at === 'directory' ? -1 : bt === 'directory' ? 1 : at.localeCompare(bt);
        return String(a.name).localeCompare(String(b.name));
      });
      return { success: true, files };
    } catch {
      return { success: false, error: 'Directory does not exist' };
    }
  }

  private async compatReadFile(args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const target = this.resolveWorkspacePath(String(args.file_path ?? args.path ?? ''));
    try {
      const stat = await fs.stat(target);
      if (!stat.isFile()) return { success: false, error: 'Path is not a file' };
      const buffer = await fs.readFile(target);
      const textLike = isTextLikePath(target);
      if (!textLike && args.start_line === undefined && args.end_line === undefined && args.max_chars === undefined) {
        return { success: true, data: buffer.toString('base64'), size: buffer.length, encoding: 'base64' };
      }
      const text = buffer.toString('utf-8');
      const lines = text.split(/\r?\n/);
      const totalLines = lines.length === 1 && lines[0] === '' ? 0 : lines.length;
      let content = text;
      const startLine = args.start_line === undefined ? undefined : Number(args.start_line);
      const endLine = args.end_line === undefined ? undefined : Number(args.end_line);
      if (startLine !== undefined || endLine !== undefined) {
        const start = Math.max(1, startLine ?? 1);
        const end = Math.min(totalLines, endLine ?? totalLines);
        if (start > Math.max(totalLines, 1)) return { success: false, error: `start_line ${start} is out of range (file has ${totalLines} lines)` };
        content = lines.slice(start - 1, end).join('\n');
      }
      const maxChars = args.max_chars === undefined ? undefined : Number(args.max_chars);
      const truncated = maxChars !== undefined && maxChars >= 0 && content.length > maxChars;
      if (truncated) content = content.slice(0, maxChars);
      return {
        success: true,
        content,
        data: buffer.toString('base64'),
        size: buffer.length,
        total_lines: totalLines,
        format: path.extname(target).toLowerCase(),
        truncated,
      };
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  private async compatWriteFile(args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const filePath = String(args.file_path ?? args.path ?? '');
    if (!filePath) return { success: false, error: 'file_path is required' };
    const target = this.resolveWorkspacePath(filePath);
    const content = String(args.content ?? '');
    const append = !!args.append;
    const overwrite = args.overwrite !== false;
    try {
      await fs.mkdir(path.dirname(target), { recursive: true });
      if (append) {
        await fs.appendFile(target, content, 'utf-8');
        return { success: true, appended_chars: content.length };
      }
      if (!overwrite) {
        try {
          await fs.writeFile(target, content, { encoding: 'utf-8', flag: 'wx' });
          await this.syncSkillTemplateFromWorkspaceWrite(target, content);
          return { success: true, overwritten: false };
        } catch (err: any) {
          if (err?.code === 'EEXIST' && this.projectSkillMarkdownId(target)) {
            await fs.writeFile(target, content, 'utf-8');
            await this.syncSkillTemplateFromWorkspaceWrite(target, content);
            return { success: true, overwritten: true, reason: 'project_skill_save' };
          }
          if (err?.code === 'EEXIST') return { success: false, error: 'File already exists', reason: 'overwrite_disabled' };
          throw err;
        }
      }
      await fs.writeFile(target, content, 'utf-8');
      await this.syncSkillTemplateFromWorkspaceWrite(target, content);
      return { success: true, overwritten: true };
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  private async compatCreateDirectory(args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = args.sub_dir ?? args.path ?? args.directory_path;
    const paths = Array.isArray(value) ? value.map(String) : [String(value ?? '')];
    const results: Record<string, unknown>[] = [];
    for (const item of paths.filter(Boolean)) {
      try {
        await fs.mkdir(this.resolveWorkspacePath(item), { recursive: true });
        results.push({ path: item, success: true });
      } catch (err) {
        results.push({ path: item, success: false, error: err instanceof Error ? err.message : String(err) });
      }
    }
    if (results.length === 1) return { success: !!results[0].success };
    return { success: results.every((item) => item.success), results };
  }

  private async compatDeletePath(methodName: string, args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = args.path ?? args.file_path ?? args.sub_dir;
    const recursive = methodName === 'delete_directory' ? true : !!args.recursive;
    const paths = Array.isArray(value) ? value.map(String) : [String(value ?? '')];
    const results: Record<string, unknown>[] = [];
    for (const item of paths.filter(Boolean)) {
      try {
        const target = this.resolveWorkspacePath(item);
        const stat = await fs.lstat(target);
        await fs.rm(target, { recursive: stat.isDirectory() ? recursive : false, force: false });
        results.push({ path: item, success: true });
      } catch (err) {
        results.push({ path: item, success: false, error: err instanceof Error ? err.message : String(err) });
      }
    }
    if (results.length === 1) return results[0];
    return { success: results.every((item) => item.success), results };
  }

  private async compatMoveFile(args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const oldPath = String(args.old_path ?? args.source_path ?? args.from ?? '');
    const newPath = String(args.new_path ?? args.target_path ?? args.to ?? '');
    if (!oldPath || !newPath) return { success: false, error: 'old_path and new_path are required' };
    try {
      const from = this.resolveWorkspacePath(oldPath);
      const to = this.resolveWorkspacePath(newPath);
      await fs.mkdir(path.dirname(to), { recursive: true });
      await fs.rename(from, to);
      return { success: true };
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  private async compatFetchImageBase64(args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const imagePath = String(args.image_path ?? args.file_path ?? '');
    if (!imagePath) return { success: false, error: 'image_path is required' };
    try {
      const target = this.resolveWorkspacePath(imagePath);
      const stat = await fs.stat(target);
      if (!stat.isFile()) return { success: false, error: 'Path is not a file' };
      if (stat.size > 10 * 1024 * 1024) return { success: false, error: 'Image is larger than 10MB' };
      const data = await fs.readFile(target);
      return { success: true, data_uri: `data:${mimeForPath(target)};base64,${data.toString('base64')}` };
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  private async compatFetchResourcesBatch(args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const resourcePaths = Array.isArray(args.resource_paths) ? args.resource_paths.map(String) : [];
    const basePath = optionalString(args.base_path);
    const resources: Record<string, unknown>[] = [];
    for (const resourcePath of resourcePaths) {
      const result: Record<string, unknown> = { path: resourcePath, success: false };
      try {
        const target = path.isAbsolute(resourcePath)
          ? resourcePath
          : this.resolveWorkspacePath(basePath ? path.join(basePath, resourcePath) : resourcePath);
        const data = await fs.readFile(target);
        const mime = mimeForPath(target);
        result.resolved_path = this.displayWorkspacePath(target);
        result.mime_type = mime;
        result.content = mime.startsWith('image/') ? `data:${mime};base64,${data.toString('base64')}` : data.toString('utf-8');
        result.success = true;
      } catch (err) {
        result.error = err instanceof Error ? err.message : String(err);
      }
      resources.push(result);
    }
    return {
      success: true,
      resources,
      total: resources.length,
      loaded: resources.filter((item) => item.success).length,
      failed: resources.filter((item) => !item.success).length,
    };
  }

  private async compatOpenFileForRead(args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const target = this.resolveWorkspacePath(String(args.file_path ?? args.path ?? ''));
    try {
      const data = await fs.readFile(target);
      const handleId = randomUUID();
      this.fileHandles.set(handleId, { mode: 'read', filePath: target, offset: 0, data, chunks: [] });
      return { success: true, handle_id: handleId, total_size: data.length };
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  private async compatReadChunk(args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const handleId = String(args.handle_id ?? '');
    const handle = this.fileHandles.get(handleId);
    if (!handle || handle.mode !== 'read' || !handle.data) return { success: false, error: 'Invalid read handle' };
    const size = Math.max(1, Number(args.size ?? args.chunk_size ?? 512 * 1024));
    const chunk = handle.data.subarray(handle.offset, handle.offset + size);
    handle.offset += chunk.length;
    return {
      success: true,
      data: chunk.toString('base64'),
      bytes_read: chunk.length,
      eof: handle.offset >= handle.data.length,
    };
  }

  private async compatOpenFileForWrite(args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const filePath = String(args.file_path ?? args.path ?? '');
    if (!filePath) return { success: false, error: 'file_path is required' };
    const target = this.resolveWorkspacePath(filePath);
    const handleId = randomUUID();
    this.fileHandles.set(handleId, { mode: 'write', filePath: target, offset: 0, chunks: [] });
    return { success: true, handle_id: handleId };
  }

  private async compatWriteChunk(args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const handleId = String(args.handle_id ?? '');
    const handle = this.fileHandles.get(handleId);
    if (!handle || handle.mode !== 'write') return { success: false, error: 'Invalid write handle' };
    const data = args.data;
    const chunk = typeof data === 'string' ? Buffer.from(data, 'base64') : Buffer.from([]);
    handle.chunks.push(chunk);
    handle.offset += chunk.length;
    return { success: true, bytes_written: chunk.length };
  }

  private async compatCloseFile(args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const handleId = String(args.handle_id ?? '');
    const handle = this.fileHandles.get(handleId);
    if (!handle) return { success: false, error: 'Invalid file handle' };
    this.fileHandles.delete(handleId);
    if (handle.mode === 'write') {
      try {
        await fs.mkdir(path.dirname(handle.filePath), { recursive: true });
        await fs.writeFile(handle.filePath, Buffer.concat(handle.chunks));
        return { success: true, bytes_written: handle.offset };
      } catch (err) {
        return { success: false, error: err instanceof Error ? err.message : String(err) };
      }
    }
    return { success: true };
  }

  private async compatManageService(args: Record<string, unknown>): Promise<Record<string, unknown>> {
    const action = String(args.action ?? 'list');
    const serviceType = String(args.service_type ?? 'toolset');
    if (action !== 'list') return { success: true, message: 'Service registry updated for this local session.', services: [] };
    if (serviceType === 'mcp') return { success: true, services: [] };
    const toolsets = await this.getToolsets();
    return { success: true, services: asRecord(toolsets).services ?? [] };
  }

  private async compatNotebook(methodName: string, args: Record<string, unknown>): Promise<Record<string, unknown>> {
    if (methodName === 'complete_request') return { success: true, data: { items: [] }, metadata: { source: 'AutOmicScience' } };
    if (methodName === 'inspect_request') return { success: true, data: { found: false } };
    if (methodName === 'manage_kernel') return { success: true, action: args.action ?? 'status', status: 'not_started', variables: [] };
    if (methodName === 'read_notebook' || methodName === 'notebook_read') {
      const notebookPath = String(args.notebook_path ?? args.file_path ?? '');
      if (!notebookPath) return { success: false, error: 'notebook_path is required' };
      const read = await this.compatReadFile({ file_path: notebookPath });
      if (read.success === false) return read;
      try {
        return { success: true, notebook: JSON.parse(String(read.content ?? '{}')) };
      } catch {
        return { success: false, error: 'Notebook is not valid JSON' };
      }
    }
    return { success: false, error: `${methodName} requires the integrated notebook runtime, which is not active in this local AutOmicScience server.` };
  }

  private resolveWorkspacePath(input: string): string {
    if (!input || input === '.' || input === '/') return this.rootDir;
    const normalized = decodeFilePath(input.replace(/^file:\/\//, ''));
    if (normalized === '~') return this.virtualHomeDir;
    if (normalized.startsWith('~/') || normalized.startsWith('~\\')) {
      return path.resolve(this.virtualHomeDir, normalized.slice(2));
    }
    if (path.isAbsolute(normalized)) return path.normalize(normalized);
    return path.resolve(this.rootDir, normalized);
  }

  private displayWorkspacePath(input: string): string {
    const relative = path.relative(this.rootDir, input);
    return relative && !relative.startsWith('..') && !path.isAbsolute(relative) ? relative.replace(/\\/g, '/') : input;
  }

  private async publishAgentEvent(chatId: string, event: AgentEvent): Promise<void> {
    if (event.type === 'text') {
      await this.publishStream(chatId, { type: 'chunk', chunk: { content: String(event.data ?? '') } });
    } else if (event.type === 'tool_call') {
      for (const call of Array.isArray(event.data) ? event.data : []) {
        const obj = asRecord(call);
        const id = optionalString(obj.id);
        const name = optionalString(obj.name);
        if (id && name) this.streamToolNames.set(`${chatId}:${id}`, name);
        await this.publishStream(chatId, {
          type: 'step_message',
          step_message: {
            id: id ? `${id}:start` : randomUUID(),
            role: 'tool',
            tool_call_id: id,
            name,
            content: JSON.stringify({
              status: 'started',
              name,
              arguments: obj.arguments ?? {},
            }),
            raw_content: call,
          },
        });
      }
    } else if (event.type === 'tool_result') {
      const result = asRecord(event.data);
      const toolCallId = optionalString(result.tool_call_id);
      const toolName = toolCallId ? this.streamToolNames.get(`${chatId}:${toolCallId}`) : undefined;
      await this.publishStream(chatId, {
        type: 'step_message',
        step_message: {
          id: toolCallId ? `${toolCallId}:result` : randomUUID(),
          role: 'tool',
          tool_call_id: toolCallId,
          name: toolName,
          content: typeof event.data === 'string' ? event.data : JSON.stringify(event.data),
          raw_content: event.data,
        },
      });
    } else if (event.type === 'error') {
      await this.publishStream(chatId, { type: 'error', error: event.data });
    }
  }

  private async publishFinalAssistantMessage(chatId: string, message: Message, index: number): Promise<void> {
    const finalMessage = toAOSMessage(message, index);
    const streamMessageId = this.streamMessageIds.get(chatId);
    if (streamMessageId) finalMessage.id = streamMessageId;
    if (!finalMessage.agent_name) finalMessage.agent_name = 'AOS';
    await this.publishStream(chatId, {
      type: 'step_message',
      step_message: finalMessage,
    });
  }

  private async trySlashCommand(chat: CompatChat, incoming: Message[]): Promise<string | null> {
    const last = incoming[incoming.length - 1];
    if (!last || last.role !== 'user') return null;
    const input = messageContentText(last.content).trim();
    if (!input.startsWith('/')) return null;
    if (input === '/help') return 'Available commands: /help, /agents, /status, /clear';
    if (input === '/agents') {
      const agents = await this.getAgents(chat.id);
      return JSON.stringify(agents, null, 2);
    }
    if (input === '/status') {
      return JSON.stringify({ chat_id: chat.id, messages: chat.messages.length, running: chat.running, nats: this.status().nats }, null, 2);
    }
    if (input === '/clear') {
      chat.messages = [];
      await this.saveChat(chat);
      return 'Chat history cleared.';
    }
    return `Unknown command: ${input}`;
  }

  private async agentForChat(chat: CompatChat): Promise<Agent> {
    let agent = this.chatAgents.get(chat.id);
    if (!agent) {
      agent = this.baseAgent.clone({ name: `aos-chat-${chat.id.slice(0, 8)}` });
      await agent.setHistory(chat.messages);
      this.chatAgents.set(chat.id, agent);
    }
    return agent;
  }

  private async savedModelsMethod(parameters: Record<string, unknown>): Promise<Record<string, unknown>> {
    const incoming = asNullableRecord(parameters.saved_models);
    if (incoming) this.savedModels = Object.fromEntries(
      Object.entries(incoming).map(([key, value]) => [key, Array.isArray(value) ? value.map(String) : [String(value)]]),
    );
    return { success: true, saved_models: this.savedModelsView() };
  }

  private setAgentModel(parameters: Record<string, unknown>): Record<string, unknown> {
    const rawModels = Array.isArray(parameters.models)
      ? parameters.models.map(String)
      : (optionalString(parameters.model) ?? optionalString(parameters.models) ?? '')
        .split('->')
        .map((m) => m.trim())
        .filter(Boolean);
    if (rawModels.length === 0) return { success: false, message: 'model is required' };
    const models = this.resolveModelAliases(rawModels);
    this.baseAgent.setModel(models.length > 1 ? models : models[0]);
    for (const agent of this.chatAgents.values()) agent.setModel(models.length > 1 ? models : models[0]);
    return { success: true, message: 'Model updated', models, requested_models: rawModels };
  }

  private async getTokenStats(chatId: string): Promise<Record<string, unknown>> {
    const chat = await this.readChat(chatId);
    const text = (chat?.messages ?? []).map((msg) => typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)).join('\n');
    return {
      success: true,
      chat_id: chatId,
      total: Math.ceil(text.length / 4),
      input: Math.ceil(text.length / 5),
      output: Math.ceil(text.length / 20),
      total_cost: 0,
    };
  }

  private async compressChat(chatId: string): Promise<Record<string, unknown>> {
    const agent = this.chatAgents.get(chatId);
    if (!agent) return { success: true, message: 'No active in-memory agent to compress.' };
    const messages = await agent.compactHistory();
    const chat = await this.readChat(chatId);
    if (chat) {
      chat.messages = messages;
      await this.saveChat(chat);
    }
    return { success: true, message: 'Chat compressed', messages: messages.length };
  }

  private async installStorePackage(parameters: Record<string, unknown>): Promise<Record<string, unknown>> {
    const id = String(parameters.package_id ?? parameters.packageId ?? '');
    if (!id) return { success: false, message: 'package_id is required' };
    const pkg = await this.findPackage(id);
    if (!pkg) return { success: false, message: `Package '${id}' not found` };
    const written = this.installer.install(pkg.type, pkg.name, pkg.content ?? `# ${pkg.display_name}\n\n${pkg.description}`, pkg.files);
    const version = String(parameters.version ?? pkg.latest_version ?? 'latest');
    const recorded = await this.recordInstall(pkg.id, version);
    return {
      success: true,
      id: pkg.id,
      name: pkg.name,
      type: pkg.type,
      version,
      package: pkg,
      install: recorded.install,
      written,
    };
  }

  private async getInstalledStorePackages(): Promise<Record<string, unknown>> {
    const packages = await this.installedPackages();
    return {
      success: true,
      packages,
      installs: installedPackageMap(packages),
    };
  }

  private checkApiKeys(): Record<string, unknown> {
    const providers = {
      openai: !!(process.env.OPENAI_API_KEY || process.env.AOS_OPENAI_API_KEY),
      gemini: !!process.env.GOOGLE_API_KEY,
      anthropic: !!process.env.ANTHROPIC_API_KEY,
    };
    return { success: true, providers };
  }

  private async ollamaStatus(url: string): Promise<Record<string, unknown>> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1500);
    try {
      const resp = await fetch(`${url.replace(/\/$/, '')}/api/tags`, { signal: controller.signal });
      return { success: resp.ok, status: resp.status, available: resp.ok };
    } catch (err) {
      return { success: false, available: false, message: err instanceof Error ? err.message : String(err) };
    } finally {
      clearTimeout(timer);
    }
  }

  private async installedPackages(): Promise<InstalledPackage[]> {
    try {
      const raw = await fs.readFile(this.installedFile, 'utf-8');
      return JSON.parse(raw) as InstalledPackage[];
    } catch {
      const names = (['agent', 'team', 'skill', 'tool'] as const).flatMap((type) =>
        this.installer.listInstalled(type).map((name) => ({
          package_id: name,
          package_name: name,
          package_type: type,
          version: 'local',
          installed_at: new Date(0).toISOString(),
        })),
      );
      return names;
    }
  }

  private async recordInstall(packageId: string, version: string): Promise<Record<string, unknown>> {
    const pkg = await this.findPackage(packageId);
    const installs = (await this.installedPackages()).filter((item) => item.package_id !== packageId);
    const install: InstalledPackage = {
      package_id: packageId,
      package_name: pkg?.name ?? packageId,
      package_type: pkg?.type ?? 'skill',
      version,
      installed_at: new Date().toISOString(),
    };
    installs.push(install);
    await fs.mkdir(this.dataDir, { recursive: true });
    await fs.writeFile(this.installedFile, JSON.stringify(installs, null, 2), 'utf-8');
    return { success: true, install };
  }

  private async recordUninstall(packageId: string): Promise<void> {
    const installs = (await this.installedPackages()).filter((item) => item.package_id !== packageId);
    await fs.mkdir(this.dataDir, { recursive: true });
    await fs.writeFile(this.installedFile, JSON.stringify(installs, null, 2), 'utf-8');
  }

  private models(): string[] {
    return this.modelsProvider ? this.modelsProvider() : this.baseAgent.modelsList;
  }

  private savedModelsView(): Record<string, string[]> {
    const runtimeDefault = this.primaryRuntimeModel();
    return {
      normal: [runtimeDefault],
      deep: [runtimeDefault],
      fast: ['gpt-5.4'],
      ...this.savedModels,
    };
  }

  private resolveModelAliases(models: string[]): string[] {
    const saved = this.savedModelsView();
    const resolved = models.flatMap((model) => {
      const key = model.trim();
      const lower = key.toLowerCase();
      if (lower === 'normal' || lower === 'default' || lower === 'auto') return saved.normal;
      if (lower === 'deep' || lower === 'expensive' || lower === 'smart') return saved.deep;
      if (lower === 'fast') return saved.fast;
      return saved[key] ?? [key];
    });
    return uniquePreserveOrder(resolved);
  }

  private primaryRuntimeModel(): string {
    const envModel = process.env.AOS_MODEL?.trim();
    if (envModel && !isModelAlias(envModel)) return envModel;
    const active = this.baseAgent.modelsList.find((model) => !isModelAlias(model));
    return active ?? 'gpt-5.5';
  }

  private defaultTemplateSummary(): Record<string, unknown> {
    return normalizeTeamTemplate(defaultAOSTemplate(), 'aos-bio-mas');
  }

  private async defaultTemplate(): Promise<Record<string, unknown>> {
    await this.ensureTemplateStorage();
    const read = await this.readTemplateFile('teams/aos-bio-mas.md', true);
    if (read.success && asRecord(read).content) return asRecord(asRecord(read).content);
    return this.defaultTemplateSummary();
  }

  private async ensureTemplateStorage(): Promise<void> {
    await Promise.all([
      fs.mkdir(this.templateKindDir('teams'), { recursive: true }),
      fs.mkdir(this.templateKindDir('agents'), { recursive: true }),
      fs.mkdir(this.templateKindDir('skills'), { recursive: true }),
      fs.mkdir(path.join(this.projectAOSDir, 'skills'), { recursive: true }),
      fs.mkdir(path.join(this.virtualHomeDir, '.aos'), { recursive: true }),
    ]);
    await this.ensureAOSConfigDefaults();
    await this.seedTemplate('agents', 'aos', defaultAOSAgent());
    await this.seedTemplate('agents', 'bioinformatics_planner', {
      id: 'bioinformatics_planner',
      name: 'Bioinformatics Planner',
      description: 'Plans omics analysis, checks inputs, and selects suitable AutOmicScience tools.',
      instructions: 'You are a AutOmicScience planning specialist for reproducible bioinformatics workflows. Decompose the user request, inspect data and dependencies first, then choose the smallest valid workflow.',
      model: 'normal',
      icon: 'compass',
      toolsets: ['aos_default'],
      tags: ['planning', 'bioinformatics'],
    });
    await this.seedTemplate('agents', 'omics_analyst', {
      id: 'omics_analyst',
      name: 'Omics Analyst',
      description: 'Runs single-cell, spatial, annotation, enrichment, and evolutionary analysis steps.',
      instructions: 'You are a AutOmicScience omics analyst. Prefer reproducible scripts, tiny smoke tests, clear artifacts, and explicit dependency reports.',
      model: 'normal',
      icon: 'dna',
      toolsets: ['aos_default'],
      tags: ['omics', 'single-cell', 'spatial'],
    });
    await this.seedTemplate('agents', 'reporter', {
      id: 'reporter',
      name: 'Scientific Reporter',
      description: 'Summarizes methods, outputs, limitations, and next steps for biological analysis.',
      instructions: 'You are a AutOmicScience scientific reporter. Write concise, auditable conclusions with file references, parameters, and limitations.',
      model: 'normal',
      icon: 'file-text',
      toolsets: ['aos_default'],
      tags: ['reporting', 'scientific-writing'],
    });
    await this.seedTemplate('teams', 'aos-bio-mas', defaultAOSTemplate());
    await this.seedTemplate('skills', 'bio-mas-preflight', {
      id: 'bio-mas-preflight',
      name: 'Bio MAS Preflight',
      description: 'Check data availability, runtimes, optional heavy assets, and tiny-test readiness before a bioinformatics MAS run.',
      tags: ['preflight', 'bioinformatics', 'quality-control'],
      content: '# Bio MAS Preflight\n\n1. Confirm the requested organism, assay, input file paths, and expected outputs.\n2. Check installed Python/R/CLI dependencies before running heavy workflows.\n3. Use tiny synthetic data only for smoke tests; never present synthetic data as biological evidence.\n4. Report missing datasets, model weights, or large references explicitly.',
    });
  }

  private async ensureAOSConfigDefaults(): Promise<void> {
    const defaults: Array<[string, string]> = [
      [
        path.join(this.virtualHomeDir, '.aos', '.env'),
        [
          '# AutOmicScience local environment',
          '# Add provider keys in your local runtime only. Do not commit this file.',
          'AOS_MODEL=',
          'AOS_OPENAI_BASE_URL=',
          'AOS_OPENAI_API_KEY=',
          '',
        ].join('\n'),
      ],
      [
        path.join(this.projectAOSDir, 'settings.json'),
        JSON.stringify({
          app_name: 'AutOmicScience',
          workspace_mode: 'project',
          default_team: 'aos-bio-mas',
          telemetry: false,
        }, null, 2) + '\n',
      ],
      [
        path.join(this.projectAOSDir, 'mcp.json'),
        JSON.stringify({
          mcpServers: {},
          notes: 'Register local MCP servers here when the project needs them.',
        }, null, 2) + '\n',
      ],
      [
        path.join(this.projectAOSDir, 'skills', 'SKILLS.md'),
        [
          '# AutOmicScience Skills',
          '',
          'Add project skills under `.aos/skills/<skill-id>/SKILL.md`.',
          'Each skill should describe when to use it, required inputs, dependencies, and expected outputs.',
          '',
        ].join('\n'),
      ],
    ];

    for (const [target, content] of defaults) {
      if (await pathExists(target)) continue;
      await fs.mkdir(path.dirname(target), { recursive: true });
      await fs.writeFile(target, content, 'utf-8');
    }
  }

  private async seedTemplate(kind: TemplateKind, id: string, content: Record<string, unknown>): Promise<void> {
    const target = this.templateFilePath(kind, id);
    if (await pathExists(target)) return;
    const normalized = normalizeTemplateContent(kind, { ...content, id }, id);
    await fs.mkdir(path.dirname(target), { recursive: true });
    await fs.writeFile(target, JSON.stringify(normalized, null, 2), 'utf-8');
    if (kind === 'skills') await this.syncSkillMarkdown(id, normalized);
  }

  private async listTemplateKind(kind: TemplateKind): Promise<Record<string, unknown>[]> {
    const dir = this.templateKindDir(kind);
    const files = await fs.readdir(dir).catch(() => []);
    const entries: Record<string, unknown>[] = [];
    for (const file of files.filter((name) => name.endsWith('.json'))) {
      const id = file.replace(/\.json$/, '');
      const target = path.join(dir, file);
      try {
        const content = normalizeTemplateContent(kind, JSON.parse(await fs.readFile(target, 'utf-8')), id);
        const stat = await fs.stat(target);
        entries.push({
          id: String(content.id ?? id),
          name: String(content.name ?? content.display_name ?? id),
          path: templateRelPath(kind, id),
          source_path: templateRelPath(kind, id),
          scope: 'project',
          type: singularTemplateKind(kind),
          updated_at: stat.mtime.toISOString(),
        });
      } catch {
        // Skip malformed local template files so one bad draft does not break the UI.
      }
    }
    return entries;
  }

  private async resolveTeamAgents(agents: Record<string, unknown>[]): Promise<Record<string, unknown>[]> {
    const resolved: Record<string, unknown>[] = [];
    for (const agent of agents) {
      const id = optionalString(agent.id);
      const hasInlinePrompt = optionalString(agent.instructions) || optionalString(agent.system_prompt);
      if (id && !hasInlinePrompt) {
        const read = await this.readTemplateFile(`agents/${id}.md`, false);
        if (read.success) {
          resolved.push({ ...asRecord(asRecord(read).content), ...agent });
          continue;
        }
      }
      resolved.push(agent);
    }
    return resolved;
  }

  private async syncSkillMarkdown(id: string, content: Record<string, unknown>): Promise<void> {
    const skillDir = path.join(this.projectAOSDir, 'skills', id);
    const skillPath = path.join(skillDir, 'SKILL.md');
    const body = optionalString(content.content) ?? optionalString(content.body) ?? `# ${content.name ?? id}\n\n${content.description ?? ''}`;
    await fs.mkdir(skillDir, { recursive: true });
    await fs.writeFile(skillPath, body.endsWith('\n') ? body : `${body}\n`, 'utf-8');
  }

  private projectSkillMarkdownId(target: string): string | null {
    const rel = path.relative(path.join(this.projectAOSDir, 'skills'), target);
    if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) return null;
    const normalized = rel.replace(/\\/g, '/');
    const match = normalized.match(/^(.+)\/SKILL\.md$/i);
    if (!match) return null;
    return sanitizeTemplateId(match[1]) || slugify(match[1]);
  }

  private async syncSkillTemplateFromWorkspaceWrite(target: string, body: string): Promise<void> {
    const id = this.projectSkillMarkdownId(target);
    if (!id) return;
    const metadata = parseSkillMarkdown(body, id);
    const normalized = normalizeSkillTemplate({ ...metadata, content: body }, id);
    const templatePath = this.templateFilePath('skills', id);
    await fs.mkdir(path.dirname(templatePath), { recursive: true });
    await fs.writeFile(templatePath, JSON.stringify(normalized, null, 2), 'utf-8');
  }

  private async readSkillMarkdownFile(filePath: string): Promise<Record<string, unknown>> {
    const id = skillIdFromMarkdownPath(filePath);
    const target = path.join(this.projectAOSDir, 'skills', id, 'SKILL.md');
    try {
      const body = await fs.readFile(target, 'utf-8');
      const metadata = parseSkillMarkdown(body, id);
      return {
        success: true,
        file_path: `skills/${id}.md`,
        type: 'skill',
        content: normalizeSkillTemplate({ ...metadata, content: body }, id),
      };
    } catch (err: any) {
      if (err?.code === 'ENOENT') return { success: false, error: `Skill file '${filePath}' not found` };
      return { success: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  private templateKindDir(kind: TemplateKind): string {
    return path.join(this.templatesDir, kind);
  }

  private templateFilePath(kind: TemplateKind, id: string): string {
    return path.join(this.templateKindDir(kind), `${sanitizeTemplateId(id)}.json`);
  }

  private chatFile(chatId: string): string {
    return path.join(this.chatsDir, `${chatId}.json`);
  }

  private async readChat(chatId: string): Promise<CompatChat | null> {
    if (!chatId) return null;
    try {
      const raw = await fs.readFile(this.chatFile(chatId), 'utf-8');
      return JSON.parse(raw) as CompatChat;
    } catch {
      return null;
    }
  }

  private async saveChat(chat: CompatChat): Promise<void> {
    await fs.mkdir(this.chatsDir, { recursive: true });
    await fs.writeFile(this.chatFile(chat.id), JSON.stringify(chat, null, 2), 'utf-8');
  }

  private json(res: ServerResponse, status: number, body: unknown): true {
    res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify(body));
    return true;
  }
}

function messageContentText(content: Message['content']): string {
  if (typeof content === 'string') return content;
  return content.map((part) => {
    const obj = asRecord(part);
    if (obj.type === 'text') return String(obj.text ?? '');
    return '';
  }).join('');
}

function normalizeMessage(input: unknown): Message {
  const obj = asRecord(input);
  const role = ['system', 'user', 'assistant', 'tool'].includes(String(obj.role)) ? String(obj.role) as Message['role'] : 'user';
  const content = typeof obj.content === 'string' || Array.isArray(obj.content)
    ? obj.content as Message['content']
    : JSON.stringify(obj.content ?? '');
  const msg: Message = { role, content };
  if (typeof obj.name === 'string') msg.name = obj.name;
  if (typeof obj.tool_call_id === 'string') msg.tool_call_id = obj.tool_call_id;
  if (Array.isArray(obj.tool_calls)) msg.tool_calls = obj.tool_calls as ToolCall[];
  return msg;
}

function formatToolCalls(data: unknown): unknown[] {
  if (!Array.isArray(data)) return [];
  return data.map((call) => {
    const obj = asRecord(call);
    return {
      id: String(obj.id ?? randomUUID()),
      type: 'function',
      function: {
        name: String(obj.name ?? ''),
        arguments: JSON.stringify(obj.arguments ?? {}),
      },
    };
  });
}

function toAOSMessage(message: Message, index: number): Record<string, unknown> {
  const raw = message as Message & Record<string, unknown>;
  const out: Record<string, unknown> = {
    ...raw,
    id: optionalString(raw.id) ?? stableMessageId(message, index),
    role: message.role,
    content: message.content,
    text: messageContentText(message.content),
  };
  if (Array.isArray(raw.tool_calls)) out.tool_calls = raw.tool_calls.map(toAOSToolCall);
  return out;
}

function toAOSToolCall(call: unknown): Record<string, unknown> {
  const obj = asRecord(call);
  const fn = asRecord(obj.function);
  const name = optionalString(fn.name) ?? optionalString(obj.name) ?? '';
  const args = fn.arguments ?? obj.arguments ?? {};
  return {
    ...obj,
    id: optionalString(obj.id) ?? randomUUID(),
    type: optionalString(obj.type) ?? 'function',
    function: {
      ...fn,
      name,
      arguments: typeof args === 'string' ? args : JSON.stringify(args),
    },
  };
}

function stableMessageId(message: Message, index: number): string {
  const content = typeof message.content === 'string' ? message.content : JSON.stringify(message.content);
  return createHash('sha1').update(`${index}:${message.role}:${content}`).digest('hex');
}

function formatLocalTimestamp(date: Date): string {
  const pad = (value: number) => String(value).padStart(2, '0');
  return [
    date.getFullYear(),
    '-',
    pad(date.getMonth() + 1),
    '-',
    pad(date.getDate()),
    ' ',
    pad(date.getHours()),
    ':',
    pad(date.getMinutes()),
    ':',
    pad(date.getSeconds()),
  ].join('');
}

function decodeFilePath(input: string): string {
  try {
    return decodeURIComponent(input);
  } catch {
    return input;
  }
}

function isTextLikePath(filePath: string): boolean {
  const ext = path.extname(filePath).toLowerCase();
  if (!ext) return true;
  return new Set([
    '.txt', '.md', '.markdown', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.csv', '.tsv',
    '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs', '.css', '.scss', '.html', '.xml', '.svg',
    '.py', '.r', '.jl', '.sh', '.ps1', '.bat', '.cmd', '.sql', '.log', '.ini', '.env',
    '.fa', '.fasta', '.fq', '.fastq', '.gff', '.gtf', '.bed', '.vcf', '.sam',
  ]).has(ext);
}

function mimeForPath(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase();
  const map: Record<string, string> = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.svg': 'image/svg+xml',
    '.css': 'text/css',
    '.js': 'application/javascript',
    '.mjs': 'application/javascript',
    '.json': 'application/json',
    '.html': 'text/html',
    '.htm': 'text/html',
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.csv': 'text/csv',
    '.tsv': 'text/tab-separated-values',
    '.pdf': 'application/pdf',
  };
  return map[ext] ?? 'application/octet-stream';
}

function findLastAssistantMessageIndex(messages: Message[]): number {
  for (let index = messages.length - 1; index >= 0; index--) {
    if (messages[index].role === 'assistant') return index;
  }
  return -1;
}

function chatInfo(chat: CompatChat): Record<string, unknown> {
  return {
    id: chat.id,
    name: chat.name,
    running: chat.running,
    last_activity_date: chat.updatedAt,
    project: chat.project,
    workspace_mode: chat.workspaceMode,
    workspace_path: chat.workspacePath,
    chat_config: chat.chatConfig,
    template: chat.template,
    message_count: chat.messages.length,
    memory_path: `aos-compat/chats/${chat.id}.json`,
  };
}

function bodyToStoreEntry(body: Record<string, unknown>): StoreEntry & { content: string } {
  const name = String(body.name ?? body.id ?? `package-${Date.now()}`);
  const type = normalizePackageType(String(body.type ?? body.category ?? 'skill'));
  return {
    id: String(body.id ?? name),
    name,
    category: type,
    version: String(body.version ?? body.latest_version ?? '1.0.0'),
    description: String(body.description ?? ''),
    author: String(body.author ?? body.author_username ?? 'AutOmicScience'),
    downloads: Number(body.downloads ?? 0),
    createdAt: optionalString(body.created_at) ?? optionalString(body.createdAt),
    updatedAt: optionalString(body.updated_at) ?? optionalString(body.updatedAt),
    files: asStringRecord(body.files),
    tags: Array.isArray(body.tags) ? body.tags.map(String) : [],
    content: String(body.content ?? `# ${body.display_name ?? name}\n\n${body.description ?? ''}`),
  };
}

function toAOSPackage(entry: StoreEntry & { content?: string }): any {
  return {
    ...entry,
    type: entry.category,
    display_name: (entry as any).display_name ?? entry.name,
    author_username: entry.author,
    latest_version: entry.version,
    is_public: true,
    source: (entry as any).source ?? 'AutOmicScience',
    source_url: (entry as any).source_url ?? null,
    created_at: entry.createdAt ?? new Date(0).toISOString(),
    updated_at: entry.updatedAt ?? entry.createdAt ?? new Date(0).toISOString(),
    content: (entry as any).content,
  };
}

function versionForPackage(pkg: any): Record<string, unknown> {
  return {
    version: pkg.latest_version ?? pkg.version ?? '1.0.0',
    content: pkg.content ?? `# ${pkg.display_name ?? pkg.name}\n\n${pkg.description ?? ''}`,
    files: pkg.files ?? {},
    created_at: pkg.updated_at ?? pkg.created_at,
    published_at: pkg.updated_at ?? pkg.created_at,
    changelog: pkg.changelog ?? '',
  };
}

function builtinPackages(): any[] {
  const created = new Date(0).toISOString();
  return [
    {
      id: 'aos-bio-mas',
      name: 'aos-bio-mas',
      type: 'skill',
      category: 'bioinformatics',
      display_name: 'AOS Bio MAS',
      description: 'Autonomous bioinformatics MAS with preflight, tiny synthetic smoke tests, model/source selection, adapter execution, and consensus reporting.',
      author: 'AutOmicScience',
      author_username: 'AutOmicScience',
      downloads: 0,
      latest_version: '1.0.0',
      version: '1.0.0',
      is_public: true,
      source: 'AutOmicScience',
      source_url: null,
      tags: ['bioinformatics', 'single-cell', 'MAS', 'agent'],
      created_at: created,
      updated_at: created,
      content: '# AOS Bio MAS\n\nUse `aos annotate bio-mas-preflight` before production runs.',
      files: {},
    },
    {
      id: 'aos-tool-catalog',
      name: 'aos-tool-catalog',
      type: 'skill',
      category: 'runtime',
      display_name: 'AOS Tool Catalog',
      description: 'Searchable tool catalog for file, shell, Python, R, Julia, benchmark, evolution, and bioinformatics workflows.',
      author: 'AutOmicScience',
      author_username: 'AutOmicScience',
      downloads: 0,
      latest_version: '1.0.0',
      version: '1.0.0',
      is_public: true,
      source: 'AutOmicScience',
      source_url: null,
      tags: ['tools', 'catalog'],
      created_at: created,
      updated_at: created,
      content: '# AOS Tool Catalog\n\nAvailable through the default AutOmicScience agent.',
      files: {},
    },
  ];
}

async function readBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  const raw = Buffer.concat(chunks).toString('utf-8').trim();
  if (!raw) return {};
  const contentType = String(req.headers['content-type'] ?? '');
  if (contentType.includes('application/x-www-form-urlencoded')) {
    return Object.fromEntries(new URLSearchParams(raw).entries());
  }
  return asRecord(JSON.parse(raw));
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function asNullableRecord(value: unknown): Record<string, unknown> | null {
  const record = asRecord(value);
  return Object.keys(record).length ? record : null;
}

function asStringRecord(value: unknown): Record<string, string> | undefined {
  const record = asRecord(value);
  const entries = Object.entries(record);
  if (entries.length === 0) return undefined;
  return Object.fromEntries(entries.map(([key, item]) => [key, String(item)]));
}

function optionalString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined;
}

function normalizePackageType(value: string): StoreEntry['category'] {
  if (value === 'agent' || value === 'team' || value === 'skill' || value === 'tool') return value;
  return 'skill';
}

function templateKindsFor(value: string): TemplateKind[] {
  const normalized = value.trim().toLowerCase();
  if (!normalized || normalized === 'teams' || normalized === 'team' || normalized === 'templates' || normalized === 'chatrooms') return ['teams'];
  if (normalized === 'agents' || normalized === 'agent') return ['agents'];
  if (normalized === 'skills' || normalized === 'skill') return ['skills'];
  if (normalized === 'all') return ['teams', 'agents', 'skills'];
  return [];
}

function singularTemplateKind(kind: TemplateKind): string {
  if (kind === 'teams') return 'team';
  if (kind === 'agents') return 'agent';
  return 'skill';
}

function templateRelPath(kind: TemplateKind, id: string): string {
  return `${kind}/${sanitizeTemplateId(id)}.md`;
}

function parseTemplatePath(filePath: string): { kind: TemplateKind; id: string } | null {
  let cleaned = filePath.replace(/\\/g, '/').replace(/\s*\/\s*/g, '/').trim();
  cleaned = cleaned.replace(/^file:\/+/, '');
  cleaned = cleaned.replace(/^\.aos\//, '').replace(/^templates\//, '');
  const parts = cleaned.split('/').filter(Boolean);
  let kindIndex = parts.findIndex((part) => templateKindsFor(part).length === 1);
  if (kindIndex < 0) {
    const match = cleaned.match(/(?:^|\/)(teams|team|templates|chatrooms|agents|agent|skills|skill)\//i);
    if (!match) return null;
    const before = cleaned.slice(0, match.index ?? 0);
    kindIndex = before.split('/').filter(Boolean).length;
  }
  const kind = templateKindsFor(parts[kindIndex])[0];
  if (!kind) return null;
  let id = parts.slice(kindIndex + 1).join('/');
  id = id.replace(/\.(md|json)$/i, '');
  if (kind === 'skills') id = id.replace(/\/SKILL$/i, '');
  id = sanitizeTemplateId(id);
  return id ? { kind, id } : null;
}

function isSkillMarkdownPath(filePath: string): boolean {
  const cleaned = filePath.replace(/\\/g, '/');
  return /(?:^|\/)\.aos\/skills\/.+\/SKILL\.md$/i.test(cleaned)
    || /(?:^|\/)skills\/.+\/SKILL\.md$/i.test(cleaned);
}

function skillIdFromMarkdownPath(filePath: string): string {
  const cleaned = filePath.replace(/\\/g, '/');
  const match = cleaned.match(/(?:^|\/)(?:\.aos\/)?skills\/(.+?)\/SKILL\.md$/i);
  const raw = match?.[1] ?? cleaned.replace(/\/SKILL\.md$/i, '').split('/').pop() ?? 'skill';
  return sanitizeTemplateId(raw) || slugify(raw);
}

function inferTemplateKind(content: Record<string, unknown>): TemplateKind {
  const type = String(content.type ?? content.kind ?? '').toLowerCase();
  if (type === 'team' || type === 'teams' || Array.isArray(content.agents)) return 'teams';
  if (type === 'agent' || type === 'agents' || content.instructions || content.system_prompt) return 'agents';
  return 'skills';
}

function normalizeTemplateContent(kind: TemplateKind, content: Record<string, unknown>, fallbackId: string): Record<string, unknown> {
  if (kind === 'teams') return normalizeTeamTemplate(content, fallbackId);
  if (kind === 'agents') return normalizeAgentTemplate(content, fallbackId);
  return normalizeSkillTemplate(content, fallbackId);
}

function normalizeTeamTemplate(content: Record<string, unknown>, fallbackId: string): Record<string, unknown> {
  const id = optionalString(content.id) ?? (sanitizeTemplateId(fallbackId) || slugify(String(content.name ?? 'team')));
  const agents = arrayOfRecords(content.agents).map((agent, index) => {
    const agentId = optionalString(agent.id) ?? optionalString(agent.name) ?? `agent_${index + 1}`;
    return { ...agent, id: sanitizeTemplateId(agentId) || agentId };
  });
  return {
    ...content,
    id,
    name: String(content.name ?? content.display_name ?? id),
    display_name: String(content.display_name ?? content.name ?? id),
    description: String(content.description ?? ''),
    category: String(content.category ?? 'bioinformatics'),
    icon: String(content.icon ?? 'aos'),
    version: String(content.version ?? '1.0.0'),
    tags: stringArray(content.tags),
    agents,
    type: 'team',
  };
}

function normalizeAgentTemplate(content: Record<string, unknown>, fallbackId: string): Record<string, unknown> {
  const id = optionalString(content.id) ?? (sanitizeTemplateId(fallbackId) || slugify(String(content.name ?? 'agent')));
  return {
    ...content,
    id,
    name: String(content.name ?? id),
    description: String(content.description ?? ''),
    instructions: String(content.instructions ?? content.system_prompt ?? ''),
    model: String(content.model ?? 'normal'),
    icon: String(content.icon ?? 'aos'),
    toolsets: stringArray(content.toolsets),
    mcp_servers: stringArray(content.mcp_servers ?? content.mcp),
    tags: stringArray(content.tags),
    type: 'agent',
  };
}

function normalizeSkillTemplate(content: Record<string, unknown>, fallbackId: string): Record<string, unknown> {
  const id = optionalString(content.id) ?? (sanitizeTemplateId(fallbackId) || slugify(String(content.name ?? 'skill')));
  return {
    ...content,
    id,
    name: String(content.name ?? id),
    description: String(content.description ?? ''),
    content: String(content.content ?? content.body ?? `# ${content.name ?? id}\n\n${content.description ?? ''}`),
    tags: stringArray(content.tags),
    type: 'skill',
  };
}

function sanitizeTemplateId(value: string): string {
  return value
    .replace(/\\/g, '/')
    .split('/')
    .map((part) => part.trim())
    .filter((part) => part && part !== '.' && part !== '..')
    .join('__')
    .replace(/[\0:*?"<>|]/g, '_')
    .replace(/^\.+/, '')
    .trim();
}

function slugify(value: string): string {
  const slug = value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  return slug || `item-${Date.now()}`;
}

function stringArray(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(String).map((item) => item.trim()).filter(Boolean);
  if (typeof value === 'string') return value.split(',').map((item) => item.trim()).filter(Boolean);
  return [];
}

function arrayOfRecords(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.map(asRecord).filter((item) => Object.keys(item).length > 0) : [];
}

function uniqueSorted(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

function uniquePreserveOrder(values: string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter(Boolean))];
}

function isModelAlias(value: string): boolean {
  const normalized = value.trim().toLowerCase();
  return normalized === 'normal' || normalized === 'default' || normalized === 'auto' || normalized === 'deep' || normalized === 'fast' || normalized === 'expensive' || normalized === 'smart';
}

function parseSkillMarkdown(body: string, fallbackId: string): Record<string, unknown> {
  const title = body.match(/^#\s+(.+)$/m)?.[1]?.trim();
  const frontmatter = body.match(/^---\n([\s\S]*?)\n---/);
  const meta: Record<string, unknown> = { id: fallbackId, name: title ?? fallbackId, description: '' };
  if (!frontmatter) return meta;
  const yaml = frontmatter[1];
  const id = yaml.match(/^id:\s*(.+)$/m)?.[1]?.trim();
  const name = yaml.match(/^name:\s*(.+)$/m)?.[1]?.trim();
  const description = yaml.match(/^description:\s*(.+)$/m)?.[1]?.trim();
  if (id) meta.id = id;
  if (name) meta.name = name;
  if (description) meta.description = description;
  return meta;
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function defaultAOSAgent(): Record<string, unknown> {
  return {
    id: 'aos',
    name: 'AOS',
    description: 'Bioinformatics intelligent agent developed by AutOmicScience.',
    instructions: 'You are AOS, a bioinformatics intelligent agent developed by AutOmicScience. Plan multi-step work, inspect files, call tools, run reproducible analysis, and report results with clear limitations.',
    model: 'normal',
    icon: 'aos',
    toolsets: ['aos_default'],
    tags: ['bioinformatics', 'agent'],
  };
}

function defaultAOSTemplate(): Record<string, unknown> {
  const planner = normalizeAgentTemplate({
    id: 'bioinformatics_planner',
    name: 'Bioinformatics Planner',
    description: 'Plans omics analysis, checks inputs, and selects suitable AutOmicScience tools.',
    instructions: 'You are a AutOmicScience planning specialist for reproducible bioinformatics workflows. Decompose the user request, inspect data and dependencies first, then choose the smallest valid workflow.',
    model: 'normal',
    icon: 'compass',
    toolsets: ['aos_default'],
    tags: ['planning', 'bioinformatics'],
  }, 'bioinformatics_planner');
  const analyst = normalizeAgentTemplate({
    id: 'omics_analyst',
    name: 'Omics Analyst',
    description: 'Runs single-cell, spatial, annotation, enrichment, and evolutionary analysis steps.',
    instructions: 'You are a AutOmicScience omics analyst. Prefer reproducible scripts, tiny smoke tests, clear artifacts, and explicit dependency reports.',
    model: 'normal',
    icon: 'dna',
    toolsets: ['aos_default'],
    tags: ['omics', 'single-cell', 'spatial'],
  }, 'omics_analyst');
  const reporter = normalizeAgentTemplate({
    id: 'reporter',
    name: 'Scientific Reporter',
    description: 'Summarizes methods, outputs, limitations, and next steps for biological analysis.',
    instructions: 'You are a AutOmicScience scientific reporter. Write concise, auditable conclusions with file references, parameters, and limitations.',
    model: 'normal',
    icon: 'file-text',
    toolsets: ['aos_default'],
    tags: ['reporting', 'scientific-writing'],
  }, 'reporter');
  return normalizeTeamTemplate({
    id: 'aos-bio-mas',
    name: 'AOS Bio MAS',
    display_name: 'AOS Bio MAS',
    description: 'Autonomous bioinformatics MAS with planning, omics analysis, dependency preflight, tool execution, and scientific reporting.',
    category: 'bioinformatics',
    icon: 'aos',
    version: '1.0.0',
    tags: ['bioinformatics', 'single-cell', 'spatial', 'MAS'],
    agents: [planner, analyst, reporter],
  }, 'aos-bio-mas');
}

function countBy(items: any[], key: string): Record<string, number> {
  const result: Record<string, number> = {};
  for (const item of items) {
    const value = String(item[key] ?? 'unknown');
    result[value] = (result[value] ?? 0) + 1;
  }
  return result;
}

function countEntries(counts: Record<string, number>): { name: string; count: number }[] {
  return Object.entries(counts)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

function installedPackageMap(installs: InstalledPackage[]): Record<string, Record<string, unknown>> {
  return Object.fromEntries(installs.map((install) => [
    install.package_id,
    {
      id: install.package_id,
      name: install.package_name,
      type: install.package_type,
      version: install.version,
      installed_at: install.installed_at,
    },
  ]));
}
