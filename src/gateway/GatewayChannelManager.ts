import type { ChannelType } from './Channel.js';
import { ALL_CHANNELS } from './Channel.js';
import type { ChannelAdapter, ChannelBridge, ChannelState } from './ChannelAdapter.js';
import { GatewayConfigStore, channelConfigured } from './GatewayConfig.js';
import { RouteRegistry } from './RouteRegistry.js';
import type { ConversationRoute } from './Route.js';
import type { GatewayMessage } from './GatewayMessage.js';

type MessageHandler = (route: ConversationRoute, msg: GatewayMessage) => Promise<string>;

const ADAPTER_MODULES: Record<string, () => Promise<{ new(): ChannelAdapter }>> = {
  slack: async () => (await import('./channels/SlackAdapter.js')).SlackAdapter,
  telegram: async () => (await import('./channels/TelegramAdapter.js')).TelegramAdapter,
  discord: async () => (await import('./channels/DiscordAdapter.js')).DiscordAdapter,
  wechat: async () => (await import('./channels/WeChatAdapter.js')).WeChatAdapter,
  feishu: async () => (await import('./channels/FeishuAdapter.js')).FeishuAdapter,
  qq: async () => (await import('./channels/QQAdapter.js')).QQAdapter,
  imessage: async () => (await import('./channels/IMessageAdapter.js')).IMessageAdapter,
};

export class GatewayChannelManager {
  private readonly configStore: GatewayConfigStore;
  private readonly registry: RouteRegistry;
  private readonly adapters: Map<string, ChannelAdapter> = new Map();
  private readonly states: Map<string, ChannelState> = new Map();
  private readonly logs: Map<string, string[]> = new Map();
  private messageHandler: MessageHandler | null = null;

  constructor(opts?: { configPath?: string; registryPath?: string }) {
    this.configStore = new GatewayConfigStore(opts?.configPath);
    this.registry = new RouteRegistry(opts?.registryPath);
  }

  onMessage(handler: MessageHandler): void {
    this.messageHandler = handler;
  }

  getConfig(masked = true): Record<string, any> {
    return masked ? this.configStore.loadMasked() : this.configStore.load();
  }

  saveConfig(config: Record<string, any>): Record<string, any> {
    return this.configStore.save(config);
  }

  listStates(): ChannelState[] {
    const cfg = this.configStore.load();
    return ALL_CHANNELS.map((channel) => {
      const configured = channelConfigured(channel, cfg);
      const existing = this.states.get(channel);
      const status = existing?.status ?? (configured ? 'stopped' : 'not_configured');
      const running = existing?.running ?? false;
      return {
        channel,
        status,
        running,
        configured,
        supported: channel in ADAPTER_MODULES,
        error: existing?.error,
        logLines: this.logs.get(channel)?.length ?? 0,
      };
    });
  }

  async startChannel(channel: ChannelType): Promise<{ ok: boolean; error?: string }> {
    const cfg = this.configStore.load();
    if (!(channel in ADAPTER_MODULES)) {
      this.setState(channel, 'failed', false, `${channel} is not supported`);
      return { ok: false, error: `${channel} is not supported` };
    }
    if (!channelConfigured(channel, cfg)) {
      this.setState(channel, 'failed', false, `${channel} is not configured`);
      return { ok: false, error: `${channel} is not configured` };
    }
    if (this.adapters.has(channel)) {
      return { ok: false, error: `${channel} is already running` };
    }

    try {
      const AdapterClass = await ADAPTER_MODULES[channel]();
      const adapter = new AdapterClass();
      const bridge = this.createBridge();
      this.adapters.set(channel, adapter);
      this.setState(channel, 'starting', false);
      this.log(channel, `[start] channel=${channel}`);

      await adapter.start(cfg[channel] ?? {}, bridge);
      this.setState(channel, 'running', true);
      this.log(channel, `[running] channel=${channel}`);
      return { ok: true };
    } catch (err: any) {
      this.setState(channel, 'failed', false, err.message);
      this.log(channel, `[error] ${err.message}`);
      this.adapters.delete(channel);
      return { ok: false, error: err.message };
    }
  }

  async stopChannel(channel: ChannelType): Promise<{ ok: boolean; error?: string }> {
    const adapter = this.adapters.get(channel);
    if (!adapter) {
      return { ok: false, error: `${channel} is not running` };
    }
    this.setState(channel, 'stopping', false);
    try {
      await adapter.stop();
    } catch { /* ignore */ }
    this.adapters.delete(channel);
    this.setState(channel, 'stopped', false);
    this.log(channel, `[stopped] channel=${channel}`);
    return { ok: true };
  }

  async stopAll(): Promise<void> {
    const channels = [...this.adapters.keys()] as ChannelType[];
    await Promise.all(channels.map((ch) => this.stopChannel(ch)));
  }

  getLogs(channel: string): string {
    return (this.logs.get(channel) ?? []).join('\n');
  }

  getRegistry(): RouteRegistry {
    return this.registry;
  }

  private createBridge(): ChannelBridge {
    return {
      handleMessage: async (route, msg) => {
        if (!this.messageHandler) return '';
        return this.messageHandler(route, msg);
      },
      handleCommand: async (route, text) => {
        return this.handleControlCommand(route, text);
      },
    };
  }

  private async handleControlCommand(
    route: ConversationRoute,
    text: string,
  ): Promise<{ handled: boolean; message?: string }> {
    const parts = text.trim().split(/\s+/);
    const cmd = parts[0]?.toLowerCase() ?? '';
    const args = parts.slice(1).join(' ');

    switch (cmd) {
      case '/help':
      case '/menu':
        return { handled: true, message: this.helpMenu() };
      case '/status':
        return { handled: true, message: this.formatStatus(route) };
      case '/new':
        this.registry.remove(route);
        return { handled: true, message: 'Started new conversation for this route.' };
      case '/reset':
        this.registry.remove(route);
        return { handled: true, message: 'Route reset. Next message will create a new chat.' };
      case '/list': {
        const entries = this.registry.listEntries()
          .filter((e) => e.channel === route.channel && e.scopeId === route.scopeId);
        if (entries.length === 0) return { handled: true, message: 'No routed chats found.' };
        const lines = entries.map((e, i) => `${i + 1}. ${e.chatName} (${e.chatId})`);
        return { handled: true, message: `Routed chats:\n${lines.join('\n')}` };
      }
      default:
        return { handled: false };
    }
  }

  private helpMenu(): string {
    return [
      'AutOmicScience Gateway Commands',
      '/help    - Show this menu',
      '/status  - Show route status',
      '/new     - Start a fresh conversation',
      '/list    - List routed chats',
      '/reset   - Delete route mapping',
    ].join('\n');
  }

  private formatStatus(route: ConversationRoute): string {
    const entry = this.registry.get(route);
    if (!entry) return 'Status: idle (no active route)';
    return `Status: active\nChat: ${entry.chatName} (${entry.chatId})\nUpdated: ${entry.updatedAt}`;
  }

  private setState(channel: string, status: ChannelState['status'], running: boolean, error?: string): void {
    const cfg = this.configStore.load();
    this.states.set(channel, {
      channel,
      status,
      running,
      configured: channelConfigured(channel as ChannelType, cfg),
      supported: channel in ADAPTER_MODULES,
      error,
      logLines: this.logs.get(channel)?.length ?? 0,
    });
  }

  private log(channel: string, message: string): void {
    const logs = this.logs.get(channel) ?? [];
    logs.push(`${new Date().toISOString()} ${message}`);
    if (logs.length > 200) logs.shift();
    this.logs.set(channel, logs);
  }
}
