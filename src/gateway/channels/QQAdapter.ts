import type { ChannelAdapter, ChannelBridge } from '../ChannelAdapter.js';
import type { ConversationRoute } from '../Route.js';

export class QQAdapter implements ChannelAdapter {
  readonly channel = 'qq';
  private running = false;
  private bridge: ChannelBridge | null = null;
  private appId = '';
  private clientSecret = '';
  private accessToken = '';
  private ws: any = null;
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;

  async start(config: Record<string, unknown>, bridge: ChannelBridge): Promise<void> {
    this.appId = config.appId as string;
    this.clientSecret = config.clientSecret as string;
    if (!this.appId || !this.clientSecret) throw new Error('QQ requires appId and clientSecret');
    this.bridge = bridge;
    this.running = true;

    await this.authenticate();
    await this.connectGateway();
  }

  async stop(): Promise<void> {
    this.running = false;
    if (this.heartbeatInterval) clearInterval(this.heartbeatInterval);
    if (this.ws) this.ws.close();
  }

  async sendReply(route: ConversationRoute, text: string): Promise<void> {
    await this.sendMessage(route.scopeId, text);
  }

  private async authenticate(): Promise<void> {
    const resp = await fetch('https://bots.qq.com/app/getAppAccessToken', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ appId: this.appId, clientSecret: this.clientSecret }),
    });
    const data = await resp.json() as any;
    this.accessToken = data.access_token ?? '';
  }

  private async connectGateway(): Promise<void> {
    const resp = await fetch('https://api.sgroup.qq.com/gateway', {
      headers: { Authorization: `QQBot ${this.accessToken}` },
    });
    const data = await resp.json() as any;
    const wsUrl = data.url;
    if (!wsUrl) throw new Error('Failed to get QQ gateway URL');

    const WebSocket = (await import('ws')).default;
    this.ws = new WebSocket(wsUrl);

    this.ws.on('message', async (raw: Buffer) => {
      try {
        const payload = JSON.parse(raw.toString());
        switch (payload.op) {
          case 10: // Hello
            this.startHeartbeat(payload.d.heartbeat_interval);
            this.identify();
            break;
          case 0: // Dispatch
            if (payload.t === 'AT_MESSAGE_CREATE' || payload.t === 'MESSAGE_CREATE') {
              await this.handleMessage(payload.d);
            }
            break;
        }
      } catch { /* ignore */ }
    });
  }

  private identify(): void {
    this.ws.send(JSON.stringify({
      op: 2,
      d: { token: `QQBot ${this.accessToken}`, intents: 1 | (1 << 30), shard: [0, 1] },
    }));
  }

  private startHeartbeat(interval: number): void {
    this.heartbeatInterval = setInterval(() => {
      this.ws?.send(JSON.stringify({ op: 1, d: null }));
    }, interval);
  }

  private async handleMessage(msg: any): Promise<void> {
    const text = msg.content?.replace(/<@!?\d+>\s*/g, '').trim() ?? '';
    if (!text) return;

    const channelId = msg.channel_id ?? msg.group_id ?? '';
    const senderId = msg.author?.id ?? '';
    const route: ConversationRoute = {
      channel: 'qq',
      scopeType: msg.guild_id ? 'guild' : 'group',
      scopeId: channelId,
      senderId,
    };

    if (text.startsWith('/')) {
      const cmdResult = await this.bridge!.handleCommand(route, text);
      if (cmdResult.handled && cmdResult.message) {
        await this.replyToMessage(channelId, msg.id, cmdResult.message);
      }
      return;
    }

    const reply = await this.bridge!.handleMessage(route, {
      channelId,
      channelType: 'qq',
      from: msg.author?.username ?? senderId,
      text,
      ts: Date.now(),
      metadata: { messageId: msg.id, guildId: msg.guild_id },
    });
    if (reply) await this.replyToMessage(channelId, msg.id, reply);
  }

  private async replyToMessage(channelId: string, msgId: string, text: string): Promise<void> {
    await fetch(`https://api.sgroup.qq.com/channels/${channelId}/messages`, {
      method: 'POST',
      headers: { Authorization: `QQBot ${this.accessToken}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: text, msg_id: msgId }),
    });
  }

  private async sendMessage(channelId: string, text: string): Promise<void> {
    await fetch(`https://api.sgroup.qq.com/channels/${channelId}/messages`, {
      method: 'POST',
      headers: { Authorization: `QQBot ${this.accessToken}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: text }),
    });
  }
}
