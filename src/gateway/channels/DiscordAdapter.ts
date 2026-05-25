import type { ChannelAdapter, ChannelBridge } from '../ChannelAdapter.js';
import type { ConversationRoute } from '../Route.js';

export class DiscordAdapter implements ChannelAdapter {
  readonly channel = 'discord';
  private running = false;
  private bridge: ChannelBridge | null = null;
  private token = '';
  private ws: any = null;
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  private lastSequence: number | null = null;

  async start(config: Record<string, unknown>, bridge: ChannelBridge): Promise<void> {
    this.token = config.token as string;
    if (!this.token) throw new Error('Discord requires a bot token');
    this.bridge = bridge;
    this.running = true;

    const gatewayResp = await fetch('https://discord.com/api/v10/gateway/bot', {
      headers: { Authorization: `Bot ${this.token}` },
    });
    const gatewayData = await gatewayResp.json() as any;
    const wsUrl = gatewayData.url + '?v=10&encoding=json';

    const WebSocket = (await import('ws')).default;
    this.ws = new WebSocket(wsUrl);

    this.ws.on('message', async (data: Buffer) => {
      const payload = JSON.parse(data.toString());
      this.lastSequence = payload.s;

      switch (payload.op) {
        case 10: // Hello
          this.startHeartbeat(payload.d.heartbeat_interval);
          this.identify();
          break;
        case 0: // Dispatch
          if (payload.t === 'MESSAGE_CREATE') {
            await this.handleMessage(payload.d);
          }
          break;
      }
    });

    this.ws.on('close', () => { this.running = false; });
  }

  async stop(): Promise<void> {
    this.running = false;
    if (this.heartbeatInterval) clearInterval(this.heartbeatInterval);
    if (this.ws) this.ws.close();
  }

  async sendReply(route: ConversationRoute, text: string): Promise<void> {
    await this.sendChannelMessage(route.scopeId, text);
  }

  private identify(): void {
    this.ws.send(JSON.stringify({
      op: 2,
      d: {
        token: this.token,
        intents: 513, // GUILDS + GUILD_MESSAGES
        properties: { os: 'linux', browser: 'aos', device: 'aos' },
      },
    }));
  }

  private startHeartbeat(interval: number): void {
    this.heartbeatInterval = setInterval(() => {
      this.ws?.send(JSON.stringify({ op: 1, d: this.lastSequence }));
    }, interval);
  }

  private async handleMessage(msg: any): Promise<void> {
    if (msg.author?.bot) return;
    if (!msg.content) return;

    const route: ConversationRoute = {
      channel: 'discord',
      scopeType: msg.guild_id ? 'guild' : 'dm',
      scopeId: msg.channel_id,
      senderId: msg.author?.id,
    };

    const text = msg.content as string;
    if (text.startsWith('/')) {
      const cmdResult = await this.bridge!.handleCommand(route, text);
      if (cmdResult.handled && cmdResult.message) {
        await this.sendChannelMessage(msg.channel_id, cmdResult.message);
      }
      return;
    }

    const reply = await this.bridge!.handleMessage(route, {
      channelId: msg.channel_id,
      channelType: 'discord',
      from: msg.author?.username ?? msg.author?.id,
      text,
      ts: Date.now(),
      metadata: { messageId: msg.id, guildId: msg.guild_id },
    });
    if (reply) await this.sendChannelMessage(msg.channel_id, reply);
  }

  private async sendChannelMessage(channelId: string, text: string): Promise<void> {
    await fetch(`https://discord.com/api/v10/channels/${channelId}/messages`, {
      method: 'POST',
      headers: { Authorization: `Bot ${this.token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: text }),
    });
  }
}
