import type { ChannelAdapter, ChannelBridge } from '../ChannelAdapter.js';
import type { ConversationRoute } from '../Route.js';

export class FeishuAdapter implements ChannelAdapter {
  readonly channel = 'feishu';
  private running = false;
  private bridge: ChannelBridge | null = null;
  private appId = '';
  private appSecret = '';
  private tenantAccessToken = '';
  private ws: any = null;

  async start(config: Record<string, unknown>, bridge: ChannelBridge): Promise<void> {
    this.appId = config.appId as string;
    this.appSecret = config.appSecret as string;
    if (!this.appId || !this.appSecret) throw new Error('Feishu requires appId and appSecret');
    this.bridge = bridge;
    this.running = true;

    await this.refreshToken();
    const connectionMode = (config.connectionMode as string) ?? 'websocket';

    if (connectionMode === 'websocket') {
      await this.connectWebSocket();
    } else {
      // Webhook mode would require an HTTP server - simplified here
      await this.connectWebSocket();
    }
  }

  async stop(): Promise<void> {
    this.running = false;
    if (this.ws) this.ws.close();
  }

  async sendReply(route: ConversationRoute, text: string): Promise<void> {
    await this.sendMessage(route.scopeId, text);
  }

  private async refreshToken(): Promise<void> {
    const resp = await fetch('https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ app_id: this.appId, app_secret: this.appSecret }),
    });
    const data = await resp.json() as any;
    this.tenantAccessToken = data.tenant_access_token ?? '';
  }

  private async connectWebSocket(): Promise<void> {
    const resp = await fetch('https://open.feishu.cn/open-apis/callback/ws/endpoint', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.tenantAccessToken}`,
      },
      body: JSON.stringify({}),
    });
    const data = await resp.json() as any;
    const wsUrl = data.data?.URL ?? data.data?.url;
    if (!wsUrl) throw new Error('Failed to get Feishu WebSocket URL');

    const WebSocket = (await import('ws')).default;
    this.ws = new WebSocket(wsUrl);

    this.ws.on('message', async (raw: Buffer) => {
      try {
        const payload = JSON.parse(raw.toString());
        if (payload.header?.event_type === 'im.message.receive_v1') {
          await this.handleEvent(payload.event);
        }
      } catch { /* ignore */ }
    });
  }

  private async handleEvent(event: any): Promise<void> {
    const msg = event?.message;
    if (!msg) return;
    const content = JSON.parse(msg.content ?? '{}');
    const text = content.text ?? '';
    if (!text) return;

    const chatId = msg.chat_id;
    const senderId = event.sender?.sender_id?.user_id ?? '';
    const route: ConversationRoute = {
      channel: 'feishu',
      scopeType: msg.chat_type === 'p2p' ? 'dm' : 'group',
      scopeId: chatId,
      senderId,
    };

    if (text.startsWith('/')) {
      const cmdResult = await this.bridge!.handleCommand(route, text);
      if (cmdResult.handled && cmdResult.message) {
        await this.sendMessage(chatId, cmdResult.message);
      }
      return;
    }

    const reply = await this.bridge!.handleMessage(route, {
      channelId: chatId,
      channelType: 'feishu',
      from: senderId,
      text,
      ts: Date.now(),
      metadata: { messageId: msg.message_id },
    });
    if (reply) await this.sendMessage(chatId, reply);
  }

  private async sendMessage(chatId: string, text: string): Promise<void> {
    await this.refreshToken();
    await fetch('https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.tenantAccessToken}`,
      },
      body: JSON.stringify({
        receive_id: chatId,
        msg_type: 'text',
        content: JSON.stringify({ text }),
      }),
    });
  }
}
