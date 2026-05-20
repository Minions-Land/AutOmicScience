import type { ChannelAdapter, ChannelBridge } from '../ChannelAdapter.js';
import type { ConversationRoute } from '../Route.js';

export class WeChatAdapter implements ChannelAdapter {
  readonly channel = 'wechat';
  private running = false;
  private bridge: ChannelBridge | null = null;
  private token = '';
  private baseUrl = 'https://ilinkai.weixin.qq.com';
  private allowFrom: string[] = [];
  private pollTimer: ReturnType<typeof setTimeout> | null = null;

  async start(config: Record<string, unknown>, bridge: ChannelBridge): Promise<void> {
    this.token = config.token as string;
    if (!this.token) throw new Error('WeChat requires a bot token');
    this.baseUrl = (config.baseUrl as string) ?? this.baseUrl;
    this.allowFrom = (config.allowFrom as string[]) ?? [];
    this.bridge = bridge;
    this.running = true;
    this.poll();
  }

  async stop(): Promise<void> {
    this.running = false;
    if (this.pollTimer) clearTimeout(this.pollTimer);
  }

  async sendReply(route: ConversationRoute, text: string): Promise<void> {
    await this.sendMessage(route.scopeId, text);
  }

  private async poll(): Promise<void> {
    while (this.running) {
      try {
        const resp = await fetch(`${this.baseUrl}/ilink/bot/get_messages`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'AuthorizationType': 'ilink_bot_token',
            'Authorization': `Bearer ${this.token}`,
          },
          body: JSON.stringify({ timeout: 30 }),
          signal: AbortSignal.timeout(35000),
        });
        const data = await resp.json() as any;
        const messages = data.messages ?? data.data ?? [];
        for (const msg of messages) {
          await this.handleMessage(msg);
        }
      } catch {
        if (this.running) await new Promise((r) => setTimeout(r, 3000));
      }
    }
  }

  private async handleMessage(msg: any): Promise<void> {
    const senderId = msg.from_user ?? msg.sender ?? '';
    if (this.allowFrom.length > 0 && !this.allowFrom.includes(senderId)) return;

    const text = msg.content ?? msg.text ?? '';
    if (!text) return;

    const chatId = msg.conversation_id ?? msg.chat_id ?? senderId;
    const route: ConversationRoute = {
      channel: 'wechat',
      scopeType: msg.is_group ? 'group' : 'dm',
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
      channelType: 'wechat',
      from: senderId,
      text,
      ts: Date.now(),
      metadata: { msgId: msg.msg_id },
    });
    if (reply) await this.sendMessage(chatId, reply);
  }

  private async sendMessage(chatId: string, text: string): Promise<void> {
    await fetch(`${this.baseUrl}/ilink/bot/send_message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'AuthorizationType': 'ilink_bot_token',
        'Authorization': `Bearer ${this.token}`,
      },
      body: JSON.stringify({ conversation_id: chatId, content: text, msg_type: 'text' }),
    });
  }
}
