import type { ChannelAdapter, ChannelBridge } from '../ChannelAdapter.js';
import type { ConversationRoute } from '../Route.js';

export class TelegramAdapter implements ChannelAdapter {
  readonly channel = 'telegram';
  private running = false;
  private bridge: ChannelBridge | null = null;
  private token = '';
  private allowedUsers: string[] = [];
  private offset = 0;
  private pollTimer: ReturnType<typeof setTimeout> | null = null;

  async start(config: Record<string, unknown>, bridge: ChannelBridge): Promise<void> {
    this.token = config.token as string;
    if (!this.token) throw new Error('Telegram requires a bot token');
    this.allowedUsers = (config.allowedUsers as string[]) ?? [];
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
        const resp = await fetch(
          `https://api.telegram.org/bot${this.token}/getUpdates?offset=${this.offset}&timeout=30`,
          { signal: AbortSignal.timeout(35000) },
        );
        const data = await resp.json() as any;
        if (data.ok && Array.isArray(data.result)) {
          for (const update of data.result) {
            this.offset = update.update_id + 1;
            await this.handleUpdate(update);
          }
        }
      } catch {
        if (this.running) await new Promise((r) => setTimeout(r, 3000));
      }
    }
  }

  private async handleUpdate(update: any): Promise<void> {
    const msg = update.message ?? update.edited_message;
    if (!msg?.text) return;

    const userId = String(msg.from?.id ?? '');
    if (this.allowedUsers.length > 0 && !this.allowedUsers.includes(userId)) return;

    const chatId = String(msg.chat.id);
    const route: ConversationRoute = {
      channel: 'telegram',
      scopeType: msg.chat.type === 'private' ? 'dm' : 'group',
      scopeId: chatId,
      senderId: userId,
    };

    const text = msg.text as string;
    if (text.startsWith('/')) {
      const cmdResult = await this.bridge!.handleCommand(route, text);
      if (cmdResult.handled && cmdResult.message) {
        await this.sendMessage(chatId, cmdResult.message);
      }
      return;
    }

    const reply = await this.bridge!.handleMessage(route, {
      channelId: chatId,
      channelType: 'telegram',
      from: msg.from?.username ?? userId,
      text,
      ts: msg.date * 1000,
      metadata: { messageId: msg.message_id },
    });
    if (reply) await this.sendMessage(chatId, reply);
  }

  private async sendMessage(chatId: string, text: string): Promise<void> {
    await fetch(`https://api.telegram.org/bot${this.token}/sendMessage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chat_id: chatId, text, parse_mode: 'Markdown' }),
    });
  }
}
