import type { Channel } from './Channel.js';
import type { GatewayMessage } from './GatewayMessage.js';

/**
 * Multi-channel gateway that connects external messaging platforms
 * (Slack, Telegram, Discord, Lark, WeChat, webhooks) to MedrixAI agents.
 */
export interface Gateway {
  /** Connect a new channel to the gateway. */
  connect(channel: Channel): Promise<void>;
  /** Disconnect a channel by its id. */
  disconnect(channelId: string): Promise<void>;
  /** Register a handler for incoming messages. Returns the agent's reply. */
  onMessage(handler: (msg: GatewayMessage) => Promise<string>): void;
}

/**
 * Stub gateway implementation that stores channels in memory.
 */
export class StubGateway implements Gateway {
  private channels = new Map<string, Channel>();
  private handler: ((msg: GatewayMessage) => Promise<string>) | null = null;

  /** Connect a new channel. */
  async connect(channel: Channel): Promise<void> {
    this.channels.set(channel.id, channel);
  }

  /** Disconnect a channel by id. */
  async disconnect(channelId: string): Promise<void> {
    this.channels.delete(channelId);
  }

  /** Register the message handler. */
  onMessage(handler: (msg: GatewayMessage) => Promise<string>): void {
    this.handler = handler;
  }

  /**
   * Simulate receiving a message (for testing).
   * Calls the registered handler and returns the response.
   */
  async simulateMessage(msg: GatewayMessage): Promise<string> {
    if (!this.handler) return '';
    return this.handler(msg);
  }
}

/**
 * Webhook-based gateway that receives messages via HTTP POST
 * and sends replies back to a configured webhook URL.
 *
 * Channel config shape for webhook:
 * ```
 * { id: string, type: 'webhook', config: { url: string } }
 * ```
 *
 * Channel config shapes for other platforms (stubs):
 * - Slack: `{ config: { botToken: string, signingSecret: string, channel: string } }`
 * - Telegram: `{ config: { botToken: string, chatId: string } }`
 * - Discord: `{ config: { botToken: string, guildId: string, channelId: string } }`
 * - Lark: `{ config: { appId: string, appSecret: string, chatId: string } }`
 * - WeChat: `{ config: { appId: string, appSecret: string, token: string } }`
 */
export class WebhookGateway implements Gateway {
  private channels = new Map<string, Channel>();
  private webhookUrls = new Map<string, string>();
  private handler: ((msg: GatewayMessage) => Promise<string>) | null = null;

  /**
   * Connect a channel. For webhook type, stores the webhook URL.
   * For other types, throws 'not implemented'.
   */
  async connect(channel: Channel): Promise<void> {
    if (channel.type !== 'webhook') {
      throw new Error(
        `Gateway adapter for '${channel.type}' is not implemented. ` +
        `Supported types: webhook.`,
      );
    }
    const url = channel.config.url as string | undefined;
    if (!url) {
      throw new Error(`Webhook channel '${channel.id}' requires a 'url' in config.`);
    }
    this.channels.set(channel.id, channel);
    this.webhookUrls.set(channel.id, url);
  }

  /** Disconnect a channel by id. */
  async disconnect(channelId: string): Promise<void> {
    this.channels.delete(channelId);
    this.webhookUrls.delete(channelId);
  }

  /** Register the message handler. */
  onMessage(handler: (msg: GatewayMessage) => Promise<string>): void {
    this.handler = handler;
  }

  /**
   * Receive an incoming webhook payload, parse it, call the handler,
   * and POST the reply back to the channel's webhook URL.
   *
   * Expected payload shape: `{ from: string, text: string, channelId?: string }`
   */
  async receiveWebhook(body: unknown): Promise<string> {
    const payload = body as { from?: string; text?: string; channelId?: string };
    const from = payload.from ?? 'webhook-user';
    const text = payload.text ?? '';

    // Determine which channel this belongs to
    let channelId: string;
    if (payload.channelId && this.channels.has(payload.channelId)) {
      channelId = payload.channelId;
    } else {
      // Default to first connected webhook channel
      const firstId = Array.from(this.channels.keys())[0];
      if (!firstId) throw new Error('No webhook channels connected.');
      channelId = firstId;
    }

    const channel = this.channels.get(channelId)!;
    const msg: GatewayMessage = {
      channelId,
      channelType: channel.type,
      from,
      text,
      ts: Date.now(),
    };

    if (!this.handler) return '';
    const reply = await this.handler(msg);

    // Send reply back to webhook URL
    const webhookUrl = this.webhookUrls.get(channelId);
    if (webhookUrl) {
      try {
        await fetch(webhookUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: reply, channelId, ts: Date.now() }),
        });
      } catch {
        // Best-effort delivery; don't throw on webhook failure
      }
    }

    return reply;
  }
}
