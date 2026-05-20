import type { Channel } from './Channel.js';
import type { GatewayMessage } from './GatewayMessage.js';

/**
 * Multi-channel gateway that connects external messaging platforms
 * (Slack, Telegram, Discord, Lark, WeChat, webhooks) to Novaeve agents.
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
