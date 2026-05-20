import type { Channel } from './Channel.js';

/**
 * An inbound message received from a gateway channel.
 */
export interface GatewayMessage {
  /** The channel this message came from. */
  channelId: string;
  /** The type of the originating channel. */
  channelType: Channel['type'];
  /** Sender identifier (platform-specific user id or name). */
  from: string;
  /** The text content of the message. */
  text: string;
  /** Unix timestamp in milliseconds. */
  ts: number;
  /** Optional platform-specific metadata. */
  metadata?: Record<string, unknown>;
}
