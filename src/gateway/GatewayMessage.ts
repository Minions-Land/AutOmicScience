import type { ChannelType } from './Channel.js';

export interface GatewayMessage {
  channelId: string;
  channelType: ChannelType;
  from: string;
  text: string;
  ts: number;
  threadId?: string;
  imageUris?: string[];
  metadata?: Record<string, unknown>;
}

export interface GatewayReply {
  text: string;
  channelId: string;
  ts: number;
  metadata?: Record<string, unknown>;
}
