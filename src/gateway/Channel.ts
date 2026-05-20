/**
 * A communication channel configuration for the gateway.
 */
export interface Channel {
  id: string;
  type: ChannelType;
  config: Record<string, unknown>;
}

export type ChannelType =
  | 'slack'
  | 'telegram'
  | 'discord'
  | 'wechat'
  | 'feishu'
  | 'qq'
  | 'imessage'
  | 'webhook';

export const ALL_CHANNELS: ChannelType[] = [
  'slack',
  'telegram',
  'discord',
  'wechat',
  'feishu',
  'qq',
  'imessage',
];

export interface ChannelConfig {
  slack: { appToken: string; botToken: string };
  telegram: { token: string; allowedUsers?: string[] };
  discord: { token: string };
  wechat: { token: string; baseUrl?: string; allowFrom?: string[] };
  feishu: { appId: string; appSecret: string; connectionMode?: 'websocket' | 'webhook' };
  qq: { appId: string; clientSecret: string; markdown?: boolean };
  imessage: { cliPath?: string; dbPath?: string };
  webhook: { url: string };
}
