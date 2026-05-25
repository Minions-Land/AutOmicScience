import type { GatewayMessage } from './GatewayMessage.js';
import type { ConversationRoute } from './Route.js';

export interface ChannelAdapter {
  readonly channel: string;
  start(config: Record<string, unknown>, bridge: ChannelBridge): Promise<void>;
  stop(): Promise<void>;
  sendReply(route: ConversationRoute, text: string): Promise<void>;
}

export interface ChannelBridge {
  handleMessage(route: ConversationRoute, msg: GatewayMessage): Promise<string>;
  handleCommand(route: ConversationRoute, text: string): Promise<{ handled: boolean; message?: string }>;
}

export type ChannelStatus = 'stopped' | 'starting' | 'running' | 'stopping' | 'failed' | 'not_configured';

export interface ChannelState {
  channel: string;
  status: ChannelStatus;
  running: boolean;
  configured: boolean;
  supported: boolean;
  error?: string;
  logLines: number;
}
