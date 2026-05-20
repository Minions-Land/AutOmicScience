/**
 * A communication channel configuration for the gateway.
 */
export interface Channel {
  /** Unique channel identifier. */
  id: string;
  /** The type of messaging platform. */
  type: 'slack' | 'telegram' | 'discord' | 'lark' | 'wechat' | 'webhook';
  /** Platform-specific configuration (tokens, webhook URLs, etc.). */
  config: Record<string, unknown>;
}
