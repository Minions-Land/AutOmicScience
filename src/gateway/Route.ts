import { createHash } from 'crypto';
import type { ChannelType } from './Channel.js';

export interface ConversationRoute {
  channel: ChannelType;
  scopeType: string;
  scopeId: string;
  threadId?: string;
  senderId?: string;
}

export function routeKey(route: ConversationRoute): string {
  if (route.threadId) {
    return `${route.channel}:${route.scopeType}:${route.scopeId}:thread:${route.threadId}`;
  }
  return `${route.channel}:${route.scopeType}:${route.scopeId}`;
}

export function stableShortId(route: ConversationRoute): string {
  return createHash('sha1').update(routeKey(route)).digest('hex').slice(0, 12);
}

export function isDirect(route: ConversationRoute): boolean {
  return ['dm', 'direct', 'private', 'p2p'].includes(route.scopeType);
}
