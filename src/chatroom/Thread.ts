import type { Message } from '../types.js';

export interface Thread {
  id: string;
  roomName: string;
  messages: Message[];
}

/**
 * NATS-backed thread that subscribes to a thread-specific subject
 * and accumulates messages in order.
 */
export class NatsThread implements Thread {
  readonly id: string;
  readonly roomName: string;
  messages: Message[] = [];
  private nc: any = null;
  private codec: any = null;
  private sub: any = null;

  constructor(roomName: string, threadId: string) {
    this.roomName = roomName;
    this.id = threadId;
  }

  private subject(): string {
    return `medrix.room.${this.roomName}.thread.${this.id}`;
  }

  /** Start listening for messages on this thread. */
  async subscribe(natsUrl?: string): Promise<void> {
    const mod: any = await import('nats');
    const { connect, JSONCodec } = mod;
    this.nc = await connect({ servers: natsUrl ?? process.env.NATS_URL ?? 'nats://localhost:4222' });
    this.codec = JSONCodec();
    this.sub = this.nc.subscribe(this.subject());
    (async () => {
      for await (const m of this.sub) {
        try {
          const msg = this.codec.decode(m.data) as Message;
          this.messages.push(msg);
        } catch { /* ignore malformed */ }
      }
    })();
  }

  /** Publish a message to this thread. */
  async publish(message: Message): Promise<void> {
    if (!this.nc) throw new Error('Thread not subscribed — call subscribe() first');
    this.nc.publish(this.subject(), this.codec.encode(message));
    this.messages.push(message);
  }

  /** Unsubscribe and disconnect. */
  async close(): Promise<void> {
    if (this.sub) this.sub.unsubscribe();
    if (this.nc) await this.nc.drain().catch(() => {});
    this.nc = null;
  }
}
