import type { AgentEvent } from '../types.js';

export interface MessageStream {
  /** Subscribe to real-time agent events. Returns an unsubscribe function. */
  subscribe(handler: (ev: AgentEvent) => void): () => void;
}

/**
 * NATS-backed stream for real-time agent event delivery.
 * Subscribes to `medrix.room.<room>.stream.<agentName>`.
 */
export class NatsStream implements MessageStream {
  private nc: any = null;
  private codec: any = null;
  private sub: any = null;
  private handlers: Array<(ev: AgentEvent) => void> = [];

  constructor(
    private readonly roomName: string,
    private readonly agentName: string,
  ) {}

  private subject(): string {
    return `medrix.room.${this.roomName}.stream.${this.agentName}`;
  }

  /** Connect to NATS and start receiving events. */
  async connect(natsUrl?: string): Promise<void> {
    const mod: any = await import('nats');
    const { connect, JSONCodec } = mod;
    this.nc = await connect({ servers: natsUrl ?? process.env.NATS_URL ?? 'nats://localhost:4222' });
    this.codec = JSONCodec();
    this.sub = this.nc.subscribe(this.subject());
    (async () => {
      for await (const m of this.sub) {
        try {
          const ev = this.codec.decode(m.data) as AgentEvent;
          for (const h of this.handlers) h(ev);
        } catch { /* ignore */ }
      }
    })();
  }

  /** Subscribe a handler to incoming events. */
  subscribe(handler: (ev: AgentEvent) => void): () => void {
    this.handlers.push(handler);
    return () => {
      this.handlers = this.handlers.filter((h) => h !== handler);
    };
  }

  /** Publish an event to this stream (used by the agent-side). */
  async publish(ev: AgentEvent): Promise<void> {
    if (!this.nc) throw new Error('Stream not connected');
    this.nc.publish(this.subject(), this.codec.encode(ev));
  }

  /** Disconnect. */
  async close(): Promise<void> {
    if (this.sub) this.sub.unsubscribe();
    if (this.nc) await this.nc.drain().catch(() => {});
    this.nc = null;
    this.handlers = [];
  }
}
