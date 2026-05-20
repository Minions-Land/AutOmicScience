import { Room, type RoomMessage } from './Room.js';

export interface NatsRoomOptions {
  url?: string;
  identity: string;
  roomName: string;
}

/** NATS-backed pub/sub chatroom. Lazy-imports `nats`. */
export class NatsRoom extends Room {
  public readonly name: string;
  private readonly url: string;
  private readonly identity: string;
  private nc: any = null;
  private codec: any = null;

  constructor(opts: NatsRoomOptions) {
    super();
    this.name = opts.roomName;
    this.identity = opts.identity;
    this.url = opts.url ?? process.env.NATS_URL ?? 'nats://localhost:4222';
  }

  private subjectFor(subject: string): string {
    return `pantheon.room.${this.name}.${subject}`;
  }

  private async ensure(): Promise<void> {
    if (this.nc) return;
    const mod: any = await import('nats');
    const { connect, JSONCodec } = mod;
    this.nc = await connect({ servers: this.url });
    this.codec = JSONCodec();
  }

  async publish(subject: string, body: unknown): Promise<void> {
    await this.ensure();
    const msg: RoomMessage = { from: this.identity, subject, body, ts: Date.now() };
    this.nc.publish(this.subjectFor(subject), this.codec.encode(msg));
  }

  async subscribe(
    subject: string,
    handler: (msg: RoomMessage) => void | Promise<void>,
  ): Promise<() => Promise<void>> {
    await this.ensure();
    const sub = this.nc.subscribe(this.subjectFor(subject));
    (async () => {
      for await (const m of sub) {
        try {
          const data = this.codec.decode(m.data) as RoomMessage;
          await handler(data);
        } catch {
          // ignore malformed
        }
      }
    })();
    return async () => {
      try {
        await sub.unsubscribe();
      } catch {
        // ignore
      }
    };
  }

  async close(): Promise<void> {
    if (this.nc) {
      try {
        await this.nc.drain();
      } catch {
        // ignore
      }
      this.nc = null;
    }
  }
}
