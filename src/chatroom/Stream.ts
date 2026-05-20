import type { AgentEvent } from '../types.js';

// --- Stream Types ---

export interface StreamMessage {
  type: 'chat' | 'system' | 'notification';
  sessionId: string;
  timestamp: number;
  data: Record<string, unknown>;
}

export interface StreamSubscriber {
  id: string;
  handler: (ev: AgentEvent) => void;
  filter?: (ev: AgentEvent) => boolean;
}

export interface MessageStream {
  subscribe(handler: (ev: AgentEvent) => void): () => void;
}

// --- NatsStream (expanded with backpressure, fan-out, replay) ---

/**
 * NATS-backed stream for real-time agent event delivery.
 * Supports backpressure handling, multi-subscriber fan-out,
 * and stream replay from offset.
 */
export class NatsStream implements MessageStream {
  private nc: any = null;
  private codec: any = null;
  private sub: any = null;
  private js: any = null;
  private subscribers: StreamSubscriber[] = [];
  private buffer: AgentEvent[] = [];
  private readonly maxBufferSize: number;
  private readonly replayCapacity: number;
  private replayBuffer: AgentEvent[] = [];
  private offset = 0;
  private _connected = false;
  private _paused = false;
  private _draining = false;

  constructor(
    private readonly roomName: string,
    private readonly agentName: string,
    opts?: {
      maxBufferSize?: number;
      replayCapacity?: number;
    },
  ) {
    this.maxBufferSize = opts?.maxBufferSize ?? 1000;
    this.replayCapacity = opts?.replayCapacity ?? 500;
  }

  private subject(): string {
    return `medrix.room.${this.roomName}.stream.${this.agentName}`;
  }

  get connected(): boolean {
    return this._connected;
  }

  get paused(): boolean {
    return this._paused;
  }

  get currentOffset(): number {
    return this.offset;
  }

  /** Connect to NATS and start receiving events. */
  async connect(natsUrl?: string): Promise<void> {
    const mod: any = await import('nats');
    const { connect, JSONCodec } = mod;
    this.nc = await connect({
      servers: natsUrl ?? process.env.NATS_URL ?? 'nats://localhost:4222',
    });
    this.codec = JSONCodec();
    this._connected = true;

    // Try to get JetStream for persistence (optional)
    try {
      this.js = this.nc.jetstream();
    } catch {
      this.js = null;
    }

    this.sub = this.nc.subscribe(this.subject());
    this._startConsumer();
  }

  private _startConsumer(): void {
    (async () => {
      for await (const m of this.sub) {
        if (this._draining) break;
        try {
          const ev = this.codec.decode(m.data) as AgentEvent;
          this.offset++;

          // Store in replay buffer (circular)
          this.replayBuffer.push(ev);
          if (this.replayBuffer.length > this.replayCapacity) {
            this.replayBuffer.shift();
          }

          // Backpressure: buffer if paused
          if (this._paused) {
            if (this.buffer.length < this.maxBufferSize) {
              this.buffer.push(ev);
            }
            // Drop if buffer full (backpressure signal)
            continue;
          }

          this._fanOut(ev);
        } catch {
          /* ignore malformed */
        }
      }
    })();
  }

  /** Fan out event to all subscribers. */
  private _fanOut(ev: AgentEvent): void {
    for (const sub of this.subscribers) {
      try {
        if (sub.filter && !sub.filter(ev)) continue;
        sub.handler(ev);
      } catch {
        /* subscriber error - don't crash the stream */
      }
    }
  }

  /** Subscribe a handler to incoming events. Returns unsubscribe function. */
  subscribe(handler: (ev: AgentEvent) => void, filter?: (ev: AgentEvent) => boolean): () => void {
    const id = `sub_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const subscriber: StreamSubscriber = { id, handler, filter };
    this.subscribers.push(subscriber);
    return () => {
      this.subscribers = this.subscribers.filter((s) => s.id !== id);
    };
  }

  /** Publish an event to this stream (used by the agent-side). */
  async publish(ev: AgentEvent): Promise<void> {
    if (!this.nc) throw new Error('Stream not connected');
    this.nc.publish(this.subject(), this.codec.encode(ev));
  }

  /** Publish a typed stream message with session context. */
  async publishMessage(msg: StreamMessage): Promise<void> {
    if (!this.nc) throw new Error('Stream not connected');
    this.nc.publish(this.subject(), this.codec.encode(msg));
  }

  // --- Backpressure ---

  /** Pause event delivery (events are buffered). */
  pause(): void {
    this._paused = true;
  }

  /** Resume event delivery and flush buffered events. */
  resume(): void {
    this._paused = false;
    // Flush buffer
    const buffered = this.buffer.splice(0);
    for (const ev of buffered) {
      this._fanOut(ev);
    }
  }

  /** Get current buffer size (backpressure indicator). */
  get bufferSize(): number {
    return this.buffer.length;
  }

  // --- Replay ---

  /**
   * Replay events from a given offset.
   * Returns events from the replay buffer starting at the requested offset.
   */
  replay(fromOffset: number): AgentEvent[] {
    const bufferStartOffset = this.offset - this.replayBuffer.length;
    if (fromOffset < bufferStartOffset) {
      // Requested offset is before our buffer - return what we have
      return [...this.replayBuffer];
    }
    const startIdx = fromOffset - bufferStartOffset;
    return this.replayBuffer.slice(startIdx);
  }

  /** Get the oldest available offset for replay. */
  get oldestReplayOffset(): number {
    return Math.max(0, this.offset - this.replayBuffer.length);
  }

  // --- Lifecycle ---

  /** Disconnect and clean up. */
  async close(): Promise<void> {
    this._draining = true;
    if (this.sub) this.sub.unsubscribe();
    if (this.nc) await this.nc.drain().catch(() => {});
    this.nc = null;
    this.js = null;
    this._connected = false;
    this.subscribers = [];
    this.buffer = [];
  }
}

// --- NatsStreamAdapter ---

/**
 * Adapter for adding NATS streaming capability to the RoomManager.
 * Creates hooks for chunk and step message streaming.
 */
export class NatsStreamAdapter {
  private nc: any = null;
  private codec: any = null;

  async ensureConnected(natsUrl?: string): Promise<void> {
    if (this.nc) return;
    const mod: any = await import('nats');
    const { connect, JSONCodec } = mod;
    this.nc = await connect({
      servers: natsUrl ?? process.env.NATS_URL ?? 'nats://localhost:4222',
    });
    this.codec = JSONCodec();
  }

  /** Publish a message to a chat stream. */
  async publish(chatId: string, messageType: string, data: Record<string, unknown>): Promise<void> {
    if (!this.nc) return;
    const subject = `medrix.chat.${chatId}.stream`;
    const payload: StreamMessage = {
      type: 'chat',
      sessionId: `chat_${chatId}`,
      timestamp: Date.now(),
      data: { ...data, chat_id: chatId },
    };
    this.nc.publish(subject, this.codec.encode(payload));
  }

  /**
   * Create streaming hooks for a chat session.
   * Returns [chunkHook, stepHook] to attach to a thread.
   */
  createHooks(chatId: string): [
    (chunk: Record<string, unknown>) => Promise<void>,
    (stepMessage: Record<string, unknown>) => Promise<void>,
  ] {
    const toolCallState: Record<string, string> = {};

    const chunkHook = async (chunk: Record<string, unknown>): Promise<void> => {
      // Detect tool_calls argument deltas
      const toolCalls = chunk.tool_calls;
      if (Array.isArray(toolCalls)) {
        for (const tc of toolCalls) {
          if (typeof tc !== 'object' || tc === null) continue;
          const fn = (tc as Record<string, unknown>).function as Record<string, unknown> | undefined;
          if (!fn) continue;
          const name = fn.name as string | undefined;
          const argsDelta = fn.arguments as string | undefined;
          if (name) toolCallState.name = name;
          if (argsDelta && toolCallState.name) {
            await this.publish(chatId, 'tool_delta', {
              type: 'tool_delta',
              tool_name: toolCallState.name,
              delta: argsDelta,
            });
          }
        }
        return;
      }

      // Check for begin/stop signals
      if (chunk.begin || chunk.stop) {
        delete toolCallState.name;
        if (chunk.stop) {
          await this.publish(chatId, 'chunk', { type: 'chunk', chunk });
        }
        return;
      }

      // Regular text chunk
      if (chunk.content) {
        delete toolCallState.name;
      }
      await this.publish(chatId, 'chunk', { type: 'chunk', chunk });
    };

    const stepHook = async (stepMessage: Record<string, unknown>): Promise<void> => {
      // Filter out user messages to avoid duplication
      if (stepMessage.role === 'user') return;
      delete toolCallState.name;
      await this.publish(chatId, 'step', {
        type: 'step_message',
        step_message: stepMessage,
      });
    };

    return [chunkHook, stepHook];
  }

  /** Publish chat finished signal. */
  async publishChatFinished(chatId: string): Promise<void> {
    await this.publish(chatId, 'chat_finished', { type: 'chat_finished' });
  }

  /** Disconnect. */
  async close(): Promise<void> {
    if (this.nc) {
      await this.nc.drain().catch(() => {});
      this.nc = null;
    }
  }
}
