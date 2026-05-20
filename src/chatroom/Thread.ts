import type { Message } from '../types.js';

// --- Thread Types ---

export type ThreadState = 'open' | 'closed' | 'archived';

export interface ThreadMetadata {
  title?: string;
  createdAt: number;
  updatedAt: number;
  closedAt?: number;
  archivedAt?: number;
  tags?: string[];
  /** Custom metadata attached by agents or users. */
  extra?: Record<string, unknown>;
}

export interface ThreadParticipant {
  id: string;
  name: string;
  role: 'user' | 'agent' | 'system';
  joinedAt: number;
  lastActiveAt: number;
}

export interface Thread {
  id: string;
  roomName: string;
  parentThreadId?: string;
  childThreadIds: string[];
  state: ThreadState;
  participants: ThreadParticipant[];
  metadata: ThreadMetadata;
  messages: Message[];
}

// --- Hooks ---

export type ChunkHook = (chunk: Record<string, unknown>) => void | Promise<void>;
export type StepMessageHook = (stepMessage: Record<string, unknown>) => void | Promise<void>;

// --- NatsThread ---

/**
 * Full-featured NATS-backed thread with parent/child relationships,
 * participant tracking, state management, and hook-based streaming.
 */
export class NatsThread implements Thread {
  readonly id: string;
  readonly roomName: string;
  parentThreadId?: string;
  childThreadIds: string[] = [];
  state: ThreadState = 'open';
  participants: ThreadParticipant[] = [];
  metadata: ThreadMetadata;
  messages: Message[] = [];

  private nc: any = null;
  private codec: any = null;
  private sub: any = null;
  private _chunkHooks: ChunkHook[] = [];
  private _stepMessageHooks: StepMessageHook[] = [];
  private _stopFlag = false;
  private _hookTimeout: number;
  private _hookRetries: number;

  constructor(
    roomName: string,
    threadId: string,
    opts?: {
      parentThreadId?: string;
      title?: string;
      hookTimeout?: number;
      hookRetries?: number;
    },
  ) {
    this.roomName = roomName;
    this.id = threadId;
    this.parentThreadId = opts?.parentThreadId;
    this._hookTimeout = opts?.hookTimeout ?? 1000;
    this._hookRetries = opts?.hookRetries ?? 5;
    this.metadata = {
      title: opts?.title,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
  }

  private subject(): string {
    return `medrix.room.${this.roomName}.thread.${this.id}`;
  }

  private controlSubject(): string {
    return `medrix.room.${this.roomName}.thread.${this.id}.control`;
  }

  // --- Lifecycle ---

  /** Start listening for messages on this thread. */
  async subscribe(natsUrl?: string): Promise<void> {
    const mod: any = await import('nats');
    const { connect, JSONCodec } = mod;
    this.nc = await connect({
      servers: natsUrl ?? process.env.NATS_URL ?? 'nats://localhost:4222',
    });
    this.codec = JSONCodec();

    // Subscribe to messages
    this.sub = this.nc.subscribe(this.subject());
    (async () => {
      for await (const m of this.sub) {
        try {
          const msg = this.codec.decode(m.data) as Message;
          this.messages.push(msg);
          this.metadata.updatedAt = Date.now();
        } catch {
          /* ignore malformed */
        }
      }
    })();

    // Subscribe to control channel (state changes, participant updates)
    const controlSub = this.nc.subscribe(this.controlSubject());
    (async () => {
      for await (const m of controlSub) {
        try {
          const ctrl = this.codec.decode(m.data) as Record<string, unknown>;
          this._handleControl(ctrl);
        } catch {
          /* ignore */
        }
      }
    })();
  }

  /** Publish a message to this thread. */
  async publish(message: Message): Promise<void> {
    if (!this.nc) throw new Error('Thread not subscribed - call subscribe() first');
    this.nc.publish(this.subject(), this.codec.encode(message));
    this.messages.push(message);
    this.metadata.updatedAt = Date.now();
  }

  /** Unsubscribe and disconnect. */
  async close(): Promise<void> {
    if (this.sub) this.sub.unsubscribe();
    if (this.nc) await this.nc.drain().catch(() => {});
    this.nc = null;
  }

  // --- State Management ---

  /** Close the thread (no more messages accepted). */
  async closeThread(): Promise<void> {
    this.state = 'closed';
    this.metadata.closedAt = Date.now();
    this.metadata.updatedAt = Date.now();
    await this._publishControl({ type: 'state_change', state: 'closed' });
  }

  /** Archive the thread. */
  async archive(): Promise<void> {
    this.state = 'archived';
    this.metadata.archivedAt = Date.now();
    this.metadata.updatedAt = Date.now();
    await this._publishControl({ type: 'state_change', state: 'archived' });
  }

  /** Reopen a closed thread. */
  async reopen(): Promise<void> {
    if (this.state === 'archived') {
      throw new Error('Cannot reopen an archived thread');
    }
    this.state = 'open';
    this.metadata.closedAt = undefined;
    this.metadata.updatedAt = Date.now();
    await this._publishControl({ type: 'state_change', state: 'open' });
  }

  // --- Participants ---

  addParticipant(participant: Omit<ThreadParticipant, 'joinedAt' | 'lastActiveAt'>): void {
    const existing = this.participants.find((p) => p.id === participant.id);
    if (existing) {
      existing.lastActiveAt = Date.now();
      return;
    }
    this.participants.push({
      ...participant,
      joinedAt: Date.now(),
      lastActiveAt: Date.now(),
    });
  }

  removeParticipant(participantId: string): boolean {
    const idx = this.participants.findIndex((p) => p.id === participantId);
    if (idx === -1) return false;
    this.participants.splice(idx, 1);
    return true;
  }

  getParticipant(participantId: string): ThreadParticipant | undefined {
    return this.participants.find((p) => p.id === participantId);
  }

  // --- Child Threads ---

  addChildThread(childId: string): void {
    if (!this.childThreadIds.includes(childId)) {
      this.childThreadIds.push(childId);
    }
  }

  removeChildThread(childId: string): boolean {
    const idx = this.childThreadIds.indexOf(childId);
    if (idx === -1) return false;
    this.childThreadIds.splice(idx, 1);
    return true;
  }

  // --- Hooks ---

  addChunkHook(hook: ChunkHook): void {
    this._chunkHooks.push(hook);
  }

  addStepMessageHook(hook: StepMessageHook): void {
    this._stepMessageHooks.push(hook);
  }

  removeChunkHook(hook: ChunkHook): void {
    this._chunkHooks = this._chunkHooks.filter((h) => h !== hook);
  }

  removeStepMessageHook(hook: StepMessageHook): void {
    this._stepMessageHooks = this._stepMessageHooks.filter((h) => h !== hook);
  }

  /** Process a streaming chunk through all registered hooks. */
  async processChunk(chunk: Record<string, unknown>): Promise<void> {
    const enriched = { ...chunk, chat_id: this.id };
    const promises = this._chunkHooks.map((hook) =>
      this._runHookWithRetry(hook, enriched),
    );
    await Promise.allSettled(promises);
  }

  /** Process a step message through all registered hooks. */
  async processStepMessage(stepMessage: Record<string, unknown>): Promise<void> {
    const enriched = { ...stepMessage, chat_id: this.id };
    const promises = this._stepMessageHooks.map((hook) =>
      this._runHookWithRetry(hook, enriched),
    );
    await Promise.allSettled(promises);
  }

  // --- Stop Control ---

  /** Signal the thread to stop processing. */
  stop(): void {
    this._stopFlag = true;
  }

  /** Check if the thread has been signaled to stop. */
  get shouldStop(): boolean {
    return this._stopFlag;
  }

  /** Reset the stop flag (for reuse). */
  resetStop(): void {
    this._stopFlag = false;
  }

  // --- Private Helpers ---

  private async _runHookWithRetry(
    hook: ChunkHook | StepMessageHook,
    data: Record<string, unknown>,
  ): Promise<void> {
    for (let attempt = 0; attempt < this._hookRetries; attempt++) {
      try {
        const result = (hook as (d: Record<string, unknown>) => void | Promise<void>)(data);
        if (result instanceof Promise) {
          await Promise.race([
            result,
            new Promise<never>((_, reject) =>
              setTimeout(() => reject(new Error('Hook timeout')), this._hookTimeout),
            ),
          ]);
        }
        return;
      } catch {
        // Retry on failure
      }
    }
    // Remove hook after exhausting retries
    this._chunkHooks = this._chunkHooks.filter((h) => h !== hook);
    this._stepMessageHooks = this._stepMessageHooks.filter((h) => h !== hook);
  }

  private _handleControl(ctrl: Record<string, unknown>): void {
    switch (ctrl.type) {
      case 'state_change':
        this.state = ctrl.state as ThreadState;
        this.metadata.updatedAt = Date.now();
        break;
      case 'participant_join': {
        const p = ctrl.participant as ThreadParticipant;
        if (p) this.addParticipant(p);
        break;
      }
      case 'participant_leave':
        this.removeParticipant(ctrl.participantId as string);
        break;
    }
  }

  private async _publishControl(data: Record<string, unknown>): Promise<void> {
    if (!this.nc) return;
    this.nc.publish(this.controlSubject(), this.codec.encode(data));
  }

  // --- Serialization ---

  toJSON(): Record<string, unknown> {
    return {
      id: this.id,
      roomName: this.roomName,
      parentThreadId: this.parentThreadId,
      childThreadIds: this.childThreadIds,
      state: this.state,
      participants: this.participants,
      metadata: this.metadata,
      messageCount: this.messages.length,
    };
  }
}
