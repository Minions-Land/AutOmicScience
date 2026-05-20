import type { AgentEvent } from '../types.js';
import type { RemoteConfig } from './RemoteConfig.js';
import { defaultRemoteConfig } from './RemoteConfig.js';
import { uid } from '../utils/misc.js';

/**
 * A proxy for an agent running on a remote worker node.
 * Communicates over NATS to invoke the agent and stream back events.
 */
export interface RemoteAgent {
  /** The name of the remote agent. */
  name: string;
  /** Invoke the remote agent with the given input, streaming back events. */
  invoke(input: string): AsyncGenerator<AgentEvent>;
}

/**
 * NATS-backed RemoteAgent that publishes invocation requests and
 * subscribes to the reply subject for streamed events.
 *
 * Publishes to: `medrix.remote.<namespace>.<agentName>.invoke`
 */
export class NatsRemoteAgent implements RemoteAgent {
  readonly name: string;
  private config: RemoteConfig;

  constructor(name: string, config?: Partial<RemoteConfig>) {
    this.name = name;
    this.config = { ...defaultRemoteConfig, ...config };
  }

  /**
   * Invoke the remote agent by publishing to its NATS subject.
   * Subscribes to a unique reply subject and yields events as they arrive.
   */
  async *invoke(input: string): AsyncGenerator<AgentEvent> {
    const mod: any = await import('nats');
    const { connect, JSONCodec, createInbox } = mod;
    const nc = await connect({ servers: this.config.natsUrl });
    const codec = JSONCodec();

    const subject = `medrix.remote.${this.config.namespace}.${this.name}.invoke`;
    const replySubject = createInbox?.() ?? `_INBOX.${uid('reply')}`;

    try {
      // Subscribe to the reply subject before publishing
      const sub = nc.subscribe(replySubject, { max: 1, timeout: this.config.timeout });

      // Publish the invocation request with reply subject
      nc.publish(subject, codec.encode({ input }), { reply: replySubject });

      for await (const msg of sub) {
        const response = codec.decode(msg.data) as { events: AgentEvent[] };
        for (const ev of response.events) {
          yield ev;
        }
      }
    } catch (err) {
      yield { type: 'error', data: { message: (err as Error).message } };
      yield { type: 'done', data: '' };
    } finally {
      try {
        await nc.drain();
      } catch {
        // ignore
      }
    }
  }
}
