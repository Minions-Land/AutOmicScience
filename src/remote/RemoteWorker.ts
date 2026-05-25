import type { Agent } from '../agent/Agent.js';
import type { AgentEvent } from '../types.js';
import type { RemoteConfig } from './RemoteConfig.js';
import { defaultRemoteConfig } from './RemoteConfig.js';
import { logger } from '../utils/logger.js';

/**
 * A worker node that registers local agents and listens for
 * remote invocations over NATS.
 */
export interface RemoteWorker {
  /** Register an agent to be available for remote invocation. */
  register(agent: Agent): Promise<void>;
  /** Start listening for invocations on NATS. */
  start(): Promise<void>;
  /** Stop listening and disconnect. */
  stop(): Promise<void>;
}

/**
 * NATS-backed RemoteWorker that subscribes to invocation subjects
 * and dispatches to registered agents.
 *
 * Subject pattern: `aos.remote.<namespace>.<agentName>.invoke`
 */
export class NatsRemoteWorker implements RemoteWorker {
  private config: RemoteConfig;
  private agents = new Map<string, Agent>();
  private running = false;
  private nc: any = null;
  private codec: any = null;
  private subscriptions: any[] = [];

  constructor(config?: Partial<RemoteConfig>) {
    this.config = { ...defaultRemoteConfig, ...config };
  }

  /** Register an agent to be available for remote invocation. */
  async register(agent: Agent): Promise<void> {
    this.agents.set(agent.name, agent);
  }

  /**
   * Start listening for invocations on NATS.
   * Subscribes to `aos.remote.<namespace>.<agentName>.invoke` for each registered agent.
   */
  async start(): Promise<void> {
    const mod: any = await import('nats');
    const { connect, JSONCodec } = mod;
    this.nc = await connect({ servers: this.config.natsUrl });
    this.codec = JSONCodec();
    this.running = true;

    for (const [name, agent] of this.agents.entries()) {
      const subject = `aos.remote.${this.config.namespace}.${name}.invoke`;
      const sub = this.nc.subscribe(subject);
      this.subscriptions.push(sub);

      (async () => {
        for await (const msg of sub) {
          try {
            const payload = this.codec.decode(msg.data) as { input: string };
            const events: AgentEvent[] = [];
            for await (const ev of agent.run(payload.input)) {
              events.push(ev);
            }
            if (msg.reply) {
              this.nc.publish(msg.reply, this.codec.encode({ events }));
            }
          } catch (err) {
            logger.warn(`RemoteWorker error handling invocation for '${name}':`, (err as Error).message);
            if (msg.reply) {
              this.nc.publish(
                msg.reply,
                this.codec.encode({ events: [{ type: 'error', data: { message: (err as Error).message } }] }),
              );
            }
          }
        }
      })();
    }
  }

  /** Stop listening and disconnect from NATS. */
  async stop(): Promise<void> {
    this.running = false;
    for (const sub of this.subscriptions) {
      try {
        sub.unsubscribe();
      } catch {
        // ignore
      }
    }
    this.subscriptions = [];
    if (this.nc) {
      try {
        await this.nc.drain();
      } catch {
        // ignore
      }
      this.nc = null;
    }
  }

  /** Whether the worker is currently running. */
  get isRunning(): boolean {
    return this.running;
  }
}
