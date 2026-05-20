import type { RemoteConfig } from './RemoteConfig.js';
import { defaultRemoteConfig } from './RemoteConfig.js';

/**
 * A worker node that registers local agents and listens for
 * remote invocations over NATS.
 */
export interface RemoteWorker {
  /** Register an agent to be available for remote invocation. */
  register(agent: { name: string }): Promise<void>;
  /** Start listening for invocations on NATS. */
  start(): Promise<void>;
  /** Stop listening and disconnect. */
  stop(): Promise<void>;
}

/**
 * Stub implementation of RemoteWorker.
 * Does not actually connect to NATS — placeholder for real implementation.
 */
export class NatsRemoteWorker implements RemoteWorker {
  private config: RemoteConfig;
  private agents: Array<{ name: string }> = [];
  private running = false;

  constructor(config?: Partial<RemoteConfig>) {
    this.config = { ...defaultRemoteConfig, ...config };
  }

  /** Register an agent to be available for remote invocation. */
  async register(agent: { name: string }): Promise<void> {
    this.agents.push(agent);
  }

  /** Start listening for invocations on NATS. Stub: sets running flag. */
  async start(): Promise<void> {
    void this.config;
    void this.agents;
    this.running = true;
  }

  /** Stop listening and disconnect. */
  async stop(): Promise<void> {
    this.running = false;
  }

  /** Whether the worker is currently running. */
  get isRunning(): boolean {
    return this.running;
  }
}
