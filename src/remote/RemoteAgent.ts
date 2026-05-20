import type { AgentEvent } from '../types.js';
import type { RemoteConfig } from './RemoteConfig.js';
import { defaultRemoteConfig } from './RemoteConfig.js';

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
 * Stub implementation of RemoteAgent.
 * Returns a single 'not_implemented' event.
 */
export class NatsRemoteAgent implements RemoteAgent {
  readonly name: string;
  private config: RemoteConfig;

  constructor(name: string, config?: Partial<RemoteConfig>) {
    this.name = name;
    this.config = { ...defaultRemoteConfig, ...config };
  }

  /** Invoke the remote agent. Stub: yields a done event immediately. */
  async *invoke(_input: string): AsyncGenerator<AgentEvent> {
    // Stub: in production, publish to NATS and subscribe to response subject
    void this.config;
    yield { type: 'error', data: { message: 'Remote agent not connected (stub)' } };
    yield { type: 'done', data: '' };
  }
}
