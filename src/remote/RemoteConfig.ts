/**
 * Configuration for remote agent/worker NATS connections.
 */
export interface RemoteConfig {
  /** NATS server URL (e.g. 'nats://localhost:4222'). */
  natsUrl: string;
  /** Namespace prefix for NATS subjects. */
  namespace: string;
  /** Timeout in milliseconds for remote invocations. */
  timeout: number;
}

/** Default remote configuration. */
export const defaultRemoteConfig: RemoteConfig = {
  natsUrl: 'nats://localhost:4222',
  namespace: 'novaeve',
  timeout: 30_000,
};
