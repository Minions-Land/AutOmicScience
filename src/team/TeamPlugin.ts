import type { Agent } from '../agent/Agent.js';
import type { AgentEvent } from '../types.js';

/**
 * Plugin interface for extending team behavior.
 * Plugins hook into the team lifecycle and can modify behavior
 * at various points during execution.
 */
export interface TeamPlugin {
  /** Unique plugin name. */
  readonly name: string;

  /** Priority for ordering (lower = earlier execution). Default: 100. */
  readonly priority?: number;

  /** Called after the team is fully set up, before any runs. */
  onTeamCreated?(agents: Agent[]): Promise<void>;

  /** Called before each team run. Return modified input or undefined to pass through. */
  beforeRun?(input: string, context: PluginContext): Promise<string | undefined>;

  /** Called after each team run completes. */
  afterRun?(input: string, output: string, context: PluginContext): Promise<void>;

  /** Called before an individual agent is invoked. */
  beforeAgentCall?(agent: Agent, input: string, context: PluginContext): Promise<string | undefined>;

  /** Called after an individual agent completes. */
  afterAgentCall?(agent: Agent, input: string, output: string, context: PluginContext): Promise<void>;

  /** Called when a tool is executed by any agent. */
  onToolCall?(agentName: string, toolName: string, args: unknown, result: unknown): Promise<void>;

  /** Called on team shutdown for cleanup. */
  onShutdown?(): Promise<void>;
}

export interface PluginContext {
  /** Current iteration number. */
  iteration: number;
  /** Accumulated events so far. */
  events: AgentEvent[];
  /** Arbitrary metadata plugins can share. */
  metadata: Record<string, unknown>;
}

// ── Built-in Plugins ──────────────────────────────────────────────────────────

/**
 * Logging plugin: logs all lifecycle events to console or a custom logger.
 */
export class LoggingPlugin implements TeamPlugin {
  readonly name = 'logging';
  readonly priority = 10;
  private log: (msg: string) => void;

  constructor(logger?: (msg: string) => void) {
    this.log = logger ?? console.log;
  }

  async beforeRun(input: string, ctx: PluginContext): Promise<undefined> {
    this.log(`[team] run start (iteration=${ctx.iteration}): ${input.slice(0, 100)}`);
    return undefined;
  }

  async afterRun(input: string, output: string): Promise<void> {
    this.log(`[team] run complete: output length=${output.length}`);
  }

  async beforeAgentCall(agent: Agent, input: string): Promise<undefined> {
    this.log(`[team] agent '${agent.name}' starting`);
    return undefined;
  }

  async afterAgentCall(agent: Agent, _input: string, output: string): Promise<void> {
    this.log(`[team] agent '${agent.name}' done: ${output.slice(0, 80)}`);
  }

  async onToolCall(agentName: string, toolName: string): Promise<void> {
    this.log(`[team] tool call: ${agentName} -> ${toolName}`);
  }
}

/**
 * Rate-limiting plugin: enforces a minimum delay between agent calls.
 */
export class RateLimitPlugin implements TeamPlugin {
  readonly name = 'rate-limit';
  readonly priority = 20;
  private minDelayMs: number;
  private lastCall = 0;

  constructor(minDelayMs = 500) {
    this.minDelayMs = minDelayMs;
  }

  async beforeAgentCall(): Promise<undefined> {
    const now = Date.now();
    const elapsed = now - this.lastCall;
    if (elapsed < this.minDelayMs) {
      await new Promise((r) => setTimeout(r, this.minDelayMs - elapsed));
    }
    this.lastCall = Date.now();
    return undefined;
  }
}

/**
 * Retry plugin: retries failed agent calls with exponential backoff.
 */
export class RetryPlugin implements TeamPlugin {
  readonly name = 'retry';
  readonly priority = 30;
  private maxRetries: number;
  private baseDelayMs: number;
  private failureCounts: Map<string, number> = new Map();

  constructor(maxRetries = 3, baseDelayMs = 1000) {
    this.maxRetries = maxRetries;
    this.baseDelayMs = baseDelayMs;
  }

  getRetryCount(agentName: string): number {
    return this.failureCounts.get(agentName) ?? 0;
  }

  shouldRetry(agentName: string): boolean {
    return this.getRetryCount(agentName) < this.maxRetries;
  }

  async recordFailure(agentName: string): Promise<void> {
    const count = this.getRetryCount(agentName) + 1;
    this.failureCounts.set(agentName, count);
    const delay = this.baseDelayMs * Math.pow(2, count - 1);
    await new Promise((r) => setTimeout(r, delay));
  }

  recordSuccess(agentName: string): void {
    this.failureCounts.delete(agentName);
  }
}
