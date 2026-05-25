import type { Agent } from '../agent/Agent.js';
import type { AgentEvent } from '../types.js';
import type { TeamPlugin } from './TeamPlugin.js';

export interface TeamRunResult {
  finalText: string;
  perAgent: { name: string; output: string }[];
}

export interface TeamConfig {
  /** Maximum iterations before the team stops (prevents infinite loops). */
  maxIterations?: number;
  /** Interval in ms between health checks (0 = disabled). */
  healthCheckInterval?: number;
  /** Timeout in ms for a single agent step. */
  agentTimeout?: number;
}

export interface AgentHealth {
  name: string;
  status: 'idle' | 'running' | 'error' | 'timeout';
  lastActive: number;
  errorCount: number;
  lastError?: string;
}

/**
 * Base class for all team orchestration patterns.
 * Provides shared event queue, agent health monitoring, plugin integration,
 * and team-level configuration.
 */
export abstract class Team {
  abstract readonly name: string;
  protected agents: Agent[];
  protected plugins: TeamPlugin[];
  protected config: Required<TeamConfig>;
  protected healthMap: Map<string, AgentHealth> = new Map();
  private eventBuffer: AgentEvent[] = [];
  private eventResolvers: Array<(value: IteratorResult<AgentEvent>) => void> = [];

  constructor(agents: Agent[], plugins: TeamPlugin[] = [], config: TeamConfig = {}) {
    this.agents = agents;
    this.plugins = plugins;
    this.config = {
      maxIterations: config.maxIterations ?? 20,
      healthCheckInterval: config.healthCheckInterval ?? 0,
      agentTimeout: config.agentTimeout ?? 120_000,
    };
    for (const agent of agents) {
      this.healthMap.set(agent.name, {
        name: agent.name,
        status: 'idle',
        lastActive: Date.now(),
        errorCount: 0,
      });
    }
  }

  /** Stream of events from all member agents (annotated with agent name). */
  abstract run(input: string): AsyncGenerator<AgentEvent>;

  /** Emit an event to the shared event queue. */
  protected emitEvent(event: AgentEvent): void {
    if (this.eventResolvers.length > 0) {
      const resolve = this.eventResolvers.shift()!;
      resolve({ value: event, done: false });
    } else {
      this.eventBuffer.push(event);
    }
  }

  /** Consume the shared event queue as an async iterable. */
  async *events(): AsyncGenerator<AgentEvent> {
    while (true) {
      if (this.eventBuffer.length > 0) {
        yield this.eventBuffer.shift()!;
      } else {
        const event = await new Promise<IteratorResult<AgentEvent>>((resolve) => {
          this.eventResolvers.push(resolve);
        });
        if (event.done) return;
        yield event.value;
      }
    }
  }

  /** Mark an agent as running and update health. */
  protected markAgentRunning(agentName: string): void {
    const health = this.healthMap.get(agentName);
    if (health) {
      health.status = 'running';
      health.lastActive = Date.now();
    }
  }

  /** Mark an agent as idle after completing work. */
  protected markAgentIdle(agentName: string): void {
    const health = this.healthMap.get(agentName);
    if (health) {
      health.status = 'idle';
      health.lastActive = Date.now();
    }
  }

  /** Mark an agent as errored. */
  protected markAgentError(agentName: string, error: string): void {
    const health = this.healthMap.get(agentName);
    if (health) {
      health.status = 'error';
      health.errorCount++;
      health.lastError = error;
    }
  }

  /** Get health status for all agents. */
  getHealth(): AgentHealth[] {
    return [...this.healthMap.values()];
  }

  /** Call a lifecycle hook on all plugins. */
  protected async callPluginHook(
    hook: keyof TeamPlugin,
    ...args: unknown[]
  ): Promise<void> {
    for (const plugin of this.plugins) {
      const fn = plugin[hook];
      if (typeof fn === 'function') {
        await (fn as (...a: unknown[]) => Promise<void>).call(plugin, ...args);
      }
    }
  }

  async runToText(input: string): Promise<TeamRunResult> {
    const perAgent: { name: string; output: string }[] = [];
    let finalText = '';
    let curAgent = '';
    let curBuf = '';
    for await (const ev of this.run(input)) {
      if (ev.type === 'agent_start') {
        if (curAgent) perAgent.push({ name: curAgent, output: curBuf });
        curAgent = String((ev.data as { name: string }).name);
        curBuf = '';
      } else if (ev.type === 'done') {
        const text = String(ev.data ?? '');
        curBuf = text;
        finalText = text;
      }
    }
    if (curAgent) perAgent.push({ name: curAgent, output: curBuf });
    return { finalText, perAgent };
  }
}
