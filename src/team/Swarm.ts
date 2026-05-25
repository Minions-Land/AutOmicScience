import { Team } from './Team.js';
import type { TeamConfig } from './Team.js';
import type { Agent } from '../agent/Agent.js';
import type { AgentEvent } from '../types.js';
import type { TeamPlugin } from './TeamPlugin.js';

/** Condition that triggers a handoff between agents. */
export type HandoffCondition =
  | { type: 'explicit'; targetAgent: string }
  | { type: 'keyword'; keywords: string[]; targetAgent: string }
  | { type: 'tool'; toolName: string; targetAgent: string };

/** Record of a single handoff event. */
export interface HandoffRecord {
  from: string;
  to: string;
  reason: string;
  timestamp: number;
  context?: string;
}

interface SwarmOptions {
  name?: string;
  plugins?: TeamPlugin[];
  config?: TeamConfig;
  /** Handoff conditions that trigger agent transfers. */
  handoffConditions?: HandoffCondition[];
  /** If true, after a subtask completes, control returns to the original agent. */
  returnToSender?: boolean;
}

/**
 * Swarm team with handoff logic: agents can transfer control to other agents.
 *
 * Features:
 * - Handoff conditions: explicit request, tool-based, keyword-based
 * - Context passing between agents on handoff
 * - Handoff history tracking
 * - Return-to-sender after subtask completion
 */
export class Swarm extends Team {
  public readonly name: string;
  private handoffConditions: HandoffCondition[];
  private returnToSender: boolean;
  private handoffHistory: HandoffRecord[] = [];
  private activeAgentName: string;

  constructor(agents: Agent[], opts: SwarmOptions = {}) {
    super(agents, opts.plugins ?? [], opts.config);
    this.name = opts.name ?? 'swarm';
    this.handoffConditions = opts.handoffConditions ?? [];
    this.returnToSender = opts.returnToSender ?? false;
    this.activeAgentName = agents[0]?.name ?? '';
  }

  /** Check if output text triggers a keyword-based handoff. */
  private checkKeywordHandoff(output: string): HandoffCondition | undefined {
    return this.handoffConditions.find(
      (c) =>
        c.type === 'keyword' &&
        c.keywords.some((kw) => output.toLowerCase().includes(kw.toLowerCase())),
    );
  }

  /** Check if a tool call triggers a tool-based handoff. */
  private checkToolHandoff(toolName: string): HandoffCondition | undefined {
    return this.handoffConditions.find(
      (c) => c.type === 'tool' && c.toolName === toolName,
    );
  }

  /** Check if output contains an explicit handoff request. */
  private checkExplicitHandoff(output: string): HandoffCondition | undefined {
    // Look for patterns like "HANDOFF: agent_name" or "transfer to agent_name"
    const transferMatch = output.match(
      /(?:HANDOFF|transfer\s+to|hand\s*off\s+to):\s*(\w+)/i,
    );
    if (transferMatch) {
      const targetName = transferMatch[1];
      if (this.agents.some((a) => a.name === targetName)) {
        return { type: 'explicit', targetAgent: targetName };
      }
    }
    return undefined;
  }

  /** Perform a handoff: record it and switch active agent. */
  private performHandoff(
    from: string,
    to: string,
    reason: string,
    context?: string,
  ): void {
    this.handoffHistory.push({
      from,
      to,
      reason,
      timestamp: Date.now(),
      context,
    });
    this.activeAgentName = to;
  }

  /** Build context string from handoff history for the receiving agent. */
  private buildHandoffContext(originalInput: string, priorOutputs: Map<string, string>): string {
    const parts = [`Original request: ${originalInput}`];

    if (priorOutputs.size > 0) {
      parts.push('\nPrior agent outputs:');
      for (const [name, output] of priorOutputs) {
        parts.push(`[${name}]: ${output.slice(0, 500)}`);
      }
    }

    if (this.handoffHistory.length > 0) {
      const recent = this.handoffHistory.slice(-3);
      parts.push('\nRecent handoff history:');
      for (const h of recent) {
        parts.push(`  ${h.from} -> ${h.to} (${h.reason})`);
      }
    }

    return parts.join('\n');
  }

  async *run(input: string): AsyncGenerator<AgentEvent> {
    const priorOutputs = new Map<string, string>();
    const senderStack: string[] = [];
    let currentInput = input;
    let iterations = 0;

    while (iterations < this.config.maxIterations) {
      iterations++;
      const agent = this.agents.find((a) => a.name === this.activeAgentName);
      if (!agent) {
        yield { type: 'error', data: { message: `Agent '${this.activeAgentName}' not found` } };
        break;
      }

      this.markAgentRunning(agent.name);
      yield { type: 'agent_start', data: { name: agent.name, iteration: iterations } };

      let output = '';
      let triggeredHandoff: HandoffCondition | undefined;

      for await (const ev of agent.run(currentInput)) {
        yield { type: ev.type, data: { agent: agent.name, payload: ev.data } };

        if (ev.type === 'done') {
          output = String(ev.data ?? '');
        }

        // Check tool-based handoff
        if (ev.type === 'tool_call' && ev.data) {
          const calls = Array.isArray(ev.data) ? ev.data : [ev.data];
          for (const call of calls) {
            const toolName = (call as { name?: string }).name;
            if (toolName) {
              const toolHandoff = this.checkToolHandoff(toolName);
              if (toolHandoff) {
                triggeredHandoff = toolHandoff;
              }
            }
          }
        }
      }

      this.markAgentIdle(agent.name);
      priorOutputs.set(agent.name, output);
      yield { type: 'agent_done', data: { name: agent.name, output } };

      // Check for handoff conditions
      if (!triggeredHandoff) {
        triggeredHandoff = this.checkExplicitHandoff(output);
      }
      if (!triggeredHandoff) {
        triggeredHandoff = this.checkKeywordHandoff(output);
      }

      if (triggeredHandoff) {
        const targetAgent = triggeredHandoff.targetAgent;
        const reason =
          triggeredHandoff.type === 'keyword'
            ? `keyword match`
            : triggeredHandoff.type === 'tool'
              ? `tool '${triggeredHandoff.toolName}' triggered`
              : 'explicit request';

        yield {
          type: 'handoff',
          data: { from: agent.name, to: targetAgent, reason },
        };

        if (this.returnToSender) {
          senderStack.push(agent.name);
        }

        this.performHandoff(agent.name, targetAgent, reason, output.slice(0, 200));
        currentInput = this.buildHandoffContext(input, priorOutputs);
        continue;
      }

      // No handoff triggered — check if we should return to sender
      if (this.returnToSender && senderStack.length > 0) {
        const sender = senderStack.pop()!;
        yield {
          type: 'handoff',
          data: { from: agent.name, to: sender, reason: 'return-to-sender' },
        };
        this.performHandoff(agent.name, sender, 'subtask complete', output.slice(0, 200));
        currentInput = `Subtask result from ${agent.name}:\n${output}\n\nContinue with the original task.`;
        continue;
      }

      // No handoff, no return — we're done
      yield { type: 'done', data: output };
      return;
    }

    // Max iterations reached
    const lastOutput = priorOutputs.get(this.activeAgentName) ?? '';
    yield { type: 'error', data: { message: 'Max iterations reached' } };
    yield { type: 'done', data: lastOutput };
  }

  /** Get the full handoff history. */
  getHandoffHistory(): HandoffRecord[] {
    return [...this.handoffHistory];
  }

  /** Get the currently active agent name. */
  getActiveAgent(): string {
    return this.activeAgentName;
  }

  /** Manually set the active agent. */
  setActiveAgent(name: string): void {
    if (!this.agents.some((a) => a.name === name)) {
      throw new Error(`Agent '${name}' not found in swarm`);
    }
    this.activeAgentName = name;
  }

  /** Add a handoff condition at runtime. */
  addHandoffCondition(condition: HandoffCondition): void {
    this.handoffConditions.push(condition);
  }
}
