import { Team } from './Team.js';
import { Agent } from '../agent/Agent.js';
import type { AgentEvent } from '../types.js';

/**
 * PantheonTeam: a coordinator agent that delegates to specialist members.
 * The coordinator sees member descriptions and decides how to route work.
 * This is a minimal implementation: it asks the coordinator to produce a
 * routing plan, then runs the chosen members in sequence.
 */
export class PantheonTeam extends Team {
  public readonly name: string;
  private readonly coordinator: Agent;

  constructor(coordinator: Agent, members: Agent[], name = 'pantheon') {
    super(members);
    this.coordinator = coordinator;
    this.name = name;
  }

  private memberRoster(): string {
    return this.agents.map((a) => `- ${a.name}`).join('\n');
  }

  async *run(input: string): AsyncGenerator<AgentEvent> {
    const planPrompt = [
      `You are the coordinator for a team of agents:`,
      this.memberRoster(),
      ``,
      `User request: ${input}`,
      ``,
      `Reply with a short plan (one line per step) of which agent should handle which sub-task. Use agent names exactly as listed.`,
    ].join('\n');
    yield { type: 'agent_start', data: { name: this.coordinator.name, role: 'coordinator' } };
    const plan = await this.coordinator.runToText(planPrompt);
    yield { type: 'plan', data: plan };

    let last = '';
    for (const member of this.agents) {
      if (!plan.includes(member.name)) continue;
      yield { type: 'agent_start', data: { name: member.name } };
      for await (const ev of member.run(`${input}\n\nPlan context:\n${plan}\n\nPrior output:\n${last}`)) {
        yield { type: ev.type, data: { agent: member.name, payload: ev.data } };
        if (ev.type === 'done') last = String(ev.data ?? '');
      }
      yield { type: 'agent_done', data: { name: member.name, output: last } };
    }
    yield { type: 'done', data: last };
  }
}
