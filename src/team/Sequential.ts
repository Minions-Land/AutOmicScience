import { Team } from './Team.js';
import type { Agent } from '../agent/Agent.js';
import type { AgentEvent } from '../types.js';

/** Pipeline: input -> agent[0] -> agent[1] -> ... -> last agent's output. */
export class Sequential extends Team {
  public readonly name: string;

  constructor(agents: Agent[], name = 'sequential') {
    super(agents);
    this.name = name;
  }

  async *run(input: string): AsyncGenerator<AgentEvent> {
    let current = input;
    for (const agent of this.agents) {
      yield { type: 'agent_start', data: { name: agent.name } };
      let out = '';
      for await (const ev of agent.run(current)) {
        yield { type: ev.type, data: { agent: agent.name, payload: ev.data } };
        if (ev.type === 'done') out = String(ev.data ?? '');
      }
      yield { type: 'agent_done', data: { name: agent.name, output: out } };
      current = out;
    }
    yield { type: 'done', data: current };
  }
}
