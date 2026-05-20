import { Team } from './Team.js';
import type { Agent } from '../agent/Agent.js';
import type { AgentEvent } from '../types.js';

/** All agents receive the same input; outputs are aggregated. */
export class Swarm extends Team {
  public readonly name: string;
  private readonly aggregator: (outputs: string[]) => string;

  constructor(
    agents: Agent[],
    opts: { name?: string; aggregate?: (outputs: string[]) => string } = {},
  ) {
    super(agents);
    this.name = opts.name ?? 'swarm';
    this.aggregator =
      opts.aggregate ??
      ((outs) => outs.map((o, i) => `### Agent ${i + 1}\n${o}`).join('\n\n'));
  }

  async *run(input: string): AsyncGenerator<AgentEvent> {
    const tasks = this.agents.map(async (agent) => {
      let out = '';
      const events: AgentEvent[] = [];
      for await (const ev of agent.run(input)) {
        events.push({ type: ev.type, data: { agent: agent.name, payload: ev.data } });
        if (ev.type === 'done') out = String(ev.data ?? '');
      }
      return { name: agent.name, out, events };
    });
    const results = await Promise.all(tasks);
    for (const r of results) {
      yield { type: 'agent_start', data: { name: r.name } };
      for (const ev of r.events) yield ev;
      yield { type: 'agent_done', data: { name: r.name, output: r.out } };
    }
    const aggregated = this.aggregator(results.map((r) => r.out));
    yield { type: 'done', data: aggregated };
  }
}
