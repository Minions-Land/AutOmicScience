import { Team } from './Team.js';
import type { Agent } from '../agent/Agent.js';
import type { AgentEvent } from '../types.js';

/**
 * Mixture-of-Agents: all member agents receive the same input in parallel,
 * then an aggregator agent synthesizes their outputs into a final answer.
 */
export class MixtureOfAgents extends Team {
  public readonly name: string;
  private readonly aggregator: Agent;

  constructor(agents: Agent[], aggregator: Agent, name = 'moa') {
    super(agents);
    this.aggregator = aggregator;
    this.name = name;
  }

  async *run(input: string): AsyncGenerator<AgentEvent> {
    const tasks = this.agents.map(async (agent) => {
      let out = '';
      for await (const ev of agent.run(input)) {
        if (ev.type === 'done') out = String(ev.data ?? '');
      }
      return { name: agent.name, out };
    });

    yield { type: 'phase', data: { phase: 'parallel', agents: this.agents.map((a) => a.name) } };
    const results = await Promise.all(tasks);

    for (const r of results) {
      yield { type: 'agent_done', data: { name: r.name, output: r.out } };
    }

    const aggregatorInput = [
      `You received responses from ${results.length} agents to the query: "${input}"`,
      '',
      ...results.map((r, i) => `### Agent ${i + 1} (${r.name})\n${r.out}`),
      '',
      'Synthesize these into a single coherent answer.',
    ].join('\n');

    yield { type: 'agent_start', data: { name: this.aggregator.name, role: 'aggregator' } };
    let final = '';
    for await (const ev of this.aggregator.run(aggregatorInput)) {
      yield { type: ev.type, data: { agent: this.aggregator.name, payload: ev.data } };
      if (ev.type === 'done') final = String(ev.data ?? '');
    }
    yield { type: 'done', data: final };
  }
}
