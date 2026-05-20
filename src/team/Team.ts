import type { Agent } from '../agent/Agent.js';
import type { AgentEvent } from '../types.js';

export interface TeamRunResult {
  finalText: string;
  perAgent: { name: string; output: string }[];
}

export abstract class Team {
  abstract readonly name: string;
  protected agents: Agent[];

  constructor(agents: Agent[]) {
    this.agents = agents;
  }

  /** Stream of events from all member agents (annotated with agent name). */
  abstract run(input: string): AsyncGenerator<AgentEvent>;

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
