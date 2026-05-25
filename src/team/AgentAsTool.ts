import type { Agent } from '../agent/Agent.js';
import type { Tool } from '../toolset/Tool.js';
import { z } from 'zod';

/**
 * Wraps an Agent as a Tool so it can be registered in another agent's ToolSet.
 * The tool accepts `{ input: string }` and returns the agent's final text output.
 */
export function agentAsTool(agent: Agent): Tool<{ input: string }, string> {
  return {
    name: `agent_${agent.name.replace(/[^a-zA-Z0-9_]/g, '_')}`,
    description: `Invoke the '${agent.name}' agent with a text input and get its response.`,
    parameters: z.object({ input: z.string().describe('The input message to send to the agent.') }),
    execute: async ({ input }) => agent.runToText(input),
  };
}
