import { z } from 'zod';
import { Agent } from '../agent/index.js';
import type { AgentOptions } from '../agent/index.js';
import { ToolSet } from './ToolSet.js';
import { defineTool } from './Tool.js';

export interface AgentToolSetOptions {
  agents?: Agent[];
  createAgent?: (type: string, prompt: string) => Agent | Promise<Agent>;
  defaultAgentOptions?: Omit<AgentOptions, 'model'> & { model?: string | string[] };
}

export function agentToolSet(opts: AgentToolSetOptions = {}): ToolSet {
  const agents = new Map((opts.agents ?? []).map((agent) => [agent.name, agent]));

  return new ToolSet('agent', [
    defineTool<
      { agent: string; input: string },
      { agent: string; output: string }
    >({
      name: 'run_agent',
      aliases: ['Agent'],
      operation: 'task',
      description: 'Run a named sub-agent and return its final text output.',
      parameters: z.object({
        agent: z.string().describe('Agent name'),
        input: z.string().describe('Input to send to the sub-agent'),
      }),
      isReadOnly: () => false,
      execute: async ({ agent, input }) => {
        const target = agents.get(agent);
        if (!target) throw new Error(`Agent not found: ${agent}`);
        return { agent, output: await target.runToText(input) };
      },
    }),

    defineTool<
      { type?: string; prompt: string; input: string },
      { agent: string; output: string }
    >({
      name: 'spawn_agent',
      operation: 'task',
      description: 'Create an ad-hoc sub-agent for a scoped task and return its output.',
      parameters: z.object({
        type: z.string().optional().default('default').describe('Sub-agent type'),
        prompt: z.string().describe('System prompt or role instructions for the sub-agent'),
        input: z.string().describe('Task input'),
      }),
      isReadOnly: () => false,
      execute: async ({ type, prompt, input }) => {
        const created = opts.createAgent
          ? await opts.createAgent(type ?? 'default', prompt)
          : new Agent({
              name: `subagent_${Date.now()}`,
              model: opts.defaultAgentOptions?.model ?? 'gpt-5.5',
              ...opts.defaultAgentOptions,
              systemPrompt: prompt,
            } as AgentOptions);
        agents.set(created.name, created);
        return { agent: created.name, output: await created.runToText(input) };
      },
    }),

    defineTool<Record<string, never>, { agents: string[] }>({
      name: 'list_agents',
      operation: 'read',
      description: 'List registered sub-agents available to run_agent.',
      parameters: z.object({}),
      isReadOnly: () => true,
      execute: async () => ({ agents: [...agents.keys()] }),
    }),
  ]);
}
