import { Team } from './Team.js';
import type { TeamConfig } from './Team.js';
import { Agent } from '../agent/Agent.js';
import type { AgentEvent, Message } from '../types.js';
import type { TeamPlugin, PluginContext } from './TeamPlugin.js';

/**
 * A step in the coordinator's execution plan.
 */
interface PlanStep {
  agentName: string;
  task: string;
  /** Steps that can run in parallel share the same group index. */
  parallelGroup: number;
  status: 'pending' | 'running' | 'done' | 'failed';
  output?: string;
  error?: string;
}

interface CoordinatorOptions {
  name?: string;
  plugins?: TeamPlugin[];
  config?: TeamConfig;
  /** Maximum re-plan attempts when a step fails. */
  maxReplans?: number;
  /** Tool delegation map: agentName -> list of tool names it handles. */
  toolDelegation?: Record<string, string[]>;
}

/**
 * CoordinatorTeam: a coordinator agent that dynamically plans, delegates,
 * and re-plans work across specialist member agents.
 *
 * Features:
 * - Dynamic re-planning when steps fail
 * - Event queue aggregation from all member agents
 * - Tool delegation to specific agents
 * - Conversation memory across interactions
 * - Max iterations guard
 * - Parallel execution of independent steps
 * - Status reporting
 */
export class CoordinatorTeam extends Team {
  public readonly name: string;
  private readonly coordinator: Agent;
  private readonly maxReplans: number;
  private readonly toolDelegation: Record<string, string[]>;
  private conversationHistory: Message[] = [];
  private currentPlan: PlanStep[] = [];

  constructor(coordinator: Agent, members: Agent[], opts: CoordinatorOptions = {}) {
    super(members, opts.plugins ?? [], opts.config);
    this.coordinator = coordinator;
    this.name = opts.name ?? 'coordinator';
    this.maxReplans = opts.maxReplans ?? 3;
    this.toolDelegation = opts.toolDelegation ?? {};
  }

  private memberRoster(): string {
    return this.agents
      .map((a) => {
        const tools = this.toolDelegation[a.name];
        const toolStr = tools ? ` [tools: ${tools.join(', ')}]` : '';
        return `- ${a.name}${toolStr}`;
      })
      .join('\n');
  }

  private buildPlanPrompt(input: string, failedSteps?: PlanStep[]): string {
    const parts = [
      'You are the coordinator for a team of agents:',
      this.memberRoster(),
      '',
    ];

    if (this.conversationHistory.length > 0) {
      parts.push('## Prior conversation context:');
      const recent = this.conversationHistory.slice(-6);
      for (const msg of recent) {
        parts.push(`[${msg.role}]: ${typeof msg.content === 'string' ? msg.content.slice(0, 200) : '...'}`);
      }
      parts.push('');
    }

    parts.push(`## User request: ${input}`, '');

    if (failedSteps && failedSteps.length > 0) {
      parts.push('## Failed steps that need re-planning:');
      for (const step of failedSteps) {
        parts.push(`- ${step.agentName}: "${step.task}" FAILED: ${step.error}`);
      }
      parts.push('');
      parts.push('Create a revised plan that works around these failures.');
      parts.push('You may reassign tasks to different agents or break them into smaller steps.');
    } else {
      parts.push('Create an execution plan. Reply with one step per line in this format:');
      parts.push('[group_number] agent_name: task description');
      parts.push('');
      parts.push('Steps with the same group number will run in parallel.');
      parts.push('Use agent names exactly as listed above.');
    }

    return parts.join('\n');
  }

  private parsePlan(planText: string): PlanStep[] {
    const steps: PlanStep[] = [];
    const lines = planText.split('\n').filter((l) => l.trim());

    for (const line of lines) {
      // Try format: [group] agent_name: task
      const groupMatch = line.match(/^\[(\d+)\]\s*([^:]+):\s*(.+)$/);
      if (groupMatch) {
        const agentName = groupMatch[2].trim();
        if (this.agents.some((a) => a.name === agentName)) {
          steps.push({
            agentName,
            task: groupMatch[3].trim(),
            parallelGroup: parseInt(groupMatch[1], 10),
            status: 'pending',
          });
          continue;
        }
      }

      // Fallback: agent_name: task (sequential, each gets own group)
      const simpleMatch = line.match(/^[-*]?\s*([^:]+):\s*(.+)$/);
      if (simpleMatch) {
        const agentName = simpleMatch[1].trim();
        if (this.agents.some((a) => a.name === agentName)) {
          steps.push({
            agentName,
            task: simpleMatch[2].trim(),
            parallelGroup: steps.length,
            status: 'pending',
          });
        }
      }
    }

    return steps;
  }

  private async runAgent(agent: Agent, task: string, priorContext: string): Promise<string> {
    const fullInput = priorContext
      ? `${task}\n\nContext from prior steps:\n${priorContext}`
      : task;

    let output = '';
    for await (const ev of agent.run(fullInput)) {
      if (ev.type === 'done') output = String(ev.data ?? '');
    }
    return output;
  }

  async *run(input: string): AsyncGenerator<AgentEvent> {
    const ctx: PluginContext = { iteration: 0, events: [], metadata: {} };

    // Plugin: beforeRun
    await this.callPluginHook('beforeRun', input, ctx);

    // Store user input in conversation history
    this.conversationHistory.push({ role: 'user', content: input });

    let replanCount = 0;
    let failedSteps: PlanStep[] | undefined;
    let finalOutput = '';

    for (let iteration = 0; iteration < this.config.maxIterations; iteration++) {
      ctx.iteration = iteration;

      // Phase 1: Generate plan
      yield { type: 'status', data: { phase: 'planning', replanCount } };
      yield { type: 'agent_start', data: { name: this.coordinator.name, role: 'coordinator' } };

      const planPrompt = this.buildPlanPrompt(input, failedSteps);
      const planText = await this.coordinator.runToText(planPrompt);
      this.currentPlan = this.parsePlan(planText);

      yield { type: 'plan', data: { text: planText, steps: this.currentPlan } };

      if (this.currentPlan.length === 0) {
        // Coordinator couldn't produce a plan — use its response as final output
        finalOutput = planText;
        break;
      }

      // Phase 2: Execute plan by parallel groups
      const groups = new Map<number, PlanStep[]>();
      for (const step of this.currentPlan) {
        const group = groups.get(step.parallelGroup) ?? [];
        group.push(step);
        groups.set(step.parallelGroup, group);
      }

      let priorContext = '';
      failedSteps = [];
      let allSucceeded = true;

      const sortedGroups = [...groups.keys()].sort((a, b) => a - b);

      for (const groupIdx of sortedGroups) {
        const groupSteps = groups.get(groupIdx)!;

        yield {
          type: 'status',
          data: {
            phase: 'executing',
            group: groupIdx,
            agents: groupSteps.map((s) => s.agentName),
          },
        };

        // Run steps in this group in parallel
        const results = await Promise.allSettled(
          groupSteps.map(async (step) => {
            step.status = 'running';
            const agent = this.agents.find((a) => a.name === step.agentName);
            if (!agent) throw new Error(`Agent '${step.agentName}' not found`);

            this.markAgentRunning(step.agentName);
            await this.callPluginHook('beforeAgentCall', agent, step.task, ctx);

            yield_event: {
              // We can't yield from inside Promise.allSettled, so we emit events
              this.emitEvent({ type: 'agent_start', data: { name: step.agentName } });
            }

            const output = await this.runAgent(agent, step.task, priorContext);
            step.output = output;
            step.status = 'done';

            this.markAgentIdle(step.agentName);
            await this.callPluginHook('afterAgentCall', agent, step.task, output, ctx);

            this.emitEvent({
              type: 'agent_done',
              data: { name: step.agentName, output: output.slice(0, 200) },
            });

            return { step, output };
          }),
        );

        // Process results
        for (const result of results) {
          if (result.status === 'fulfilled') {
            const { step, output } = result.value;
            priorContext += `\n[${step.agentName}]: ${output}\n`;
          } else {
            const failedStep = groupSteps.find((s) => s.status === 'running');
            if (failedStep) {
              failedStep.status = 'failed';
              failedStep.error = String(result.reason);
              failedSteps.push(failedStep);
              this.markAgentError(failedStep.agentName, String(result.reason));
            }
            allSucceeded = false;
          }
        }

        // If any step in this group failed, break out for re-planning
        if (!allSucceeded) break;
      }

      if (allSucceeded) {
        // All steps succeeded — synthesize final output
        const synthesizePrompt = [
          `The team completed all tasks for: "${input}"`,
          '',
          'Results from each agent:',
          priorContext,
          '',
          'Synthesize these results into a coherent final response.',
        ].join('\n');

        finalOutput = await this.coordinator.runToText(synthesizePrompt);
        break;
      }

      // Re-plan if we haven't exceeded max replans
      replanCount++;
      if (replanCount > this.maxReplans) {
        yield { type: 'error', data: { message: 'Max re-plan attempts exceeded', failedSteps } };
        // Use whatever partial results we have
        finalOutput = priorContext || 'Team execution failed after maximum re-plan attempts.';
        break;
      }

      yield { type: 'status', data: { phase: 'replanning', attempt: replanCount, failedSteps } };
    }

    // Store output in conversation history
    this.conversationHistory.push({ role: 'assistant', content: finalOutput });

    // Plugin: afterRun
    await this.callPluginHook('afterRun', input, finalOutput, ctx);

    yield { type: 'done', data: finalOutput };
  }

  /** Get the current execution plan. */
  getPlan(): PlanStep[] {
    return [...this.currentPlan];
  }

  /** Get conversation history. */
  getHistory(): Message[] {
    return [...this.conversationHistory];
  }

  /** Clear conversation history. */
  clearHistory(): void {
    this.conversationHistory = [];
  }

  /** Delegate a tool call to the appropriate agent based on toolDelegation config. */
  findDelegateForTool(toolName: string): Agent | undefined {
    for (const [agentName, tools] of Object.entries(this.toolDelegation)) {
      if (tools.includes(toolName)) {
        return this.agents.find((a) => a.name === agentName);
      }
    }
    return undefined;
  }
}
