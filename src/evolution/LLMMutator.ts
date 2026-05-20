/**
 * LLM-driven mutation using an Agent for intelligent code evolution.
 * Supports multiple mutation strategies and extracts code from LLM responses.
 */

import { Agent } from '../agent/Agent.js';
import type { Program } from './Program.js';
import { createProgram, computeDiff } from './Program.js';
import { PromptBuilder, MUTATION_SYSTEM_PROMPT, CROSSOVER_SYSTEM_PROMPT } from './PromptBuilder.js';
import type { EvolutionConfig } from './EvolutionConfig.js';
import { DEFAULT_CONFIG } from './EvolutionConfig.js';

/** Mutation strategy types. */
export type MutationStrategy = 'point' | 'structural' | 'semantic' | 'guided' | 'random';

/** Options for the LLM mutator. */
export interface LLMMutatorOptions {
  /** Model to use for mutation. */
  model?: string;
  /** Temperature for LLM generation. */
  temperature?: number;
  /** Maximum retries on failure. */
  maxRetries?: number;
  /** Timeout per mutation attempt in ms. */
  timeout?: number;
  /** Optimization objective description. */
  objective?: string;
  /** Prompt builder instance. */
  promptBuilder?: PromptBuilder;
}

/**
 * LLM-driven mutator that uses an Agent to intelligently mutate code.
 * Supports multiple strategies and handles code extraction from responses.
 */
export class LLMMutator {
  private agent: Agent;
  private crossoverAgent: Agent;
  private promptBuilder: PromptBuilder;
  private objective: string;
  private maxRetries: number;
  private timeout: number;

  constructor(opts: LLMMutatorOptions = {}) {
    const model = opts.model ?? DEFAULT_CONFIG.mutatorModel;
    const temperature = opts.temperature ?? DEFAULT_CONFIG.temperature;

    this.agent = new Agent({
      name: 'evolution-mutator',
      model,
      systemPrompt: MUTATION_SYSTEM_PROMPT,
      temperature,
      maxIterations: 1,
    });

    this.crossoverAgent = new Agent({
      name: 'evolution-crossover',
      model,
      systemPrompt: CROSSOVER_SYSTEM_PROMPT,
      temperature,
      maxIterations: 1,
    });

    this.promptBuilder = opts.promptBuilder ?? new PromptBuilder();
    this.objective = opts.objective ?? 'Improve the program.';
    this.maxRetries = opts.maxRetries ?? DEFAULT_CONFIG.maxRetries;
    this.timeout = opts.timeout ?? DEFAULT_CONFIG.mutationTimeout * 1000;
  }

  /** Set the optimization objective. */
  setObjective(objective: string): void {
    this.objective = objective;
  }

  /**
   * Mutate a program using LLM-driven intelligent mutation.
   * Selects strategy based on mutation rate and program state.
   */
  async mutate(
    parent: Program,
    generation: number,
    opts?: {
      strategy?: MutationStrategy;
      topPrograms?: Program[];
      inspirations?: Program[];
      feedback?: string;
    },
  ): Promise<Program> {
    const strategy = opts?.strategy ?? this.selectStrategy(parent);

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const prompt = this.buildPromptForStrategy(strategy, parent, generation, opts);
        const code = await this.callAgent(prompt);

        if (!code || code.trim().length === 0) {
          continue; // Retry on empty response
        }

        // Validate the mutated code is actually different
        if (code.trim() === parent.code.trim()) {
          continue; // Retry if no change
        }

        const diff = computeDiff(parent.code, code);

        return createProgram(code, {
          parentIds: [parent.id],
          generation,
          islandId: parent.islandId,
          diffFromParent: diff,
          mutationInfo: {
            strategy,
            summary: `${strategy} mutation at generation ${generation}`,
            category: strategy === 'structural' ? 'architecture' : 'implementation',
            isAlgorithmic: strategy === 'structural' || strategy === 'semantic',
            parentFitness: parent.fitness,
          },
        });
      } catch (e) {
        if (attempt === this.maxRetries - 1) {
          throw new Error(`Mutation failed after ${this.maxRetries} attempts: ${(e as Error).message}`);
        }
      }
    }

    throw new Error('Mutation failed: all retries exhausted');
  }

  /**
   * Perform crossover between two parents using LLM.
   */
  async crossover(parentA: Program, parentB: Program, generation: number): Promise<Program> {
    const prompt = this.promptBuilder.buildCrossoverPrompt({
      parentA,
      parentB,
      objective: this.objective,
    });

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const code = await this.callCrossoverAgent(prompt);
        if (!code || code.trim().length === 0) continue;

        return createProgram(code, {
          parentIds: [parentA.id, parentB.id],
          generation,
          islandId: parentA.islandId,
          diffFromParent: computeDiff(parentA.code, code),
          mutationInfo: {
            strategy: 'crossover',
            summary: `Crossover of ${parentA.id.slice(0, 8)} and ${parentB.id.slice(0, 8)}`,
            category: 'recombination',
            isAlgorithmic: true,
          },
        });
      } catch (e) {
        if (attempt === this.maxRetries - 1) {
          throw new Error(`Crossover failed: ${(e as Error).message}`);
        }
      }
    }

    throw new Error('Crossover failed: all retries exhausted');
  }

  /** Select mutation strategy based on program state. */
  private selectStrategy(parent: Program): MutationStrategy {
    const rand = Math.random();
    const fitness = parent.fitness ?? 0;

    // High fitness: prefer small targeted changes
    if (fitness > 0.8) {
      if (rand < 0.5) return 'point';
      if (rand < 0.8) return 'semantic';
      return 'structural';
    }

    // Medium fitness: balanced approach
    if (fitness > 0.4) {
      if (rand < 0.3) return 'point';
      if (rand < 0.6) return 'semantic';
      if (rand < 0.85) return 'structural';
      return 'guided';
    }

    // Low fitness: prefer larger structural changes
    if (rand < 0.2) return 'point';
    if (rand < 0.5) return 'structural';
    if (rand < 0.8) return 'semantic';
    return 'guided';
  }

  /** Build prompt based on mutation strategy. */
  private buildPromptForStrategy(
    strategy: MutationStrategy,
    parent: Program,
    generation: number,
    opts?: {
      topPrograms?: Program[];
      inspirations?: Program[];
      feedback?: string;
    },
  ): string {
    const strategyHint = this.getStrategyHint(strategy);

    const basePrompt = this.promptBuilder.buildMutationPrompt({
      parent,
      objective: `${this.objective}\n\n${strategyHint}`,
      topPrograms: opts?.topPrograms,
      inspirations: opts?.inspirations,
      feedback: opts?.feedback,
      iteration: generation,
    });

    return basePrompt;
  }

  /** Get strategy-specific instructions. */
  private getStrategyHint(strategy: MutationStrategy): string {
    switch (strategy) {
      case 'point':
        return 'Strategy: Make a small, targeted change to a single function or expression. Fix a bug, optimize a calculation, or improve a constant.';
      case 'structural':
        return 'Strategy: Make a structural change to the program architecture. Reorganize logic, change data structures, or modify the algorithm approach.';
      case 'semantic':
        return 'Strategy: Make a semantically meaningful change. Improve the algorithm logic, add better error handling, or enhance the core computation.';
      case 'guided':
        return 'Strategy: Use the feedback and top programs as guidance. Identify what makes them successful and apply similar patterns.';
      case 'random':
        return 'Strategy: Make a creative, unexpected change. Try something novel that might lead to a breakthrough.';
    }
  }

  /** Call the mutation agent and extract code from response. */
  private async callAgent(prompt: string): Promise<string> {
    let fullText = '';
    const gen = this.agent.run(prompt);

    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error('LLM mutation timed out')), this.timeout);
    });

    try {
      const result = await Promise.race([
        (async () => {
          for await (const event of gen) {
            if (event.type === 'text') fullText += event.data;
            if (event.type === 'done') break;
          }
          return fullText;
        })(),
        timeoutPromise,
      ]);
      return extractCodeFromResponse(result as string);
    } catch (e) {
      throw e;
    }
  }

  /** Call the crossover agent and extract code. */
  private async callCrossoverAgent(prompt: string): Promise<string> {
    let fullText = '';
    for await (const event of this.crossoverAgent.run(prompt)) {
      if (event.type === 'text') fullText += event.data;
      if (event.type === 'done') break;
    }
    return extractCodeFromResponse(fullText);
  }
}

/**
 * Extract code from an LLM response that may contain markdown code blocks.
 * Handles various formats: ```lang\ncode```, ```\ncode```, or raw code.
 */
export function extractCodeFromResponse(response: string): string {
  if (!response || response.trim().length === 0) return '';

  // Try to find a fenced code block
  const fenceRegex = /```(?:\w+)?\n([\s\S]*?)```/g;
  const matches: string[] = [];

  let match: RegExpExecArray | null;
  while ((match = fenceRegex.exec(response)) !== null) {
    matches.push(match[1]);
  }

  if (matches.length > 0) {
    // Return the longest code block (most likely the full program)
    return matches.reduce((longest, current) =>
      current.length > longest.length ? current : longest,
    ).trim();
  }

  // No code blocks found - try to extract code heuristically
  // Remove common non-code prefixes/suffixes
  const lines = response.split('\n');
  const codeLines: string[] = [];
  let inCode = false;

  for (const line of lines) {
    // Skip obvious prose lines
    if (!inCode && /^(Here|I|The|This|Note|Let me|Sure|Below)/.test(line)) continue;
    if (!inCode && line.trim().length === 0) continue;

    // Start of code detected
    if (!inCode && (/^(import|export|const|let|var|function|class|def|from|#|\/)/.test(line) || /^\s/.test(line))) {
      inCode = true;
    }

    if (inCode) {
      codeLines.push(line);
    }
  }

  return codeLines.length > 0 ? codeLines.join('\n').trim() : response.trim();
}
