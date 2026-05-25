/**
 * Prompt construction for LLM-driven mutation and analysis.
 * Builds structured prompts with fitness context, top performers, and feedback.
 */

import type { Program } from './Program.js';

/** System prompt for the mutator agent. */
export const MUTATION_SYSTEM_PROMPT = `You are an expert code optimizer. Your task is to improve code through targeted mutations.

## Output Format
Respond with the COMPLETE improved code. Do not use diff format - output the full program.
Wrap your code in a single markdown code block.

## Guidelines
1. Make targeted, surgical improvements - don't rewrite everything
2. Preserve working functionality
3. Focus on the optimization objective
4. Learn from high-performing examples
5. Consider both correctness and performance
6. Handle edge cases properly
`;

/** System prompt for the analysis agent. */
export const ANALYSIS_SYSTEM_PROMPT = `You are an expert code analyzer. Identify issues and propose specific improvements.

## Your Task
1. Identify Issues: What are the main problems or bottlenecks?
2. Design Solutions: What specific changes should be made?
3. Prioritize: Focus on changes with the highest impact on the objective.

Be specific and actionable. Reference exact code locations.
`;

/** System prompt for crossover. */
export const CROSSOVER_SYSTEM_PROMPT = `You are an expert at combining code from multiple programs.
Given two parent programs, produce a child that combines the best traits of both.
Output the COMPLETE child program in a single markdown code block.
`;

export interface PromptBuilderOptions {
  maxCodeLength?: number;
  maxTopPrograms?: number;
  maxInspirations?: number;
  includeArtifacts?: boolean;
}

/**
 * Builds prompts for the mutation agent with full context:
 * current program, top performers, inspirations, and feedback.
 */
export class PromptBuilder {
  private maxCodeLength: number;
  private maxTopPrograms: number;
  private maxInspirations: number;
  private includeArtifacts: boolean;

  constructor(opts: PromptBuilderOptions = {}) {
    this.maxCodeLength = opts.maxCodeLength ?? 10000;
    this.maxTopPrograms = opts.maxTopPrograms ?? 3;
    this.maxInspirations = opts.maxInspirations ?? 2;
    this.includeArtifacts = opts.includeArtifacts ?? true;
  }

  /** Build a mutation prompt with full context. */
  buildMutationPrompt(opts: {
    parent: Program;
    objective: string;
    topPrograms?: Program[];
    inspirations?: Program[];
    feedback?: string;
    iteration?: number;
  }): string {
    const parts: string[] = [];

    // Objective
    parts.push(this.buildObjectiveSection(opts.objective, opts.iteration));

    // Current program
    parts.push(this.buildCurrentProgramSection(opts.parent));

    // Top performers
    if (opts.topPrograms && opts.topPrograms.length > 0) {
      parts.push(this.buildTopProgramsSection(opts.topPrograms));
    }

    // Inspirations
    if (opts.inspirations && opts.inspirations.length > 0) {
      parts.push(this.buildInspirationsSection(opts.inspirations));
    }

    // Feedback
    if (opts.feedback && this.includeArtifacts) {
      parts.push(this.buildFeedbackSection(opts.feedback));
    }

    // Task
    parts.push(this.buildTaskSection());

    return parts.join('\n\n');
  }

  /** Build a crossover prompt combining two parents. */
  buildCrossoverPrompt(opts: {
    parentA: Program;
    parentB: Program;
    objective: string;
  }): string {
    const parts: string[] = [];

    parts.push(`## Objective\n${opts.objective}`);

    parts.push(`## Parent A (Fitness: ${(opts.parentA.fitness ?? 0).toFixed(4)})\n\`\`\`\n${this.truncateCode(opts.parentA.code)}\n\`\`\``);

    parts.push(`## Parent B (Fitness: ${(opts.parentB.fitness ?? 0).toFixed(4)})\n\`\`\`\n${this.truncateCode(opts.parentB.code)}\n\`\`\``);

    parts.push(`## Task\nCombine the best aspects of both parents into a single improved program.
Consider what makes each parent successful and merge those traits.
Output the complete child program in a code block.`);

    return parts.join('\n\n');
  }

  /** Build an analysis prompt for understanding program quality. */
  buildAnalysisPrompt(opts: {
    program: Program;
    objective: string;
    topPrograms?: Program[];
  }): string {
    const parts: string[] = [];

    parts.push(`## Objective\n${opts.objective}`);
    parts.push(this.buildCurrentProgramSection(opts.program));

    if (opts.topPrograms && opts.topPrograms.length > 0) {
      parts.push(this.buildTopProgramsSection(opts.topPrograms));
    }

    parts.push(`## Analysis Task
Analyze this program and explain:
1. What it does well (strengths)
2. What it does poorly (weaknesses)
3. Specific improvements that would increase its fitness score
4. Which aspects of the top programs could be incorporated

Be concrete and reference specific code sections.`);

    return parts.join('\n\n');
  }

  private buildObjectiveSection(objective: string, iteration?: number): string {
    const header = iteration !== undefined
      ? `## Optimization Objective (Iteration ${iteration})`
      : '## Optimization Objective';
    return `${header}\n\n${objective}`;
  }

  private buildCurrentProgramSection(program: Program): string {
    const parts: string[] = [];
    parts.push(`## Current Program (Fitness: ${(program.fitness ?? 0).toFixed(4)})`);

    // Metrics
    if (Object.keys(program.metrics).length > 0) {
      const metricsLines = Object.entries(program.metrics)
        .filter(([, v]) => isFinite(v))
        .map(([k, v]) => `  - ${k}: ${v.toFixed(4)}`)
        .join('\n');
      if (metricsLines) {
        parts.push(`Metrics:\n${metricsLines}`);
      }
    }

    // Code
    parts.push(`\`\`\`\n${this.truncateCode(program.code)}\n\`\`\``);

    return parts.join('\n');
  }

  private buildTopProgramsSection(programs: Program[]): string {
    const parts: string[] = ['## Top Performing Programs', 'Learn from these high-scoring examples:'];

    for (let i = 0; i < Math.min(programs.length, this.maxTopPrograms); i++) {
      const prog = programs[i];
      parts.push(`### #${i + 1} (Fitness: ${(prog.fitness ?? 0).toFixed(4)})`);

      if (prog.diffFromParent) {
        parts.push(`Key changes:\n\`\`\`diff\n${prog.diffFromParent.slice(0, 500)}\n\`\`\``);
      } else {
        parts.push(`\`\`\`\n${this.truncateCode(prog.code, 2000)}\n\`\`\``);
      }
    }

    return parts.join('\n\n');
  }

  private buildInspirationsSection(programs: Program[]): string {
    const parts: string[] = ['## Diverse Inspirations', 'Consider these alternative approaches:'];

    for (let i = 0; i < Math.min(programs.length, this.maxInspirations); i++) {
      const prog = programs[i];
      parts.push(`### Inspiration ${i + 1} (Fitness: ${(prog.fitness ?? 0).toFixed(4)})`);
      parts.push(`\`\`\`\n${this.truncateCode(prog.code, 1500)}\n\`\`\``);
    }

    return parts.join('\n\n');
  }

  private buildFeedbackSection(feedback: string): string {
    return `## Evaluation Feedback\n${feedback.slice(0, 2000)}`;
  }

  private buildTaskSection(): string {
    return `## Your Task
Generate an improved version of the current program.
Focus on the optimization objective and learn from top performers.
Output the COMPLETE improved program in a single code block.`;
  }

  private truncateCode(code: string, maxLen?: number): string {
    const limit = maxLen ?? this.maxCodeLength;
    if (code.length <= limit) return code;
    const truncated = code.slice(0, limit);
    const lastNewline = truncated.lastIndexOf('\n');
    if (lastNewline > limit * 0.8) {
      return truncated.slice(0, lastNewline) + '\n// ... (truncated)';
    }
    return truncated + '\n// ... (truncated)';
  }
}
