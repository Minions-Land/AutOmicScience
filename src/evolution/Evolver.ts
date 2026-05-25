/**
 * Full evolution driver integrating LLM mutation, island model,
 * database persistence, and visualization.
 * Equivalent to AutOmicScience evolution/team.py.
 */

import type { Program } from './Program.js';
import { createProgram } from './Program.js';
import type { Evaluator, EvaluationResult } from './Evaluator.js';
import { EvolutionDatabase, type Generation } from './Database.js';
import { LLMMutator, type MutationStrategy } from './LLMMutator.js';
import { IslandModel } from './IslandModel.js';
import { PromptBuilder } from './PromptBuilder.js';
import { Visualizer, type FitnessDataPoint } from './Visualizer.js';
import type { EvolutionConfig } from './EvolutionConfig.js';
import { DEFAULT_CONFIG, createConfig } from './EvolutionConfig.js';

// ── Types ────────────────────────────────────────────────────────────────────

export type { Evaluator } from './Evaluator.js';

export interface Mutator {
  mutate(p: Program, gen: number): Promise<Program>;
}

export interface Crossover {
  cross(a: Program, b: Program, gen: number): Promise<Program>;
}

/** Events emitted during evolution. */
export type EvolutionEvent =
  | { type: 'generation'; data: Generation }
  | { type: 'improvement'; data: { program: Program; fitness: number; generation: number } }
  | { type: 'migration'; data: { migrated: number; generation: number } }
  | { type: 'evaluation'; data: { programId: string; result: EvaluationResult } }
  | { type: 'stagnation'; data: { generations: number } }
  | { type: 'complete'; data: { bestProgram: Program; totalGenerations: number } }
  | { type: 'error'; data: { error: string; generation: number } }
  | { type: 'paused'; data: { generation: number } }
  | { type: 'resumed'; data: { generation: number } };

/** Evolution run result. */
export interface EvolutionResult {
  bestProgram: Program;
  bestFitness: number;
  totalGenerations: number;
  totalEvaluations: number;
  improvements: number;
  duration: number;
  fitnessHistory: FitnessDataPoint[];
  database: EvolutionDatabase;
}

/** Options for the legacy Evolver class. */
export interface EvolverOptions {
  population: Program[];
  evaluator: Evaluator;
  mutator: Mutator;
  crossover?: Crossover;
  generations?: number;
  eliteCount?: number;
}

/** Configuration for the evolve async generator. */
export interface EvolutionRunConfig {
  generations: number;
  eliteCount: number;
  populationSize: number;
  mutationRate: number;
}

// ── Full Evolution Engine ────────────────────────────────────────────────────

/**
 * Full-featured evolution engine with LLM mutation, island model,
 * persistence, and event streaming.
 */
export class EvolutionEngine {
  private database: EvolutionDatabase;
  private mutator: LLMMutator;
  private islandModel: IslandModel;
  private visualizer: Visualizer;
  private promptBuilder: PromptBuilder;
  private config: EvolutionConfig;
  private evaluator: Evaluator;
  private objective: string;

  // State
  private paused = false;
  private stopped = false;
  private currentGeneration = 0;
  private totalEvaluations = 0;
  private improvements = 0;
  private bestFitnessEver = -Infinity;
  private generationsWithoutImprovement = 0;

  constructor(opts: {
    evaluator: Evaluator;
    objective: string;
    config?: Partial<EvolutionConfig>;
    initialPopulation?: Program[];
    database?: EvolutionDatabase;
  }) {
    this.config = createConfig(opts.config);
    this.evaluator = opts.evaluator;
    this.objective = opts.objective;

    this.database = opts.database ?? new EvolutionDatabase(this.config);
    this.islandModel = new IslandModel(this.config);
    this.visualizer = new Visualizer();
    this.promptBuilder = new PromptBuilder();

    this.mutator = new LLMMutator({
      model: this.config.mutatorModel,
      temperature: this.config.temperature,
      maxRetries: this.config.maxRetries,
      timeout: this.config.mutationTimeout * 1000,
      objective: opts.objective,
      promptBuilder: this.promptBuilder,
    });

    // Seed initial population
    if (opts.initialPopulation) {
      for (const prog of opts.initialPopulation) {
        this.database.add(prog);
        this.islandModel.assignToIsland(prog.id, prog.islandId);
      }
    }
  }

  /** Pause evolution (can be resumed). */
  pause(): void {
    this.paused = true;
  }

  /** Resume paused evolution. */
  resume(): void {
    this.paused = false;
  }

  /** Stop evolution permanently. */
  stop(): void {
    this.stopped = true;
  }

  /** Get current state. */
  getState(): { generation: number; paused: boolean; stopped: boolean } {
    return {
      generation: this.currentGeneration,
      paused: this.paused,
      stopped: this.stopped,
    };
  }

  /** Get the database for external access. */
  getDatabase(): EvolutionDatabase {
    return this.database;
  }

  /** Get the visualizer for external access. */
  getVisualizer(): Visualizer {
    return this.visualizer;
  }

  /**
   * Run evolution as an async generator, yielding events as they occur.
   * This is the primary API for running evolution with full features.
   */
  async *run(): AsyncGenerator<EvolutionEvent> {
    const startTime = Date.now();

    for (let gen = 0; gen < this.config.maxIterations; gen++) {
      this.currentGeneration = gen;

      // Check stop/pause
      if (this.stopped) break;
      if (this.paused) {
        yield { type: 'paused', data: { generation: gen } };
        while (this.paused && !this.stopped) {
          await sleep(100);
        }
        if (this.stopped) break;
        yield { type: 'resumed', data: { generation: gen } };
      }

      // Early stopping check
      if (this.generationsWithoutImprovement >= this.config.earlyStopGenerations) {
        yield { type: 'stagnation', data: { generations: this.generationsWithoutImprovement } };
        break;
      }

      // Sample parent and inspirations from database
      let parent: Program;
      let inspirations: Program[] = [];
      let topPrograms: Program[] = [];

      if (this.database.getAllPrograms().length > 0) {
        parent = this.database.sampleParent(gen % this.config.numIslands);
        inspirations = this.database.sampleInspirations(
          this.config.numInspirations,
          new Set([parent.id]),
        );
        topPrograms = this.database.getTopPrograms(this.config.numTopPrograms);
      } else {
        // No programs yet - this shouldn't happen if initial population was provided
        yield { type: 'error', data: { error: 'No programs in database', generation: gen } };
        break;
      }

      // Determine mutation strategy from island model
      const islandId = gen % this.config.numIslands;
      const strategy = this.islandModel.getStrategy(islandId);

      // Generate child via LLM mutation
      let child: Program;
      try {
        if (Math.random() < this.config.crossoverRate && inspirations.length > 0) {
          // Crossover
          child = await this.mutator.crossover(parent, inspirations[0], gen);
        } else {
          // Mutation
          child = await this.mutator.mutate(parent, gen, {
            strategy,
            topPrograms,
            inspirations,
            feedback: parent.llmFeedback,
          });
        }
      } catch (e) {
        yield { type: 'error', data: { error: (e as Error).message, generation: gen } };
        continue;
      }

      // Evaluate child
      let evalResult: EvaluationResult;
      try {
        evalResult = await this.evaluator.evaluate(child);
        this.totalEvaluations++;
      } catch (e) {
        yield { type: 'error', data: { error: `Evaluation failed: ${(e as Error).message}`, generation: gen } };
        continue;
      }

      // Apply evaluation results
      child.fitness = evalResult.fitness;
      child.metrics = evalResult.metrics;
      child.llmFeedback = evalResult.stderr || evalResult.error;
      child.evaluationError = evalResult.error;

      yield { type: 'evaluation', data: { programId: child.id, result: evalResult } };

      // Add to database
      const improved = this.database.add(child, islandId);
      this.islandModel.assignToIsland(child.id, islandId);

      // Track improvements
      if (child.fitness > this.bestFitnessEver) {
        this.bestFitnessEver = child.fitness;
        this.improvements++;
        this.generationsWithoutImprovement = 0;
        yield { type: 'improvement', data: { program: child, fitness: child.fitness, generation: gen } };
      } else {
        this.generationsWithoutImprovement++;
      }

      // Record generation for visualization
      const allPrograms = this.database.getAllPrograms();
      this.visualizer.recordGeneration(gen, allPrograms);
      const generation = this.database.recordGeneration(gen, allPrograms);
      yield { type: 'generation', data: generation };

      // Island migration
      if (this.islandModel.shouldMigrate(gen)) {
        const events = this.islandModel.performMigration(
          (id) => this.database.getProgram(id),
          gen,
        );
        const migrated = this.database.migrate(this.config.migrationRate);
        yield { type: 'migration', data: { migrated, generation: gen } };
      }

      // Persist if configured
      if (this.config.dbPath && gen % 10 === 0) {
        try {
          this.database.save(this.config.dbPath);
        } catch {
          // Non-fatal: log but continue
        }
      }
    }

    // Final save
    if (this.config.dbPath) {
      try {
        this.database.save(this.config.dbPath);
      } catch {
        // Non-fatal
      }
    }

    const bestProgram = this.database.getBestProgram();
    if (bestProgram) {
      yield {
        type: 'complete',
        data: {
          bestProgram,
          totalGenerations: this.currentGeneration + 1,
        },
      };
    }
  }

  /** Run evolution and return the final result (non-streaming). */
  async runToCompletion(): Promise<EvolutionResult> {
    const startTime = Date.now();

    for await (const event of this.run()) {
      // Consume all events
      if (event.type === 'complete') break;
    }

    const bestProgram = this.database.getBestProgram() ?? createProgram('// empty');

    return {
      bestProgram,
      bestFitness: bestProgram.fitness ?? 0,
      totalGenerations: this.currentGeneration + 1,
      totalEvaluations: this.totalEvaluations,
      improvements: this.improvements,
      duration: Date.now() - startTime,
      fitnessHistory: this.visualizer.getFitnessHistory(),
      database: this.database,
    };
  }
}

// ── Legacy Evolver (backward compatible) ─────────────────────────────────────

/** Minimal genetic-algorithm driver (backward compatible). */
export class Evolver {
  constructor(private opts: EvolverOptions) {}

  async run(): Promise<Program[]> {
    const generations = this.opts.generations ?? 5;
    const eliteCount = this.opts.eliteCount ?? 2;
    let pop = [...this.opts.population];

    for (let gen = 0; gen < generations; gen++) {
      for (const p of pop) {
        if (p.fitness === undefined) {
          const result = await this.opts.evaluator.evaluate(p);
          p.fitness = result.fitness;
          p.metrics = result.metrics;
        }
      }
      pop.sort((a, b) => (b.fitness ?? 0) - (a.fitness ?? 0));
      const elites = pop.slice(0, eliteCount);
      const offspring: Program[] = [];
      while (offspring.length < pop.length - elites.length) {
        const a = pop[Math.floor(Math.random() * pop.length)];
        const b = pop[Math.floor(Math.random() * pop.length)];
        let child = this.opts.crossover
          ? await this.opts.crossover.cross(a, b, gen + 1)
          : createProgram(a.code, { parentIds: [a.id], generation: gen + 1 });
        child = await this.opts.mutator.mutate(child, gen + 1);
        offspring.push(child);
      }
      pop = [...elites, ...offspring];
    }

    for (const p of pop) {
      if (p.fitness === undefined) {
        const result = await this.opts.evaluator.evaluate(p);
        p.fitness = result.fitness;
        p.metrics = result.metrics;
      }
    }
    pop.sort((a, b) => (b.fitness ?? 0) - (a.fitness ?? 0));
    return pop;
  }

  /**
   * Run evolution as an async generator, yielding each generation.
   */
  static async *evolve(
    population: Program[],
    evaluator: Evaluator,
    config: EvolutionRunConfig,
  ): AsyncGenerator<Generation> {
    let pop = [...population];
    const { generations, eliteCount } = config;

    for (let gen = 0; gen < generations; gen++) {
      for (const p of pop) {
        if (p.fitness === undefined) {
          const result = await evaluator.evaluate(p);
          p.fitness = result.fitness;
          p.metrics = result.metrics;
        }
      }
      pop.sort((a, b) => (b.fitness ?? 0) - (a.fitness ?? 0));

      const fitnesses = pop.map((p) => p.fitness ?? 0);
      const generation: Generation = {
        id: `gen_${gen}_${Date.now()}`,
        index: gen,
        programs: [...pop],
        bestFitness: fitnesses[0] ?? 0,
        averageFitness: fitnesses.reduce((s, f) => s + f, 0) / fitnesses.length,
        timestamp: Date.now(),
      };
      yield generation;

      // Produce next generation
      const elites = pop.slice(0, eliteCount);
      const offspring: Program[] = [];
      while (offspring.length < pop.length - elites.length) {
        const a = pop[Math.floor(Math.random() * pop.length)];
        const b = pop[Math.floor(Math.random() * pop.length)];
        const point = Math.floor(Math.random() * Math.min(a.code.length, b.code.length));
        const childCode = a.code.slice(0, point) + b.code.slice(point);
        const child = createProgram(childCode, {
          parentIds: [a.id, b.id],
          generation: gen + 1,
        });
        offspring.push(child);
      }
      pop = [...elites, ...offspring];
    }
  }
}

// ── Utilities ────────────────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
