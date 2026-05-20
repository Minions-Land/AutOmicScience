import { newProgram, type Program } from './Program.js';
import type { Generation } from './Database.js';
import type { Evaluator } from './Evaluator.js';

export type { Evaluator } from './Evaluator.js';

export interface Mutator {
  mutate(p: Program, gen: number): Promise<Program>;
}

export interface Crossover {
  cross(a: Program, b: Program, gen: number): Promise<Program>;
}

export interface EvolverOptions {
  population: Program[];
  evaluator: Evaluator;
  mutator: Mutator;
  crossover?: Crossover;
  generations?: number;
  eliteCount?: number;
}

/** Configuration for the `evolve` async generator. */
export interface EvolutionConfig {
  generations: number;
  eliteCount: number;
  populationSize: number;
  mutationRate: number;
}

/** Minimal genetic-algorithm driver. */
export class Evolver {
  constructor(private opts: EvolverOptions) {}

  async run(): Promise<Program[]> {
    const generations = this.opts.generations ?? 5;
    const eliteCount = this.opts.eliteCount ?? 2;
    let pop = [...this.opts.population];

    for (let gen = 0; gen < generations; gen++) {
      for (const p of pop) {
        if (p.fitness === undefined) p.fitness = await this.opts.evaluator.evaluate(p);
      }
      pop.sort((a, b) => (b.fitness ?? 0) - (a.fitness ?? 0));
      const elites = pop.slice(0, eliteCount);
      const offspring: Program[] = [];
      while (offspring.length < pop.length - elites.length) {
        const a = pop[Math.floor(Math.random() * pop.length)];
        const b = pop[Math.floor(Math.random() * pop.length)];
        let child = this.opts.crossover
          ? await this.opts.crossover.cross(a, b, gen + 1)
          : newProgram(a.code, [a.id], gen + 1);
        child = await this.opts.mutator.mutate(child, gen + 1);
        offspring.push(child);
      }
      pop = [...elites, ...offspring];
    }

    for (const p of pop) {
      if (p.fitness === undefined) p.fitness = await this.opts.evaluator.evaluate(p);
    }
    pop.sort((a, b) => (b.fitness ?? 0) - (a.fitness ?? 0));
    return pop;
  }

  /**
   * Run evolution as an async generator, yielding each generation as it completes.
   * @param population - Initial population of programs.
   * @param evaluator - Fitness evaluator.
   * @param config - Evolution configuration.
   */
  static async *evolve(
    population: Program[],
    evaluator: Evaluator,
    config: EvolutionConfig,
  ): AsyncGenerator<Generation> {
    let pop = [...population];
    const { generations, eliteCount } = config;

    for (let gen = 0; gen < generations; gen++) {
      for (const p of pop) {
        if (p.fitness === undefined) p.fitness = await evaluator.evaluate(p);
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
        const child: Program = {
          id: `prog_${Math.random().toString(36).slice(2, 10)}`,
          code: childCode,
          generation: gen + 1,
          parentIds: [a.id, b.id],
        };
        offspring.push(child);
      }
      pop = [...elites, ...offspring];
    }
  }
}
