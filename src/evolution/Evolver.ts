import { newProgram, type Program } from './Program.js';

export interface Evaluator {
  evaluate(p: Program): Promise<number>; // higher == better
}

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
}
