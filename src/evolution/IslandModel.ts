/**
 * Island model for multi-population evolution.
 * Multiple sub-populations evolve independently with periodic migration.
 * Different islands can use different mutation strategies for diversity.
 */

import type { Program } from './Program.js';
import type { MutationStrategy } from './LLMMutator.js';
import type { EvolutionConfig } from './EvolutionConfig.js';
import { DEFAULT_CONFIG } from './EvolutionConfig.js';

/** Configuration for a single island. */
export interface IslandConfig {
  id: number;
  /** Preferred mutation strategy for this island. */
  strategy: MutationStrategy;
  /** Population of program IDs on this island. */
  population: Set<string>;
  /** Island-specific mutation temperature modifier. */
  temperatureModifier: number;
}

/** Migration event record. */
export interface MigrationEvent {
  fromIsland: number;
  toIsland: number;
  programId: string;
  fitness: number;
  generation: number;
  timestamp: number;
}

/** Island model statistics. */
export interface IslandStats {
  islandId: number;
  populationSize: number;
  bestFitness: number;
  avgFitness: number;
  strategy: MutationStrategy;
  migrationsSent: number;
  migrationsReceived: number;
}

/**
 * Island model manager.
 * Coordinates multiple sub-populations with different strategies
 * and handles periodic migration between islands.
 */
export class IslandModel {
  private islands: IslandConfig[];
  private migrationHistory: MigrationEvent[] = [];
  private migrationsSent: number[];
  private migrationsReceived: number[];
  private config: EvolutionConfig;

  constructor(config: EvolutionConfig = DEFAULT_CONFIG) {
    this.config = config;
    const strategies: MutationStrategy[] = ['semantic', 'structural', 'point', 'guided', 'random'];

    this.islands = Array.from({ length: config.numIslands }, (_, i) => ({
      id: i,
      strategy: strategies[i % strategies.length],
      population: new Set<string>(),
      temperatureModifier: 0.8 + (i * 0.1), // Vary temperature across islands
    }));

    this.migrationsSent = new Array(config.numIslands).fill(0);
    this.migrationsReceived = new Array(config.numIslands).fill(0);
  }

  /** Get the number of islands. */
  get numIslands(): number {
    return this.islands.length;
  }

  /** Get island configuration. */
  getIsland(id: number): IslandConfig {
    return this.islands[id];
  }

  /** Get the preferred strategy for an island. */
  getStrategy(islandId: number): MutationStrategy {
    return this.islands[islandId].strategy;
  }

  /** Get the temperature modifier for an island. */
  getTemperatureModifier(islandId: number): number {
    return this.islands[islandId].temperatureModifier;
  }

  /** Assign a program to an island. */
  assignToIsland(programId: string, islandId?: number): number {
    const target = islandId ?? this.selectIslandForAssignment();
    this.islands[target].population.add(programId);
    return target;
  }

  /** Select island for new program assignment (least populated). */
  private selectIslandForAssignment(): number {
    let minSize = Infinity;
    let minIsland = 0;
    for (let i = 0; i < this.islands.length; i++) {
      if (this.islands[i].population.size < minSize) {
        minSize = this.islands[i].population.size;
        minIsland = i;
      }
    }
    return minIsland;
  }

  /**
   * Perform migration between islands using ring topology.
   * Top programs from each island migrate to the next island.
   */
  performMigration(
    getProgram: (id: string) => Program | undefined,
    generation: number,
  ): MigrationEvent[] {
    if (this.islands.length < 2) return [];

    const events: MigrationEvent[] = [];
    const rate = this.config.migrationRate;

    for (let src = 0; src < this.islands.length; src++) {
      const target = (src + 1) % this.islands.length;
      const srcPop = [...this.islands[src].population];

      // Get top programs from source
      const numMigrants = Math.max(1, Math.floor(srcPop.length * rate));
      const migrants = srcPop
        .map((id) => ({ id, program: getProgram(id) }))
        .filter((m) => m.program !== undefined)
        .sort((a, b) => (b.program!.fitness ?? 0) - (a.program!.fitness ?? 0))
        .slice(0, numMigrants);

      // Migrate to target island
      for (const { id, program } of migrants) {
        this.islands[target].population.add(id);
        this.migrationsSent[src]++;
        this.migrationsReceived[target]++;

        const event: MigrationEvent = {
          fromIsland: src,
          toIsland: target,
          programId: id,
          fitness: program!.fitness ?? 0,
          generation,
          timestamp: Date.now(),
        };
        events.push(event);
        this.migrationHistory.push(event);
      }
    }

    return events;
  }

  /** Check if migration should occur at this generation. */
  shouldMigrate(generation: number): boolean {
    return generation > 0 && generation % this.config.migrationInterval === 0;
  }

  /** Get programs from a specific island. */
  getIslandPopulation(islandId: number): string[] {
    return [...this.islands[islandId].population];
  }

  /** Get the best programs across all islands for final selection. */
  mergeForFinalSelection(
    getProgram: (id: string) => Program | undefined,
    topN: number,
  ): Program[] {
    const allPrograms: Program[] = [];

    for (const island of this.islands) {
      for (const id of island.population) {
        const prog = getProgram(id);
        if (prog) allPrograms.push(prog);
      }
    }

    return allPrograms
      .sort((a, b) => (b.fitness ?? 0) - (a.fitness ?? 0))
      .slice(0, topN);
  }

  /** Get statistics for all islands. */
  getStats(getProgram: (id: string) => Program | undefined): IslandStats[] {
    return this.islands.map((island, i) => {
      const programs = [...island.population]
        .map((id) => getProgram(id))
        .filter((p): p is Program => p !== undefined);

      const fitnesses = programs.map((p) => p.fitness ?? 0);

      return {
        islandId: island.id,
        populationSize: island.population.size,
        bestFitness: fitnesses.length > 0 ? Math.max(...fitnesses) : 0,
        avgFitness: fitnesses.length > 0
          ? fitnesses.reduce((s, f) => s + f, 0) / fitnesses.length
          : 0,
        strategy: island.strategy,
        migrationsSent: this.migrationsSent[i],
        migrationsReceived: this.migrationsReceived[i],
      };
    });
  }

  /** Get migration history. */
  getMigrationHistory(): MigrationEvent[] {
    return [...this.migrationHistory];
  }

  /** Get diversity metric: how different are the islands from each other. */
  getInterIslandDiversity(getProgram: (id: string) => Program | undefined): number {
    if (this.islands.length < 2) return 0;

    // Compare average fitness between islands
    const avgFitnesses = this.islands.map((island) => {
      const programs = [...island.population]
        .map((id) => getProgram(id))
        .filter((p): p is Program => p !== undefined);
      const fitnesses = programs.map((p) => p.fitness ?? 0);
      return fitnesses.length > 0
        ? fitnesses.reduce((s, f) => s + f, 0) / fitnesses.length
        : 0;
    });

    // Compute variance of average fitnesses
    const mean = avgFitnesses.reduce((s, f) => s + f, 0) / avgFitnesses.length;
    const variance = avgFitnesses.reduce((s, f) => s + (f - mean) ** 2, 0) / avgFitnesses.length;
    return Math.sqrt(variance);
  }
}
