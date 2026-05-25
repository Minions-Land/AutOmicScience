/**
 * Evolution progress visualization.
 * Provides data structures for charting fitness over generations,
 * population diversity, and best program history.
 */

import type { Program } from './Program.js';
import type { EvolutionDatabase, DatabaseStats, Generation } from './Database.js';
import type { IslandModel, IslandStats } from './IslandModel.js';

/** A data point for fitness over time charts. */
export interface FitnessDataPoint {
  generation: number;
  bestFitness: number;
  averageFitness: number;
  minFitness: number;
  diversity: number;
  populationSize: number;
}

/** Summary of the best program at each generation. */
export interface BestProgramEntry {
  generation: number;
  programId: string;
  fitness: number;
  codeLength: number;
  mutationStrategy?: string;
  improvement: number; // delta from previous best
}

/** Complete evolution visualization data. */
export interface VisualizationData {
  fitnessHistory: FitnessDataPoint[];
  bestProgramHistory: BestProgramEntry[];
  islandStats: IslandStats[];
  databaseStats: DatabaseStats;
  diversityOverTime: { generation: number; diversity: number }[];
  config: Record<string, unknown>;
}

/**
 * Collects and formats evolution data for visualization.
 */
export class Visualizer {
  private fitnessHistory: FitnessDataPoint[] = [];
  private bestProgramHistory: BestProgramEntry[] = [];
  private diversityHistory: { generation: number; diversity: number }[] = [];
  private previousBestFitness = 0;

  /** Record a generation's data. */
  recordGeneration(
    generation: number,
    programs: Program[],
    diversity?: number,
  ): void {
    const fitnesses = programs.map((p) => p.fitness ?? 0);
    const best = Math.max(...fitnesses, 0);
    const avg = fitnesses.length > 0
      ? fitnesses.reduce((s, f) => s + f, 0) / fitnesses.length
      : 0;
    const min = Math.min(...fitnesses, 0);

    // Compute diversity as ratio of unique code prefixes
    const uniqueCodes = new Set(programs.map((p) => p.code.slice(0, 200)));
    const computedDiversity = diversity ?? uniqueCodes.size / Math.max(programs.length, 1);

    this.fitnessHistory.push({
      generation,
      bestFitness: best,
      averageFitness: avg,
      minFitness: min,
      diversity: computedDiversity,
      populationSize: programs.length,
    });

    this.diversityHistory.push({ generation, diversity: computedDiversity });

    // Track best program
    const bestProgram = programs.reduce((a, b) =>
      (b.fitness ?? 0) > (a.fitness ?? 0) ? b : a,
    );

    if (bestProgram) {
      const improvement = best - this.previousBestFitness;
      this.bestProgramHistory.push({
        generation,
        programId: bestProgram.id,
        fitness: best,
        codeLength: bestProgram.code.length,
        mutationStrategy: bestProgram.mutationInfo?.strategy,
        improvement,
      });
      this.previousBestFitness = Math.max(this.previousBestFitness, best);
    }
  }

  /** Get fitness history data for charting. */
  getFitnessHistory(): FitnessDataPoint[] {
    return [...this.fitnessHistory];
  }

  /** Get best program history. */
  getBestProgramHistory(): BestProgramEntry[] {
    return [...this.bestProgramHistory];
  }

  /** Get diversity over time. */
  getDiversityHistory(): { generation: number; diversity: number }[] {
    return [...this.diversityHistory];
  }

  /** Get complete visualization data snapshot. */
  getVisualizationData(
    database: EvolutionDatabase,
    islandModel?: IslandModel,
    config?: Record<string, unknown>,
  ): VisualizationData {
    const dbStats = database.getStatistics();
    const islandStats = islandModel
      ? islandModel.getStats((id) => database.getProgram(id))
      : [];

    return {
      fitnessHistory: this.fitnessHistory,
      bestProgramHistory: this.bestProgramHistory,
      islandStats,
      databaseStats: dbStats,
      diversityOverTime: this.diversityHistory,
      config: config ?? {},
    };
  }

  /** Export all visualization data as JSON string. */
  exportJSON(
    database: EvolutionDatabase,
    islandModel?: IslandModel,
    config?: Record<string, unknown>,
  ): string {
    const data = this.getVisualizationData(database, islandModel, config);
    return JSON.stringify(data, null, 2);
  }

  /** Get a text summary of evolution progress. */
  getSummary(): string {
    if (this.fitnessHistory.length === 0) {
      return 'No evolution data recorded yet.';
    }

    const latest = this.fitnessHistory[this.fitnessHistory.length - 1];
    const first = this.fitnessHistory[0];
    const totalImprovement = latest.bestFitness - first.bestFitness;
    const improvementPct = first.bestFitness > 0
      ? (totalImprovement / first.bestFitness) * 100
      : 0;

    const lines = [
      '='.repeat(50),
      'Evolution Progress Summary',
      '='.repeat(50),
      `Generations completed: ${this.fitnessHistory.length}`,
      `Best fitness: ${latest.bestFitness.toFixed(4)}`,
      `Average fitness: ${latest.averageFitness.toFixed(4)}`,
      `Total improvement: ${totalImprovement.toFixed(4)} (${improvementPct.toFixed(1)}%)`,
      `Current diversity: ${latest.diversity.toFixed(3)}`,
      `Population size: ${latest.populationSize}`,
      '='.repeat(50),
    ];

    return lines.join('\n');
  }

  /** Reset all recorded data. */
  reset(): void {
    this.fitnessHistory = [];
    this.bestProgramHistory = [];
    this.diversityHistory = [];
    this.previousBestFitness = 0;
  }
}
