import type { Program } from './Program.js';

/** A single generation snapshot in the evolutionary process. */
export interface Generation {
  id: string;
  index: number;
  programs: Program[];
  bestFitness: number;
  averageFitness: number;
  timestamp: number;
}

/**
 * Persistence layer for evolution runs.
 */
export interface EvolutionDB {
  /** Save a generation snapshot. */
  save(gen: Generation): Promise<void>;
  /** Load a generation by its id. */
  load(id: string): Promise<Generation | null>;
  /** List all stored generation ids. */
  listGenerations(): Promise<string[]>;
}

/**
 * In-memory implementation of EvolutionDB for testing and prototyping.
 */
export class InMemoryEvolutionDB implements EvolutionDB {
  private store = new Map<string, Generation>();

  /** Save a generation snapshot. */
  async save(gen: Generation): Promise<void> {
    this.store.set(gen.id, gen);
  }

  /** Load a generation by its id. */
  async load(id: string): Promise<Generation | null> {
    return this.store.get(id) ?? null;
  }

  /** List all stored generation ids. */
  async listGenerations(): Promise<string[]> {
    return [...this.store.keys()];
  }
}
