/**
 * Full population store with lineage tracking, island model, and persistence.
 * Equivalent to AutOmicScience evolution/database.py.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'fs';
import { join } from 'path';
import type { Program, FitnessMetrics } from './Program.js';
import { serializeProgram, deserializeProgram, computeFitness } from './Program.js';
import type { EvolutionConfig } from './EvolutionConfig.js';
import { DEFAULT_CONFIG } from './EvolutionConfig.js';

/** A single generation snapshot. */
export interface Generation {
  id: string;
  index: number;
  programs: Program[];
  bestFitness: number;
  averageFitness: number;
  timestamp: number;
}

/** Statistics about the database state. */
export interface DatabaseStats {
  totalPrograms: number;
  totalAdded: number;
  totalImproved: number;
  archiveSize: number;
  numIslands: number;
  islandSizes: number[];
  bestFitness: number;
  avgFitness: number;
  minFitness: number;
  generationCount: number;
}

/** Lineage edge in the DAG. */
interface LineageEdge {
  parentId: string;
  childId: string;
  generation: number;
}

/**
 * Full evolution database with:
 * - Multi-island populations
 * - Elite archive
 * - Lineage DAG tracking
 * - Fitness-weighted sampling
 * - JSON persistence
 */
export class EvolutionDatabase {
  private programs = new Map<string, Program>();
  private islands: Set<string>[];
  private archive = new Set<string>();
  private lineageEdges: LineageEdge[] = [];
  private generations: Generation[] = [];
  private bestProgramId: string | null = null;
  private fitnessWeights: Record<string, number> = {};

  // Metric ranges for normalization
  private metricRanges = new Map<string, [number, number]>();

  // Counters
  private totalAdded = 0;
  private totalImproved = 0;
  private nextOrder = 0;

  constructor(private config: EvolutionConfig = DEFAULT_CONFIG) {
    this.islands = Array.from({ length: config.numIslands }, () => new Set<string>());
  }

  /** Set fitness weights for score computation. */
  setFitnessWeights(weights: Record<string, number>): void {
    this.fitnessWeights = weights;
  }

  /** Add a program to the database. Returns true if it improved its niche. */
  add(program: Program, targetIsland?: number): boolean {
    this.totalAdded++;
    program.metadata._order = this.nextOrder++;

    // Store program
    this.programs.set(program.id, program);

    // Assign to island
    const island = targetIsland ?? Math.floor(Math.random() * this.config.numIslands);
    program.islandId = island;
    this.islands[island].add(program.id);

    // Track lineage
    for (const parentId of program.parentIds) {
      this.lineageEdges.push({
        parentId,
        childId: program.id,
        generation: program.generation,
      });
    }

    // Update metric ranges
    this.updateMetricRanges(program.metrics);

    // Compute fitness with current ranges
    const fitness = this.computeProgramFitness(program);
    program.fitness = fitness;

    // Update archive (top programs)
    const improved = this.updateArchive(program);
    if (improved) this.totalImproved++;

    // Update best
    this.updateBest(program);

    return improved;
  }

  /** Compute fitness for a program using current metric ranges for normalization. */
  computeProgramFitness(program: Program): number {
    if (Object.keys(this.fitnessWeights).length === 0) {
      return computeFitness(program.metrics);
    }

    // Normalize metrics using observed ranges, then apply weights
    const normalized: FitnessMetrics = {};
    for (const [key, value] of Object.entries(program.metrics)) {
      const range = this.metricRanges.get(key);
      if (range) {
        const [min, max] = range;
        const span = max - min;
        normalized[key] = span > 1e-8 ? (value - min) / span : 0.5;
      } else {
        normalized[key] = value;
      }
    }

    return computeFitness(normalized, this.fitnessWeights);
  }

  /** Update observed metric ranges. */
  private updateMetricRanges(metrics: FitnessMetrics): void {
    for (const [key, value] of Object.entries(metrics)) {
      if (!isFinite(value)) continue;
      const existing = this.metricRanges.get(key);
      if (!existing) {
        this.metricRanges.set(key, [value, value]);
      } else {
        this.metricRanges.set(key, [
          Math.min(existing[0], value),
          Math.max(existing[1], value),
        ]);
      }
    }
  }

  /** Update the elite archive. Keeps top archiveRatio of programs. */
  private updateArchive(program: Program): boolean {
    this.archive.add(program.id);

    const targetSize = Math.max(1, Math.floor(this.programs.size * 0.25));
    if (this.archive.size <= targetSize) return true;

    // Trim: remove lowest fitness
    const archivePrograms = [...this.archive]
      .map((id) => ({ id, fitness: this.programs.get(id)?.fitness ?? 0 }))
      .sort((a, b) => b.fitness - a.fitness);

    this.archive = new Set(archivePrograms.slice(0, targetSize).map((p) => p.id));
    return this.archive.has(program.id);
  }

  /** Update best program tracking. */
  private updateBest(program: Program): void {
    if (!this.bestProgramId) {
      this.bestProgramId = program.id;
      return;
    }
    const best = this.programs.get(this.bestProgramId);
    if (!best || (program.fitness ?? 0) > (best.fitness ?? 0)) {
      this.bestProgramId = program.id;
    }
  }

  /** Record a generation snapshot. */
  recordGeneration(index: number, programs: Program[]): Generation {
    const fitnesses = programs.map((p) => p.fitness ?? 0);
    const gen: Generation = {
      id: `gen_${index}_${Date.now()}`,
      index,
      programs: [...programs],
      bestFitness: Math.max(...fitnesses, 0),
      averageFitness: fitnesses.length > 0
        ? fitnesses.reduce((s, f) => s + f, 0) / fitnesses.length
        : 0,
      timestamp: Date.now(),
    };
    this.generations.push(gen);
    return gen;
  }

  // ── Queries ──────────────────────────────────────────────────────────────

  /** Get the best program found so far. */
  getBestProgram(): Program | undefined {
    return this.bestProgramId ? this.programs.get(this.bestProgramId) : undefined;
  }

  /** Get a program by ID. */
  getProgram(id: string): Program | undefined {
    return this.programs.get(id);
  }

  /** Get all programs. */
  getAllPrograms(): Program[] {
    return [...this.programs.values()];
  }

  /** Get programs in a specific generation. */
  getProgramsByGeneration(generation: number): Program[] {
    return [...this.programs.values()].filter((p) => p.generation === generation);
  }

  /** Get top N programs by fitness. */
  getTopPrograms(n: number, islandId?: number): Program[] {
    let candidates = [...this.programs.values()];
    if (islandId !== undefined) {
      const islandIds = this.islands[islandId];
      candidates = candidates.filter((p) => islandIds.has(p.id));
    }
    return candidates
      .sort((a, b) => (b.fitness ?? 0) - (a.fitness ?? 0))
      .slice(0, n);
  }

  /** Get direct children of a program. */
  getChildren(parentId: string): Program[] {
    return this.lineageEdges
      .filter((e) => e.parentId === parentId)
      .map((e) => this.programs.get(e.childId))
      .filter((p): p is Program => p !== undefined);
  }

  /** Get the full ancestor chain from root to program (excluding the program itself). */
  getAncestorChain(programId: string): Program[] {
    const chain: Program[] = [];
    let currentId: string | undefined = programId;
    const visited = new Set<string>();

    while (currentId && !visited.has(currentId)) {
      visited.add(currentId);
      const program = this.programs.get(currentId);
      if (!program) break;

      const parentId = program.parentIds[0];
      if (parentId && this.programs.has(parentId)) {
        chain.unshift(this.programs.get(parentId)!);
        currentId = parentId;
      } else {
        break;
      }
    }

    return chain;
  }

  /** Get generation history. */
  getGenerations(): Generation[] {
    return [...this.generations];
  }

  /** Get the latest generation index. */
  getLatestGeneration(): number {
    if (this.generations.length === 0) return -1;
    return this.generations[this.generations.length - 1].index;
  }

  // ── Sampling ─────────────────────────────────────────────────────────────

  /** Sample a parent program using exploration/exploitation strategy. */
  sampleParent(islandId?: number): Program {
    if (this.programs.size === 0) {
      throw new Error('Cannot sample from empty database');
    }

    const rand = Math.random();
    if (rand < this.config.explorationRatio) {
      return this.sampleRandom(islandId);
    } else if (rand < this.config.explorationRatio + this.config.exploitationRatio) {
      return this.sampleFromArchive();
    } else {
      return this.sampleWeighted(islandId);
    }
  }

  /** Sample random program. */
  private sampleRandom(islandId?: number): Program {
    if (islandId !== undefined && this.islands[islandId].size > 0) {
      const ids = [...this.islands[islandId]];
      const id = ids[Math.floor(Math.random() * ids.length)];
      return this.programs.get(id)!;
    }
    const all = [...this.programs.values()];
    return all[Math.floor(Math.random() * all.length)];
  }

  /** Sample from elite archive with fitness-weighted probability. */
  private sampleFromArchive(): Program {
    if (this.archive.size === 0) return this.sampleRandom();

    const candidates = [...this.archive]
      .map((id) => this.programs.get(id))
      .filter((p): p is Program => p !== undefined);

    if (candidates.length === 0) return this.sampleRandom();

    // Fitness-weighted selection
    const weights = candidates.map((p) => Math.max(p.fitness ?? 0, 0.001));
    return weightedChoice(candidates, weights);
  }

  /** Sample with fitness-weighted probability. */
  private sampleWeighted(islandId?: number): Program {
    let candidates: Program[];
    if (islandId !== undefined && this.islands[islandId].size > 0) {
      candidates = [...this.islands[islandId]]
        .map((id) => this.programs.get(id))
        .filter((p): p is Program => p !== undefined);
    } else {
      candidates = [...this.programs.values()];
    }

    if (candidates.length === 0) return this.sampleRandom();

    const weights = candidates.map((p) => Math.max(p.fitness ?? 0, 0.001));
    return weightedChoice(candidates, weights);
  }

  /** Sample diverse inspiration programs from different islands. */
  sampleInspirations(count: number, excludeIds: Set<string> = new Set()): Program[] {
    const inspirations: Program[] = [];
    const islandOrder = shuffle([...Array(this.config.numIslands).keys()]);

    for (const islandId of islandOrder) {
      if (inspirations.length >= count) break;
      const candidates = [...this.islands[islandId]]
        .filter((id) => !excludeIds.has(id))
        .map((id) => this.programs.get(id))
        .filter((p): p is Program => p !== undefined)
        .sort((a, b) => (b.fitness ?? 0) - (a.fitness ?? 0));

      if (candidates.length > 0) {
        inspirations.push(candidates[0]);
        excludeIds.add(candidates[0].id);
      }
    }

    // Fill remaining from random
    if (inspirations.length < count) {
      const remaining = [...this.programs.values()]
        .filter((p) => !excludeIds.has(p.id))
        .sort(() => Math.random() - 0.5)
        .slice(0, count - inspirations.length);
      inspirations.push(...remaining);
    }

    return inspirations;
  }

  // ── Island Migration ─────────────────────────────────────────────────────

  /** Perform migration between islands (ring topology). */
  migrate(migrationRate?: number): number {
    if (this.config.numIslands < 2) return 0;
    const rate = migrationRate ?? this.config.migrationRate;
    let migrated = 0;

    for (let src = 0; src < this.config.numIslands; src++) {
      const target = (src + 1) % this.config.numIslands;
      const topFromSrc = this.getTopPrograms(
        Math.max(1, Math.floor(this.islands[src].size * rate)),
        src,
      );

      for (const prog of topFromSrc) {
        this.islands[target].add(prog.id);
        migrated++;
      }
    }

    return migrated;
  }

  // ── Statistics ───────────────────────────────────────────────────────────

  /** Get comprehensive database statistics. */
  getStatistics(): DatabaseStats {
    const fitnesses = [...this.programs.values()].map((p) => p.fitness ?? 0);
    const maxGen = this.programs.size > 0
      ? Math.max(...[...this.programs.values()].map((p) => p.generation))
      : 0;

    return {
      totalPrograms: this.programs.size,
      totalAdded: this.totalAdded,
      totalImproved: this.totalImproved,
      archiveSize: this.archive.size,
      numIslands: this.config.numIslands,
      islandSizes: this.islands.map((s) => s.size),
      bestFitness: fitnesses.length > 0 ? Math.max(...fitnesses) : 0,
      avgFitness: fitnesses.length > 0
        ? fitnesses.reduce((s, f) => s + f, 0) / fitnesses.length
        : 0,
      minFitness: fitnesses.length > 0 ? Math.min(...fitnesses) : 0,
      generationCount: maxGen + 1,
    };
  }

  /** Get fitness history over generations. */
  getFitnessHistory(): { generation: number; best: number; average: number; min: number }[] {
    return this.generations.map((g) => {
      const fitnesses = g.programs.map((p) => p.fitness ?? 0);
      return {
        generation: g.index,
        best: Math.max(...fitnesses, 0),
        average: fitnesses.reduce((s, f) => s + f, 0) / Math.max(fitnesses.length, 1),
        min: Math.min(...fitnesses, 0),
      };
    });
  }

  /** Get diversity metric: number of unique content hashes. */
  getDiversity(): number {
    const hashes = new Set<string>();
    for (const p of this.programs.values()) {
      hashes.add(p.code.slice(0, 200)); // Approximate uniqueness
    }
    return hashes.size / Math.max(this.programs.size, 1);
  }

  // ── Persistence ──────────────────────────────────────────────────────────

  /** Save database to a directory as JSON files. */
  save(dirPath: string): void {
    mkdirSync(dirPath, { recursive: true });
    const programsDir = join(dirPath, 'programs');
    mkdirSync(programsDir, { recursive: true });

    // Save metadata
    const metadata = {
      bestProgramId: this.bestProgramId,
      archive: [...this.archive],
      islands: this.islands.map((s) => [...s]),
      totalAdded: this.totalAdded,
      totalImproved: this.totalImproved,
      nextOrder: this.nextOrder,
      fitnessWeights: this.fitnessWeights,
      metricRanges: Object.fromEntries(this.metricRanges),
      config: this.config,
    };
    writeFileSync(join(dirPath, 'metadata.json'), JSON.stringify(metadata, null, 2));

    // Save programs
    for (const [id, program] of this.programs) {
      writeFileSync(
        join(programsDir, `${id}.json`),
        JSON.stringify(serializeProgram(program), null, 2),
      );
    }

    // Save generations
    const genData = this.generations.map((g) => ({
      ...g,
      programs: g.programs.map((p) => p.id), // Store only IDs
    }));
    writeFileSync(join(dirPath, 'generations.json'), JSON.stringify(genData, null, 2));

    // Save lineage
    writeFileSync(join(dirPath, 'lineage.json'), JSON.stringify(this.lineageEdges, null, 2));
  }

  /** Load database from a directory. */
  static load(dirPath: string, config?: EvolutionConfig): EvolutionDatabase {
    const metadataPath = join(dirPath, 'metadata.json');
    if (!existsSync(metadataPath)) {
      throw new Error(`Database not found at ${dirPath}`);
    }

    const metadata = JSON.parse(readFileSync(metadataPath, 'utf-8'));
    const dbConfig = config ?? metadata.config ?? DEFAULT_CONFIG;
    const db = new EvolutionDatabase(dbConfig);

    db.bestProgramId = metadata.bestProgramId;
    db.archive = new Set(metadata.archive ?? []);
    db.islands = (metadata.islands ?? []).map((ids: string[]) => new Set(ids));
    db.totalAdded = metadata.totalAdded ?? 0;
    db.totalImproved = metadata.totalImproved ?? 0;
    db.nextOrder = metadata.nextOrder ?? 0;
    db.fitnessWeights = metadata.fitnessWeights ?? {};
    db.metricRanges = new Map(Object.entries(metadata.metricRanges ?? {}));

    // Ensure correct island count
    while (db.islands.length < dbConfig.numIslands) {
      db.islands.push(new Set());
    }

    // Load programs
    const programsDir = join(dirPath, 'programs');
    if (existsSync(programsDir)) {
      for (const file of readdirSync(programsDir)) {
        if (!file.endsWith('.json')) continue;
        const data = JSON.parse(readFileSync(join(programsDir, file), 'utf-8'));
        const program = deserializeProgram(data);
        db.programs.set(program.id, program);
      }
    }

    // Load lineage
    const lineagePath = join(dirPath, 'lineage.json');
    if (existsSync(lineagePath)) {
      db.lineageEdges = JSON.parse(readFileSync(lineagePath, 'utf-8'));
    }

    // Load generations (reconstruct with program references)
    const genPath = join(dirPath, 'generations.json');
    if (existsSync(genPath)) {
      const genData = JSON.parse(readFileSync(genPath, 'utf-8'));
      db.generations = genData.map((g: Record<string, unknown>) => ({
        id: g.id as string,
        index: g.index as number,
        programs: (g.programs as string[])
          .map((id) => db.programs.get(id))
          .filter((p): p is Program => p !== undefined),
        bestFitness: g.bestFitness as number,
        averageFitness: g.averageFitness as number,
        timestamp: g.timestamp as number,
      }));
    }

    return db;
  }
}

// ── Utility functions ────────────────────────────────────────────────────────

/** Weighted random selection. */
function weightedChoice<T>(items: T[], weights: number[]): T {
  const total = weights.reduce((s, w) => s + w, 0);
  let r = Math.random() * total;
  for (let i = 0; i < items.length; i++) {
    r -= weights[i];
    if (r <= 0) return items[i];
  }
  return items[items.length - 1];
}

/** Fisher-Yates shuffle. */
function shuffle<T>(arr: T[]): T[] {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

/**
 * In-memory implementation of EvolutionDB for testing and prototyping.
 */
export class InMemoryEvolutionDB {
  private store = new Map<string, Generation>();

  async save(gen: Generation): Promise<void> {
    this.store.set(gen.id, gen);
  }

  async load(id: string): Promise<Generation | null> {
    return this.store.get(id) ?? null;
  }

  async listGenerations(): Promise<string[]> {
    return [...this.store.keys()];
  }
}
