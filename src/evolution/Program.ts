/**
 * Program model for evolutionary search.
 * Represents a candidate solution with metadata, lineage, and fitness tracking.
 */

import { createHash } from 'crypto';

/** Fitness metrics for a program. */
export interface FitnessMetrics {
  [key: string]: number;
}

/** Mutation metadata describing how a program was created. */
export interface MutationInfo {
  strategy: string;
  summary: string;
  category: string;
  isAlgorithmic: boolean;
  parentFitness?: number;
}

/** A program in the evolutionary search: code + metadata + lineage. */
export interface Program {
  id: string;
  code: string;
  generation: number;
  parentIds: string[];
  fitness?: number;
  metrics: FitnessMetrics;
  metadata: Record<string, unknown>;

  // Lineage
  createdAt: number;
  mutationInfo?: MutationInfo;
  diffFromParent?: string;

  // Island model
  islandId: number;

  // Evaluation artifacts
  llmFeedback?: string;
  evaluationError?: string;
}

/** Create a new program with a unique ID. */
export function createProgram(
  code: string,
  opts: {
    parentIds?: string[];
    generation?: number;
    islandId?: number;
    metrics?: FitnessMetrics;
    mutationInfo?: MutationInfo;
    diffFromParent?: string;
    metadata?: Record<string, unknown>;
  } = {},
): Program {
  return {
    id: generateId(),
    code,
    generation: opts.generation ?? 0,
    parentIds: opts.parentIds ?? [],
    metrics: opts.metrics ?? {},
    metadata: opts.metadata ?? {},
    createdAt: Date.now(),
    islandId: opts.islandId ?? 0,
    mutationInfo: opts.mutationInfo,
    diffFromParent: opts.diffFromParent,
  };
}

/** Generate a unique program ID. */
function generateId(): string {
  const rand = Math.random().toString(36).slice(2, 10);
  const ts = Date.now().toString(36).slice(-4);
  return `prog_${ts}_${rand}`;
}

/** Compute a content hash for deduplication. */
export function contentHash(code: string): string {
  return createHash('sha256').update(code).digest('hex').slice(0, 16);
}

/** Validate program code (basic syntax check). */
export function validateCode(code: string): { valid: boolean; error?: string } {
  try {
    // Use Function constructor for syntax validation without execution
    new Function(code);
    return { valid: true };
  } catch (e) {
    return { valid: false, error: (e as Error).message };
  }
}

/** Compute a unified diff between parent and child code. */
export function computeDiff(parentCode: string, childCode: string): string {
  const parentLines = parentCode.split('\n');
  const childLines = childCode.split('\n');
  const diff: string[] = [];

  // Simple line-by-line diff (Myers-like output)
  let i = 0;
  let j = 0;
  while (i < parentLines.length || j < childLines.length) {
    if (i < parentLines.length && j < childLines.length && parentLines[i] === childLines[j]) {
      diff.push(` ${parentLines[i]}`);
      i++;
      j++;
    } else if (j < childLines.length && (i >= parentLines.length || !parentLines.slice(i).includes(childLines[j]))) {
      diff.push(`+${childLines[j]}`);
      j++;
    } else if (i < parentLines.length) {
      diff.push(`-${parentLines[i]}`);
      i++;
    }
  }

  return diff.join('\n');
}

/** Compute fitness score from metrics using weighted aggregation. */
export function computeFitness(
  metrics: FitnessMetrics,
  weights?: Record<string, number>,
): number {
  if (!weights || Object.keys(weights).length === 0) {
    // Default: average all metrics
    const values = Object.values(metrics).filter((v) => typeof v === 'number' && isFinite(v));
    if (values.length === 0) return 0;
    return values.reduce((sum, v) => sum + v, 0) / values.length;
  }

  let totalWeight = 0;
  let weightedSum = 0;
  for (const [key, weight] of Object.entries(weights)) {
    const value = metrics[key];
    if (value !== undefined && isFinite(value)) {
      weightedSum += weight * value;
      totalWeight += weight;
    }
  }

  return totalWeight > 0 ? weightedSum / totalWeight : 0;
}

/** Serialize a program to a plain object for JSON storage. */
export function serializeProgram(p: Program): Record<string, unknown> {
  return {
    id: p.id,
    code: p.code,
    generation: p.generation,
    parentIds: p.parentIds,
    fitness: p.fitness,
    metrics: p.metrics,
    metadata: p.metadata,
    createdAt: p.createdAt,
    mutationInfo: p.mutationInfo,
    diffFromParent: p.diffFromParent,
    islandId: p.islandId,
    llmFeedback: p.llmFeedback,
    evaluationError: p.evaluationError,
  };
}

/** Deserialize a program from stored JSON. */
export function deserializeProgram(data: Record<string, unknown>): Program {
  return {
    id: data.id as string,
    code: data.code as string,
    generation: (data.generation as number) ?? 0,
    parentIds: (data.parentIds as string[]) ?? [],
    fitness: data.fitness as number | undefined,
    metrics: (data.metrics as FitnessMetrics) ?? {},
    metadata: (data.metadata as Record<string, unknown>) ?? {},
    createdAt: (data.createdAt as number) ?? Date.now(),
    mutationInfo: data.mutationInfo as MutationInfo | undefined,
    diffFromParent: data.diffFromParent as string | undefined,
    islandId: (data.islandId as number) ?? 0,
    llmFeedback: data.llmFeedback as string | undefined,
    evaluationError: data.evaluationError as string | undefined,
  };
}

// Re-export the legacy helper for backward compat
export function newProgram(code: string, parentIds: string[] = [], generation = 0): Program {
  return createProgram(code, { parentIds, generation });
}
