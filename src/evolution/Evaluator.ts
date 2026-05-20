import type { Program } from './Program.js';

/**
 * Evaluates a program's fitness. Higher scores are better.
 */
export interface Evaluator {
  /** Evaluate a single program and return its fitness score. */
  evaluate(program: Program): Promise<number>;
}
