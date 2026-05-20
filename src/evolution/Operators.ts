import type { Program } from './Program.js';

/**
 * Genetic operators for evolving programs.
 */

/**
 * Apply a random mutation to a program, producing a new variant.
 * @param p - The program to mutate.
 * @returns A new program with a mutation applied.
 */
export function mutate(p: Program): Program {
  // Stub: flip a random character in the code
  const idx = Math.floor(Math.random() * p.code.length);
  const chars = p.code.split('');
  chars[idx] = String.fromCharCode(
    (chars[idx]?.charCodeAt(0) ?? 97) + 1,
  );
  return {
    id: `prog_${Math.random().toString(36).slice(2, 10)}`,
    code: chars.join(''),
    generation: p.generation + 1,
    parentIds: [p.id],
  };
}

/**
 * Combine two parent programs via single-point crossover.
 * @param a - First parent.
 * @param b - Second parent.
 * @returns A child program combining code from both parents.
 */
export function crossover(a: Program, b: Program): Program {
  const point = Math.floor(Math.random() * Math.min(a.code.length, b.code.length));
  const childCode = a.code.slice(0, point) + b.code.slice(point);
  return {
    id: `prog_${Math.random().toString(36).slice(2, 10)}`,
    code: childCode,
    generation: Math.max(a.generation, b.generation) + 1,
    parentIds: [a.id, b.id],
  };
}

/**
 * Tournament selection: pick k individuals from the population,
 * sorted by fitness (descending).
 * @param pop - The population to select from.
 * @param k - Number of individuals to select.
 * @returns The top-k individuals by fitness.
 */
export function select(pop: Program[], k: number): Program[] {
  const sorted = [...pop].sort((a, b) => (b.fitness ?? 0) - (a.fitness ?? 0));
  return sorted.slice(0, k);
}
