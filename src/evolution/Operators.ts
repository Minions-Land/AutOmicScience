/**
 * Genetic operators for evolving programs.
 * Provides basic mutation, crossover, and selection operators
 * as building blocks for custom evolution strategies.
 */

import type { Program } from './Program.js';
import { createProgram } from './Program.js';

/**
 * Apply a random character mutation to a program.
 * Simple baseline operator for testing; prefer LLMMutator for real use.
 */
export function mutate(p: Program): Program {
  const idx = Math.floor(Math.random() * p.code.length);
  const chars = p.code.split('');
  chars[idx] = String.fromCharCode(
    (chars[idx]?.charCodeAt(0) ?? 97) + 1,
  );
  return createProgram(chars.join(''), {
    parentIds: [p.id],
    generation: p.generation + 1,
    islandId: p.islandId,
  });
}

/**
 * Single-point crossover: combine code from two parents at a random split point.
 */
export function crossover(a: Program, b: Program): Program {
  const point = Math.floor(Math.random() * Math.min(a.code.length, b.code.length));
  const childCode = a.code.slice(0, point) + b.code.slice(point);
  return createProgram(childCode, {
    parentIds: [a.id, b.id],
    generation: Math.max(a.generation, b.generation) + 1,
    islandId: a.islandId,
  });
}

/**
 * Two-point crossover: take a segment from parent B and insert into parent A.
 */
export function twoPointCrossover(a: Program, b: Program): Program {
  const len = Math.min(a.code.length, b.code.length);
  let p1 = Math.floor(Math.random() * len);
  let p2 = Math.floor(Math.random() * len);
  if (p1 > p2) [p1, p2] = [p2, p1];

  const childCode = a.code.slice(0, p1) + b.code.slice(p1, p2) + a.code.slice(p2);
  return createProgram(childCode, {
    parentIds: [a.id, b.id],
    generation: Math.max(a.generation, b.generation) + 1,
    islandId: a.islandId,
  });
}

/**
 * Line-level crossover: randomly select lines from each parent.
 * Better suited for code than character-level crossover.
 */
export function lineCrossover(a: Program, b: Program): Program {
  const aLines = a.code.split('\n');
  const bLines = b.code.split('\n');
  const maxLines = Math.max(aLines.length, bLines.length);
  const childLines: string[] = [];

  for (let i = 0; i < maxLines; i++) {
    if (Math.random() < 0.5 && i < aLines.length) {
      childLines.push(aLines[i]);
    } else if (i < bLines.length) {
      childLines.push(bLines[i]);
    } else if (i < aLines.length) {
      childLines.push(aLines[i]);
    }
  }

  return createProgram(childLines.join('\n'), {
    parentIds: [a.id, b.id],
    generation: Math.max(a.generation, b.generation) + 1,
    islandId: a.islandId,
  });
}

/**
 * Tournament selection: pick the best k individuals from a random subset.
 */
export function tournamentSelect(pop: Program[], k: number, tournamentSize = 3): Program[] {
  const selected: Program[] = [];

  for (let i = 0; i < k; i++) {
    // Pick random tournament participants
    const tournament: Program[] = [];
    for (let j = 0; j < tournamentSize; j++) {
      tournament.push(pop[Math.floor(Math.random() * pop.length)]);
    }
    // Select the best from tournament
    tournament.sort((a, b) => (b.fitness ?? 0) - (a.fitness ?? 0));
    selected.push(tournament[0]);
  }

  return selected;
}

/**
 * Elitist selection: return the top k individuals by fitness.
 */
export function select(pop: Program[], k: number): Program[] {
  const sorted = [...pop].sort((a, b) => (b.fitness ?? 0) - (a.fitness ?? 0));
  return sorted.slice(0, k);
}

/**
 * Roulette wheel selection: probability proportional to fitness.
 */
export function rouletteSelect(pop: Program[], k: number): Program[] {
  const fitnesses = pop.map((p) => Math.max(p.fitness ?? 0, 0.001));
  const total = fitnesses.reduce((s, f) => s + f, 0);
  const selected: Program[] = [];

  for (let i = 0; i < k; i++) {
    let r = Math.random() * total;
    for (let j = 0; j < pop.length; j++) {
      r -= fitnesses[j];
      if (r <= 0) {
        selected.push(pop[j]);
        break;
      }
    }
    if (selected.length <= i) {
      selected.push(pop[pop.length - 1]);
    }
  }

  return selected;
}

/**
 * Line-level mutation: randomly modify, insert, or delete a line.
 */
export function lineMutate(p: Program): Program {
  const lines = p.code.split('\n');
  const operation = Math.random();

  if (operation < 0.33 && lines.length > 1) {
    // Delete a random line
    const idx = Math.floor(Math.random() * lines.length);
    lines.splice(idx, 1);
  } else if (operation < 0.66) {
    // Duplicate a random line
    const idx = Math.floor(Math.random() * lines.length);
    lines.splice(idx, 0, lines[idx]);
  } else {
    // Swap two random lines
    const i = Math.floor(Math.random() * lines.length);
    const j = Math.floor(Math.random() * lines.length);
    [lines[i], lines[j]] = [lines[j], lines[i]];
  }

  return createProgram(lines.join('\n'), {
    parentIds: [p.id],
    generation: p.generation + 1,
    islandId: p.islandId,
  });
}
