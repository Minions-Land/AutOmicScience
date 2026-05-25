import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';
import { Evolver, SandboxEvaluator, createProgram, type Program } from '../evolution/index.js';

const EVOLUTION_CAPABILITIES = [
  'LLM-driven mutation through EvolutionEngine and LLMMutator',
  'Island-model populations with migration and lineage tracking',
  'Sandbox evaluation for Node/Python/Deno programs',
  'Legacy genetic operators, crossover, selection, and visualization data',
  'Persistent EvolutionDatabase for longer optimization runs',
];

export function evolutionToolSet(): ToolSet {
  return new ToolSet('evolution', [
    defineTool<Record<string, never>, { capabilities: string[]; tools: string[] }>({
      name: 'evolution_capabilities',
      operation: 'read',
      description: 'List AOS genetic evolution capabilities: mutation, crossover, island model, sandbox evaluation, lineage, and persistence.',
      parameters: z.object({}),
      isReadOnly: () => true,
      execute: async () => ({
        capabilities: EVOLUTION_CAPABILITIES,
        tools: ['evolution_capabilities', 'evolution_run_smoke'],
      }),
    }),
    defineTool<
      { generations?: number; populationSize?: number; runtime?: 'node' | 'python' },
      { ok: boolean; runtime: string; generations: number; populationSize: number; bestFitness: number; bestProgram: Pick<Program, 'id' | 'code' | 'generation' | 'metrics' | 'fitness'>; generationSummaries: { index: number; bestFitness: number; averageFitness: number }[] }
    >({
      name: 'evolution_run_smoke',
      operation: 'execute',
      description:
        'Run a small local genetic-evolution smoke test using the original AOS Evolver and sandbox evaluator. This validates evolution plumbing without an LLM mutation run.',
      parameters: z.object({
        generations: z.number().int().min(1).max(5).optional().default(2),
        populationSize: z.number().int().min(2).max(8).optional().default(4),
        runtime: z.enum(['node', 'python']).optional().default('node'),
      }),
      isReadOnly: () => false,
      isDestructive: () => false,
      execute: async ({ generations = 2, populationSize = 4, runtime = 'node' }) => {
        const population = makePopulation(populationSize, runtime);
        const evaluator = new SandboxEvaluator({
          runtime,
          timeout: 5000,
          scoreFn: (stdout, stderr, code) => ({
            execSuccess: stderr.trim().length === 0 ? 1 : 0,
            printsTarget: stdout.includes('AOS_EVOLUTION_TARGET') ? 1 : 0,
            concise: Math.min(1, 160 / Math.max(code.length, 1)),
          }),
        });
        const generationSummaries = [];
        for await (const gen of Evolver.evolve(population, evaluator, {
          generations,
          eliteCount: 1,
          populationSize,
          mutationRate: 0,
        })) {
          generationSummaries.push({
            index: gen.index,
            bestFitness: gen.bestFitness,
            averageFitness: gen.averageFitness,
          });
        }
        const evaluated = await new Evolver({
          population,
          evaluator,
          mutator: { mutate: async (program) => program },
          generations: 0,
          eliteCount: 1,
        }).run();
        const best = evaluated[0];
        return {
          ok: true,
          runtime,
          generations,
          populationSize,
          bestFitness: best.fitness ?? 0,
          bestProgram: {
            id: best.id,
            code: best.code,
            generation: best.generation,
            metrics: best.metrics,
            fitness: best.fitness,
          },
          generationSummaries,
        };
      },
    }),
  ]);
}

function makePopulation(size: number, runtime: 'node' | 'python'): Program[] {
  const goodCode = runtime === 'python'
    ? 'print("AOS_EVOLUTION_TARGET")'
    : 'console.log("AOS_EVOLUTION_TARGET");';
  const okCode = runtime === 'python'
    ? 'print("baseline")'
    : 'console.log("baseline");';
  const silentCode = runtime === 'python' ? 'x = 1 + 1' : 'const x = 1 + 1;';
  const seeds = [goodCode, okCode, silentCode];
  return Array.from({ length: size }, (_, idx) =>
    createProgram(seeds[idx % seeds.length], {
      islandId: idx % 2,
      metadata: { smoke: true, candidate: idx },
    }),
  );
}
