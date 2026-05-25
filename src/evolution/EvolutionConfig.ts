/**
 * Evolution configuration system.
 * Provides typed config with defaults and validation.
 */

export interface EvolutionConfig {
  // Evolution parameters
  maxIterations: number;
  populationSize: number;
  eliteCount: number;
  earlyStopGenerations: number;

  // Island model
  numIslands: number;
  migrationInterval: number;
  migrationRate: number;

  // Mutation parameters
  mutationRate: number;
  crossoverRate: number;
  temperature: number;
  maxRetries: number;
  mutationTimeout: number;
  maxCodeLength: number;

  // Evaluation parameters
  evaluationTimeout: number;
  maxParallelEvaluations: number;

  // Sampling parameters
  explorationRatio: number;
  exploitationRatio: number;
  numInspirations: number;
  numTopPrograms: number;

  // LLM configuration
  mutatorModel: string;
  feedbackModel: string;

  // Persistence
  dbPath?: string;
  saveAllPrograms: boolean;

  // Logging
  logIterations: boolean;
  logImprovements: boolean;
}

export const DEFAULT_CONFIG: EvolutionConfig = {
  maxIterations: 100,
  populationSize: 50,
  eliteCount: 5,
  earlyStopGenerations: 20,

  numIslands: 3,
  migrationInterval: 20,
  migrationRate: 0.1,

  mutationRate: 0.8,
  crossoverRate: 0.3,
  temperature: 0.7,
  maxRetries: 3,
  mutationTimeout: 120,
  maxCodeLength: 50000,

  evaluationTimeout: 60,
  maxParallelEvaluations: 4,

  explorationRatio: 0.2,
  exploitationRatio: 0.7,
  numInspirations: 2,
  numTopPrograms: 3,

  mutatorModel: 'gpt-5.5',
  feedbackModel: 'gpt-5.4',

  dbPath: undefined,
  saveAllPrograms: true,

  logIterations: true,
  logImprovements: true,
};

export function createConfig(overrides: Partial<EvolutionConfig> = {}): EvolutionConfig {
  return { ...DEFAULT_CONFIG, ...overrides };
}

export function validateConfig(config: EvolutionConfig): string[] {
  const warnings: string[] = [];

  const totalRatio = config.explorationRatio + config.exploitationRatio;
  if (totalRatio > 1.0) {
    warnings.push(
      `explorationRatio + exploitationRatio = ${totalRatio} > 1.0`,
    );
  }

  if (config.eliteCount >= config.populationSize) {
    warnings.push('eliteCount >= populationSize, no offspring will be generated');
  }

  if (config.migrationInterval > config.maxIterations) {
    warnings.push('migrationInterval > maxIterations, no migrations will occur');
  }

  if (config.numIslands < 1) {
    warnings.push('numIslands must be at least 1');
  }

  return warnings;
}

/** Preset: fast iteration for prototyping. */
export function getFastConfig(): EvolutionConfig {
  return createConfig({
    maxIterations: 20,
    populationSize: 20,
    eliteCount: 3,
    numIslands: 1,
    evaluationTimeout: 30,
    numInspirations: 1,
    numTopPrograms: 2,
  });
}

/** Preset: thorough exploration. */
export function getThoroughConfig(): EvolutionConfig {
  return createConfig({
    maxIterations: 500,
    populationSize: 100,
    eliteCount: 10,
    numIslands: 5,
    migrationInterval: 50,
    evaluationTimeout: 300,
    numInspirations: 3,
    numTopPrograms: 5,
  });
}
