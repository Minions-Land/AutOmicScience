/**
 * Evaluation system for evolved programs.
 * Supports sandbox execution, timeout handling, multi-metric evaluation,
 * and test case runners.
 */

import { spawn } from 'child_process';
import { writeFileSync, mkdirSync, rmSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { randomBytes } from 'crypto';
import type { Program } from './Program.js';

/** Evaluates a program's fitness. Higher scores are better. */
export interface Evaluator {
  evaluate(program: Program): Promise<EvaluationResult>;
}

/** Result of evaluating a program. */
export interface EvaluationResult {
  success: boolean;
  fitness: number;
  metrics: Record<string, number>;
  stdout?: string;
  stderr?: string;
  error?: string;
  executionTime: number;
}

/** A test case: input and expected output. */
export interface TestCase {
  input: string;
  expectedOutput: string;
  weight?: number;
  name?: string;
}

/**
 * Sandbox evaluator that spawns a child process to run code safely.
 * Supports timeout, multi-metric scoring, and test case validation.
 */
export class SandboxEvaluator implements Evaluator {
  private workspaceBase: string;
  private counter = 0;

  constructor(
    private options: {
      /** Timeout in milliseconds. Default: 30000 (30s). */
      timeout?: number;
      /** Test cases to validate against. */
      testCases?: TestCase[];
      /** Custom scoring function (receives stdout, returns metrics). */
      scoreFn?: (stdout: string, stderr: string, code: string) => Record<string, number>;
      /** Language/runtime to use. Default: 'node'. */
      runtime?: 'node' | 'python' | 'deno';
      /** Working directory base. */
      workspaceBase?: string;
    } = {},
  ) {
    this.workspaceBase = options.workspaceBase ?? join(tmpdir(), 'aos-eval');
    mkdirSync(this.workspaceBase, { recursive: true });
  }

  async evaluate(program: Program): Promise<EvaluationResult> {
    const startTime = Date.now();
    const workspace = this.createWorkspace();

    try {
      // Write program code to workspace
      const ext = this.getExtension();
      const filePath = join(workspace, `program${ext}`);
      writeFileSync(filePath, program.code, 'utf-8');

      // Run the program
      const { stdout, stderr, exitCode } = await this.runInSandbox(filePath);
      const executionTime = Date.now() - startTime;

      // Compute metrics
      let metrics: Record<string, number>;

      if (this.options.scoreFn) {
        // Custom scoring
        metrics = this.options.scoreFn(stdout, stderr, program.code);
      } else if (this.options.testCases && this.options.testCases.length > 0) {
        // Test case evaluation
        metrics = await this.runTestCases(program.code, workspace);
      } else {
        // Default: score based on successful execution and output
        metrics = this.defaultScoring(stdout, stderr, exitCode, program.code);
      }

      // Aggregate fitness from metrics
      const fitness = this.aggregateFitness(metrics);

      return {
        success: exitCode === 0,
        fitness,
        metrics,
        stdout: stdout.slice(0, 5000),
        stderr: stderr.slice(0, 2000),
        executionTime,
      };
    } catch (e) {
      return {
        success: false,
        fitness: 0,
        metrics: { error: 1 },
        error: (e as Error).message,
        executionTime: Date.now() - startTime,
      };
    } finally {
      // Cleanup workspace
      try {
        rmSync(workspace, { recursive: true, force: true });
      } catch {
        // Ignore cleanup errors
      }
    }
  }

  /** Run code in a sandboxed child process with timeout. */
  private runInSandbox(filePath: string): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    const timeout = this.options.timeout ?? 30000;
    const runtime = this.options.runtime ?? 'node';

    const cmd = runtime === 'python' ? 'python3' : runtime === 'deno' ? 'deno' : 'node';
    const args = runtime === 'deno' ? ['run', '--allow-none', filePath] : [filePath];

    return new Promise((resolve, reject) => {
      const child = spawn(cmd, args, {
        timeout,
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, NODE_OPTIONS: '--max-old-space-size=256' },
      });

      let stdout = '';
      let stderr = '';
      let killed = false;

      child.stdout.on('data', (data) => {
        stdout += data.toString();
        // Prevent memory explosion from infinite output
        if (stdout.length > 1_000_000) {
          child.kill('SIGKILL');
          killed = true;
        }
      });

      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      const timer = setTimeout(() => {
        child.kill('SIGKILL');
        killed = true;
      }, timeout);

      child.on('close', (code) => {
        clearTimeout(timer);
        if (killed && stdout.length > 1_000_000) {
          reject(new Error('Output exceeded 1MB limit'));
        } else if (killed) {
          reject(new Error(`Execution timed out after ${timeout}ms`));
        } else {
          resolve({ stdout, stderr, exitCode: code ?? 1 });
        }
      });

      child.on('error', (err) => {
        clearTimeout(timer);
        reject(err);
      });
    });
  }

  /** Run test cases and compute correctness metrics. */
  private async runTestCases(code: string, workspace: string): Promise<Record<string, number>> {
    const testCases = this.options.testCases!;
    let passed = 0;
    let totalWeight = 0;
    let weightedScore = 0;

    for (let i = 0; i < testCases.length; i++) {
      const tc = testCases[i];
      const weight = tc.weight ?? 1;
      totalWeight += weight;

      // Create test wrapper that feeds input and captures output
      const testCode = this.wrapWithTestCase(code, tc.input);
      const testFile = join(workspace, `test_${i}${this.getExtension()}`);
      writeFileSync(testFile, testCode, 'utf-8');

      try {
        const { stdout, exitCode } = await this.runInSandbox(testFile);
        const actual = stdout.trim();
        const expected = tc.expectedOutput.trim();

        if (exitCode === 0 && actual === expected) {
          passed++;
          weightedScore += weight;
        } else if (exitCode === 0) {
          // Partial credit for close answers
          const similarity = computeStringSimilarity(actual, expected);
          weightedScore += weight * similarity * 0.5;
        }
      } catch {
        // Test case failed (timeout, crash)
      }
    }

    return {
      correctness: totalWeight > 0 ? weightedScore / totalWeight : 0,
      passRate: testCases.length > 0 ? passed / testCases.length : 0,
      testsPassed: passed,
      testsTotal: testCases.length,
    };
  }

  /** Default scoring when no custom scorer or test cases provided. */
  private defaultScoring(
    stdout: string,
    stderr: string,
    exitCode: number,
    code: string,
  ): Record<string, number> {
    const metrics: Record<string, number> = {};

    // Execution success
    metrics.execSuccess = exitCode === 0 ? 1 : 0;

    // Output quality (has meaningful output)
    metrics.hasOutput = stdout.trim().length > 0 ? 1 : 0;

    // No errors
    metrics.noErrors = stderr.trim().length === 0 ? 1 : 0;

    // Code quality heuristics
    metrics.codeLength = Math.min(1, 100 / Math.max(code.length, 1)); // Prefer concise
    metrics.hasComments = /\/\/|#|\*\//.test(code) ? 0.5 : 0;

    return metrics;
  }

  /** Aggregate multiple metrics into a single fitness score. */
  private aggregateFitness(metrics: Record<string, number>): number {
    const values = Object.entries(metrics)
      .filter(([key]) => key !== 'error' && key !== 'testsTotal' && key !== 'testsPassed')
      .map(([, v]) => v)
      .filter((v) => isFinite(v));

    if (values.length === 0) return 0;
    return values.reduce((sum, v) => sum + v, 0) / values.length;
  }

  /** Create a unique workspace directory. */
  private createWorkspace(): string {
    const id = `ws_${++this.counter}_${randomBytes(4).toString('hex')}`;
    const dir = join(this.workspaceBase, id);
    mkdirSync(dir, { recursive: true });
    return dir;
  }

  /** Get file extension for the runtime. */
  private getExtension(): string {
    switch (this.options.runtime) {
      case 'python': return '.py';
      case 'deno': return '.ts';
      default: return '.js';
    }
  }

  /** Wrap code with test case input injection. */
  private wrapWithTestCase(code: string, input: string): string {
    const runtime = this.options.runtime ?? 'node';
    if (runtime === 'python') {
      return `import sys\nfrom io import StringIO\nsys.stdin = StringIO(${JSON.stringify(input)})\n${code}`;
    }
    // Node.js: override stdin
    return `process.stdin.isTTY = false;
const __input = ${JSON.stringify(input)};
let __inputIdx = 0;
const __origRead = process.stdin.read;
process.stdin.read = () => __inputIdx === 0 ? (__inputIdx++, __input) : null;
process.stdin.resume = () => {};
${code}`;
  }
}

/**
 * Function-based evaluator: wraps a simple scoring function.
 */
export class FunctionEvaluator implements Evaluator {
  constructor(
    private scoreFn: (program: Program) => Promise<Record<string, number>> | Record<string, number>,
  ) {}

  async evaluate(program: Program): Promise<EvaluationResult> {
    const startTime = Date.now();
    try {
      const metrics = await this.scoreFn(program);
      const values = Object.values(metrics).filter((v) => isFinite(v));
      const fitness = values.length > 0
        ? values.reduce((s, v) => s + v, 0) / values.length
        : 0;

      return {
        success: true,
        fitness,
        metrics,
        executionTime: Date.now() - startTime,
      };
    } catch (e) {
      return {
        success: false,
        fitness: 0,
        metrics: {},
        error: (e as Error).message,
        executionTime: Date.now() - startTime,
      };
    }
  }
}

/** Compute string similarity (Levenshtein-based, 0-1). */
function computeStringSimilarity(a: string, b: string): number {
  if (a === b) return 1;
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 1;

  // Simple character-level comparison for efficiency
  let matches = 0;
  const minLen = Math.min(a.length, b.length);
  for (let i = 0; i < minLen; i++) {
    if (a[i] === b[i]) matches++;
  }
  return matches / maxLen;
}
