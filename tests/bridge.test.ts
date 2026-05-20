import { describe, it, expect } from 'vitest';
import { buildPythonArgv } from '../src/bridge/PythonBridge.js';

describe('PythonBridge.buildPythonArgv', () => {
  it('skips null/undefined/empty values', () => {
    const argv = buildPythonArgv('novaeve_bio', 'select-models', [
      ['--query-profile', 'q.json'],
      ['--output-dir', undefined],
      ['--seed', null],
      ['--device', ''],
      ['--top-k', 3],
    ]);
    expect(argv).toEqual(['-m', 'novaeve_bio', 'select-models', '--query-profile', 'q.json', '--top-k', '3']);
  });

  it('includes boolean flags only when true', () => {
    const argv = buildPythonArgv('novaeve_bio', 'prepare-sources', [
      ['--include-smartseq', true],
      ['--include-seaad-reference', false],
    ]);
    expect(argv).toEqual(['-m', 'novaeve_bio', 'prepare-sources', '--include-smartseq']);
  });

  it('treats string entries as positional args', () => {
    const argv = buildPythonArgv('novaeve_bio', 'foo', ['raw', '--bar']);
    expect(argv).toEqual(['-m', 'novaeve_bio', 'foo', 'raw', '--bar']);
  });
});
