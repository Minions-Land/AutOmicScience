import { describe, it, expect } from 'vitest';
import { buildScmasArgv } from '../../src/plugins/scmas/tools/PythonBridge.js';

describe('scmas PythonBridge', () => {
  it('skips null/undefined/empty values', () => {
    const argv = buildScmasArgv('select-models', [
      ['--query-profile', 'q.json'],
      ['--output-dir', undefined],
      ['--seed', null],
      ['--device', ''],
      ['--top-k', 3],
    ]);
    expect(argv).toEqual(['-m', 'scmas', 'select-models', '--query-profile', 'q.json', '--top-k', '3']);
  });

  it('includes boolean flags only when true', () => {
    const argv = buildScmasArgv('prepare-sources', [
      ['--include-smartseq', true],
      ['--include-seaad-reference', false],
    ]);
    expect(argv).toEqual(['-m', 'scmas', 'prepare-sources', '--include-smartseq']);
  });

  it('treats string entries as positional args', () => {
    const argv = buildScmasArgv('foo', ['raw', '--bar']);
    expect(argv).toEqual(['-m', 'scmas', 'foo', 'raw', '--bar']);
  });
});
