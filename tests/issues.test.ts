import { describe, expect, it } from 'vitest';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { HookManager } from '../src/hooks/index.js';
import { installIssueReportingHook, IssueReporter } from '../src/issues/index.js';

describe('IssueReporter', () => {
  it('saves local gitissues without requiring gh', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'aos-issues-'));
    const reporter = new IssueReporter({
      autoSubmit: false,
      issueDir: dir,
      cwd: dir,
    });

    const result = await reporter.report({
      context: { token: 'hf_secret_token_should_not_leak' },
      error: new Error('synthetic failure'),
      source: 'test',
    });

    expect(result.submitted).toBe(false);
    expect(result.localPath).toContain(dir);
    const body = await readFile(result.localPath, 'utf-8');
    expect(body).toContain('synthetic failure');
    expect(body).toContain('[REDACTED]');
    expect(body).not.toContain('hf_secret_token_should_not_leak');

    const records = await reporter.listLocalIssues();
    expect(records).toHaveLength(1);
    expect(records[0].title).toContain('AutOmicScience test error');

    await rm(dir, { recursive: true, force: true });
  });

  it('reports hook errors in the background without throwing', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'aos-issues-hook-'));
    const hooks = new HookManager();
    const reporter = new IssueReporter({
      backgroundSubmit: false,
      issueDir: dir,
      cwd: dir,
    });
    installIssueReportingHook(hooks, reporter, { source: 'test-hook' });

    await hooks.emit('agent:error', { error: new Error('background hook failure') });
    await new Promise((resolve) => setTimeout(resolve, 50));

    const records = await reporter.listLocalIssues();
    expect(records.length).toBeGreaterThanOrEqual(1);
    expect(records[0].title).toContain('background hook failure');

    await rm(dir, { recursive: true, force: true });
  });

  it('can be disabled without writing local records', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'aos-issues-disabled-'));
    const reporter = new IssueReporter({
      enabled: false,
      issueDir: dir,
      cwd: dir,
    });

    reporter.reportInBackground({
      error: new Error('should not be saved'),
      source: 'disabled-test',
    });
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(await reporter.listLocalIssues()).toHaveLength(0);
    await rm(dir, { recursive: true, force: true });
  });
});
