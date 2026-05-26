import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import type { HookManager } from '../hooks/HookManager.js';

const execFileAsync = promisify(execFile);
const DEFAULT_GITHUB_REPO = 'Minions-Land/AutOmicScience';
const DEFAULT_ISSUE_DIR = path.join(os.homedir(), '.aos', 'gitissues');
const DEFAULT_BODY_LIMIT = 64_000;
const DEFAULT_DEDUPE_WINDOW_MS = 60_000;
const GH_SETUP_HINT = 'Run `gh auth login` and make sure the account can create issues in Minions-Land/AutOmicScience.';

export interface IssueReporterOptions {
  autoSubmit?: boolean;
  backgroundSubmit?: boolean;
  cwd?: string;
  enabled?: boolean;
  ghCommand?: string;
  issueDir?: string;
  maxBodyChars?: number;
  repo?: string;
  silent?: boolean;
}

export interface IssueReportInput {
  autoSubmit?: boolean;
  context?: Record<string, unknown>;
  error: unknown;
  severity?: 'error' | 'fatal' | 'warning';
  source: string;
  title?: string;
}

export interface IssueReportResult {
  githubUrl?: string;
  id: string;
  localPath: string;
  prompt?: string;
  submitted: boolean;
  title: string;
}

export interface LocalIssueRecord {
  createdAt: string;
  path: string;
  title: string;
}

interface NormalizedError {
  message: string;
  name: string;
  stack?: string;
}

interface HookInstallOptions {
  getContext?: () => Record<string, unknown>;
  source?: string;
}

export class IssueReporter {
  private readonly autoSubmit: boolean;
  private readonly backgroundSubmit: boolean;
  private readonly cwd: string;
  private readonly enabled: boolean;
  private readonly ghCommand: string;
  private readonly issueDir: string;
  private readonly maxBodyChars: number;
  private readonly repo: string;
  private readonly silent: boolean;
  private readonly recentReports = new Map<string, number>();

  constructor(options: IssueReporterOptions = {}) {
    this.enabled = options.enabled ?? process.env.AOS_GITHUB_ISSUES !== '0';
    this.autoSubmit = options.autoSubmit ?? this.enabled;
    this.cwd = options.cwd ?? process.cwd();
    this.ghCommand = options.ghCommand ?? process.env.AOS_GH_COMMAND ?? 'gh';
    this.issueDir = options.issueDir ?? process.env.AOS_ISSUE_DIR ?? DEFAULT_ISSUE_DIR;
    this.maxBodyChars = options.maxBodyChars ?? DEFAULT_BODY_LIMIT;
    this.repo = options.repo ?? process.env.AOS_GITHUB_REPO ?? DEFAULT_GITHUB_REPO;
    this.backgroundSubmit = options.backgroundSubmit ?? (this.enabled && process.env.AOS_GITHUB_ISSUES_AUTO_SUBMIT !== '0');
    this.silent = options.silent ?? true;
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  reportInBackground(input: IssueReportInput): void {
    if (!this.enabled) return;
    const error = normalizeError(input.error);
    const key = `${input.source}:${error.name}:${error.message}:${error.stack?.split('\n')[0] ?? ''}`;
    const now = Date.now();
    const last = this.recentReports.get(key) ?? 0;
    if (now - last < DEFAULT_DEDUPE_WINDOW_MS) return;
    this.recentReports.set(key, now);

    void this.report({ ...input, autoSubmit: input.autoSubmit ?? this.backgroundSubmit }).catch((err) => {
      if (!this.silent) {
        process.stderr.write(`[aos issue reporter] ${normalizeError(err).message}\n`);
      }
    });
  }

  async report(input: IssueReportInput): Promise<IssueReportResult> {
    if (!this.enabled) {
      return {
        id: 'AOS-ISSUE-REPORTING-DISABLED',
        localPath: '',
        prompt: 'AutOmicScience issue reporting is disabled.',
        submitted: false,
        title: input.title ?? 'AutOmicScience issue reporting disabled',
      };
    }
    const createdAt = new Date();
    const error = normalizeError(input.error);
    const id = createIssueId(createdAt, input.source, error);
    const title = buildTitle(input, error);
    const body = this.buildBody({ ...input, title }, error, id, createdAt);
    await fs.mkdir(this.issueDir, { recursive: true });

    const fileName = `${fileTimestamp(createdAt)}-${slugify(title)}-${id.slice(-8)}.md`;
    const localPath = path.join(this.issueDir, fileName);
    await fs.writeFile(localPath, body, 'utf-8');

    const shouldSubmit = input.autoSubmit ?? this.autoSubmit;
    if (!shouldSubmit) {
      return {
        id,
        localPath,
        prompt: `GitHub issue submission is disabled. ${GH_SETUP_HINT}`,
        submitted: false,
        title,
      };
    }

    const ghReady = await this.isGitHubCliReady();
    if (!ghReady) {
      return {
        id,
        localPath,
        prompt: `A local gitissue was saved. ${GH_SETUP_HINT}`,
        submitted: false,
        title,
      };
    }

    try {
      const { stdout, stderr } = await execFileAsync(
        this.ghCommand,
        ['issue', 'create', '--repo', this.repo, '--title', title, '--body-file', localPath],
        { cwd: this.cwd, timeout: 30_000, maxBuffer: 1024 * 1024 },
      );
      const githubUrl = findIssueUrl(`${stdout}\n${stderr}`);
      return {
        githubUrl,
        id,
        localPath,
        submitted: Boolean(githubUrl),
        title,
      };
    } catch (err) {
      const reason = normalizeError(err).message;
      return {
        id,
        localPath,
        prompt: `A local gitissue was saved, but GitHub issue creation failed: ${reason}. ${GH_SETUP_HINT}`,
        submitted: false,
        title,
      };
    }
  }

  async listLocalIssues(limit = 30): Promise<LocalIssueRecord[]> {
    try {
      const entries = await fs.readdir(this.issueDir, { withFileTypes: true });
      const records = await Promise.all(
        entries
          .filter((entry) => entry.isFile() && entry.name.endsWith('.md'))
          .map(async (entry) => {
            const fullPath = path.join(this.issueDir, entry.name);
            const [stat, head] = await Promise.all([
              fs.stat(fullPath),
              fs.readFile(fullPath, 'utf-8').then((text) => text.slice(0, 800)).catch(() => ''),
            ]);
            return {
              createdAt: stat.mtime.toISOString(),
              path: fullPath,
              title: extractTitle(head) ?? entry.name,
            };
          }),
      );
      return records
        .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
        .slice(0, limit);
    } catch {
      return [];
    }
  }

  async submitLocalIssue(localPath: string): Promise<IssueReportResult> {
    const body = await fs.readFile(localPath, 'utf-8');
    const title = extractTitle(body.slice(0, 1200)) ?? `AutOmicScience issue: ${path.basename(localPath)}`;
    const id = extractIssueId(body) ?? createHash('sha256').update(localPath).digest('hex').slice(0, 12);
    const ghReady = await this.isGitHubCliReady();
    if (!ghReady) {
      return {
        id,
        localPath,
        prompt: GH_SETUP_HINT,
        submitted: false,
        title,
      };
    }
    try {
      const { stdout, stderr } = await execFileAsync(
        this.ghCommand,
        ['issue', 'create', '--repo', this.repo, '--title', title, '--body-file', localPath],
        { cwd: this.cwd, timeout: 30_000, maxBuffer: 1024 * 1024 },
      );
      const githubUrl = findIssueUrl(`${stdout}\n${stderr}`);
      return {
        githubUrl,
        id,
        localPath,
        submitted: Boolean(githubUrl),
        title,
      };
    } catch (err) {
      return {
        id,
        localPath,
        prompt: `GitHub issue creation failed: ${normalizeError(err).message}. ${GH_SETUP_HINT}`,
        submitted: false,
        title,
      };
    }
  }

  private async isGitHubCliReady(): Promise<boolean> {
    try {
      await execFileAsync(this.ghCommand, ['auth', 'status'], {
        cwd: this.cwd,
        timeout: 10_000,
        maxBuffer: 256 * 1024,
      });
      return true;
    } catch {
      return false;
    }
  }

  private buildBody(
    input: IssueReportInput & { title: string },
    error: NormalizedError,
    id: string,
    createdAt: Date,
  ): string {
    const context = redactObject({
      ...(input.context ?? {}),
      runtime: {
        argv: process.argv,
        cwd: this.cwd,
        node: process.version,
        platform: `${process.platform}/${process.arch}`,
      },
    });
    const sections = [
      `# ${input.title}`,
      '',
      '## Summary',
      '',
      `- ID: ${id}`,
      `- Source: ${input.source}`,
      `- Severity: ${input.severity ?? 'error'}`,
      `- Created: ${createdAt.toISOString()}`,
      `- Repository: ${this.repo}`,
      '',
      '## Error',
      '',
      `- Name: ${redactText(error.name)}`,
      `- Message: ${redactText(error.message)}`,
      '',
      '## Stack',
      '',
      '```text',
      redactText(error.stack ?? '(no stack available)'),
      '```',
      '',
      '## Context',
      '',
      '```json',
      safeJson(context),
      '```',
      '',
      '## GitHub Submission',
      '',
      `AutOmicScience will try to create a GitHub Issue in ${this.repo} when GitHub CLI is configured.`,
      '',
      `If no issue was created automatically, run: \`gh auth login\`, then \`aos issues submit "${path.join(this.issueDir, '<this-file>.md')}"\`.`,
      '',
      '_Generated automatically by AutOmicScience issue reporting._',
      '',
    ];
    return truncateReport(sections.join('\n'), this.maxBodyChars).replace('<this-file>.md', 'this-file.md');
  }
}

export function installIssueReportingHook(
  hooks: HookManager,
  reporter: IssueReporter,
  options: HookInstallOptions = {},
): HookManager {
  hooks.on('agent:error', (payload) => {
    reporter.reportInBackground({
      context: options.getContext?.(),
      error: payload.error,
      source: options.source ?? 'agent',
    });
  });
  return hooks;
}

export function formatIssueReportSummary(result: IssueReportResult): string {
  const parts = [`AutOmicScience saved an issue record: ${result.localPath}`];
  if (result.githubUrl) parts.push(`GitHub Issue: ${result.githubUrl}`);
  if (!result.githubUrl && result.prompt) parts.push(result.prompt);
  return parts.join('\n');
}

function normalizeError(error: unknown): NormalizedError {
  if (error instanceof Error) {
    return {
      message: error.message || String(error),
      name: error.name || 'Error',
      stack: error.stack,
    };
  }
  if (typeof error === 'object' && error !== null) {
    const record = error as Record<string, unknown>;
    if (typeof record.message === 'string' || typeof record.stack === 'string') {
      return {
        message: typeof record.message === 'string' ? record.message : safeJson(redactObject(error)),
        name: typeof record.name === 'string' ? record.name : 'Error',
        stack: typeof record.stack === 'string' ? record.stack : undefined,
      };
    }
    return {
      message: safeJson(redactObject(error)),
      name: 'NonError',
    };
  }
  return {
    message: String(error),
    name: 'Error',
  };
}

function buildTitle(input: IssueReportInput, error: NormalizedError): string {
  const raw = input.title ?? `AutOmicScience ${input.source} error: ${error.message}`;
  const oneLine = redactText(raw).replace(/\s+/g, ' ').trim();
  return oneLine.length > 120 ? `${oneLine.slice(0, 117)}...` : oneLine || 'AutOmicScience error report';
}

function createIssueId(createdAt: Date, source: string, error: NormalizedError): string {
  const hash = createHash('sha256')
    .update(`${source}\n${error.name}\n${error.message}\n${error.stack ?? ''}`)
    .digest('hex')
    .slice(0, 12);
  return `AOS-${createdAt.toISOString().replace(/[-:.TZ]/g, '').slice(0, 14)}-${hash}`;
}

function fileTimestamp(date: Date): string {
  return date.toISOString().replace(/[-:.TZ]/g, '').slice(0, 14);
}

function slugify(value: string): string {
  const slug = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60);
  return slug || 'issue';
}

function extractTitle(markdownHead: string): string | null {
  const match = markdownHead.match(/^#\s+(.+)$/m);
  return match?.[1]?.trim() ?? null;
}

function extractIssueId(markdownHead: string): string | null {
  const match = markdownHead.match(/^- ID:\s+(.+)$/m);
  return match?.[1]?.trim() ?? null;
}

function findIssueUrl(output: string): string | undefined {
  return output.match(/https:\/\/github\.com\/[^\s]+\/issues\/\d+/)?.[0];
}

function safeJson(value: unknown): string {
  const seen = new WeakSet<object>();
  return JSON.stringify(value, (_key, item) => {
    if (typeof item === 'bigint') return item.toString();
    if (typeof item === 'object' && item !== null) {
      if (seen.has(item)) return '[Circular]';
      seen.add(item);
    }
    return item;
  }, 2) ?? 'null';
}

function redactObject(value: unknown): unknown {
  return JSON.parse(safeJson(value), (key, item) => {
    if (isSensitiveKey(key)) return '[REDACTED]';
    return typeof item === 'string' ? redactText(item) : item;
  });
}

function isSensitiveKey(key: string): boolean {
  return /(api[_-]?key|token|secret|password|passwd|authorization|cookie|credential)/i.test(key);
}

function redactText(value: string): string {
  return value
    .replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, '$1[REDACTED]')
    .replace(/\b(hf_[A-Za-z0-9]{8,})\b/g, '[REDACTED_HF_TOKEN]')
    .replace(/\b(sk-[A-Za-z0-9]{8,})\b/g, '[REDACTED_API_KEY]')
    .replace(/([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|AUTHORIZATION)[A-Z0-9_]*\s*[:=]\s*)(["']?)[^"',\s}]+/gi, '$1$2[REDACTED]');
}

function truncateReport(body: string, maxChars: number): string {
  if (body.length <= maxChars) return body;
  return `${body.slice(0, maxChars)}\n\n_Report truncated at ${maxChars} characters._\n`;
}
