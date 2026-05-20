import readline from 'node:readline';
import { promises as fs } from 'node:fs';
import { existsSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  KNOWN_MODELS_BY_PROVIDER,
  getDefaultModel,
} from '../utils/modelDiscovery.js';

/**
 * Interactive setup wizard.
 *
 * Detects which provider keys are already in env / ~/.medrix/.env, prompts
 * for the rest, optionally validates each key with a tiny live API call,
 * lets the user pick a default model, and persists everything to
 * ~/.medrix/.env without clobbering existing entries (asks first).
 */

export type ProviderId = 'openai' | 'anthropic' | 'gemini';

export interface ProviderConfig {
  id: ProviderId;
  envKey: string;
  altEnvKey?: string;
  baseUrlEnvKey: string;
  defaultBaseUrl: string;
  displayName: string;
  /** Hits a low-cost endpoint. Returns true on success. */
  validate: (apiKey: string, baseUrl?: string) => Promise<boolean>;
}

const PROVIDERS: ProviderConfig[] = [
  {
    id: 'openai',
    envKey: 'OPENAI_API_KEY',
    baseUrlEnvKey: 'OPENAI_BASE_URL',
    defaultBaseUrl: 'https://api.openai.com/v1',
    displayName: 'OpenAI',
    validate: async (apiKey, baseUrl) => {
      const url = `${(baseUrl ?? 'https://api.openai.com/v1').replace(/\/$/, '')}/models`;
      const res = await fetchWithTimeout(url, {
        headers: { Authorization: `Bearer ${apiKey}` },
      }, 8000);
      return res.ok;
    },
  },
  {
    id: 'anthropic',
    envKey: 'ANTHROPIC_API_KEY',
    baseUrlEnvKey: 'ANTHROPIC_BASE_URL',
    defaultBaseUrl: 'https://api.anthropic.com',
    displayName: 'Anthropic',
    validate: async (apiKey, baseUrl) => {
      const url = `${(baseUrl ?? 'https://api.anthropic.com').replace(/\/$/, '')}/v1/models`;
      const res = await fetchWithTimeout(url, {
        headers: {
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
        },
      }, 8000);
      return res.ok;
    },
  },
  {
    id: 'gemini',
    envKey: 'GOOGLE_API_KEY',
    altEnvKey: 'GEMINI_API_KEY',
    baseUrlEnvKey: 'GEMINI_BASE_URL',
    defaultBaseUrl: 'https://generativelanguage.googleapis.com/v1beta',
    displayName: 'Google Gemini',
    validate: async (apiKey, baseUrl) => {
      const url = `${(baseUrl ?? 'https://generativelanguage.googleapis.com/v1beta').replace(/\/$/, '')}/models?key=${encodeURIComponent(apiKey)}`;
      const res = await fetchWithTimeout(url, {}, 8000);
      return res.ok;
    },
  },
];

async function fetchWithTimeout(url: string, init: RequestInit, ms: number): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    return await fetch(url, { ...init, signal: ctrl.signal });
  } finally {
    clearTimeout(timer);
  }
}

export interface SetupWizardOptions {
  /** Override env file location. Default: ~/.medrix/.env */
  envFilePath?: string;
  /** Skip live key validation (useful for tests). Default: false. */
  skipValidation?: boolean;
  /** Don't actually prompt — useful for CI; uses existing env only. Default: false. */
  nonInteractive?: boolean;
}

export interface SetupResult {
  providers: string[];
  defaultModel: string;
  envPath: string;
}

export class SetupWizard {
  private readonly envPath: string;
  private readonly skipValidation: boolean;
  private readonly nonInteractive: boolean;
  private rl: readline.Interface | null = null;

  constructor(opts: SetupWizardOptions = {}) {
    this.envPath = opts.envFilePath ?? path.join(os.homedir(), '.medrix', '.env');
    this.skipValidation = opts.skipValidation ?? false;
    this.nonInteractive = opts.nonInteractive ?? false;
  }

  async run(): Promise<SetupResult> {
    this.println('MedrixAI setup');
    this.println(`Will write to: ${this.envPath}`);
    this.println('');

    const existingEnv = await this.readExistingEnv();
    const configured: ProviderId[] = [];

    for (const provider of PROVIDERS) {
      const result = await this.configureProvider(provider, existingEnv);
      if (result) configured.push(provider.id);
    }

    if (configured.length === 0) {
      this.println('');
      this.println('No providers configured. You can re-run `medrix setup` later.');
    }

    const defaultModel = await this.pickDefaultModel(configured, existingEnv);
    if (defaultModel) {
      existingEnv['MEDRIX_MODEL'] = defaultModel;
    }

    // NATS — keep this tiny prompt for chatroom users.
    if (!existingEnv['NATS_URL']) {
      const nats = await this.ask('NATS_URL [nats://localhost:4222]: ');
      existingEnv['NATS_URL'] = nats || 'nats://localhost:4222';
    }

    await this.writeEnv(existingEnv);
    this.close();

    this.println('');
    this.println(`Wrote ${this.envPath}`);
    return {
      providers: configured.map(String),
      defaultModel: existingEnv['MEDRIX_MODEL'] ?? getDefaultModel(),
      envPath: this.envPath,
    };
  }

  // ---------------------------------------------------------------------
  // Provider configuration
  // ---------------------------------------------------------------------

  private async configureProvider(
    provider: ProviderConfig,
    env: Record<string, string>,
  ): Promise<boolean> {
    const existingKey = env[provider.envKey] || (provider.altEnvKey ? env[provider.altEnvKey] : undefined);

    if (existingKey && !this.nonInteractive) {
      const overwrite = await this.askYesNo(
        `${provider.displayName} key already configured. Re-enter? [y/N]: `,
        false,
      );
      if (!overwrite) {
        await this.maybeValidate(provider, existingKey, env[provider.baseUrlEnvKey]);
        return true;
      }
    } else if (existingKey) {
      // Non-interactive — keep existing.
      return true;
    }

    if (this.nonInteractive) return !!existingKey;

    const want = await this.askYesNo(`Configure ${provider.displayName}? [y/N]: `, false);
    if (!want) return !!existingKey;

    const apiKey = await this.askSecret(`  ${provider.envKey}: `);
    if (!apiKey) {
      this.println('  (skipped — empty value)');
      return !!existingKey;
    }

    const baseUrl = await this.ask(
      `  ${provider.baseUrlEnvKey} [${provider.defaultBaseUrl}]: `,
    );

    env[provider.envKey] = apiKey;
    if (baseUrl && baseUrl !== provider.defaultBaseUrl) {
      env[provider.baseUrlEnvKey] = baseUrl;
    }
    if (provider.altEnvKey && env[provider.altEnvKey]) {
      // Keep alt env aligned for downstream tools that look at it.
      env[provider.altEnvKey] = apiKey;
    }

    await this.maybeValidate(provider, apiKey, baseUrl || provider.defaultBaseUrl);
    return true;
  }

  private async maybeValidate(
    provider: ProviderConfig,
    apiKey: string,
    baseUrl: string | undefined,
  ): Promise<void> {
    if (this.skipValidation) return;
    this.print(`  Validating ${provider.displayName} key...`);
    try {
      const ok = await provider.validate(apiKey, baseUrl);
      this.println(ok ? ' OK' : ' INVALID (key was rejected)');
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      this.println(` could not reach API (${reason})`);
    }
  }

  // ---------------------------------------------------------------------
  // Default model
  // ---------------------------------------------------------------------

  private async pickDefaultModel(
    configured: ProviderId[],
    env: Record<string, string>,
  ): Promise<string | undefined> {
    if (this.nonInteractive) {
      return env['MEDRIX_MODEL'] || getDefaultModel();
    }
    if (configured.length === 0) return env['MEDRIX_MODEL'];

    // Build candidate list from configured providers.
    const candidates: string[] = [];
    for (const id of configured) {
      const models = KNOWN_MODELS_BY_PROVIDER[id] ?? [];
      candidates.push(...models);
    }
    if (candidates.length === 0) return env['MEDRIX_MODEL'];

    this.println('');
    this.println('Available models:');
    candidates.forEach((m, i) => this.println(`  ${i + 1}. ${m}`));
    const fallback = env['MEDRIX_MODEL'] || candidates[0];
    const ans = await this.ask(`Default model [${fallback}]: `);
    if (!ans) return fallback;

    const idx = Number.parseInt(ans, 10);
    if (!Number.isNaN(idx) && idx >= 1 && idx <= candidates.length) {
      return candidates[idx - 1];
    }
    return ans;
  }

  // ---------------------------------------------------------------------
  // Env file I/O
  // ---------------------------------------------------------------------

  private async readExistingEnv(): Promise<Record<string, string>> {
    const merged: Record<string, string> = {};

    // Start from process.env for any keys we care about.
    for (const provider of PROVIDERS) {
      if (process.env[provider.envKey]) merged[provider.envKey] = process.env[provider.envKey] as string;
      if (provider.altEnvKey && process.env[provider.altEnvKey]) {
        merged[provider.altEnvKey] = process.env[provider.altEnvKey] as string;
      }
      if (process.env[provider.baseUrlEnvKey]) {
        merged[provider.baseUrlEnvKey] = process.env[provider.baseUrlEnvKey] as string;
      }
    }
    if (process.env.MEDRIX_MODEL) merged['MEDRIX_MODEL'] = process.env.MEDRIX_MODEL;
    if (process.env.NATS_URL) merged['NATS_URL'] = process.env.NATS_URL;

    if (!existsSync(this.envPath)) return merged;
    try {
      const raw = await fs.readFile(this.envPath, 'utf8');
      for (const line of raw.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        const eq = trimmed.indexOf('=');
        if (eq < 0) continue;
        const k = trimmed.slice(0, eq).trim();
        let v = trimmed.slice(eq + 1).trim();
        if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
          v = v.slice(1, -1);
        }
        // File values do not override process.env (env wins).
        if (!merged[k]) merged[k] = v;
      }
    } catch {
      // Ignore read errors; treat as empty.
    }
    return merged;
  }

  private async writeEnv(env: Record<string, string>): Promise<void> {
    await fs.mkdir(path.dirname(this.envPath), { recursive: true });
    const lines: string[] = [];
    for (const [k, v] of Object.entries(env)) {
      if (v === undefined || v === null || v === '') continue;
      const escaped = /\s|"/.test(v) ? `"${v.replace(/"/g, '\\"')}"` : v;
      lines.push(`${k}=${escaped}`);
    }
    const tmp = `${this.envPath}.tmp`;
    await fs.writeFile(tmp, lines.join('\n') + '\n', { mode: 0o600 });
    await fs.rename(tmp, this.envPath);
  }

  // ---------------------------------------------------------------------
  // Prompt helpers
  // ---------------------------------------------------------------------

  private getRl(): readline.Interface {
    if (!this.rl) {
      this.rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    }
    return this.rl;
  }

  private close(): void {
    if (this.rl) {
      this.rl.close();
      this.rl = null;
    }
  }

  private ask(question: string): Promise<string> {
    if (this.nonInteractive) return Promise.resolve('');
    return new Promise((resolve) => this.getRl().question(question, (a) => resolve(a.trim())));
  }

  /** Like ask() but does not echo the user's input to the terminal. */
  private askSecret(question: string): Promise<string> {
    if (this.nonInteractive) return Promise.resolve('');
    return new Promise((resolve) => {
      const rl = this.getRl();
      const stdout = process.stdout as NodeJS.WriteStream & { _writeToOutput?: (s: string) => void };
      let muted = false;
      const original = stdout._writeToOutput;
      // Patch readline's internal echo so passwords don't print.
      stdout._writeToOutput = function (this: unknown, s: string) {
        if (muted) {
          // Allow newline through.
          if (s.includes('\n')) {
            (original as (s: string) => void).call(this, '\n');
          }
          return;
        }
        (original as (s: string) => void).call(this, s);
      };
      rl.question(question, (answer) => {
        muted = false;
        stdout._writeToOutput = original;
        process.stdout.write('\n');
        resolve(answer.trim());
      });
      muted = true;
    });
  }

  private async askYesNo(question: string, defaultYes: boolean): Promise<boolean> {
    const ans = (await this.ask(question)).toLowerCase();
    if (!ans) return defaultYes;
    return ans === 'y' || ans === 'yes';
  }

  private println(msg: string): void {
    process.stdout.write(msg + '\n');
  }

  private print(msg: string): void {
    process.stdout.write(msg);
  }
}
