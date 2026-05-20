import { existsSync, readFileSync, mkdirSync, writeFileSync } from 'fs';
import { join, resolve, dirname } from 'path';
import { homedir } from 'os';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export const PROJECT_ROOT = process.cwd();
export const MEDRIX_DIR = process.env.MEDRIX_DIR ?? join(homedir(), '.medrix');
export const CONFIG_FILE = join(MEDRIX_DIR, 'settings.json');
export const CLI_HISTORY_FILE = join(MEDRIX_DIR, 'cli_history');

export const FILE_COMPLETION_IGNORED = new Set([
  '.git', '.svn', '.hg',
  '__pycache__', '.pytest_cache', '.mypy_cache',
  'node_modules',
  'dist', 'build',
  '.medrix', '.endpoint-logs',
]);

function stripJsoncComments(content: string): string {
  content = content.replace(/\/\*[\s\S]*?\*\//g, '');
  const lines: string[] = [];
  for (const line of content.split('\n')) {
    let inString = false;
    let escapeNext = false;
    let result = '';
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (escapeNext) { result += char; escapeNext = false; continue; }
      if (char === '\\') { result += char; escapeNext = true; continue; }
      if (char === '"') { inString = !inString; result += char; continue; }
      if (!inString && i + 1 < line.length && line.slice(i, i + 2) === '//') break;
      result += char;
    }
    lines.push(result);
  }
  return lines.join('\n');
}

function loadJsonc(path: string): Record<string, any> {
  if (!existsSync(path)) return {};
  try {
    const content = readFileSync(path, 'utf-8');
    return JSON.parse(stripJsoncComments(content));
  } catch {
    return {};
  }
}

function deepMerge(base: Record<string, any>, override: Record<string, any>): Record<string, any> {
  const result = { ...base };
  for (const [key, value] of Object.entries(override)) {
    if (key in result && typeof result[key] === 'object' && !Array.isArray(result[key]) && typeof value === 'object' && !Array.isArray(value)) {
      result[key] = deepMerge(result[key], value);
    } else {
      result[key] = value;
    }
  }
  return result;
}

export interface SettingsData {
  model?: string;
  provider?: string;
  apiKeys?: Record<string, string>;
  endpoint?: { workspacePath?: string; port?: number };
  chatroom?: { natsUrl?: string; autoStart?: boolean };
  evolution?: { defaultGenerations?: number; populationSize?: number };
  gateway?: { autoStart?: string[] };
  [key: string]: unknown;
}

export class Settings {
  readonly workDir: string;
  readonly userHome: string;
  readonly projectDir: string;
  private _settings: SettingsData = {};
  private _loaded = false;

  constructor(workDir?: string) {
    this.workDir = workDir ?? PROJECT_ROOT;
    this.userHome = MEDRIX_DIR;
    this.projectDir = join(this.workDir, '.medrix');
  }

  get agentsDir(): string { return join(this.userHome, 'agents'); }
  get teamsDir(): string { return join(this.userHome, 'teams'); }
  get skillsDir(): string { return join(this.userHome, 'skills'); }
  get templatesDir(): string { return join(this.userHome, 'templates'); }
  get memoryDir(): string { return join(this.userHome, 'memory'); }
  get sessionsDir(): string { return join(this.userHome, 'sessions'); }
  get learningDir(): string { return join(this.userHome, 'learning'); }

  private ensureLoaded(): void {
    if (this._loaded) return;
    this.loadEnv();
    const packageDefaults = loadJsonc(join(__dirname, '..', 'factory', 'templates', 'settings.json'));
    const userGlobal = loadJsonc(join(this.userHome, 'settings.json'));
    const projectLocal = loadJsonc(join(this.projectDir, 'settings.json'));
    this._settings = deepMerge(deepMerge(packageDefaults, userGlobal), projectLocal);
    this.applyEnvOverrides();
    this._loaded = true;
  }

  private loadEnv(): void {
    const envPaths = [
      join(this.workDir, '.env'),
      join(this.userHome, '.env'),
      join(homedir(), '.env'),
    ];
    for (const envPath of envPaths) {
      if (existsSync(envPath)) {
        const content = readFileSync(envPath, 'utf-8');
        for (const line of content.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith('#')) continue;
          const eqIdx = trimmed.indexOf('=');
          if (eqIdx < 0) continue;
          const key = trimmed.slice(0, eqIdx).trim();
          let value = trimmed.slice(eqIdx + 1).trim();
          if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
            value = value.slice(1, -1);
          }
          if (!process.env[key]) process.env[key] = value;
        }
        break;
      }
    }
  }

  private applyEnvOverrides(): void {
    if (process.env.MEDRIX_MODEL) this._settings.model = process.env.MEDRIX_MODEL;
    if (process.env.OPENAI_API_KEY) {
      if (!this._settings.apiKeys) this._settings.apiKeys = {};
      this._settings.apiKeys.openai = process.env.OPENAI_API_KEY;
    }
    if (process.env.ANTHROPIC_API_KEY) {
      if (!this._settings.apiKeys) this._settings.apiKeys = {};
      this._settings.apiKeys.anthropic = process.env.ANTHROPIC_API_KEY;
    }
    if (process.env.GOOGLE_API_KEY) {
      if (!this._settings.apiKeys) this._settings.apiKeys = {};
      this._settings.apiKeys.gemini = process.env.GOOGLE_API_KEY;
    }
  }

  get(key: string): unknown {
    this.ensureLoaded();
    const parts = key.split('.');
    let current: any = this._settings;
    for (const part of parts) {
      if (current == null || typeof current !== 'object') return undefined;
      current = current[part];
    }
    return current;
  }

  set(key: string, value: unknown): void {
    this.ensureLoaded();
    const parts = key.split('.');
    let current: any = this._settings;
    for (const part of parts.slice(0, -1)) {
      if (!current[part] || typeof current[part] !== 'object') current[part] = {};
      current = current[part];
    }
    current[parts[parts.length - 1]] = value;
  }

  save(): void {
    this.ensureLoaded();
    const dir = this.userHome;
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'settings.json'), JSON.stringify(this._settings, null, 2), 'utf-8');
  }

  getModel(): string {
    return (this.get('model') as string) ?? process.env.MEDRIX_MODEL ?? 'gpt-4o';
  }

  getApiKey(provider: string): string | undefined {
    return (this.get(`apiKeys.${provider}`) as string) ?? process.env[`${provider.toUpperCase()}_API_KEY`];
  }

  getEndpointWorkspace(): string {
    const configured = this.get('endpoint.workspacePath') as string | undefined;
    if (configured) return resolve(this.workDir, configured);
    return this.workDir;
  }

  getMcpConfig(): Record<string, any> {
    return loadJsonc(join(this.userHome, 'mcp.json'));
  }

  all(): SettingsData {
    this.ensureLoaded();
    return { ...this._settings };
  }
}

let _instance: Settings | null = null;

export function getSettings(workDir?: string): Settings {
  if (!_instance || workDir) _instance = new Settings(workDir);
  return _instance;
}
