import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve, basename } from 'node:path';
import { homedir } from 'node:os';

// --- Types ---

export interface ProjectInfo {
  path: string;
  name: string;
  createdAt: string;
  lastAccessed: string;
}

export interface ProjectListEntry extends ProjectInfo {
  isActive: boolean;
  exists: boolean;
  hasMedrix: boolean;
}

interface RegistryData {
  active: string | null;
  projects: ProjectInfo[];
}

// --- Helpers ---

function globalMedrixDir(): string {
  return join(homedir(), '.medrix');
}

function registryPath(): string {
  return join(globalMedrixDir(), 'projects.json');
}

// --- ProjectManager ---

/**
 * Manages the global project registry and active project state.
 *
 * A project is a directory containing (or that will contain) a `.medrix/` folder.
 * The global registry lives at `~/.medrix/projects.json`.
 */
export class ProjectManager {
  private registryFile: string;
  private projects: Map<string, ProjectInfo> = new Map();
  private activePath: string | null = null;

  constructor(activePath?: string) {
    this.registryFile = registryPath();
    this.load();

    if (activePath) {
      const resolved = resolve(activePath);
      this.register(resolved);
      this.setActive(resolved);
    }
  }

  // --- Persistence ---

  private load(): void {
    if (!existsSync(this.registryFile)) return;
    try {
      const raw = readFileSync(this.registryFile, 'utf-8');
      const data: RegistryData = JSON.parse(raw);
      this.activePath = data.active;
      for (const entry of data.projects ?? []) {
        this.projects.set(entry.path, entry);
      }
    } catch {
      // Corrupted file - start fresh
    }
  }

  private save(): void {
    const dir = globalMedrixDir();
    mkdirSync(dir, { recursive: true });
    const data: RegistryData = {
      active: this.activePath,
      projects: Array.from(this.projects.values()),
    };
    writeFileSync(this.registryFile, JSON.stringify(data, null, 2), 'utf-8');
  }

  // --- Active Project ---

  get activeProject(): ProjectInfo | null {
    if (this.activePath && this.projects.has(this.activePath)) {
      return this.projects.get(this.activePath)!;
    }
    // Auto-register if path exists but not registered
    if (this.activePath && existsSync(this.activePath)) {
      this.register(this.activePath);
      return this.projects.get(this.activePath) ?? null;
    }
    return null;
  }

  // --- CRUD ---

  listProjects(): ProjectListEntry[] {
    const entries = Array.from(this.projects.values());
    entries.sort((a, b) => b.lastAccessed.localeCompare(a.lastAccessed));
    return entries.map((p) => ({
      ...p,
      isActive: p.path === this.activePath,
      exists: existsSync(p.path),
      hasMedrix: existsSync(join(p.path, '.medrix')),
    }));
  }

  register(path: string, name?: string): ProjectInfo {
    const resolved = resolve(path);
    const existing = this.projects.get(resolved);
    if (existing) {
      if (name) {
        existing.name = name;
        this.save();
      }
      return existing;
    }

    const info: ProjectInfo = {
      path: resolved,
      name: name ?? basename(resolved),
      createdAt: new Date().toISOString(),
      lastAccessed: new Date().toISOString(),
    };
    this.projects.set(resolved, info);
    this.save();
    return info;
  }

  remove(path: string): boolean {
    const resolved = resolve(path);
    if (!this.projects.has(resolved)) return false;
    this.projects.delete(resolved);
    if (this.activePath === resolved) {
      this.activePath = null;
    }
    this.save();
    return true;
  }

  setActive(path: string): ProjectInfo | null {
    const resolved = resolve(path);
    const info = this.projects.get(resolved);
    if (!info) return null;
    this.activePath = resolved;
    info.lastAccessed = new Date().toISOString();
    this.save();
    return info;
  }

  getProject(path: string): ProjectInfo | null {
    return this.projects.get(resolve(path)) ?? null;
  }

  // --- Settings Scope ---

  getConfigScope(projectPath: string): { global: Record<string, unknown>; project: Record<string, unknown> } {
    const globalSettingsPath = join(globalMedrixDir(), 'settings.json');
    const projectSettingsPath = join(projectPath, '.medrix', 'settings.json');

    let globalSettings: Record<string, unknown> = {};
    let projectSettings: Record<string, unknown> = {};

    if (existsSync(globalSettingsPath)) {
      try {
        globalSettings = JSON.parse(readFileSync(globalSettingsPath, 'utf-8'));
      } catch { /* ignore */ }
    }

    if (existsSync(projectSettingsPath)) {
      try {
        projectSettings = JSON.parse(readFileSync(projectSettingsPath, 'utf-8'));
      } catch { /* ignore */ }
    }

    return { global: globalSettings, project: projectSettings };
  }
}
