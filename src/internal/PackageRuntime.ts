import { readdir, stat, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { homedir } from 'node:os';

// ── Types ─────────────────────────────────────────────────────────────────────

export type PackageStatus = 'uninitialized' | 'ready' | 'error' | 'missing';

export interface PackageMethod {
  name: string;
  description: string;
  isAsync: boolean;
}

export interface PackageRecord {
  name: string;
  path: string | null;
  status: PackageStatus;
  description: string;
  methods: PackageMethod[];
  lastLoaded: number | null;
  lastError: string | null;
  origin: 'user' | 'system' | 'store';
  dependencies: string[];
}

export interface PackageManifest {
  name: string;
  version: string;
  description: string;
  entry: string;
  dependencies?: string[];
  agentScope?: string[];
  tools?: string[];
}

// ── PackageManager ────────────────────────────────────────────────────────────

/**
 * Manages installed packages (agents, tools, skills from the store).
 *
 * Features:
 * - Package discovery from ~/.medrix/packages/
 * - Runtime context for package execution
 * - Package isolation (each package gets its own context)
 * - Dependency resolution between packages
 */
export class PackageManager {
  private packagesPath: string;
  private packages: Map<string, PackageRecord> = new Map();
  private loaded = false;

  constructor(packagesPath?: string) {
    this.packagesPath = packagesPath ?? join(homedir(), '.medrix', 'packages');
  }

  /** Discover all packages in the packages directory. */
  async discover(): Promise<string[]> {
    try {
      const entries = await readdir(this.packagesPath, { withFileTypes: true });
      return entries
        .filter((e) => e.isDirectory() && !e.name.startsWith('.'))
        .map((e) => e.name)
        .sort();
    } catch {
      return [];
    }
  }

  /** Load and register all discovered packages. */
  async loadAll(): Promise<PackageRecord[]> {
    const names = await this.discover();
    const records: PackageRecord[] = [];

    for (const name of names) {
      const record = await this.loadPackage(name);
      records.push(record);
    }

    this.loaded = true;
    return records;
  }

  /** Load a single package by name. */
  async loadPackage(name: string): Promise<PackageRecord> {
    const pkgPath = join(this.packagesPath, name);

    try {
      const manifestPath = join(pkgPath, 'package.json');
      const manifestContent = await readFile(manifestPath, 'utf-8');
      const manifest = JSON.parse(manifestContent) as PackageManifest;

      const record: PackageRecord = {
        name: manifest.name || name,
        path: pkgPath,
        status: 'ready',
        description: manifest.description || '',
        methods: await this.discoverMethods(pkgPath, manifest),
        lastLoaded: Date.now(),
        lastError: null,
        origin: 'user',
        dependencies: manifest.dependencies ?? [],
      };

      this.packages.set(name, record);
      return record;
    } catch (err) {
      const record: PackageRecord = {
        name,
        path: pkgPath,
        status: 'error',
        description: '',
        methods: [],
        lastLoaded: null,
        lastError: err instanceof Error ? err.message : String(err),
        origin: 'user',
        dependencies: [],
      };
      this.packages.set(name, record);
      return record;
    }
  }

  /** List all loaded packages with their metadata. */
  listPackages(): PackageRecord[] {
    return [...this.packages.values()];
  }

  /** Get a specific package record. */
  getPackage(name: string): PackageRecord | undefined {
    return this.packages.get(name);
  }

  /** Reload a specific package. */
  async reloadPackage(name: string): Promise<PackageRecord> {
    this.packages.delete(name);
    return this.loadPackage(name);
  }

  /** Resolve dependencies for a package (topological sort). */
  resolveDependencies(name: string): string[] {
    const visited = new Set<string>();
    const sorted: string[] = [];

    const visit = (pkgName: string): void => {
      if (visited.has(pkgName)) return;
      visited.add(pkgName);

      const record = this.packages.get(pkgName);
      if (record) {
        for (const dep of record.dependencies) {
          visit(dep);
        }
      }
      sorted.push(pkgName);
    };

    visit(name);
    return sorted;
  }

  /** Check if all dependencies for a package are satisfied. */
  checkDependencies(name: string): { satisfied: boolean; missing: string[] } {
    const record = this.packages.get(name);
    if (!record) return { satisfied: false, missing: [name] };

    const missing: string[] = [];
    for (const dep of record.dependencies) {
      const depRecord = this.packages.get(dep);
      if (!depRecord || depRecord.status !== 'ready') {
        missing.push(dep);
      }
    }

    return { satisfied: missing.length === 0, missing };
  }

  private async discoverMethods(pkgPath: string, manifest: PackageManifest): Promise<PackageMethod[]> {
    // Read the entry file and extract exported function names
    const entryPath = join(pkgPath, manifest.entry || 'index.js');
    try {
      const content = await readFile(entryPath, 'utf-8');
      const methods: PackageMethod[] = [];

      // Simple regex-based extraction of exported functions
      const exportMatches = content.matchAll(
        /export\s+(?:async\s+)?function\s+(\w+)/g,
      );
      for (const match of exportMatches) {
        methods.push({
          name: match[1],
          description: '',
          isAsync: content.includes(`async function ${match[1]}`),
        });
      }

      // Also check for tools declared in manifest
      if (manifest.tools) {
        for (const toolName of manifest.tools) {
          if (!methods.some((m) => m.name === toolName)) {
            methods.push({ name: toolName, description: '', isAsync: true });
          }
        }
      }

      return methods;
    } catch {
      return [];
    }
  }

  isLoaded(): boolean {
    return this.loaded;
  }
}

// ── PackageRuntime ────────────────────────────────────────────────────────────

/**
 * Runtime context for executing package methods.
 * Provides isolation and dependency injection.
 */
export class PackageRuntime {
  private manager: PackageManager;
  private contexts: Map<string, PackageContext> = new Map();

  constructor(manager: PackageManager) {
    this.manager = manager;
  }

  /** Get or create an execution context for a package. */
  getContext(packageName: string): PackageContext {
    let ctx = this.contexts.get(packageName);
    if (!ctx) {
      const record = this.manager.getPackage(packageName);
      ctx = new PackageContext(packageName, record?.path ?? null);
      this.contexts.set(packageName, ctx);
    }
    return ctx;
  }

  /** List all available packages. */
  listPackages(): PackageRecord[] {
    return this.manager.listPackages();
  }

  /** Describe a package. */
  describe(name: string): PackageRecord | undefined {
    return this.manager.getPackage(name);
  }

  /** Reload packages. */
  async reload(name?: string): Promise<void> {
    if (name) {
      await this.manager.reloadPackage(name);
      this.contexts.delete(name);
    } else {
      await this.manager.loadAll();
      this.contexts.clear();
    }
  }
}

/**
 * Isolated execution context for a single package.
 */
export class PackageContext {
  readonly packageName: string;
  readonly packagePath: string | null;
  private variables: Map<string, unknown> = new Map();

  constructor(packageName: string, packagePath: string | null) {
    this.packageName = packageName;
    this.packagePath = packagePath;
  }

  /** Set a context variable. */
  set(key: string, value: unknown): void {
    this.variables.set(key, value);
  }

  /** Get a context variable. */
  get<T>(key: string): T | undefined {
    return this.variables.get(key) as T | undefined;
  }

  /** Check if a context variable exists. */
  has(key: string): boolean {
    return this.variables.has(key);
  }

  /** Clear all context variables. */
  clear(): void {
    this.variables.clear();
  }
}
