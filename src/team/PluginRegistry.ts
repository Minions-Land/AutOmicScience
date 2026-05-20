import type { TeamPlugin } from './TeamPlugin.js';

/**
 * Plugin definition for the registry.
 * Describes how to create and configure a plugin.
 */
export interface PluginDef {
  /** Unique plugin name. */
  name: string;
  /** Configuration key in settings (e.g., "logging", "rate_limit"). */
  configKey: string;
  /** Key within config to check for enabled state. */
  enabledKey: string;
  /** Factory function that creates the plugin from config. */
  factory: (config: Record<string, unknown>) => TeamPlugin | null;
  /** Priority for ordering (lower = earlier). Default: 100. */
  priority?: number;
  /** Plugin dependencies (names of plugins that must be loaded first). */
  dependencies?: string[];
}

/**
 * Global registry of team plugins.
 * Manages registration, dependency resolution, and instantiation.
 */
export class PluginRegistry {
  private static instance: PluginRegistry | null = null;
  private definitions: PluginDef[] = [];

  private constructor() {}

  static getInstance(): PluginRegistry {
    if (!PluginRegistry.instance) {
      PluginRegistry.instance = new PluginRegistry();
    }
    return PluginRegistry.instance;
  }

  /** Register a plugin definition. */
  register(def: PluginDef): void {
    // Remove existing definition with same name
    this.definitions = this.definitions.filter((d) => d.name !== def.name);
    this.definitions.push(def);
    this.definitions.sort((a, b) => (a.priority ?? 100) - (b.priority ?? 100));
  }

  /** Unregister a plugin by name. */
  unregister(name: string): boolean {
    const before = this.definitions.length;
    this.definitions = this.definitions.filter((d) => d.name !== name);
    return this.definitions.length < before;
  }

  /** List all registered plugin definitions. */
  list(): PluginDef[] {
    return [...this.definitions];
  }

  /** Get a plugin definition by name. */
  get(name: string): PluginDef | undefined {
    return this.definitions.find((d) => d.name === name);
  }

  /**
   * Create all enabled plugins from a settings object.
   * Resolves dependencies and returns plugins in priority order.
   */
  createPlugins(settings: Record<string, unknown>): TeamPlugin[] {
    const resolved = this.resolveDependencies();
    const plugins: TeamPlugin[] = [];

    for (const def of resolved) {
      try {
        const config = (settings[def.configKey] as Record<string, unknown>) ?? {};
        if (!config[def.enabledKey]) continue;

        const plugin = def.factory(config);
        if (plugin !== null) {
          plugins.push(plugin);
        }
      } catch (err) {
        console.warn(`[PluginRegistry] Failed to create plugin '${def.name}':`, err);
      }
    }

    return plugins;
  }

  /**
   * Resolve plugin dependencies using topological sort.
   * Returns definitions in dependency-safe order.
   */
  private resolveDependencies(): PluginDef[] {
    const nameMap = new Map(this.definitions.map((d) => [d.name, d]));
    const visited = new Set<string>();
    const sorted: PluginDef[] = [];

    const visit = (def: PluginDef): void => {
      if (visited.has(def.name)) return;
      visited.add(def.name);

      for (const depName of def.dependencies ?? []) {
        const dep = nameMap.get(depName);
        if (dep) visit(dep);
      }
      sorted.push(def);
    };

    for (const def of this.definitions) {
      visit(def);
    }

    return sorted;
  }

  /** Reset the registry (useful for testing). */
  reset(): void {
    this.definitions = [];
  }
}
