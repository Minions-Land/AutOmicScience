export type PermissionMode = 'default' | 'plan' | 'auto' | 'bypassPermissions';
export type PermissionBehavior = 'allow' | 'deny' | 'ask';
export type ToolOperation = 'read' | 'write' | 'execute' | 'network' | 'task' | 'unknown';

export interface PermissionRule {
  id?: string;
  effect: PermissionBehavior;
  tool?: string | RegExp;
  operation?: ToolOperation | ToolOperation[];
  commandPrefix?: string | string[];
  pathPrefix?: string | string[];
  readOnly?: boolean;
  destructive?: boolean;
  reason?: string;
}

export interface PermissionRequest {
  toolName: string;
  args: unknown;
  agentName?: string;
  operation?: ToolOperation;
  readOnly?: boolean;
  destructive?: boolean;
  command?: string;
  path?: string;
  metadata?: Record<string, unknown>;
}

export interface PermissionDecision {
  behavior: PermissionBehavior;
  reason?: string;
  updatedArgs?: unknown;
  rule?: PermissionRule;
}

export interface PermissionManagerOptions {
  mode?: PermissionMode;
  rules?: PermissionRule[];
  askFallback?: 'allow' | 'deny';
}

export class PermissionManager {
  private mode: PermissionMode;
  private rules: PermissionRule[];
  private askFallback: 'allow' | 'deny';

  constructor(opts: PermissionManagerOptions = {}) {
    this.mode = opts.mode ?? 'default';
    this.rules = [...(opts.rules ?? [])];
    this.askFallback = opts.askFallback ?? 'deny';
  }

  getMode(): PermissionMode {
    return this.mode;
  }

  setMode(mode: PermissionMode): void {
    this.mode = mode;
  }

  addRule(rule: PermissionRule): this {
    this.rules.push(rule);
    return this;
  }

  removeRule(id: string): boolean {
    const before = this.rules.length;
    this.rules = this.rules.filter((rule) => rule.id !== id);
    return this.rules.length !== before;
  }

  listRules(): PermissionRule[] {
    return [...this.rules];
  }

  check(request: PermissionRequest): PermissionDecision {
    if (this.mode === 'bypassPermissions') {
      return { behavior: 'allow', reason: 'permission mode bypasses checks' };
    }

    for (const rule of this.rules) {
      if (matchesRule(rule, request)) {
        return {
          behavior: rule.effect,
          reason: rule.reason,
          rule,
        };
      }
    }

    if (this.mode === 'plan') {
      if (request.operation !== 'read' || request.destructive) {
        return {
          behavior: this.askFallback,
          reason: 'plan mode blocks non-read-only tool use without approval',
        };
      }
    }

    if (this.mode === 'auto') {
      if (request.destructive) {
        return {
          behavior: this.askFallback,
          reason: 'auto mode requires approval for destructive tool use',
        };
      }
    }

    return { behavior: 'allow' };
  }
}

export function parsePermissionRule(input: string): PermissionRule {
  const trimmed = input.trim();
  const match = /^(allow|deny|ask)\s+(.+)$/i.exec(trimmed);
  if (!match) {
    throw new Error(`Invalid permission rule: ${input}`);
  }

  const effect = match[1].toLowerCase() as PermissionBehavior;
  const target = match[2].trim();
  const rule: PermissionRule = { effect };

  if (target === 'destructive') {
    rule.destructive = true;
    return rule;
  }
  if (target === 'readonly' || target === 'read-only') {
    rule.readOnly = true;
    return rule;
  }

  const idx = target.indexOf(':');
  if (idx < 0) {
    rule.tool = target;
    return rule;
  }

  const kind = target.slice(0, idx).trim();
  const value = target.slice(idx + 1).trim();
  if (kind === 'tool') rule.tool = value;
  else if (kind === 'shell' || kind === 'command') rule.commandPrefix = value;
  else if (kind === 'path') rule.pathPrefix = value;
  else if (kind === 'op' || kind === 'operation') rule.operation = value as ToolOperation;
  else rule.tool = target;
  return rule;
}

function matchesRule(rule: PermissionRule, request: PermissionRequest): boolean {
  if (rule.tool !== undefined && !matchesTool(rule.tool, request.toolName)) return false;
  if (rule.operation !== undefined && !matchesOperation(rule.operation, request.operation)) return false;
  if (rule.commandPrefix !== undefined && !matchesPrefix(rule.commandPrefix, request.command)) return false;
  if (rule.pathPrefix !== undefined && !matchesPrefix(rule.pathPrefix, request.path)) return false;
  if (rule.readOnly !== undefined && rule.readOnly !== Boolean(request.readOnly)) return false;
  if (rule.destructive !== undefined && rule.destructive !== Boolean(request.destructive)) return false;
  return true;
}

function matchesTool(pattern: string | RegExp, toolName: string): boolean {
  return typeof pattern === 'string' ? pattern === toolName : pattern.test(toolName);
}

function matchesOperation(expected: ToolOperation | ToolOperation[], actual?: ToolOperation): boolean {
  const values = Array.isArray(expected) ? expected : [expected];
  return values.includes(actual ?? 'unknown');
}

function matchesPrefix(prefix: string | string[], value?: string): boolean {
  if (!value) return false;
  const prefixes = Array.isArray(prefix) ? prefix : [prefix];
  return prefixes.some((p) => value.startsWith(p));
}
