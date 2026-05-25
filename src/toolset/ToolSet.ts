import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { zodToJsonSchema } from 'zod-to-json-schema';
import type { ExecutionContext, OpenAIToolDef, ToolResult } from '../types.js';
import { getToolMetadata, type Tool } from './Tool.js';
import { z } from 'zod';
import { PermissionManager } from '../permissions/index.js';
import type { PermissionDecision, PermissionManagerOptions } from '../permissions/index.js';

export interface ToolSetOptions {
  permissionManager?: PermissionManager;
  permissions?: PermissionManagerOptions;
  resultStorageDir?: string;
  defaultMaxResultSizeChars?: number;
}

export class ToolSet {
  private tools = new Map<string, Tool>();
  private aliases = new Map<string, string>();
  private permissionManager?: PermissionManager;
  private resultStorageDir: string;
  private defaultMaxResultSizeChars: number;
  public readonly name: string;

  constructor(name = 'default', tools: Tool[] = [], opts: ToolSetOptions = {}) {
    this.name = name;
    this.permissionManager = opts.permissionManager ?? (opts.permissions ? new PermissionManager(opts.permissions) : undefined);
    this.resultStorageDir = opts.resultStorageDir ?? path.join(os.tmpdir(), 'aos-tool-results');
    this.defaultMaxResultSizeChars = opts.defaultMaxResultSizeChars ?? 200_000;
    for (const t of tools) this.register(t);
  }

  register(tool: Tool): this {
    if (this.tools.has(tool.name) || this.aliases.has(tool.name)) {
      throw new Error(`Tool '${tool.name}' is already registered in toolset '${this.name}'`);
    }
    this.tools.set(tool.name, tool);
    for (const alias of tool.aliases ?? []) {
      if (this.aliases.has(alias) || this.tools.has(alias)) {
        throw new Error(`Tool alias '${alias}' is already registered in toolset '${this.name}'`);
      }
      this.aliases.set(alias, tool.name);
    }
    return this;
  }

  unregister(name: string): boolean {
    const canonical = this.resolveName(name);
    const tool = this.tools.get(canonical);
    if (!tool) return false;
    this.tools.delete(canonical);
    for (const [alias, target] of this.aliases) {
      if (target === canonical) this.aliases.delete(alias);
    }
    return true;
  }

  has(name: string): boolean {
    return this.tools.has(name) || this.aliases.has(name);
  }

  get(name: string): Tool | undefined {
    return this.tools.get(this.resolveName(name));
  }

  list(): Tool[] {
    return Array.from(this.tools.values());
  }

  size(): number {
    return this.tools.size;
  }

  /** Merge another toolset into this one (later wins on collision). */
  merge(other: ToolSet): this {
    for (const t of other.list()) {
      if (this.tools.has(t.name)) this.unregister(t.name);
      this.register(t);
    }
    return this;
  }

  setPermissionManager(permissionManager?: PermissionManager): this {
    this.permissionManager = permissionManager;
    return this;
  }

  getPermissionManager(): PermissionManager | undefined {
    return this.permissionManager;
  }

  /** Standard OpenAI tool-definition format, also accepted by most other providers. */
  toOpenAITools(): OpenAIToolDef[] {
    return this.list().map((t) => ({
      type: 'function' as const,
      function: {
        name: t.name,
        description: t.description,
        parameters: sanitizeJsonSchema(
          zodToJsonSchema(t.parameters, { target: 'jsonSchema7' }) as Record<string, unknown>,
        ),
      },
    }));
  }

  async execute(name: string, args: unknown, ctx: ExecutionContext): Promise<ToolResult> {
    const tool = this.tools.get(this.resolveName(name));
    if (!tool) {
      return {
        tool_call_id: ctx.metadata?.tool_call_id as string ?? '',
        content: JSON.stringify({ error: `Tool '${name}' not found` }),
      };
    }
    try {
      if (tool.isEnabled && !tool.isEnabled(ctx)) {
        return this.errorResult(ctx, `Tool '${tool.name}' is disabled`);
      }

      const parsed = tool.parameters.safeParse(args);
      if (!parsed.success) {
        return this.errorResult(ctx, parsed.error.message, { errorCode: 'invalid_input' });
      }

      const arg = parsed.data;
      const validation = await tool.validateInput?.(arg, ctx);
      if (validation && validation.ok === false) {
        return this.errorResult(ctx, validation.message ?? 'Tool input validation failed', {
          errorCode: validation.errorCode ?? 'validation_failed',
        });
      }

      const permission = await this.checkPermission(tool, arg, ctx);
      if (permission.behavior !== 'allow') {
        return this.errorResult(ctx, permission.reason ?? `Tool '${tool.name}' was not permitted`, {
          permission: permission.behavior,
          rule: permission.rule,
        });
      }

      const effectiveArg = permission.updatedArgs ?? arg;
      const result = await tool.execute(effectiveArg, ctx);
      const content = typeof result === 'string' ? result : JSON.stringify(result);
      const stored = await this.maybeStoreLargeResult(tool, content);
      const output: ToolResult = {
        tool_call_id: (ctx.metadata?.tool_call_id as string) ?? '',
        content: stored.content,
      };
      if (stored.metadata) output.metadata = stored.metadata;
      return output;
    } catch (err) {
      return {
        tool_call_id: (ctx.metadata?.tool_call_id as string) ?? '',
        content: JSON.stringify({ error: (err as Error).message }),
      };
    }
  }

  /**
   * Build a ToolSet from a class instance whose methods were decorated with `@tool()`.
   * Each decorated method becomes a Tool whose `execute` invokes the bound method.
   */
  static fromClass(instance: object, name?: string): ToolSet {
    const ts = new ToolSet(name ?? instance.constructor.name);
    const meta = getToolMetadata(instance);
    if (!meta) return ts;
    for (const [key, info] of meta.entries()) {
      const fn = (instance as Record<string | symbol, unknown>)[key];
      if (typeof fn !== 'function') continue;
      const toolName = info.name ?? String(key);
      const params = info.parameters ?? z.object({}).passthrough();
      ts.register({
        name: toolName,
        description: info.description,
        parameters: params,
        execute: async (args, ctx) => (fn as Function).call(instance, args, ctx),
      });
    }
    return ts;
  }

  private resolveName(name: string): string {
    return this.aliases.get(name) ?? name;
  }

  private async checkPermission(tool: Tool, args: unknown, ctx: ExecutionContext): Promise<PermissionDecision> {
    const toolDecision = await tool.checkPermissions?.(args, ctx);
    if (toolDecision) return toolDecision;
    if (!this.permissionManager) return { behavior: 'allow' };
    return this.permissionManager.check({
      toolName: tool.name,
      args,
      agentName: ctx.agentName,
      operation: tool.operation,
      readOnly: tool.isReadOnly?.(args),
      destructive: tool.isDestructive?.(args),
      command: tool.getCommand?.(args),
      path: tool.getPath?.(args),
      metadata: ctx.metadata,
    });
  }

  private errorResult(ctx: ExecutionContext, message: string, metadata?: Record<string, unknown>): ToolResult {
    return {
      tool_call_id: (ctx.metadata?.tool_call_id as string) ?? '',
      content: JSON.stringify({ error: message }),
      metadata,
    };
  }

  private async maybeStoreLargeResult(
    tool: Tool,
    content: string,
  ): Promise<{ content: string; metadata?: Record<string, unknown> }> {
    const max = tool.maxResultSizeChars ?? this.defaultMaxResultSizeChars;
    if (content.length <= max) return { content };

    await fs.mkdir(this.resultStorageDir, { recursive: true });
    const file = path.join(this.resultStorageDir, `${Date.now()}-${tool.name}.txt`);
    await fs.writeFile(file, content, 'utf-8');
    const preview = content.slice(0, max);
    return {
      content: JSON.stringify({
        preview,
        truncated: true,
        fullResultPath: file,
        originalLength: content.length,
      }),
      metadata: { fullResultPath: file, originalLength: content.length, truncated: true },
    };
  }
}

function sanitizeJsonSchema(schema: Record<string, unknown>): Record<string, unknown> {
  return sanitizeJsonSchemaValue(schema) as Record<string, unknown>;
}

function sanitizeJsonSchemaValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sanitizeJsonSchemaValue);
  if (!value || typeof value !== 'object') return value;

  const clone: Record<string, unknown> = { ...(value as Record<string, unknown>) };
  delete clone['$schema'];

  const type = clone['type'];
  const exclusiveMinimum = clone['exclusiveMinimum'];
  const exclusiveMaximum = clone['exclusiveMaximum'];

  if (typeof exclusiveMinimum === 'number') {
    clone['minimum'] = type === 'integer' && Number.isInteger(exclusiveMinimum)
      ? exclusiveMinimum + 1
      : exclusiveMinimum;
    delete clone['exclusiveMinimum'];
  } else if (exclusiveMinimum === true) {
    const minimum = clone['minimum'];
    if (typeof minimum === 'number' && type === 'integer' && Number.isInteger(minimum)) {
      clone['minimum'] = minimum + 1;
    }
    delete clone['exclusiveMinimum'];
  } else if (exclusiveMinimum === false) {
    delete clone['exclusiveMinimum'];
  }

  if (typeof exclusiveMaximum === 'number') {
    clone['maximum'] = type === 'integer' && Number.isInteger(exclusiveMaximum)
      ? exclusiveMaximum - 1
      : exclusiveMaximum;
    delete clone['exclusiveMaximum'];
  } else if (exclusiveMaximum === true) {
    const maximum = clone['maximum'];
    if (typeof maximum === 'number' && type === 'integer' && Number.isInteger(maximum)) {
      clone['maximum'] = maximum - 1;
    }
    delete clone['exclusiveMaximum'];
  } else if (exclusiveMaximum === false) {
    delete clone['exclusiveMaximum'];
  }

  for (const [key, item] of Object.entries(clone)) {
    clone[key] = sanitizeJsonSchemaValue(item);
  }
  return clone;
}
