import { zodToJsonSchema } from 'zod-to-json-schema';
import type { ExecutionContext, OpenAIToolDef, ToolResult } from '../types.js';
import { getToolMetadata, type Tool } from './Tool.js';
import { z } from 'zod';

export class ToolSet {
  private tools = new Map<string, Tool>();
  public readonly name: string;

  constructor(name = 'default', tools: Tool[] = []) {
    this.name = name;
    for (const t of tools) this.register(t);
  }

  register(tool: Tool): this {
    if (this.tools.has(tool.name)) {
      throw new Error(`Tool '${tool.name}' is already registered in toolset '${this.name}'`);
    }
    this.tools.set(tool.name, tool);
    return this;
  }

  unregister(name: string): boolean {
    return this.tools.delete(name);
  }

  has(name: string): boolean {
    return this.tools.has(name);
  }

  get(name: string): Tool | undefined {
    return this.tools.get(name);
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
      this.tools.set(t.name, t);
    }
    return this;
  }

  /** Standard OpenAI tool-definition format, also accepted by most other providers. */
  toOpenAITools(): OpenAIToolDef[] {
    return this.list().map((t) => ({
      type: 'function' as const,
      function: {
        name: t.name,
        description: t.description,
        parameters: zodToJsonSchema(t.parameters, { target: 'openApi3' }) as Record<string, unknown>,
      },
    }));
  }

  async execute(name: string, args: unknown, ctx: ExecutionContext): Promise<ToolResult> {
    const tool = this.tools.get(name);
    if (!tool) {
      return {
        tool_call_id: ctx.metadata?.tool_call_id as string ?? '',
        content: JSON.stringify({ error: `Tool '${name}' not found` }),
      };
    }
    try {
      const parsed = tool.parameters.safeParse(args);
      const arg = parsed.success ? parsed.data : args;
      const result = await tool.execute(arg, ctx);
      const content = typeof result === 'string' ? result : JSON.stringify(result);
      return {
        tool_call_id: (ctx.metadata?.tool_call_id as string) ?? '',
        content,
      };
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
}
