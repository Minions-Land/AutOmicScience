import { z, ZodTypeAny } from 'zod';
import type { ExecutionContext } from '../types.js';
import type { PermissionDecision, PermissionRequest, ToolOperation } from '../permissions/index.js';

export interface ToolValidationResult {
  ok: boolean;
  message?: string;
  errorCode?: string;
}

export interface ToolLifecycle<TArgs = any> {
  aliases?: string[];
  searchHint?: string;
  shouldDefer?: boolean;
  maxResultSizeChars?: number;
  operation?: ToolOperation;
  validateInput?: (args: TArgs, ctx: ExecutionContext) => Promise<ToolValidationResult | void> | ToolValidationResult | void;
  checkPermissions?: (args: TArgs, ctx: ExecutionContext) => Promise<PermissionDecision | void> | PermissionDecision | void;
  isEnabled?: (ctx?: ExecutionContext) => boolean;
  isReadOnly?: (args: TArgs) => boolean;
  isDestructive?: (args: TArgs) => boolean;
  isConcurrencySafe?: (args: TArgs) => boolean;
  getPath?: (args: TArgs) => string | undefined;
  getCommand?: (args: TArgs) => string | undefined;
  userFacingName?: (args: TArgs) => string;
  prompt?: () => string;
}

export interface Tool<TArgs = any, TResult = any> {
  name: string;
  description: string;
  parameters: ZodTypeAny;
  execute(args: TArgs, ctx: ExecutionContext): Promise<TResult>;
  aliases?: string[];
  searchHint?: string;
  shouldDefer?: boolean;
  maxResultSizeChars?: number;
  operation?: ToolOperation;
  validateInput?: (args: TArgs, ctx: ExecutionContext) => Promise<ToolValidationResult | void> | ToolValidationResult | void;
  checkPermissions?: (args: TArgs, ctx: ExecutionContext) => Promise<PermissionDecision | void> | PermissionDecision | void;
  isEnabled?: (ctx?: ExecutionContext) => boolean;
  isReadOnly?: (args: TArgs) => boolean;
  isDestructive?: (args: TArgs) => boolean;
  isConcurrencySafe?: (args: TArgs) => boolean;
  getPath?: (args: TArgs) => string | undefined;
  getCommand?: (args: TArgs) => string | undefined;
  userFacingName?: (args: TArgs) => string;
  prompt?: () => string;
}

/**
 * Metadata attached to class methods decorated with `@tool()`.
 * `ToolSet.fromClass(instance)` reads this metadata to register tools.
 */
export interface ToolMetadata {
  description: string;
  parameters?: ZodTypeAny;
  name?: string;
}

export const TOOL_METADATA = Symbol.for('aos.tool.metadata');

type DecoratedTarget = Record<string | symbol, unknown> & {
  constructor: { name: string };
  [TOOL_METADATA]?: Map<string | symbol, ToolMetadata>;
};

/**
 * `@tool('what it does')` — marks a class method as a tool.
 * Optionally provide a Zod schema for the args:
 *
 *   @tool('add two numbers', z.object({ a: z.number(), b: z.number() }))
 *   add(args: { a: number; b: number }) { return args.a + args.b; }
 */
export function tool(
  description: string,
  parameters?: ZodTypeAny,
  name?: string,
): MethodDecorator {
  return (target, propertyKey) => {
    const t = target as DecoratedTarget;
    let map = (t[TOOL_METADATA] as Map<string | symbol, ToolMetadata> | undefined);
    if (!map) {
      map = new Map();
      Object.defineProperty(t, TOOL_METADATA, {
        value: map,
        enumerable: false,
        configurable: true,
        writable: true,
      });
    }
    map.set(propertyKey, { description, parameters, name });
  };
}

export function getToolMetadata(
  instance: object,
): Map<string | symbol, ToolMetadata> | undefined {
  const proto = Object.getPrototypeOf(instance) as DecoratedTarget | null;
  return proto ? proto[TOOL_METADATA] : undefined;
}

/** Convenience: build a Tool from a plain function. */
export function defineTool<TArgs, TResult>(spec: {
  name: string;
  description: string;
  parameters: ZodTypeAny;
  execute: (args: TArgs, ctx: ExecutionContext) => Promise<TResult> | TResult;
} & ToolLifecycle<TArgs>): Tool<TArgs, TResult> {
  return {
    name: spec.name,
    description: spec.description,
    parameters: spec.parameters,
    execute: async (args, ctx) => spec.execute(args as TArgs, ctx),
    aliases: spec.aliases,
    searchHint: spec.searchHint,
    shouldDefer: spec.shouldDefer,
    maxResultSizeChars: spec.maxResultSizeChars,
    operation: spec.operation,
    validateInput: spec.validateInput,
    checkPermissions: spec.checkPermissions,
    isEnabled: spec.isEnabled,
    isReadOnly: spec.isReadOnly,
    isDestructive: spec.isDestructive,
    isConcurrencySafe: spec.isConcurrencySafe,
    getPath: spec.getPath,
    getCommand: spec.getCommand,
    userFacingName: spec.userFacingName,
    prompt: spec.prompt,
  };
}

export { z };
export type { PermissionDecision, PermissionRequest, ToolOperation };
