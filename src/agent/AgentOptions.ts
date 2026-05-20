import type { Memory } from '../memory/Memory.js';
import type { McpPlugin } from '../mcp/McpPlugin.js';
import type { Skill } from '../skill/Skill.js';
import type { ToolSet } from '../toolset/ToolSet.js';
import type { LLMProvider } from '../provider/Provider.js';

export interface AgentOptions {
  name?: string;
  /** Single model id, or fallback chain. May include `+think[:level]` suffix. */
  model: string | string[];
  provider?: LLMProvider;
  toolset?: ToolSet;
  skills?: Skill[];
  mcpPlugins?: McpPlugin[];
  memory?: Memory;
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
  maxIterations?: number;
  maxHistoryTokens?: number;
  maxRetries?: number;
  /** Response format: 'text' (default), 'json_object', or a JSON schema for structured output. */
  responseFormat?: 'text' | 'json_object' | { type: 'json_schema'; schema: Record<string, unknown> };

  // ── Hooks ──────────────────────────────────────────────────────────────
  onToolCall?: (name: string, args: unknown) => void | Promise<void>;
  onToolResult?: (name: string, result: unknown) => void | Promise<void>;
  onMessage?: (msg: import('../types.js').Message) => void | Promise<void>;
  onBeforeRun?: (input: string | import('../types.js').Message[]) => void | Promise<void>;
  onAfterRun?: (result: string) => void | Promise<void>;
  onError?: (error: Error) => void | Promise<void>;
}
