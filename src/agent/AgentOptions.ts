import type { Memory } from '../memory/Memory.js';
import type { McpPlugin } from '../mcp/McpPlugin.js';
import type { Skill } from '../skill/Skill.js';
import type { ToolSet } from '../toolset/ToolSet.js';
import type { LLMProvider } from '../provider/Provider.js';

export interface AgentOptions {
  /** Display name for logs/events. */
  name?: string;
  /** Single model id, or fallback chain. May include `+think` suffix. */
  model: string | string[];
  /** Optional explicit provider. If omitted, ModelSelector picks one. */
  provider?: LLMProvider;
  toolset?: ToolSet;
  skills?: Skill[];
  mcpPlugins?: McpPlugin[];
  memory?: Memory;
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
  /** Hard cap on tool-call iterations to prevent runaway loops. */
  maxIterations?: number;
}
