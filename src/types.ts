// Shared types used across the framework.

export type Role = 'system' | 'user' | 'assistant' | 'tool';

/** Text content block for multi-modal messages. */
export interface TextContent {
  type: 'text';
  text: string;
}

/** Image content block for multi-modal messages. */
export interface ImageContent {
  type: 'image';
  /** URL or base64-encoded data URI (e.g. "data:image/png;base64,..."). */
  source: string;
  /** Optional media type hint, e.g. "image/png". */
  mediaType?: string;
}

export interface Message {
  role: Role;
  content: string | (TextContent | ImageContent)[];
  name?: string;
  tool_call_id?: string;
  tool_calls?: ToolCall[];
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolResult {
  tool_call_id: string;
  content: string;
}

export interface AgentEvent {
  type: 'text' | 'tool_call' | 'tool_result' | 'done' | 'error' | string;
  data: unknown;
}

export interface ExecutionContext {
  agentName?: string;
  signal?: AbortSignal;
  metadata?: Record<string, unknown>;
}

export interface ChatOptions {
  model: string;
  tools?: OpenAIToolDef[];
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
  extendedThinking?: boolean;
  signal?: AbortSignal;
  responseFormat?: 'text' | 'json_object' | { type: 'json_schema'; schema: Record<string, unknown> };
}

export interface OpenAIToolDef {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export interface ChatResponse {
  content: string;
  toolCalls: ToolCall[];
  finishReason?: string;
}
