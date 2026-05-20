import type { Tool } from '../toolset/Tool.js';

export interface McpPlugin {
  name: string;
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  getTools(): Promise<Tool[]>;
}
