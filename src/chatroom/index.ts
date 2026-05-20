// MedrixAI Chatroom Module
// Full-featured distributed chatroom with NATS-backed messaging,
// agent orchestration, project management, and conversation export.

// Core room abstractions
export * from './Room.js';
export * from './NatsRoom.js';

// Thread model (parent/child, participants, state, hooks)
export * from './Thread.js';

// Streaming pipeline (backpressure, fan-out, replay, adapter)
export * from './Stream.js';

// NATS server lifecycle management
export { NatsManager, type NatsServerInfo, type NatsManagerOptions } from './NatsManager.js';

// Room manager (the main orchestrator)
export {
  RoomManager,
  type RoomManagerOptions,
  type AgentRegistration,
  type ChatInfo,
  type ChatContext,
  type ChatResult,
  type SlashCommandHandler,
  type Permission,
} from './RoomManager.js';

// Special-purpose agents
export {
  SummarizerAgent,
  SuggestionAgent,
  ChatNameAgent,
  RouterAgent,
  NotifierAgent,
  getSummarizer,
  getSuggester,
  getChatNamer,
  getNotifier,
  type SuggestedQuestion,
  type SpecialAgentConfig,
  type Notification,
} from './SpecialAgents.js';

// Project management
export { ProjectManager, type ProjectInfo, type ProjectListEntry } from './Projects.js';

// Export/Import
export {
  exportChatBundle,
  importChatBundle,
  exportChatToMarkdown,
  exportChatToJSON,
  type ExportOptions,
  type ExportResult,
  type ImportResult,
} from './Export.js';
