import type { Message } from '../types.js';

// --- Types ---

export interface SuggestedQuestion {
  text: string;
  category: 'clarification' | 'follow_up' | 'deep_dive' | 'related';
}

export interface SpecialAgentConfig {
  /** LLM call function - takes messages and returns a string response. */
  llmCall: (messages: Message[], opts?: { maxTokens?: number; timeout?: number }) => Promise<string>;
}

// --- SummarizerAgent ---

/**
 * Generates concise summaries of conversation context for sub-agent delegation.
 * Extracts key information from message history to provide sub-agents with
 * essential context for task execution.
 */
export class SummarizerAgent {
  private config: SpecialAgentConfig;

  constructor(config: SpecialAgentConfig) {
    this.config = config;
  }

  /**
   * Generate a concise summary of conversation context.
   */
  async generateSummary(messages: Message[], maxTokens = 500): Promise<string> {
    if (messages.length === 0) return '';

    const contextText = this.formatMessagesForSummary(messages);
    if (!contextText) return '';

    const systemMsg: Message = {
      role: 'system',
      content: `You are a context summarizer for agent delegation.

Your task: Extract and summarize the parent agent's conversation history to provide sub-agents with essential context for task execution.

Guidelines:
1. Identify KEY INFORMATION: main goal, decisions made, constraints, current status.
2. Be CONCISE and FOCUSED (1-3 sentences + 2-4 bullet points).
3. Omit implementation details or tool outputs that don't affect task understanding.
4. OUTPUT ONLY the summary text, no preamble.`,
    };

    const userMsg: Message = {
      role: 'user',
      content: `Summarize the following conversation context for a sub-agent delegation.
Focus on information that affects task execution, not implementation details.
Maximum ${maxTokens} tokens.

---
CONTEXT:
${contextText}`,
    };

    try {
      const response = await this.config.llmCall([systemMsg, userMsg], {
        maxTokens,
        timeout: 15000,
      });
      return response.trim();
    } catch {
      return '';
    }
  }

  private formatMessagesForSummary(messages: Message[]): string {
    const parts: string[] = [];
    for (const msg of messages) {
      if (msg.role === 'system' || msg.role === 'tool') continue;
      const content = typeof msg.content === 'string'
        ? msg.content
        : Array.isArray(msg.content)
          ? msg.content
              .filter((b): b is { type: 'text'; text: string } => b.type === 'text')
              .map((b) => b.text)
              .join('\n')
          : '';
      if (content.trim()) {
        parts.push(`${msg.role}: ${content}`);
      }
    }
    return parts.join('\n');
  }
}

// --- SuggestionAgent ---

/**
 * Generates contextual follow-up questions based on conversation history.
 */
export class SuggestionAgent {
  private config: SpecialAgentConfig;

  constructor(config: SpecialAgentConfig) {
    this.config = config;
  }

  /**
   * Generate contextual follow-up questions.
   */
  async generateSuggestions(
    messages: Message[],
    maxSuggestions = 3,
  ): Promise<SuggestedQuestion[]> {
    if (messages.length < 2) return [];

    const context = this.buildConversationContext(messages);
    if (!context) return [];

    const systemMsg: Message = {
      role: 'system',
      content: `You are a suggestion assistant that generates contextual follow-up questions.
Rules:
1. Generate exactly ${maxSuggestions} questions that are contextual and actionable
2. Make questions specific to the conversation topic
3. Focus on clarification, follow-up details, or related exploration
4. Keep questions concise and natural
5. Return only the questions, one per line, without numbering or formatting`,
    };

    const userMsg: Message = {
      role: 'user',
      content: `Based on this conversation, generate ${maxSuggestions} follow-up questions.

Conversation:
${context}

Questions:`,
    };

    try {
      const response = await this.config.llmCall([systemMsg, userMsg], {
        timeout: 30000,
      });
      return this.parseSuggestions(response);
    } catch {
      return [];
    }
  }

  private buildConversationContext(messages: Message[]): string {
    const recent = messages.slice(-6);
    const parts: string[] = [];

    for (const msg of recent) {
      if (msg.role === 'system' || msg.role === 'tool') continue;
      let content = typeof msg.content === 'string'
        ? msg.content
        : Array.isArray(msg.content)
          ? msg.content
              .filter((b): b is { type: 'text'; text: string } => b.type === 'text')
              .map((b) => b.text)
              .join('\n')
          : '';
      if (!content) continue;
      if (content.length > 800) content = content.slice(0, 800) + '...';
      const label = msg.role === 'user' ? 'User' : 'Assistant';
      parts.push(`${label}: ${content}`);
    }

    return parts.join('\n\n');
  }

  private parseSuggestions(response: string): SuggestedQuestion[] {
    if (!response) return [];
    const categories: Array<'clarification' | 'follow_up' | 'deep_dive'> = [
      'clarification', 'follow_up', 'deep_dive',
    ];
    const suggestions: SuggestedQuestion[] = [];

    for (const rawLine of response.trim().split('\n')) {
      let line = rawLine.trim();
      if (!line) continue;
      // Remove numbering prefixes
      if (/^[0-9]+\./.test(line)) line = line.replace(/^[0-9]+\.\s*/, '');
      if (line.startsWith('- ') || line.startsWith('* ')) line = line.slice(2);
      if (line) {
        suggestions.push({
          text: line,
          category: categories[suggestions.length % categories.length],
        });
      }
      if (suggestions.length >= 3) break;
    }

    return suggestions;
  }
}

// --- ChatNameAgent ---

/**
 * Generates or updates chat names based on conversation content.
 */
export class ChatNameAgent {
  private config: SpecialAgentConfig;

  private static DEFAULT_NAMES = new Set([
    'New Chat', 'New Chat in Project', 'Untitled', '',
  ]);

  constructor(config: SpecialAgentConfig) {
    this.config = config;
  }

  /**
   * Generate a chat name from conversation messages.
   * Returns null if no meaningful name can be generated.
   */
  async generateName(messages: Message[]): Promise<string | null> {
    const userMsgs = messages.filter((m) => m.role === 'user');
    if (userMsgs.length === 0) return null;

    const systemMsg: Message = {
      role: 'system',
      content:
        'You are a helpful assistant that generates chat titles. ' +
        'Generate a concise (3-6 words) title based on the user\'s intent. ' +
        'Rules:\n' +
        '- Start with a single relevant emoji WITHOUT a space.\n' +
        '- Be specific to the technical task or topic.\n' +
        '- Avoid generic words like "Chat", "Question", "Help".\n' +
        '- Return ONLY the emoji and title text, no quotes.',
    };

    // Select most informative messages
    const finalMsg = userMsgs[userMsgs.length - 1];
    const otherMsgs = userMsgs.slice(0, -1);
    const sorted = otherMsgs
      .map((m) => ({ msg: m, len: (typeof m.content === 'string' ? m.content : '').length }))
      .sort((a, b) => b.len - a.len)
      .slice(0, 5);

    const selected = [...sorted.map((s) => s.msg), finalMsg];
    let contextStr = '';
    for (const msg of selected) {
      const content = typeof msg.content === 'string' ? msg.content : '';
      contextStr += `- ${content.slice(0, 500)}\n`;
    }

    const userMsg: Message = {
      role: 'user',
      content: `User's Messages:\n${contextStr}\nGenerate a concise title (3-6 words):`,
    };

    try {
      const response = await this.config.llmCall([systemMsg, userMsg], {
        timeout: 10000,
      });
      const name = response.trim();
      if (name && name.length > 3 && name.length < 100) {
        return name;
      }
    } catch {
      // Fall through to fallback
    }

    return this.fallbackName(messages);
  }

  /** Check if a name is a default placeholder. */
  isDefaultName(name: string): boolean {
    if (!name) return true;
    const stripped = name.trim();
    if (ChatNameAgent.DEFAULT_NAMES.has(stripped)) return true;
    if (stripped.startsWith('New Chat (')) return true;
    return false;
  }

  private fallbackName(messages: Message[]): string {
    for (const msg of messages) {
      if (msg.role === 'user') {
        const content = typeof msg.content === 'string' ? msg.content : '';
        if (content) {
          const fallback = content.slice(0, 50).trim();
          return content.length > 50 ? fallback + '...' : fallback;
        }
      }
    }
    const now = new Date();
    return `Chat ${(now.getMonth() + 1).toString().padStart(2, '0')}-${now.getDate().toString().padStart(2, '0')} ${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
  }
}

// --- RouterAgent ---

/**
 * Routes messages to the appropriate agent based on content analysis.
 */
export class RouterAgent {
  private config: SpecialAgentConfig;
  private agentDescriptions: Map<string, string> = new Map();

  constructor(config: SpecialAgentConfig) {
    this.config = config;
  }

  /** Register an agent with its description for routing decisions. */
  registerAgent(name: string, description: string): void {
    this.agentDescriptions.set(name, description);
  }

  /** Remove an agent from routing. */
  unregisterAgent(name: string): void {
    this.agentDescriptions.delete(name);
  }

  /**
   * Determine which agent should handle a message.
   * Returns the agent name or null if no suitable agent found.
   */
  async route(message: string): Promise<string | null> {
    if (this.agentDescriptions.size === 0) return null;

    const agentList = Array.from(this.agentDescriptions.entries())
      .map(([name, desc]) => `- ${name}: ${desc}`)
      .join('\n');

    const systemMsg: Message = {
      role: 'system',
      content: `You are a message router. Given a user message and a list of available agents, determine which agent should handle the message. Return ONLY the agent name, nothing else. If no agent is suitable, return "none".

Available agents:
${agentList}`,
    };

    const userMsg: Message = {
      role: 'user',
      content: message,
    };

    try {
      const response = await this.config.llmCall([systemMsg, userMsg], {
        timeout: 10000,
      });
      const agentName = response.trim().toLowerCase();
      if (agentName === 'none') return null;

      // Find matching agent (case-insensitive)
      for (const name of this.agentDescriptions.keys()) {
        if (name.toLowerCase() === agentName) return name;
      }
      return null;
    } catch {
      return null;
    }
  }
}

// --- NotifierAgent ---

export interface Notification {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  body: string;
  timestamp: number;
  read: boolean;
  chatId?: string;
  agentName?: string;
}

/**
 * Manages notifications for the chatroom system.
 * Handles sending, queuing, and retrieving notifications.
 */
export class NotifierAgent {
  private notifications: Notification[] = [];
  private subscribers: Array<(n: Notification) => void> = [];
  private maxNotifications: number;

  constructor(opts?: { maxNotifications?: number }) {
    this.maxNotifications = opts?.maxNotifications ?? 100;
  }

  /** Send a notification. */
  notify(opts: Omit<Notification, 'id' | 'timestamp' | 'read'>): Notification {
    const notification: Notification = {
      ...opts,
      id: `notif_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      timestamp: Date.now(),
      read: false,
    };

    this.notifications.push(notification);

    // Trim old notifications
    if (this.notifications.length > this.maxNotifications) {
      this.notifications = this.notifications.slice(-this.maxNotifications);
    }

    // Notify subscribers
    for (const sub of this.subscribers) {
      try { sub(notification); } catch { /* ignore */ }
    }

    return notification;
  }

  /** Subscribe to new notifications. Returns unsubscribe function. */
  subscribe(handler: (n: Notification) => void): () => void {
    this.subscribers.push(handler);
    return () => {
      this.subscribers = this.subscribers.filter((s) => s !== handler);
    };
  }

  /** Get all notifications, optionally filtered. */
  getNotifications(opts?: { unreadOnly?: boolean; chatId?: string }): Notification[] {
    let result = this.notifications;
    if (opts?.unreadOnly) {
      result = result.filter((n) => !n.read);
    }
    if (opts?.chatId) {
      result = result.filter((n) => n.chatId === opts.chatId);
    }
    return result.slice().reverse(); // Most recent first
  }

  /** Mark a notification as read. */
  markRead(notificationId: string): boolean {
    const n = this.notifications.find((x) => x.id === notificationId);
    if (!n) return false;
    n.read = true;
    return true;
  }

  /** Mark all notifications as read. */
  markAllRead(): void {
    for (const n of this.notifications) {
      n.read = true;
    }
  }

  /** Get unread count. */
  get unreadCount(): number {
    return this.notifications.filter((n) => !n.read).length;
  }

  /** Clear all notifications. */
  clear(): void {
    this.notifications = [];
  }
}

// --- Singleton Factories ---

let _summarizer: SummarizerAgent | null = null;
let _suggester: SuggestionAgent | null = null;
let _chatNamer: ChatNameAgent | null = null;
let _notifier: NotifierAgent | null = null;

export function getSummarizer(config: SpecialAgentConfig): SummarizerAgent {
  if (!_summarizer) _summarizer = new SummarizerAgent(config);
  return _summarizer;
}

export function getSuggester(config: SpecialAgentConfig): SuggestionAgent {
  if (!_suggester) _suggester = new SuggestionAgent(config);
  return _suggester;
}

export function getChatNamer(config: SpecialAgentConfig): ChatNameAgent {
  if (!_chatNamer) _chatNamer = new ChatNameAgent(config);
  return _chatNamer;
}

export function getNotifier(opts?: { maxNotifications?: number }): NotifierAgent {
  if (!_notifier) _notifier = new NotifierAgent(opts);
  return _notifier;
}
