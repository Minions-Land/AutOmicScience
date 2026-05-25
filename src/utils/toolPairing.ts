import type { Tool } from '../toolset/Tool.js';

/**
 * Filter tools to only those matching an agent's declared capabilities.
 * Capabilities are matched against tool names and descriptions.
 */
export function pairToolsToAgent(tools: Tool[], agentCapabilities: string[]): Tool[] {
  if (!agentCapabilities.length) return tools;

  const capSet = new Set(agentCapabilities.map((c) => c.toLowerCase()));

  return tools.filter((tool) => {
    const name = tool.name.toLowerCase();
    const desc = tool.description.toLowerCase();
    // A tool matches if any capability keyword appears in its name or description
    return capSet.has(name) || [...capSet].some((cap) => name.includes(cap) || desc.includes(cap));
  });
}

/**
 * Rank tools by relevance to a query string.
 * Uses simple keyword overlap scoring.
 */
export function rankToolsByRelevance(tools: Tool[], query: string): Tool[] {
  const queryWords = new Set(
    query.toLowerCase().split(/\s+/).filter((w) => w.length > 2),
  );

  const scored = tools.map((tool) => {
    const text = `${tool.name} ${tool.description}`.toLowerCase();
    let score = 0;
    for (const word of queryWords) {
      if (text.includes(word)) score++;
    }
    // Bonus for exact name match
    if (queryWords.has(tool.name.toLowerCase())) score += 3;
    return { tool, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.map((s) => s.tool);
}

/**
 * Format tool definitions for a specific provider's API format.
 * Returns the provider-native tool definition array.
 */
export function formatToolsForProvider(tools: Tool[], provider: string): any[] {
  switch (provider.toLowerCase()) {
    case 'openai':
      return tools.map((t) => ({
        type: 'function',
        function: {
          name: t.name,
          description: t.description,
          parameters: zodToJsonSchema(t.parameters),
        },
      }));
    case 'anthropic':
      return tools.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: zodToJsonSchema(t.parameters),
      }));
    case 'gemini':
      return [{
        functionDeclarations: tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: zodToJsonSchema(t.parameters),
        })),
      }];
    default:
      return tools.map((t) => ({
        type: 'function',
        function: {
          name: t.name,
          description: t.description,
          parameters: zodToJsonSchema(t.parameters),
        },
      }));
  }
}

/** Convert a Zod schema to JSON Schema (basic extraction). */
function zodToJsonSchema(schema: any): Record<string, unknown> {
  // If the schema has a _def with typeName, use zod-to-json-schema pattern
  if (schema && typeof schema.parse === 'function') {
    // Try the zod built-in if available
    if (typeof schema._def?.jsonSchema === 'object') {
      return schema._def.jsonSchema;
    }
    // Fallback: use zodToJsonSchema from zod if available
    try {
      const { zodToJsonSchema: convert } = require('zod-to-json-schema');
      return convert(schema) as Record<string, unknown>;
    } catch {
      // Last resort: return a permissive schema
      return { type: 'object', properties: {}, additionalProperties: true };
    }
  }
  // Already a plain object (JSON Schema)
  if (schema && typeof schema === 'object' && !schema.parse) {
    return schema as Record<string, unknown>;
  }
  return { type: 'object', properties: {} };
}
