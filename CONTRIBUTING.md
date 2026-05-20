# Contributing to MedrixAI

## Setup

```bash
git clone <repo-url> && cd MedrixAI
npm install
cp .env.example .env   # fill in at least one API key
npm test               # verify everything works
```

## Project Structure

```
src/
├── agent/           Core Agent class (streaming, tools, hooks, clone, +think)
├── bridge/          Python subprocess bridge for scientific compute
├── chatroom/        NATS-based distributed chatroom (room manager, threads, export)
├── cli/             CLI entry point + SetupWizard
├── endpoint/        MCP server + HTTP/SSE endpoint + ToolsetProxy + Hub
├── evolution/       Genetic algorithm engine (LLM mutation, sandbox eval, islands)
├── factory/         Template manager (~/.medrix/ agents/teams/skills/prompts)
├── gateway/         Multi-channel gateway (7 adapters) + route registry
├── internal/        Learning system, memory system, background agent, attachment pipeline
├── knowledge/       RAG: InMemoryKB (TF-IDF) + VectorStoreKB (embeddings)
├── mcp/             MCP client (stdio/SSE)
├── memory/          InMemoryMemory + FileMemory (JSONL persistence)
├── provider/        OpenAI, Anthropic, Gemini adapters + ModelSelector + ToolProvider
├── remote/          NATS-based distributed agent execution
├── repl/            Interactive REPL with slash commands
├── schemas/         Zod schemas for annotation pipeline
├── session/         File-based session persistence
├── skill/           Skill loader (markdown frontmatter + TS modules)
├── store/           Package store (auth, client, installer, publisher)
├── task/            Background task manager
├── team/            Team patterns (Sequential, Swarm, Coordinator, MoA, AgentAsTool)
├── toolset/         20+ built-in toolsets
├── ui/              Dev server placeholder
├── utils/           Logger, vision, tokens, model discovery, process, template
├── settings.ts      Multi-layer config
├── types.ts         Shared types
└── index.ts         Barrel export
tests/               vitest tests (flat, *.test.ts)
```

## Coding Rules

### TypeScript

- ESM only (`"type": "module"`). All relative imports end in `.js`.
- Strict mode. No `any` in public interfaces.
- Use `zod` for runtime validation at system boundaries.
- Lazy-import heavy optional deps (`nats`, `openai`, `@anthropic-ai/sdk`, `ws`, `better-sqlite3`, `pg`) so missing packages don't break unrelated paths.
- No comments explaining *what* — only *why* when non-obvious.

### Naming

| Thing | Convention | Example |
|-------|-----------|---------|
| Files | PascalCase for classes, camelCase for functions | `Agent.ts`, `modelDiscovery.ts` |
| Tools | snake_case with domain prefix | `bio_load_dataset`, `shell_execute` |
| Agents | kebab-case | `medrix-selector` |
| Teams | kebab-case | `annotation-pipeline` |
| Env vars | `MEDRIX_*` for framework, provider-specific otherwise | `MEDRIX_MODEL`, `OPENAI_API_KEY` |

### Dependencies

- Pin exact versions for production deps.
- Every new dep must be justified in the PR description.
- Prefer well-known, actively maintained packages.

## Frontend Integration

### SSE Streaming (primary)

```ts
// Server handler
app.post('/api/chat', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  for await (const ev of agent.run(req.body.message)) {
    res.write(`data: ${JSON.stringify(ev)}\n\n`);
  }
  res.end();
});
```

**Event types:**

| `ev.type` | `ev.data` | UI action |
|-----------|-----------|-----------|
| `text` | string chunk | Append to message |
| `tool_call` | `{ id, name, arguments }` | Show activity indicator |
| `tool_result` | `{ tool_call_id, content }` | Show/collapse result |
| `done` | final text | Mark complete |
| `error` | `{ message }` | Show error |

### NATS Chatroom (multi-agent)

```ts
import { NatsRoom } from 'medrix-ai';
const room = new NatsRoom({ identity: 'user-1', roomName: 'session-abc' });
await room.subscribe('chat', (msg) => renderMessage(msg));
await room.publish('chat', { text: userInput });
```

### REST (stateless)

```ts
const text = await agent.runToText(userMessage);
```

## Adding Things

### New toolset

1. Create `src/toolset/MyTools.ts`
2. Export tools via `defineTool<Args, Result>({...})`
3. Add `export * from './MyTools.js'` to `src/toolset/index.ts`
4. Write tests in `tests/my-tools.test.ts`

### New team pattern

1. Create `src/team/MyTeam.ts` extending `Team`
2. Implement `async *run(input: string): AsyncGenerator<AgentEvent>`
3. Export from `src/team/index.ts`

### New gateway channel

1. Create `src/gateway/channels/MyAdapter.ts` implementing `ChannelAdapter`
2. Implement `start(config, bridge)`, `stop()`, `sendReply(route, text)`
3. Add to `src/gateway/channels/index.ts`
4. Register in `GatewayChannelManager` ADAPTER_MODULES map

### New provider

1. Create `src/provider/MyProvider.ts` implementing `LLMProvider`
2. Implement `async *chat(messages, options): AsyncGenerator<ProviderStreamChunk>`
3. Export from `src/provider/index.ts`
4. Add detection logic to `src/utils/modelDiscovery.ts`

## Testing

```bash
npm test              # all tests
npm run test:watch    # watch mode
npm run typecheck     # tsc --noEmit
```

### Patterns

```ts
// Mock provider for tests
function mockProvider(reply: string): LLMProvider {
  return {
    name: 'mock',
    supportsTools: false,
    async *chat() {
      yield { type: 'text', text: reply };
      yield { type: 'done', finishReason: 'stop' };
    },
  };
}

// Tool test
const ts = new ToolSet('t', [defineTool({ ... })]);
const res = await ts.execute('tool_name', args, { agentName: 'test' });
expect(res.content).toBe('expected');
```

### What to test

| Layer | Assert |
|-------|--------|
| Tool | `execute()` returns expected shape; zod rejects bad args |
| Agent | Tool-call loop terminates; events arrive in order |
| Team | Pipeline pipes correctly; coordinator routes |
| Schema | Valid payloads parse; invalid throw |
| Provider | Streaming yields correct chunk sequence |

## PR Conventions

- Title: imperative, under 70 chars. Prefix: `feat(toolset):`, `fix(bridge):`, `docs:`
- Body: what changed, why, what was tested
- One logical change per PR
- All tests + typecheck must pass
- New public API needs at least one test

## Commit Messages

```
<type>(<scope>): <subject>

<body — what and why>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
Scopes: `agent`, `toolset`, `bridge`, `team`, `skill`, `mcp`, `provider`, `cli`, `gateway`, `chatroom`, `evolution`, `store`
