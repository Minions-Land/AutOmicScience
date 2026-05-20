# MedrixAI Developer Guide

This document is the scaffold for anyone building on, extending, or
integrating with MedrixAI. Read it before writing your first line.

---

## 1. Frontend Integration (start here)

MedrixAI exposes two integration surfaces for frontends:

### 1.1 Streaming Agent Events (primary)

Every `Agent.run(input)` returns an `AsyncGenerator<AgentEvent>`. A
frontend connects by consuming this stream — over WebSocket, SSE, or
in-process.

```ts
// Server-side handler (e.g. Express / Hono / Fastify)
import { Agent } from 'medrix-ai';

app.post('/api/chat', async (req, res) => {
  const agent = getOrCreateAgent(req.session);
  res.setHeader('Content-Type', 'text/event-stream');

  for await (const ev of agent.run(req.body.message)) {
    // ev.type: 'text' | 'tool_call' | 'tool_result' | 'done' | 'error'
    res.write(`data: ${JSON.stringify(ev)}\n\n`);
  }
  res.end();
});
```

**Event types your frontend must handle:**

| `ev.type`      | `ev.data`                              | UI action                        |
|----------------|----------------------------------------|----------------------------------|
| `text`         | `string` (chunk)                       | Append to message bubble         |
| `tool_call`    | `{ id, name, arguments }`              | Show "calling tool X..." badge   |
| `tool_result`  | `{ tool_call_id, content }`            | Show result or collapse badge    |
| `done`         | `string` (final assembled text)        | Mark message complete            |
| `error`        | `{ message: string }`                  | Show error toast                 |

### 1.2 NATS Chatroom (multi-agent UI)

For multi-agent UIs where several agents share a room:

```ts
import { NatsRoom } from 'medrix-ai';

const room = new NatsRoom({
  url: 'nats://localhost:4222',
  identity: 'frontend-user-123',
  roomName: 'session-abc',
});

// Subscribe to all messages in the room
const unsub = await room.subscribe('chat', (msg) => {
  // msg: { from, subject, body, ts }
  renderMessage(msg);
});

// Publish user input
await room.publish('chat', { text: userInput });
```

NATS subjects are scoped as `medrix.room.<roomName>.<subject>`.

### 1.3 REST / RPC pattern (stateless)

For simpler integrations that don't need streaming:

```ts
import { Agent } from 'medrix-ai';

const agent = new Agent({ model: 'gpt-4o-mini', name: 'api' });
const finalText = await agent.runToText(userMessage);
// Returns the final assembled string after all tool calls resolve.
```

### 1.4 Frontend contract summary

Your frontend needs to:
1. POST user messages to a backend that holds an `Agent` instance.
2. Consume the `AgentEvent` stream (SSE or WebSocket).
3. Render `text` chunks incrementally, show tool-call activity, handle `done`.
4. Optionally: connect to NATS for multi-agent room presence.

No special SDK required — the protocol is JSON lines over any transport.

---

## 2. Core API Surface

### 2.1 Agent

```ts
const agent = new Agent({
  name: 'my-agent',
  model: 'gpt-4o-mini',          // or ['gpt-4o', 'claude-3-5-sonnet'] fallback chain
  toolset: myToolSet,             // optional
  skills: [mySkill],              // optional — injected into system prompt
  mcpPlugins: [myMcpPlugin],      // optional — tools merged on .ready()
  memory: new InMemoryMemory(),   // optional
  systemPrompt: '...',            // optional
  temperature: 0.7,              // optional
  maxIterations: 8,              // tool-call loop cap
});

await agent.ready();                        // connect MCP plugins
for await (const ev of agent.run(input)) {} // streaming
const text = await agent.runToText(input);  // convenience
await agent.close();                        // disconnect MCP
```

### 2.2 Tool / ToolSet

```ts
import { defineTool, ToolSet, z } from 'medrix-ai';

const myTool = defineTool<{ query: string }, string>({
  name: 'search',
  description: 'Search the knowledge base.',
  parameters: z.object({ query: z.string() }),
  execute: async ({ query }) => searchKB(query),
});

const ts = new ToolSet('my-tools', [myTool]);
// or: ts.register(myTool);
// or: ToolSet.fromClass(instanceWithDecoratedMethods);
```

### 2.3 Skill

```ts
// As a markdown file (src/skill/builtin/my-skill.md):
// ---
// name: my-skill
// description: Does X
// ---
// Instructions injected into the system prompt...

import { FileSkillLoader } from 'medrix-ai';
const skill = await new FileSkillLoader(['./skills']).load('my-skill');
```

### 2.4 MCP Plugin

```ts
import { McpClient } from 'medrix-ai';

const plugin = new McpClient('my-server', {
  kind: 'stdio',
  command: 'npx',
  args: ['-y', '@my/mcp-server'],
});
// Pass to Agent via mcpPlugins: [plugin]
// Agent.ready() calls plugin.connect() and merges tools.
```

### 2.5 Team

```ts
import { Sequential, CoordinatorTeam } from 'medrix-ai';

// Pipeline: output of agent[i] becomes input of agent[i+1]
const pipeline = new Sequential([agentA, agentB, agentC], 'my-pipeline');

// Coordinator: one agent plans, members execute
const team = new CoordinatorTeam(coordinator, [worker1, worker2]);

for await (const ev of pipeline.run(input)) { ... }
const { finalText, perAgent } = await team.runToText(input);
```

### 2.6 Bridge (subprocess interop)

```ts
import { runPython } from 'medrix-ai';

const result = await runPython('my-subcommand', [
  ['--flag', value],
  ['--bool-flag', true],
], { pythonBin: 'python3', cwd: '/path/to/project' });
// result: { exitCode, stdout, stderr, parsedJson? }
```

### 2.7 Provider

```ts
import type { LLMProvider, ProviderStreamChunk } from 'medrix-ai';

// Implement your own:
const myProvider: LLMProvider = {
  name: 'my-llm',
  supportsTools: true,
  async *chat(messages, options): AsyncGenerator<ProviderStreamChunk> {
    // yield { type: 'text', text: '...' }
    // yield { type: 'tool_call', toolCall: { id, name, arguments } }
    // yield { type: 'done', finishReason: 'stop' }
  },
};
```

---

## 3. Extension Seams

| Seam | Interface | How to extend |
|------|-----------|---------------|
| **Tool** | `Tool<TArgs, TResult>` | `defineTool(...)` or `@tool()` decorator |
| **Skill** | `Skill` | Markdown file or TS module default-exporting `Skill` |
| **MCP** | `McpPlugin` | `McpClient` for stdio/SSE, or implement the interface |
| **Provider** | `LLMProvider` | Implement `chat()` as an async generator |
| **Memory** | `Memory` | Implement `append/recent/clear` + optional `remember/recall` |
| **Room** | `Room` (abstract) | Extend; `NatsRoom` is the built-in |
| **Bridge** | `runPython()` | Swap `BridgeOptions.moduleName` / `cwd` / `pythonBin` |
| **Team** | `Team` (abstract) | Extend; implement `run()` as `AsyncGenerator<AgentEvent>` |

### Adding a new built-in toolset

1. Create `src/toolset/MyTools.ts`.
2. Export a factory: `export function myToolSet(opt?): ToolSet { ... }`.
3. Add `export * from './MyTools.js';` to `src/toolset/index.ts`.
4. Write tests in `tests/my-tools.test.ts`.

### Adding a new built-in agent

1. Create a system-prompt markdown in `src/agent/prompts/my-agent.system.md`.
2. Add a factory in `src/agent/AnnotationAgents.ts` (or a new file).
3. Export from `src/agent/index.ts`.

### Adding a new team pattern

1. Create `src/team/MyTeam.ts` extending `Team`.
2. Implement `async *run(input: string): AsyncGenerator<AgentEvent>`.
3. Export from `src/team/index.ts`.

---

## 4. Project Structure

```
MedrixAI/
├── src/
│   ├── agent/          # Agent class + built-in agents + prompts/
│   │   └── compression/ # MessageCompressor + SummaryCompressor
│   ├── bridge/         # Python subprocess seam (runPython)
│   │   └── runtime/    # Bundled Python package (novaeve_agent) + configs + scripts
│   ├── chatroom/       # NATS pub/sub Room
│   ├── cli/            # `novaeve` commander CLI
│   ├── endpoint/       # McpServerEndpoint + HttpEndpoint (expose agents externally)
│   ├── evolution/      # Genetic-algorithm code evolution (full)
│   ├── factory/        # ~/.medrix/ template manager
│   ├── gateway/        # Multi-channel gateway (Slack/Telegram/Discord/Lark/WeChat/webhook)
│   ├── knowledge/      # KnowledgeBase + InMemoryKB (RAG)
│   ├── mcp/            # McpPlugin + McpClient
│   ├── memory/         # Memory interface + InMemoryMemory
│   ├── provider/       # LLMProvider + OpenAI/Anthropic/Gemini adapters
│   ├── remote/         # RemoteAgent + RemoteWorker (NATS distributed execution)
│   ├── repl/           # readline REPL
│   ├── schemas/        # zod schemas for built-in agent outputs
│   ├── session/        # SessionStore + FileSessionStore (persistence)
│   ├── skill/          # Skill interface + FileSkillLoader + builtin/
│   ├── store/          # Store interface + LocalStore (package registry)
│   ├── task/           # TaskManager + InMemoryTaskManager (background tasks)
│   ├── team/           # Team patterns (Sequential, Swarm, Coordinator, Pipeline)
│   ├── toolset/        # Tool/ToolSet + built-in toolsets (file, shell, web, notebook, bio, etc.)
│   ├── ui/             # UIServer + DevServer + route definitions
│   ├── utils/          # logger, tokens, misc
│   ├── types.ts        # shared interfaces (Message, AgentEvent, ToolCall, etc.)
│   └── index.ts        # public barrel export
├── tests/              # vitest — flat, one file per module
├── package.json
├── tsconfig.json
└── vitest.config.ts
```

---

## 5. Coding Rules

### TypeScript

- ESM only (`"type": "module"`). All relative imports end in `.js`.
- Strict mode. No `any` in public interfaces; `unknown` + narrowing instead.
- Use `zod` for runtime validation at system boundaries.
- Lazy-import heavy optional deps (`nats`, `openai`, `@anthropic-ai/sdk`,
  `@google/generative-ai`, `@modelcontextprotocol/sdk`) so missing packages
  don't break unrelated paths.
- No comments explaining *what* — only *why* when non-obvious.
- No `TODO` without a linked issue number.

### Naming

- Files: `PascalCase.ts` for classes/interfaces, `camelCase.ts` for pure functions.
- Tool names: `snake_case` with a domain prefix (`bio_`, `synth_`, `bench_`, `annotate_`).
- Agent names: `novaeve-<role>` (e.g. `medrix-selector`).
- Team names: `kebab-case` (e.g. `annotation-pipeline`).
- Env vars: `MEDRIX_*` for framework config; `OPENAI_*` / `ANTHROPIC_*` for providers.

### Dependencies

- Pin exact versions in `package.json` (no `^` for production deps in a framework).
- Prefer well-known, actively maintained packages.
- Every new dep must be justified in the PR description.

---

## 6. Issues & PRs

### Issue template

```markdown
## Context
What you're trying to do and why.

## Current behavior
What happens now (include error messages / logs).

## Expected behavior
What should happen instead.

## Reproduction
Minimal code or CLI command to reproduce.

## Environment
- Node version:
- OS:
- MedrixAI version:
```

### Labels

| Label | Meaning |
|-------|---------|
| `bug` | Something broken |
| `feature` | New capability |
| `toolset` | New or modified Tool/ToolSet |
| `agent` | New or modified built-in Agent |
| `bridge` | Python/Rust subprocess boundary |
| `frontend` | Integration surface (events, NATS, REST) |
| `docs` | Documentation only |
| `breaking` | Public API change |

### PR conventions

- Title: imperative, under 70 chars. Prefix with scope: `feat(toolset):`, `fix(bridge):`, `docs:`.
- Body: what changed, why, what was tested.
- One logical change per PR. Split large features into stacked PRs.
- All tests must pass (`npm test`). Typecheck must pass (`npm run typecheck`).
- New public API must have at least one test.

### Commit messages

```
<type>(<scope>): <subject>

<body — what and why, not how>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.
Scopes: `agent`, `toolset`, `bridge`, `team`, `skill`, `mcp`, `provider`, `cli`, `schemas`.

---

## 7. Testing

### Stack

- **vitest** — test runner.
- Tests live in `tests/` (flat, `*.test.ts`).
- `vitest.config.ts` includes `tests/**/*.test.ts`, timeout 10s.

### Patterns

```ts
// Mock provider for agent/team tests
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

// Schema test
expect(() => SelectorResponse.parse(badPayload)).toThrow();
```

### What to test

| Layer | What to assert |
|-------|----------------|
| Tool | `execute()` returns expected shape; zod rejects bad args |
| Agent | Tool-call loop terminates; events arrive in correct order |
| Team | Pipeline pipes output correctly; coordinator routes |
| Schema | Valid payloads parse; invalid payloads throw |
| Bridge | `buildPythonArgv` produces correct argv; null/empty skipped |
| Skill | `FileSkillLoader` resolves and parses markdown front-matter |

### Running

```bash
npm test              # all tests, single run
npm run test:watch    # watch mode
npm run typecheck     # tsc --noEmit (no test execution)
```

---

## 8. Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MEDRIX_MODEL` | `gpt-4o-mini` | Default LLM model for CLI / REPL |
| `MEDRIX_LOG_LEVEL` | `info` | `debug \| info \| warn \| error` |
| `MEDRIX_PYTHON_BIN` | `python` | Python interpreter for the bridge |
| `MEDRIX_PYTHON_RUNTIME` | `src/bridge/runtime` | Bridge working directory |
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic provider |
| `GOOGLE_API_KEY` | — | Required for Gemini provider |
| `NATS_URL` | `nats://localhost:4222` | NATS server for chatroom |

---

## 9. Quick-Start Recipes

### Add a custom tool and use it in the REPL

```ts
// src/toolset/MyCustomTools.ts
import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';

export function myCustomToolSet(): ToolSet {
  return new ToolSet('custom', [
    defineTool<{ x: number }, number>({
      name: 'double',
      description: 'Double a number.',
      parameters: z.object({ x: z.number() }),
      execute: async ({ x }) => x * 2,
    }),
  ]);
}
```

Then wire it into the CLI or pass it to any Agent.

### Run the annotation pipeline end-to-end

```bash
# 1. Profile a query dataset
npx novaeve annotate profile-query --dataset-id my_query --input data/query.h5ad

# 2. Select source+model pairs
npx novaeve annotate select-models --query-profile runs/selection/my_query/query_profile.json

# 3. Adapt and execute
npx novaeve annotate adapt-and-execute --plan runs/selection/my_query/selected_execution_plan.yaml

# 4. Run consensus
npx novaeve annotate run-consensus --stage3-summary runs/stage3/execution_summary.json
```

### Connect a React frontend via SSE

```ts
// Frontend (React)
const es = new EventSource('/api/chat?message=' + encodeURIComponent(input));
es.onmessage = (e) => {
  const ev = JSON.parse(e.data);
  if (ev.type === 'text') appendToUI(ev.data);
  if (ev.type === 'done') es.close();
};
```

---

## 10. Roadmap Seams (not yet implemented)

These are architectural slots left intentionally empty:

- **`src/evolution/`** — genetic-algorithm code evolution. Interface exists; driver is a stub.
- **Rust bridge backend** — `BridgeOptions` accepts `moduleName` / `cwd` / `pythonBin`; swap for a Rust binary by implementing the same subprocess protocol.
- **MCP server mode** — expose MedrixAI itself as an MCP server so other tools can call its agents.
- **Persistent memory** — `Memory` interface has `remember/recall`; implement with SQLite, Redis, or vector DB.
- **UI server** — the `novaeve ui` command is a placeholder; wire it to a Next.js / Vite frontend serving the SSE endpoint above.
