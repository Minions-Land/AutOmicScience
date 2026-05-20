# MedrixAI

A distributed, evolvable multi-agent framework for data science and
single-cell biology. TypeScript, ESM, Node 20+.

## Quick Start

```bash
npm install
npx medrix setup          # interactive wizard: configure API keys + default model
npx medrix cli            # interactive REPL
npx medrix serve          # HTTP/SSE endpoint on port 4000
npm test                  # vitest
```

## Features

- **Multi-model agents** with streaming, tool calling, vision, `+think` mode, retry + fallback chains
- **5 team patterns**: Sequential, Swarm (with handoff), CoordinatorTeam (dynamic re-plan), MixtureOfAgents, AgentAsTool
- **20+ built-in toolsets**: shell, code analysis, file ops, Python/R/Julia, Jupyter notebooks, web/scraping, database, image generation, knowledge/RAG, scFM, task management, file transfer
- **Distributed chatroom** via NATS pub/sub with room manager, special agents, projects, export
- **7-channel gateway**: Slack, Telegram, Discord, WeChat, Feishu, QQ, iMessage
- **Code evolution engine**: LLM-driven mutation, sandbox evaluation, island model, lineage tracking
- **MCP support**: both client (consume MCP servers) and server (expose agents as MCP)
- **Package store**: auth, publish, install, local + remote registries
- **Internal systems**: learning (pattern extraction), persistent memory, background tasks, attachment pipeline

## Architecture

```
src/
├── agent/           Agent core + compression + hooks + clone + smartFunc
├── bridge/          Python subprocess seam (scientific compute)
├── chatroom/        NATS room manager, special agents, projects, export, threads, streaming
├── cli/             CLI entry + SetupWizard
├── endpoint/        MCP server (3 primitives) + HTTP/SSE + ToolsetProxy + Hub
├── evolution/       GA engine: LLMMutator, Evaluator, Database, IslandModel, Visualizer
├── factory/         Template manager (agents/teams/skills/prompts) with sync
├── gateway/         Channel manager + 7 adapters + route registry + config store
├── internal/        LearningSystem, MemorySystem, BackgroundAgent, PackageRuntime, AttachmentPipeline
├── knowledge/       KnowledgeBase interface + InMemoryKB (TF-IDF) + VectorStoreKB (embeddings)
├── mcp/             McpClient (stdio/SSE) + McpPlugin interface
├── memory/          Memory interface + InMemoryMemory + FileMemory (JSONL persistence)
├── provider/        OpenAI + Anthropic + Gemini + ModelSelector + ToolProvider
├── remote/          RemoteAgent + RemoteWorker (NATS distributed execution)
├── repl/            Interactive REPL with 12 slash commands + session management
├── schemas/         Zod schemas for annotation pipeline outputs
├── session/         FileSessionStore (conversation persistence)
├── skill/           Skill interface + SkillLoader (markdown frontmatter + TS modules)
├── store/           Store client + auth + installer + publisher + local/remote
├── task/            TaskManager + InMemoryTaskManager
├── team/            Sequential, Swarm, CoordinatorTeam, MoA, AgentAsTool, Plugin system
├── toolset/         Tool/ToolSet + 20 built-in toolsets
├── ui/              Dev server (placeholder for frontend)
├── utils/           Logger (file rotation), vision, tokens, model discovery, process, template
├── settings.ts      Multi-layer config (env + JSONC + deep merge)
├── types.ts         Shared interfaces
└── index.ts         Public barrel export
```

## Usage

### Agent

```ts
import { Agent } from 'medrix-ai';

const agent = new Agent({
  name: 'assistant',
  model: 'gpt-4o',                    // or fallback chain: ['gpt-4o', 'claude-sonnet-4-20250514']
  systemPrompt: 'You are helpful.',
  // model: 'claude-sonnet-4-20250514+think:high'  // extended thinking
});

for await (const ev of agent.run('Hello')) {
  if (ev.type === 'text') process.stdout.write(ev.data as string);
}

const copy = agent.clone({ name: 'copy', model: 'gpt-4o-mini' });
```

### Tools

```ts
import { defineTool, ToolSet, z } from 'medrix-ai';

const tool = defineTool<{ q: string }, string>({
  name: 'search',
  description: 'Search the web.',
  parameters: z.object({ q: z.string() }),
  execute: async ({ q }) => `Results for: ${q}`,
});

const agent = new Agent({ model: 'gpt-4o', toolset: new ToolSet('my', [tool]) });
```

### Teams

```ts
import { CoordinatorTeam, Sequential, Agent } from 'medrix-ai';

const coordinator = new Agent({ model: 'gpt-4o', name: 'planner' });
const workers = [new Agent({ model: 'gpt-4o-mini', name: 'coder' })];
const team = new CoordinatorTeam(coordinator, workers);

for await (const ev of team.run('Build a REST API')) { ... }
```

### Gateway

```ts
import { GatewayChannelManager } from 'medrix-ai';

const gw = new GatewayChannelManager();
gw.onMessage(async (route, msg) => agent.runToText(msg.text));
await gw.startChannel('telegram');
```

### Evolution

```ts
import { EvolutionEngine } from 'medrix-ai';

const engine = new EvolutionEngine({ populationSize: 20, generations: 10, model: 'gpt-4o-mini' });
for await (const event of engine.run(initialPopulation, evaluator)) {
  if (event.type === 'generation') console.log(`Gen ${event.data.index}: best=${event.data.bestFitness}`);
}
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MEDRIX_MODEL` | `gpt-4o` | Default LLM model |
| `MEDRIX_LOG_LEVEL` | `info` | `debug` / `info` / `warn` / `error` |
| `MEDRIX_LOG_FILE` | `~/.medrix/logs/medrix.log` | Log file path |
| `MEDRIX_PYTHON_BIN` | `python` | Python interpreter for bridge |
| `OPENAI_API_KEY` | — | OpenAI provider |
| `ANTHROPIC_API_KEY` | — | Anthropic provider |
| `GOOGLE_API_KEY` | — | Gemini provider |
| `NATS_URL` | `nats://localhost:4222` | NATS server for chatroom/remote |

## Extension Points

| Seam | Interface | How to extend |
|------|-----------|---------------|
| Tool | `Tool<TArgs, TResult>` | `defineTool(...)` or `@tool()` decorator |
| Skill | `Skill` | Markdown file with frontmatter or TS module |
| MCP | `McpPlugin` | `McpClient` for stdio/SSE, or implement interface |
| Provider | `LLMProvider` | Implement `chat()` as async generator |
| Memory | `Memory` | `InMemoryMemory` (default), `FileMemory` (persistent), or custom |
| Knowledge | `KnowledgeBase` | `InMemoryKB` (TF-IDF), `VectorStoreKB` (embeddings), or custom |
| Team | `Team` (abstract) | Implement `run()` as `AsyncGenerator<AgentEvent>` |
| Gateway | `ChannelAdapter` | Implement `start/stop/sendReply` for new platforms |
| Store | `Store` | `LocalStore`, `RemoteStore`, or custom registry |

## License

MIT
