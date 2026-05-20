<div align="center">

# MedrixAI

**An evolvable, distributed multi-agent framework for data science and single-cell biology — written in TypeScript.**

[Documentation](#-medrixai) · [Quick Start](#3-installation) · [Architecture](#5-architecture) · [Contributing](./CONTRIBUTING.md) · [Issues](https://github.com/Minions-Land/MedrixAI/issues)

</div>

<div align="center">

<!-- SHIELD GROUP -->

[![][github-stars-shield]][github-stars-link]
[![][github-forks-shield]][github-forks-link]
[![][github-issues-shield]][github-issues-link]
[![][github-license-shield]][github-license-link]
[![][typescript-shield]][typescript-link]
[![][node-shield]][node-link]
[![][status-shield]]()

</div>

---

## `1` What is MedrixAI?

MedrixAI is a **TypeScript-native, distributed multi-agent framework** that brings agentic code evolution, multi-channel deployment, and bioinformatics-grade compute under one roof. Built for end-to-end data science workflows with a particular focus on single-cell biology, MedrixAI lets agents collaborate across NATS, MCP, and external messaging platforms — and improve themselves through genetic-algorithm-driven code evolution.

### Key Highlights

- **🧬 Evolvable** — Built-in genetic algorithm engine with LLM-driven mutation, sandbox evaluation, island-model populations, and full lineage tracking
- **👥 Multi-Agent Teams** — Sequential, Swarm (with handoff), CoordinatorTeam (dynamic re-plan), Mixture-of-Agents, and AgentAsTool patterns
- **🌐 Distributed** — NATS-based chatroom and remote agent execution for scalable, fault-tolerant deployments
- **📡 Multi-Channel Gateway** — Native adapters for Slack, Telegram, Discord, WeChat, Feishu, QQ, and iMessage
- **🔧 20+ Built-in Toolsets** — Shell, code analysis, file ops, Python/R/Julia, Jupyter, web/scraping, database, image generation, knowledge/RAG, scFM
- **🔌 MCP Native** — Both client (consume MCP servers) and server (expose agents as MCP) modes
- **🛍️ Package Store** — Auth, publish, install with local + remote registries
- **🧠 Internal Systems** — Pattern learning, persistent memory, background tasks, attachment pipeline, multi-layer config

## `2` Quick Start

| | |
| :--- | :--- |
| 📦 | `npm install` |
| 🪄 | `npx medrix setup` — interactive wizard for API keys + default model |
| 💬 | `npx medrix cli` — interactive REPL with slash commands |
| 🚀 | `npx medrix serve` — HTTP/SSE endpoint on port 4000 |
| 🧪 | `npm test` — vitest test suite |

## `3` Installation

### From Source

```bash
git clone https://github.com/Minions-Land/MedrixAI.git
cd MedrixAI
npm install
cp .env.example .env   # fill in at least one API key
npm test               # verify everything works
```

### Optional Native Dependencies

Some toolsets dynamically load optional packages — install only what you need:

```bash
npm install ws                # NATS WebSocket / Slack Socket Mode / Discord gateway
npm install better-sqlite3    # SQLite database tool
npm install pg                # PostgreSQL database tool
npm install sharp             # image resizing in vision utilities
npm install gpt-tokenizer     # accurate BPE token counting
```

### Python Bridge (for scientific compute)

The bridge at `src/bridge/runtime/` runs Python for `anndata`, `scanpy`, `scikit-learn`, `torch`, and R/scDesign3 workflows. Install once:

```bash
pip install -e src/bridge/runtime/
```

## `4` Usage

### REPL

```bash
npx medrix cli
```

Slash commands: `/help`, `/clear`, `/history`, `/save`, `/load`, `/sessions`, `/model`, `/tools`, `/verbose`, `/cancel`, `/export`, `/new`

### Programmatic API

```ts
import { Agent, ToolSet, defineTool, z } from 'medrix-ai';

const searchTool = defineTool<{ q: string }, string>({
  name: 'search',
  description: 'Search the web.',
  parameters: z.object({ q: z.string() }),
  execute: async ({ q }) => `Results for: ${q}`,
});

const agent = new Agent({
  name: 'assistant',
  model: 'claude-sonnet-4-20250514+think:high',  // extended thinking
  toolset: new ToolSet('default', [searchTool]),
});

for await (const event of agent.run('Find papers on single-cell ATAC-seq')) {
  if (event.type === 'text') process.stdout.write(event.data as string);
}
```

### Multi-Agent Teams

```ts
import { CoordinatorTeam, Agent } from 'medrix-ai';

const coordinator = new Agent({ model: 'gpt-4o', name: 'planner' });
const workers = [
  new Agent({ model: 'gpt-4o-mini', name: 'coder' }),
  new Agent({ model: 'gpt-4o-mini', name: 'reviewer' }),
];

const team = new CoordinatorTeam(coordinator, workers);

for await (const event of team.run('Refactor this auth module')) {
  console.log(event.type, event.data);
}
```

### Multi-Channel Gateway

```ts
import { GatewayChannelManager } from 'medrix-ai';

const gw = new GatewayChannelManager();
gw.onMessage(async (route, msg) => agent.runToText(msg.text));

await gw.startChannel('telegram');
await gw.startChannel('slack');
await gw.startChannel('discord');
```

### Evolution Engine

```ts
import { EvolutionEngine } from 'medrix-ai';

const engine = new EvolutionEngine({
  populationSize: 20,
  generations: 10,
  model: 'gpt-4o-mini',
  islands: 4,
});

for await (const event of engine.run(initialPopulation, evaluator)) {
  if (event.type === 'generation') {
    console.log(`Gen ${event.data.index}: best=${event.data.bestFitness}`);
  }
}
```

## `5` Architecture

```
src/
├── agent/           # Agent core + compression + hooks + clone + smartFunc
├── bridge/          # Python subprocess seam (scientific compute)
├── chatroom/        # NATS room manager, special agents, projects, export
├── cli/             # CLI entry + SetupWizard
├── endpoint/        # MCP server + HTTP/SSE + ToolsetProxy + Hub
├── evolution/       # GA engine: LLMMutator, Evaluator, IslandModel
├── factory/         # Template manager (~/.medrix/)
├── gateway/         # 7 channel adapters + route registry + config
├── internal/        # Learning + memory + background + attachment pipeline
├── knowledge/       # InMemoryKB (TF-IDF) + VectorStoreKB (embeddings)
├── mcp/             # McpClient (stdio/SSE) + McpPlugin
├── memory/          # InMemoryMemory + FileMemory (JSONL)
├── provider/        # OpenAI + Anthropic + Gemini + ToolProvider
├── remote/          # NATS-based distributed agent execution
├── repl/            # Interactive REPL with 12 slash commands
├── schemas/         # Zod schemas for annotation pipeline
├── session/         # FileSessionStore (conversation persistence)
├── skill/           # Skill loader (markdown + TS modules)
├── store/           # Package store (auth, client, installer, publisher)
├── task/            # TaskManager + InMemoryTaskManager
├── team/            # 5 team patterns + plugin system
├── toolset/         # Tool/ToolSet + 20 built-in toolsets
├── utils/           # logger, vision, tokens, model discovery, process
├── settings.ts      # Multi-layer config (env + JSONC + deep merge)
└── types.ts         # Shared interfaces
```

## `6` Extension Points

| Seam | Interface | How to Extend |
|------|-----------|---------------|
| **Tool** | `Tool<TArgs, TResult>` | `defineTool(...)` or `@tool()` decorator |
| **Skill** | `Skill` | Markdown file with frontmatter or TS module |
| **MCP** | `McpPlugin` | `McpClient` for stdio/SSE, or implement interface |
| **Provider** | `LLMProvider` | Implement `chat()` as async generator |
| **Memory** | `Memory` | `InMemoryMemory`, `FileMemory`, or custom |
| **Knowledge** | `KnowledgeBase` | `InMemoryKB`, `VectorStoreKB`, or custom |
| **Team** | `Team` (abstract) | Implement `run()` as `AsyncGenerator<AgentEvent>` |
| **Gateway** | `ChannelAdapter` | Implement `start/stop/sendReply` for new platforms |
| **Store** | `Store` | `LocalStore`, `RemoteStore`, or custom registry |

## `7` Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MEDRIX_MODEL` | `gpt-4o` | Default LLM model |
| `MEDRIX_LOG_LEVEL` | `info` | `debug` / `info` / `warn` / `error` |
| `MEDRIX_LOG_FILE` | `~/.medrix/logs/medrix.log` | Log file path |
| `MEDRIX_PYTHON_BIN` | `python` | Python interpreter for bridge |
| `OPENAI_API_KEY` | — | OpenAI provider |
| `ANTHROPIC_API_KEY` | — | Anthropic provider |
| `GOOGLE_API_KEY` | — | Gemini provider |
| `NATS_URL` | `nats://localhost:4222` | NATS server for chatroom + remote |

## `8` Contributing

Contributions of all kinds are welcome — new toolsets, gateway channels, providers, evolution strategies, or just bug reports. See [CONTRIBUTING.md](./CONTRIBUTING.md) for the developer guide, project structure, coding rules, and PR conventions.

[![][pr-welcome-shield]][pr-welcome-link]

## License

MIT

Copyright © 2026 [Minions-Land](https://github.com/Minions-Land).

---

<!-- LINK GROUP -->

[github-stars-shield]: https://img.shields.io/github/stars/Minions-Land/MedrixAI?color=ffcb47&labelColor=black&style=flat-square
[github-stars-link]: https://github.com/Minions-Land/MedrixAI/stargazers
[github-forks-shield]: https://img.shields.io/github/forks/Minions-Land/MedrixAI?color=8ae8ff&labelColor=black&style=flat-square
[github-forks-link]: https://github.com/Minions-Land/MedrixAI/network/members
[github-issues-shield]: https://img.shields.io/github/issues/Minions-Land/MedrixAI?color=ff80eb&labelColor=black&style=flat-square
[github-issues-link]: https://github.com/Minions-Land/MedrixAI/issues
[github-license-shield]: https://img.shields.io/badge/license-MIT-white?labelColor=black&style=flat-square
[github-license-link]: https://github.com/Minions-Land/MedrixAI/blob/main/LICENSE

[typescript-shield]: https://img.shields.io/badge/TypeScript-5.6%2B-3178C6?labelColor=black&logo=typescript&logoColor=white&style=flat-square
[typescript-link]: https://www.typescriptlang.org/
[node-shield]: https://img.shields.io/badge/Node.js-20%2B-339933?labelColor=black&logo=node.js&logoColor=white&style=flat-square
[node-link]: https://nodejs.org/
[status-shield]: https://img.shields.io/badge/status-beta-orange?labelColor=black&style=flat-square

[pr-welcome-shield]: https://img.shields.io/badge/👌_pr_welcome-%E2%86%92-ffcb47?labelColor=black&style=for-the-badge
[pr-welcome-link]: https://github.com/Minions-Land/MedrixAI/pulls
