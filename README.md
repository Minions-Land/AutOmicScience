# Novaeve-Agent

An evolvable, distributed multi-agent framework with a built-in
bioinformatics annotation pipeline. TypeScript, ESM, Node 20+.

## Install

```bash
npm install
cp .env.example .env   # fill in OPENAI_API_KEY etc.
```

## Run

```bash
npm run cli              # interactive REPL with a default agent
npm run dev -- setup     # write ~/.novaeve/.env interactively
npm run dev -- serve     # start UI server on port 3000
npm test                 # vitest
npx novaeve annotate ... # built-in annotation pipeline subcommands
npx novaeve store list   # browse the local package store
npx novaeve evolve cfg.json  # run evolutionary optimization
```

## Architecture

- **`src/agent/`** — `Agent` class (streaming tool-call loop, MCP+skill
  injection, multi-model fallback chain, `+think` suffix). Built-in
  domain agents: `createSelectorAgent`, `createAdapterAgent`,
  `createAdjudicatorAgent`.
- **`src/agent/compression/`** — `MessageCompressor` interface +
  `SummaryCompressor` for context-window management.
- **`src/toolset/`** — `Tool` interface, `@tool()` decorator, `ToolSet`.
  Built-in toolsets: `fileToolSet`, `shellToolSet`, `webToolSet`,
  `codeToolSet`, `notebookToolSet`, plus domain toolsets `bioDataToolSet`,
  `syntheticDataToolSet`, `benchmarkToolSet`,
  `annotationStageToolSet`.
- **`src/skill/`** — `Skill` interface + `FileSkillLoader`. Built-in:
  `annotation-pipeline`.
- **`src/mcp/`** — `McpPlugin` interface + `McpClient` (stdio / SSE).
- **`src/provider/`** — OpenAI (primary) + Anthropic + Gemini adapters;
  `ModelSelector` auto-detects from env keys.
- **`src/team/`** — `Sequential`, `Swarm`, `CoordinatorTeam`, plus the
  `createAnnotationPipeline` factory.
- **`src/bridge/`** — `runPython()` subprocess seam. Heavy scientific
  compute (anndata, scanpy, sklearn, torch, R/scDesign3) lives in the
  bundled Python runtime at `src/bridge/runtime/` and is reached
  exclusively through this seam.
- **`src/ui/`** — `UIServer` interface + `DevServer` (placeholder HTTP
  server) + route definitions for the REST/SSE API.
- **`src/evolution/`** — Full genetic-algorithm code-evolution system:
  `Evolver` (batch + async-generator), `Evaluator`, `Operators`
  (mutate/crossover/select), `EvolutionDB` + `InMemoryEvolutionDB`.
- **`src/store/`** — `Store` interface + `LocalStore` for discovering,
  installing, and publishing agents/skills/tools/teams.
- **`src/remote/`** — `RemoteAgent` + `RemoteWorker` for NATS-based
  distributed agent execution across nodes.
- **`src/endpoint/`** — `McpServerEndpoint` (expose agents AS an MCP
  server) + `HttpEndpoint` (REST/SSE for external clients).
- **`src/knowledge/`** — `KnowledgeBase` interface + `InMemoryKB` for
  RAG-style document ingestion and retrieval.
- **`src/gateway/`** — Multi-channel `Gateway` interface for Slack,
  Telegram, Discord, Lark, WeChat, and webhooks.
- **`src/task/`** — `TaskManager` interface + `InMemoryTaskManager` for
  background task execution with timeout and cancellation.
- **`src/session/`** — `SessionStore` interface + `FileSessionStore` for
  persisting conversation histories to disk.
- **`src/schemas/`** — zod schemas for built-in agents' JSON outputs
  (`SelectorResponse`, `AdapterSpec`, `AdjudicationResponse`).
- **`src/chatroom/`** — NATS pub/sub chatroom.
- **`src/memory/`** — in-memory short/long-term store.
- **`src/repl/`** — readline REPL.
- **`src/cli/`** — `novaeve` commander CLI (`cli`, `serve`, `setup`,
  `store`, `evolve`, `annotate <subcmd>`).
- **`src/factory/`** — `~/.novaeve/` template manager.

## Extension seams

1. **Tools** — implement the `Tool` interface or use `@tool('description')`.
2. **Skills** — implement `Skill` or drop a markdown file into a directory
   you pass to `FileSkillLoader`.
3. **MCP** — implement `McpPlugin` or use `McpClient` to wrap a stdio /
   SSE MCP server.
4. **Bridge backends** — `runPython()` is one implementation of the
   subprocess seam. Replace it with a Rust binary or remote service
   behind the same interface to swap out heavy compute.
5. **Knowledge** — implement `KnowledgeBase` with a vector DB for
   production RAG.
6. **Gateway** — implement channel adapters for new messaging platforms.
7. **Store** — implement `Store` against a remote registry for team sharing.

## Built-in annotation pipeline

`createAnnotationPipeline()` returns a `Sequential` team:
`Selector -> Adapter -> Adjudicator`. The pipeline is no-training and
follows the rules in `src/skill/builtin/annotation-pipeline.md`. The
deterministic data prep, synthetic generation, and benchmarking steps
are exposed as tools (`bio_*`, `synth_*`, `bench_*`,
`annotate_*`) so any agent can drive them.

```ts
import { createAnnotationPipeline, loadAnnotationPipelineSkill } from 'novaeve-agent';

const skill = await loadAnnotationPipelineSkill();
const team = await createAnnotationPipeline({
  model: process.env.NOVAEVE_MODEL ?? 'gpt-4o-mini',
  skills: [skill],
});
const { finalText } = await team.runToText(JSON.stringify({
  query_profile: 'runs/selection/my_query/query_profile.json',
  capability_dir: 'configs/capability',
  prepared_source_root: 'data/prepared_sources',
  artifact_bundle: 'artifacts/capability_eval',
  top_k: 3,
}));
```

## Python runtime

`src/bridge/runtime/` contains the Python scientific compute package
(`novaeve_bio`) that the bridge invokes via `python -m novaeve_bio <subcommand>`.
This is an implementation detail behind the bridge seam. Replace it with
another backend at any time by overriding `BridgeOptions.cwd`,
`pythonBin`, or `moduleName`.
