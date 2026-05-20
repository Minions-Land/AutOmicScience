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
npm test                 # vitest
npx novaeve annotate ... # built-in annotation pipeline subcommands
```

## Architecture

- **`src/agent/`** — `Agent` class (streaming tool-call loop, MCP+skill
  injection, multi-model fallback chain, `+think` suffix). Built-in
  domain agents: `createSelectorAgent`, `createAdapterAgent`,
  `createAdjudicatorAgent`.
- **`src/toolset/`** — `Tool` interface, `@tool()` decorator, `ToolSet`.
  Built-in toolsets: `fileToolSet`, `shellToolSet`, `webToolSet`,
  `codeToolSet`, plus domain toolsets `bioDataToolSet`,
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
  compute (anndata, scanpy, sklearn, torch, R/scDesign3) lives in
  vendored Python under `vendor/python/canchen-mas/` and is reached
  exclusively through this seam.
- **`src/schemas/`** — zod schemas for built-in agents' JSON outputs
  (`SelectorResponse`, `AdapterSpec`, `AdjudicationResponse`).
- **`src/chatroom/`** — NATS pub/sub chatroom.
- **`src/memory/`** — in-memory short/long-term store.
- **`src/evolution/`** — genetic-algorithm code-evolution scaffold.
- **`src/repl/`** — readline REPL.
- **`src/cli/`** — `novaeve` commander CLI (`cli`, `ui`, `setup`,
  `annotate <subcmd>`).
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

## Built-in annotation pipeline

`createAnnotationPipeline()` returns a `Sequential` team:
`Selector → Adapter → Adjudicator`. The pipeline is no-training and
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

## Vendored Python

`vendor/python/canchen-mas/` is the upstream scientific package the
bridge invokes via `python -m scmas <subcommand>`. It is the only place
in the repo where that upstream module identifier appears; everything
else refers to it as the "vendored Python" or "bridge backend". Replace
it with another backend at any time by overriding `BridgeOptions.cwd`,
`pythonBin`, or `moduleName`.
