# PantheonOS-ts

TypeScript rewrite of [PantheonOS](https://github.com/) — an evolvable, distributed multi-agent framework.

## Install

```bash
npm install
cp .env.example .env  # then fill in API keys
```

## Run

```bash
npm run cli           # start interactive REPL
npm run dev -- setup  # interactive key setup
npm test
```

## Architecture

- `src/agent` — `Agent` class with streaming tool-call loop
- `src/toolset` — `Tool` interface + `@tool()` decorator + `ToolSet`
- `src/skill` — `Skill` interface + markdown/code loader
- `src/mcp` — `McpPlugin` interface + `@modelcontextprotocol/sdk` adapter
- `src/provider` — OpenAI / Anthropic / Gemini providers (with `+think` suffix)
- `src/team` — Sequential, Swarm, PantheonTeam patterns
- `src/chatroom` — NATS pub/sub chatroom
- `src/memory` — short/long-term memory interface
- `src/evolution` — genetic-algorithm code evolution stub
- `src/repl` — readline REPL
- `src/cli` — `pantheon-ts` commander CLI
- `src/factory` — `~/.pantheon/` template manager

## Plugin seams

1. **Tools** — implement the `Tool` interface or use `@tool('description')`.
2. **Skills** — implement `Skill` or load a markdown skill file via `SkillLoader`.
3. **MCP** — implement `McpPlugin` or use `McpClient` to wrap a stdio/SSE MCP server.
