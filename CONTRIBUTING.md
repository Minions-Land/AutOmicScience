# AutOmicScience 维护指南

AutOmicScience 是 AutOmicScience 开发的生信智能体系统，面向单细胞、生信数据处理和计算生物学工作流。维护目标是让对话、规划、工具调用、真实数据接入、缺失依赖反馈和外部前端协议连接稳定可靠。

## 开发环境

```bash
npm install
npm run typecheck
npm test
npm run build
```

可选 Python bridge：

```bash
cd src/bridge/runtime
uv venv
uv pip install -e ".[llm,foundation,notebook,test]"
cd ../../..
```

本地路径只放环境变量，不写进代码：

```bash
AOS_PYTHON_BIN=/path/to/python
AOS_RSCRIPT_PATH=/path/to/Rscript
AOS_MAS_SEAAD_MERFISH_H5AD=/path/to/data.h5ad
AOS_MAS_GENEFORMER_DIR=/path/to/checkpoint
```

## 品牌与文案

- 产品名使用 AutOmicScience 或 AOS。
- README、UI 文案、示例和错误信息应使用清晰、自然的产品语言。
- Provider 名称使用通用技术名，例如 `openai`、`gemini`、`anthropic/<model>`。

## 目录职责

```text
src/
  agent/        Agent runtime、prompt、历史、工具循环
  bridge/       Python bridge 和 Bio MAS runtime
  chatroom/     ChatRoom、NATS、项目/会话协作能力
  cli/          aos CLI、setup、structured IO
  commands/     命令加载和 slash/prompt command
  endpoint/     HTTP/MCP endpoint
  evolution/    遗传/进化优化
  hooks/        生命周期 hook
  mcp/          MCP client
  permissions/  权限规则和持久化
  plugin/       插件加载和注册
  project/      项目指令发现
  session/      文件会话存储
  store/        包发布、安装、Store client
  task/         后台任务
  team/         多智能体团队模式
  toolset/      内置工具集和工具搜索
  ui/           Console server、HTML、品牌资产、AutOmicScience 兼容层和内置兼容前端
tests/          Vitest 测试
```

## 添加 Tool / ToolSet

1. 在 `src/toolset/MyTools.ts` 新建工具集。
2. 使用 `defineTool<Args, Result>({...})` 和 `zod` 参数 schema。
3. 在 `src/toolset/index.ts` 导出。
4. 如果要默认给 AOS 使用，在 `src/toolset/BuiltinToolSets.ts` 注册。
5. 如果工具会写文件、跑命令或删除数据，接入权限字段：`operation`、`isReadOnly`、`isDestructive`、`getPath`、`getCommand`。
6. 在 `tests/` 增加覆盖，至少测成功、输入错误和权限/缺依赖路径。

## 添加 Skill

- 内置 skill：`src/skill/`。
- 插件 skill：插件目录下 `skills/<name>/SKILL.md` 或 `skills/<name>.md`，由 `src/plugin/PluginLoader.ts` 加载。
- 用户本地 skill：`~/.aos/skills/`。
- 默认 agent 会暴露 `list_available_skills`、`list_active_skills`、`read_skill`、`load_skill`、`remove_skill`；新增 skill 后要确保这些工具能发现和读取。
- 生信 skill 必须写清楚输入、输出、依赖、数据来源、失败模式，以及缺数据时应该如何反馈。

## 添加 MCP

- MCP client：`src/mcp/McpClient.ts`、`src/mcp/McpPlugin.ts`。
- MCP server endpoint：`src/endpoint/McpServerEndpoint.ts`。
- 如果 MCP 工具要进入默认 agent，优先通过插件或启动配置注入，不要在 prompt 里硬编码。
- 新 MCP 需要说明启动命令、环境变量、需要的外部服务，以及断连/超时行为。

## 添加 Plugin

- 插件 loader 和 registry 在 `src/plugin/`。
- 本地插件搜索路径：
  - `~/.aos/plugins`
  - 项目内 `plugins`
- 插件可包含 tools、skills、commands、hooks。
- 插件 manifest 里不要写本机绝对路径和真实密钥。

## 添加 Command

- 核心命令系统在 `src/commands/`。
- 用户或项目命令位置：
  - `~/.aos/commands`
  - `.aos/commands`
  - `commands`
- 命令应该是可读、可审计的文本文件；涉及危险操作时交给权限系统判断。

## 修改 Prompt

- 主 prompt 文件：`src/agent/prompts/AOSSystemPrompt.ts`。
- 选择器/裁判/adapter prompt 在 `src/agent/prompts/`。
- Prompt 应聚焦 AutOmicScience 的生信、工程和工具调用能力，避免写入临时调试说明或交接备注。

## AutOmicScience 前端兼容层

兼容层代码在：

```text
src/ui/AOSCompat.ts
src/ui/Server.ts
src/ui/aos/
src/ui/routes.ts
tests/ui.test.ts
vendor/nats-server/
```

维护原则：

- AutOmicScience 原生接口与兼容接口都需要稳定可用。
- `/aos/` 内置页面、静态资源和兼容服务端协议都属于维护范围。
- 兼容接口数量和功能必须大于等于 AutOmicScience 前端/客户端实际调用。
- Store API 需要覆盖这些路径：`/api/auth/login`、`/api/store/packages`、`/api/store/packages/stats`、详情、版本、下载、发布、更新、删除、my published、my installed。
- ChatRoom 兼容方法需要通过 NATS RPC 和 `POST /api/aos/rpc` 两种方式可调用。
- 聊天上下文、团队 setup、导入导出、消息回滚、项目/工作区绑定、gateway/wechat 占位、template scope 等辅助方法也属于兼容层维护范围；即使本地没有外部服务，也要返回结构化结果或明确的 `success: false` 原因，不能让前端 404 或超时。
- NATS subject 格式为 `aos.service.<service_id>`。
- Chat stream subject 格式为 `aos.stream.chat_<chat_id>`。
- 本地 NATS 运行件优先级为 `AOS_NATS_SERVER`、系统 PATH、`vendor/nats-server/<platform>-<arch>/nats-server`。更新运行件时保留其 LICENSE，并用 `/api/aos/ready` 确认 `nats.running` 为 `true`。
- 如果某项外部能力需要账号、OAuth、语音服务或额外进程，必须返回结构化错误，不要 404 或空白失败。
- 修改兼容层后至少运行 `tests/ui.test.ts`，并确认 `/api/aos/ready` 返回正确。

## UI 维护

- Console HTML：`src/ui/AppHtml.ts`。
- Server/API：`src/ui/Server.ts`。
- AutOmicScience 兼容层：`src/ui/AOSCompat.ts`。
- AutOmicScience 兼容前端静态资源：`src/ui/aos/`。
- 品牌图片：`src/ui/assets/`。
- 页面必须保留对话、工具事件、模型、权限、插件、任务、会话、项目指令和 hook 信息。
- 不提交运行日志、临时截图、真实 key、本机路径。

## Bio MAS 维护

- Bio MAS runtime 在 `src/bridge/runtime/aos_agent/`。
- 配置在 `src/bridge/runtime/configs/`。
- 轻量 tiny demo 必须明确标记为 synthetic smoke test。
- 真实数据/权重缺失时必须写 blocker，不要造假。
- Foundation model 运行必须依赖真实 checkpoint 路径。
- 新增生信能力时要补 preflight 检查、错误反馈和 tiny/单元测试。

## 依赖和大文件

可以直接安装的小依赖由维护者安装并记录。大型内容不要提交：

- 模型权重；
- 原始数据集；
- 大型 benchmark 产物；
- 运行日志；
- checkpoint；
- `.env`、token、API key。

如果缺这些内容，在 README、Bio MAS guide 或对应工具返回里写清楚需要什么路径、如何配置、缺失时系统会如何跳过。

## 测试要求

常规：

```bash
npm run typecheck
npm test
npm run build
```

Bio MAS：

```bash
npm run dev -- annotate bio-mas-preflight
npm run dev -- annotate run-tiny-bio-mas-demo -- --output-dir runs/bio_mas_tiny_demo --cells-per-label 6 --top-k 1
```

AutOmicScience 兼容：

```bash
npm run dev -- serve --port 3127
```

如需隔离测试数据：

```bash
npm run dev -- serve --port 3127 --aos-data-dir .aos/aos-compat
```

检查：

```text
GET http://localhost:3127/api/aos/ready
GET http://localhost:3127/aos/
GET http://localhost:3127/api/store/packages
POST http://localhost:3127/api/aos/rpc
```

`tests/ui.test.ts` 会在发现可用 NATS 运行件时执行真实 NATS RPC；默认交付包在 Windows x64 上使用 `vendor/nats-server/win32-x64/nats-server.exe`。

## PR / 交接检查表

- 没有真实 key、token、`.env`、日志、本机绝对路径。
- 没有提交大型数据、权重、checkpoint。
- 缺数据/缺权重已文档化，并能在 preflight 或工具返回中体现。
- 新接口已在 `src/ui/routes.ts` 或对应文档中登记。
- 前端静态资源需要能通过 `npm run build` 复制到 `dist/ui/`。
- AutOmicScience 兼容接口没有减少。
- `npm run typecheck`、`npm test`、`npm run build` 通过。

## Commit Message

```text
<type>(<scope>): <subject>
```

常用 type：`feat`、`fix`、`refactor`、`docs`、`test`、`chore`。

常用 scope：`agent`、`toolset`、`bridge`、`bio-mas`、`provider`、`cli`、`ui`、`mcp`、`permissions`、`plugin`、`session`、`task`、`aos-compat`。
