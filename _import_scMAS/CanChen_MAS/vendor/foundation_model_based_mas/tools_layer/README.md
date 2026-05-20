# tools_layer

This tools layer stays intentionally thin and LangGraph-oriented.

## Design principle
- Use LangChain / LangGraph native tool objects whenever possible.
- Do not introduce a custom manager / registry framework unless there is a real need.
- Keep migration logic close to the original repository when the original workflow is important.

## Adapters kept in this layer
- `function_adapter.py`: returns function tools directly
- `skill_adapter.py`: wraps skill callables into `StructuredTool` only when needed
- `mcp_adapter.py`: loads MCP tools through `langchain-mcp-adapters`

## Tool discovery
Each subpackage auto-discovers modules and collects module-level exports:
- `function_tools/*` exports `TOOLS` and `TOOL_CATALOG`
- `skill_tools/*` exports `SKILL_TOOLS` and `SKILL_TOOL_CATALOG`
- `mcp_tools/*` exports `MCP_SERVERS` and `MCP_SERVER_CATALOG`

This keeps future extension cost low without adding a heavy custom framework.

---

## BRICK migration status

### Migrated source materials
The following original materials are now kept inside this repository:
- `foundation_model_based_mas/BRICK/`
- `foundation_model_based_mas/notebooks/`

### Current BRICK RAG tool
- `retrieve_brick_references_tool`

This tool is the agent-facing wrapper around the migrated BRICK reference retrieval flow.
Its internal orchestration now follows the original BRICK RAG sequence more closely:
- load vectorstore
- search code
- search notebook
- generate final answer

### Vectorstore build strategy
The vectorstore build path now follows the **original BRICK implementation style**:
- code index: built through `BRICK.embedcode.pycode_creator(...)`
- notebook index: built through `BRICK.embedcode.notebook_creator_cell(...)`
- vector database backend: `FAISS`
- embeddings backend: remote `OpenAIEmbeddings`-compatible service configured from `.env`

Generated artifacts:
- `tools_layer/vectorstores/artifacts/BRICK_code_local/`
- `tools_layer/vectorstores/artifacts/BRICK_notebook_local/`

### Important note about retrieval quality
The current retrieval pipeline is already usable, but **ranking quality tuning is intentionally deferred**.
Planned later work includes:
- function-name-aware reranking
- better BRICK-specific chunk prioritization
- more transparent hit metadata in the tool output

This is a known follow-up item and is intentionally not solved in this round.

### Important note about vectorstore source filtering
Current code vectorstore building is still close to the original BRICK behavior:
- recursively scan Python files
- extract Python functions
- write almost all extracted functions into FAISS

So at the moment, the code vectorstore is **not yet aggressively filtered**.

This is intentionally deferred for a later round.
Planned follow-up options:
- file allowlist
- file / directory denylist
- function-name-level filtering
- two-stage filtering: file first, function second

---

## LLM runtime

`foundation_model_based_mas/llm_runtime/` is the single place for:
- loading `.env`
- building chat models
- building embedding models

### Chat env priority
- `BRICK_LLM_*`
- fallback to `RAG_LLM_*`
- fallback to `OPENAI_*`

### Embedding env priority
- `BRICK_EMBEDDING_*`
- fallback to `RAG_EMBEDDING_*`
- fallback to `OPENAI_EMBEDDING_*`

This keeps BRICK and future tools on the same runtime convention.

---

## How to add a new function tool

### Step 1: create a module
Create a Python file under:
- `tools_layer/function_tools/`

Example:
- `tools_layer/function_tools/read_table.py`

### Step 2: define the tool
Prefer native `@tool`.

```python
from langchain_core.tools import tool
from pydantic import BaseModel


class ReadTableInput(BaseModel):
    file_path: str


@tool(args_schema=ReadTableInput)
def read_table_tool(file_path: str) -> dict:
    """Read a table file and return structured information."""
    return {"file_path": file_path, "success": True}
```

### Step 3: export the module-level metadata

```python
TOOLS = [read_table_tool]

TOOL_CATALOG = [
    {
        "name": "read_table_tool",
        "kind": "function",
        "description": "Read a table file and return structured information.",
        "source_repo": "foundation_model_based_mas",
        "source_file": "tools_layer/function_tools/read_table.py",
    }
]
```

### Step 4: use it in LangGraph

```python
from tools_layer import build_langgraph_tools

tools = build_langgraph_tools(function_names=["read_table_tool"])
```

### Step 5: if it needs LLM calls
Never hardcode keys or base URLs inside the tool.
Use `llm_runtime`.

```python
from llm_runtime import build_chat_model

llm = build_chat_model(prefix="OPENAI")
```

If the tool is BRICK-specific:

```python
from llm_runtime import build_brick_chat_model

llm = build_brick_chat_model()
```

### Step 6: if it needs embeddings

```python
from llm_runtime import build_brick_embedding_model

embedding_model = build_brick_embedding_model()
```

### Step 7: if it needs vectorstores
Register the vectorstore in:
- `tools_layer/vectorstores/registry.py`

Build or load it in:
- `tools_layer/vectorstores/builder.py`

If the original source repo already has a canonical build pipeline, prefer reusing that implementation path.

---

## How to add a new skill tool

Use `skill_tools` for workflow-style abilities or wrapped local capabilities that are not already plain LangChain tools.

```python
from pydantic import BaseModel


class NotebookSkillInput(BaseModel):
    notebook_path: str


def summarize_notebook_skill(notebook_path: str) -> str:
    return f"Notebook summary for: {notebook_path}"
```

Then export:
- `SKILL_TOOLS`
- `SKILL_TOOL_CATALOG`

---

## How to add a new MCP tool

Prerequisite:
- `langchain-mcp-adapters`
- `mcp`

Export from the module:
- `MCP_SERVERS`
- `MCP_SERVER_CATALOG`

Use it through:

```python
from tools_layer import build_langgraph_tools

tools = build_langgraph_tools(mcp_names=["filesystem_mcp"])
```

---

## Minimal BRICK RAG usage

```python
from tools_layer import build_langgraph_tools

tools = build_langgraph_tools(
    function_names=["retrieve_brick_references_tool"],
)
```

The tool will:
1. build or load BRICK code / notebook FAISS indexes,
2. retrieve BRICK reference context,
3. synthesize an implementation-focused answer.
