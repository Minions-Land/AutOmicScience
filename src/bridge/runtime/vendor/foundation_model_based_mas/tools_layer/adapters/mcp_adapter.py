from __future__ import annotations

import asyncio

from ..mcp_tools import MCP_SERVER_CATALOG, MCP_SERVERS


async def get_mcp_tools(names: list[str] | None = None) -> list:
    selected_servers = MCP_SERVERS if names is None else [server for server in MCP_SERVERS if server['name'] in names]
    if names is not None:
        found_names = {server['name'] for server in selected_servers}
        missing = [name for name in names if name not in found_names]
        if missing:
            raise KeyError(f'Unknown MCP server configs: {missing}')

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        raise RuntimeError(
            "Install 'langchain-mcp-adapters' and 'mcp' before using the MCP adapter."
        ) from exc

    tools = []
    for server in selected_servers:
        client = MultiServerMCPClient({server['server_name']: server['client']})
        server_tools = await client.get_tools()
        include_tools = set(server.get('include_tools') or [])
        exclude_tools = set(server.get('exclude_tools') or [])
        name_prefix = server.get('name_prefix', '')
        for tool in server_tools:
            tool_name = getattr(tool, 'name', '')
            if include_tools and tool_name not in include_tools:
                continue
            if exclude_tools and tool_name in exclude_tools:
                continue
            if name_prefix:
                setattr(tool, 'name', f'{name_prefix}{tool_name}')
            tools.append(tool)
    return tools


def get_mcp_tools_sync(names: list[str] | None = None) -> list:
    try:
        return asyncio.run(get_mcp_tools(names))
    except RuntimeError as exc:
        if 'asyncio.run() cannot be called from a running event loop' not in str(exc):
            raise
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(get_mcp_tools(names))
        finally:
            loop.close()


def describe_mcp_servers() -> list[dict]:
    return list(MCP_SERVER_CATALOG.values())
