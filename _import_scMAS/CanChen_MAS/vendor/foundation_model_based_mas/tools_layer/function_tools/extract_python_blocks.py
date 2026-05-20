from __future__ import annotations

import re

from langchain_core.tools import tool


@tool
def extract_python_blocks_tool(text: str) -> str:
    """
    Use this tool to extract all Python code blocks from the given text.

    Args:
        text: The input string containing potential Python code blocks.

    Returns:
        A string of cleaned Python code. Returns the original text if no markdown
        code blocks are found (assuming it is already pure Python code).
    """
    separator = '\n'
    pattern = r"```python\s*([\s\S]*?)\s*```"
    code_blocks = re.findall(pattern, text, re.DOTALL)

    if code_blocks:
        cleaned_blocks = [block.strip() for block in code_blocks if block.strip()]
        return separator.join(cleaned_blocks) if cleaned_blocks else ''
    return text.strip()


TOOLS = [extract_python_blocks_tool]
TOOL_CATALOG = [
    {
        'name': 'extract_python_blocks_tool',
        'kind': 'function',
        'description': 'Extract Python code blocks from markdown-style text and return cleaned executable Python code.',
        'source_repo': 'BiOmics-master',
        'source_file': 'tools/extract_code.py',
    }
]
