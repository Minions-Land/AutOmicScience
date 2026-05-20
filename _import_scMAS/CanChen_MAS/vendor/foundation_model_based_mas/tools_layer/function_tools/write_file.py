from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel


class Input(BaseModel):
    agentname: str
    filename: str
    content: str
    save_dir: str


class Output(BaseModel):
    status: str
    path: Optional[str] = None
    message: Optional[str] = None


@tool(args_schema=Input)
def write_file_tool(agentname: str, filename: str, content: str, save_dir: str) -> Output:
    """
    Use this tool to write the content to the Markdown file in the specified directory.

    Args:
        agentname: The name of the current agent, like "analyzer"
        filename: The file name suffix that identifies the task, like "calculation"
        content: The Markdown content to be saved
        save_dir: The target directory path

    Return:
        output: A dict with status and path if successful, or status and error message if failed.
    """
    try:
        output_dir = Path(save_dir or './results')
        output_dir.mkdir(exist_ok=True)

        final_name = f'{agentname}_{filename}_result.md'
        output_path = output_dir / final_name

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return Output(status='success', path=str(output_path))
    except Exception as e:
        return Output(status='error', message=str(e))


TOOLS = [write_file_tool]
TOOL_CATALOG = [
    {
        'name': 'write_file_tool',
        'kind': 'function',
        'description': 'Write markdown content to a result file under a target directory and return the saved path.',
        'source_repo': 'BiOmics-master',
        'source_file': 'tools/write_file.py',
    }
]
