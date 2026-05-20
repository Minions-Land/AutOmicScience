from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import BaseModel


class Input(BaseModel):
    agentname: str
    notebook_path: str


class Output(BaseModel):
    status: str
    cells_text: Optional[str] = None


@tool(args_schema=Input)
def read_notebook_tool(agentname: str, notebook_path: str) -> Output:
    """
    Use this tool to read and parse Jupyter notebook (.ipynb) files.
    Only extracts markdown and code cell content, skipping image outputs to reduce token usage.

    Args:
        agentname: The name of the current agent, like "analyzer"
        notebook_path: The path to the .ipynb file to be read, must be a string

    Return:
        output: A dict with status and cells_text containing markdown, code, and text outputs.
    """
    try:
        notebook_file = Path(notebook_path)
        if not notebook_file.exists():
            return Output(status='error', cells_text=f'Notebook file not found: {notebook_path}')

        if notebook_file.suffix.lower() != '.ipynb':
            return Output(status='error', cells_text=f'File is not a Jupyter notebook: {notebook_path}')

        with open(notebook_file, 'r', encoding='utf-8') as f:
            notebook_content = json.load(f)

        cells_text_parts = []
        cells_count = 0

        if 'cells' in notebook_content:
            for cell in notebook_content['cells']:
                cell_type = cell.get('cell_type', 'unknown')

                if cell_type in ['markdown', 'code']:
                    cells_count += 1
                    cells_text_parts.append(f'\n=== Cell {cells_count} ({cell_type}) ===\n')

                    if 'source' in cell:
                        source = cell['source']
                        if isinstance(source, list):
                            source_text = ''.join(source)
                        else:
                            source_text = str(source)
                        cells_text_parts.append(source_text)
                        cells_text_parts.append('\n')

                    if cell_type == 'code' and 'outputs' in cell:
                        for output in cell['outputs']:
                            output_type = output.get('output_type', '')

                            if output_type == 'stream':
                                stream_name = output.get('name', '')
                                if 'text' in output:
                                    text_content = output['text']
                                    if isinstance(text_content, list):
                                        text_content = ''.join(text_content)
                                    cells_text_parts.append(f'[{stream_name} output]:\n{text_content}\n')

                            elif output_type in ['execute_result', 'display_data']:
                                if 'data' in output:
                                    data = output['data']
                                    if 'text/plain' in data:
                                        plain_text = data['text/plain']
                                        if isinstance(plain_text, list):
                                            plain_text = ''.join(plain_text)
                                        cells_text_parts.append(f'[output]:\n{plain_text}\n')
                                    elif 'text/html' in data and not any(key.startswith('image/') for key in data.keys()):
                                        html_text = data['text/html']
                                        if isinstance(html_text, list):
                                            html_text = ''.join(html_text)
                                        cells_text_parts.append(f'[html output]:\n{html_text}\n')

                            elif output_type == 'error':
                                if 'traceback' in output:
                                    traceback_text = '\n'.join(output['traceback'])
                                    cells_text_parts.append(f'[error]:\n{traceback_text}\n')

        combined_text = '\n'.join(cells_text_parts)

        return Output(status='success', cells_text=combined_text)

    except json.JSONDecodeError as e:
        return Output(status='error', cells_text=f'Invalid JSON format in notebook file: {str(e)}')
    except Exception as e:
        return Output(status='error', cells_text=f'Error reading notebook: {str(e)}')


def _read_notebook_internal(notebook_path: str) -> Output:
    """
    Internal helper to read notebook content without tool invocation overhead.
    Only extracts markdown and code cell content.
    """
    try:
        notebook_file = Path(notebook_path)
        if not notebook_file.exists():
            return Output(status='error', cells_text=f'Notebook file not found: {notebook_path}')

        if notebook_file.suffix.lower() != '.ipynb':
            return Output(status='error', cells_text=f'File is not a Jupyter notebook: {notebook_path}')

        with open(notebook_file, 'r', encoding='utf-8') as f:
            notebook_content = json.load(f)

        cells_text_parts = []
        cells_count = 0

        if 'cells' in notebook_content:
            for cell in notebook_content['cells']:
                cell_type = cell.get('cell_type', 'unknown')
                if cell_type in ['markdown', 'code']:
                    cells_count += 1
                    cells_text_parts.append(f'\n=== Cell {cells_count} ({cell_type}) ===\n')
                    if 'source' in cell:
                        source = cell['source']
                        if isinstance(source, list):
                            source_text = ''.join(source)
                        else:
                            source_text = str(source)
                        cells_text_parts.append(source_text)
                        cells_text_parts.append('\n')

        combined_text = '\n'.join(cells_text_parts)
        return Output(status='success', cells_text=combined_text)

    except json.JSONDecodeError as e:
        return Output(status='error', cells_text=f'Invalid JSON format in notebook file: {str(e)}')
    except Exception as e:
        return Output(status='error', cells_text=f'Error reading notebook: {str(e)}')


def extract_notebook_text(notebook_path: str) -> str:
    result = _read_notebook_internal(notebook_path)
    if result.status == 'success' and result.cells_text:
        return result.cells_text
    return ''


def get_notebook_info(notebook_path: str) -> Dict[str, Any]:
    result = _read_notebook_internal(notebook_path)
    if result.status == 'success':
        cell_count = 0
        if result.cells_text:
            cell_count = result.cells_text.count('=== Cell')
        return {
            'status': 'success',
            'cells_count': cell_count,
            'file_path': notebook_path,
            'has_content': bool(result.cells_text),
        }
    return {
        'status': 'error',
        'message': result.cells_text,
        'file_path': notebook_path,
    }


TOOLS = [read_notebook_tool]
TOOL_CATALOG = [
    {
        'name': 'read_notebook_tool',
        'kind': 'function',
        'description': 'Read a Jupyter notebook and extract markdown, code, and text outputs into plain text.',
        'source_repo': 'BiOmics-master',
        'source_file': 'tools/read_notebook.py',
    }
]
