import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';

/**
 * Notebook toolset — provides tools for creating, reading, and executing
 * Jupyter-style notebook cells programmatically.
 */

const NotebookCreateParams = z.object({
  name: z.string().describe('Name for the new notebook.'),
  kernel: z.string().optional().describe('Kernel name (e.g. python3).'),
});

const NotebookExecuteCellParams = z.object({
  notebookId: z.string().describe('ID of the notebook.'),
  cellIndex: z.number().describe('Zero-based index of the cell to execute.'),
});

const NotebookReadParams = z.object({
  notebookId: z.string().describe('ID of the notebook to read.'),
});

const NotebookAddCellParams = z.object({
  notebookId: z.string().describe('ID of the notebook.'),
  cellType: z.enum(['code', 'markdown']).describe('Type of cell to add.'),
  source: z.string().describe('Source content of the cell.'),
});

const notebookCreate = defineTool<
  { name: string; kernel?: string },
  { status: string }
>({
  name: 'notebook_create',
  description: 'Create a new notebook.',
  parameters: NotebookCreateParams,
  execute: async (_args) => ({ status: 'not_implemented' }),
});

const notebookExecuteCell = defineTool<
  { notebookId: string; cellIndex: number },
  { status: string }
>({
  name: 'notebook_execute_cell',
  description: 'Execute a specific cell in a notebook by index.',
  parameters: NotebookExecuteCellParams,
  execute: async (_args) => ({ status: 'not_implemented' }),
});

const notebookRead = defineTool<
  { notebookId: string },
  { status: string }
>({
  name: 'notebook_read',
  description: 'Read the full contents of a notebook.',
  parameters: NotebookReadParams,
  execute: async (_args) => ({ status: 'not_implemented' }),
});

const notebookAddCell = defineTool<
  { notebookId: string; cellType: 'code' | 'markdown'; source: string },
  { status: string }
>({
  name: 'notebook_add_cell',
  description: 'Add a new cell to a notebook.',
  parameters: NotebookAddCellParams,
  execute: async (_args) => ({ status: 'not_implemented' }),
});

/**
 * Returns a ToolSet containing all notebook tools.
 */
export function notebookToolSet(): ToolSet {
  return new ToolSet('notebook', [
    notebookCreate,
    notebookExecuteCell,
    notebookRead,
    notebookAddCell,
  ]);
}
