/**
 * TaskTools — Task management with priorities, dependencies, and ephemeral tasks.
 */

import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';
import { InMemoryTaskManager } from '../task/InMemoryTaskManager.js';
import type { TaskManager } from '../task/TaskManager.js';

// ---------------------------------------------------------------------------
// Task state
// ---------------------------------------------------------------------------

export type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'cancelled';
export type TaskPriority = 'low' | 'medium' | 'high' | 'critical';

export interface Task {
  id: string;
  name: string;
  description: string;
  status: TaskStatus;
  priority: TaskPriority;
  createdAt: string;
  updatedAt: string;
  completedAt?: string;
  dependencies: string[];
  ephemeral: boolean;
  metadata: Record<string, unknown>;
}

export interface TaskToolSetOptions {
  manager?: TaskManager;
}

/** In-memory task store. Shared across the toolset lifetime. */
class TaskStore {
  private tasks = new Map<string, Task>();
  private counter = 0;

  create(opts: {
    name: string;
    description: string;
    priority?: TaskPriority;
    dependencies?: string[];
    ephemeral?: boolean;
    metadata?: Record<string, unknown>;
  }): Task {
    const id = `task_${++this.counter}_${Date.now().toString(36)}`;
    const now = new Date().toISOString();
    const task: Task = {
      id,
      name: opts.name,
      description: opts.description,
      status: 'pending',
      priority: opts.priority ?? 'medium',
      createdAt: now,
      updatedAt: now,
      dependencies: opts.dependencies ?? [],
      ephemeral: opts.ephemeral ?? false,
      metadata: opts.metadata ?? {},
    };
    this.tasks.set(id, task);
    return task;
  }

  get(id: string): Task | undefined {
    return this.tasks.get(id);
  }

  update(id: string, updates: Partial<Pick<Task, 'status' | 'priority' | 'description' | 'metadata'>>): Task {
    const task = this.tasks.get(id);
    if (!task) throw new Error(`Task '${id}' not found`);
    if (updates.status) task.status = updates.status;
    if (updates.priority) task.priority = updates.priority;
    if (updates.description) task.description = updates.description;
    if (updates.metadata) task.metadata = { ...task.metadata, ...updates.metadata };
    task.updatedAt = new Date().toISOString();
    if (task.status === 'completed' || task.status === 'failed') {
      task.completedAt = task.updatedAt;
    }
    return task;
  }

  list(filter?: { status?: TaskStatus; priority?: TaskPriority }): Task[] {
    let tasks = Array.from(this.tasks.values());
    if (filter?.status) tasks = tasks.filter((t) => t.status === filter.status);
    if (filter?.priority) tasks = tasks.filter((t) => t.priority === filter.priority);
    return tasks.sort((a, b) => {
      const priorityOrder: Record<TaskPriority, number> = { critical: 0, high: 1, medium: 2, low: 3 };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    });
  }

  delete(id: string): boolean {
    return this.tasks.delete(id);
  }

  /** Remove completed ephemeral tasks. */
  cleanup(): number {
    let removed = 0;
    for (const [id, task] of this.tasks) {
      if (task.ephemeral && (task.status === 'completed' || task.status === 'cancelled')) {
        this.tasks.delete(id);
        removed++;
      }
    }
    return removed;
  }

  /** Check if all dependencies of a task are completed. */
  canStart(id: string): boolean {
    const task = this.tasks.get(id);
    if (!task) return false;
    return task.dependencies.every((depId) => {
      const dep = this.tasks.get(depId);
      return dep?.status === 'completed';
    });
  }
}

// ---------------------------------------------------------------------------
// Toolset factory
// ---------------------------------------------------------------------------

export function taskToolSet(opts: TaskToolSetOptions = {}): ToolSet {
  const store = new TaskStore();
  const manager = opts.manager ?? new InMemoryTaskManager();

  return new ToolSet('task', [
    defineTool<
      { name: string; description: string; priority?: string; dependencies?: string[]; ephemeral?: boolean; metadata?: Record<string, unknown> },
      Task
    >({
      name: 'create_task',
      aliases: ['TodoWrite'],
      operation: 'task',
      description:
        'Create a new task with a name, description, priority, and optional dependencies on other task IDs.',
      parameters: z.object({
        name: z.string().describe('Short task name'),
        description: z.string().describe('Detailed task description'),
        priority: z.enum(['low', 'medium', 'high', 'critical']).optional().default('medium'),
        dependencies: z.array(z.string()).optional().describe('IDs of tasks that must complete first'),
        ephemeral: z.boolean().optional().default(false).describe('Auto-cleanup when completed'),
        metadata: z.record(z.unknown()).optional().describe('Arbitrary metadata'),
      }),
      execute: async ({ name, description, priority, dependencies, ephemeral, metadata }) => {
        return store.create({
          name,
          description,
          priority: priority as TaskPriority,
          dependencies,
          ephemeral,
          metadata,
        });
      },
    }),

    defineTool<
      { name: string; description?: string; script: string; timeoutMs?: number; metadata?: Record<string, unknown> },
      { id: string; state: string }
    >({
      name: 'start_background_task',
      aliases: ['TaskCreate'],
      operation: 'task',
      description:
        'Start a background JavaScript task. The script receives a context with signal and reportProgress.',
      parameters: z.object({
        name: z.string().describe('Task name'),
        description: z.string().optional().describe('Task description'),
        script: z.string().describe('Async JavaScript function body to execute'),
        timeoutMs: z.number().int().positive().optional().describe('Timeout in milliseconds'),
        metadata: z.record(z.unknown()).optional(),
      }),
      isReadOnly: () => false,
      isDestructive: () => false,
      execute: async ({ name, description, script, timeoutMs, metadata }) => {
        const id = await manager.submit({
          name,
          type: 'script',
          description,
          metadata,
          timeout: timeoutMs,
          fn: async (ctx) => {
            const fn = new Function('ctx', `"use strict"; return (async () => {\n${script}\n})();`);
            return fn(ctx);
          },
        });
        const status = await manager.status(id);
        return { id, state: status.state };
      },
    }),

    defineTool<
      { id: string },
      Awaited<ReturnType<TaskManager['status']>>
    >({
      name: 'get_background_task',
      aliases: ['TaskGet'],
      operation: 'task',
      description: 'Get status, progress, and result for a background task.',
      parameters: z.object({
        id: z.string().describe('Background task ID'),
      }),
      isReadOnly: () => true,
      isDestructive: () => false,
      execute: async ({ id }) => manager.status(id),
    }),

    defineTool<
      Record<string, never>,
      { tasks: Awaited<ReturnType<TaskManager['list']>>; total: number }
    >({
      name: 'list_background_tasks',
      aliases: ['TaskList'],
      operation: 'task',
      description: 'List all background tasks.',
      parameters: z.object({}),
      isReadOnly: () => true,
      isDestructive: () => false,
      execute: async () => {
        const tasks = await manager.list();
        return { tasks, total: tasks.length };
      },
    }),

    defineTool<
      { id: string },
      Awaited<ReturnType<TaskManager['onComplete']>>
    >({
      name: 'wait_background_task',
      aliases: ['TaskOutput'],
      operation: 'task',
      description: 'Wait for a background task to finish and return its final result.',
      parameters: z.object({
        id: z.string().describe('Background task ID'),
      }),
      isReadOnly: () => true,
      isDestructive: () => false,
      execute: async ({ id }) => manager.onComplete(id),
    }),

    defineTool<
      { id: string },
      { ok: boolean; id: string; state: string }
    >({
      name: 'stop_background_task',
      aliases: ['TaskStop'],
      operation: 'task',
      description: 'Cancel a running or pending background task.',
      parameters: z.object({
        id: z.string().describe('Background task ID'),
      }),
      isReadOnly: () => false,
      isDestructive: () => false,
      execute: async ({ id }) => {
        await manager.cancel(id);
        const status = await manager.status(id);
        return { ok: true, id, state: status.state };
      },
    }),

    defineTool<
      { id: string; status?: string; priority?: string; description?: string; metadata?: Record<string, unknown> },
      Task
    >({
      name: 'update_task',
      operation: 'task',
      description: 'Update a task status, priority, description, or metadata.',
      parameters: z.object({
        id: z.string().describe('Task ID'),
        status: z.enum(['pending', 'in_progress', 'completed', 'failed', 'cancelled']).optional(),
        priority: z.enum(['low', 'medium', 'high', 'critical']).optional(),
        description: z.string().optional(),
        metadata: z.record(z.unknown()).optional(),
      }),
      execute: async ({ id, status, priority, description, metadata }) => {
        return store.update(id, {
          status: status as TaskStatus | undefined,
          priority: priority as TaskPriority | undefined,
          description,
          metadata,
        });
      },
    }),

    defineTool<
      { status?: string; priority?: string },
      { tasks: Task[]; total: number }
    >({
      name: 'list_tasks',
      operation: 'task',
      description: 'List tasks with optional status and priority filters.',
      parameters: z.object({
        status: z.enum(['pending', 'in_progress', 'completed', 'failed', 'cancelled']).optional(),
        priority: z.enum(['low', 'medium', 'high', 'critical']).optional(),
      }),
      execute: async ({ status, priority }) => {
        const tasks = store.list({
          status: status as TaskStatus | undefined,
          priority: priority as TaskPriority | undefined,
        });
        return { tasks, total: tasks.length };
      },
    }),

    defineTool<
      { id: string; summary?: string },
      Task
    >({
      name: 'complete_task',
      operation: 'task',
      description: 'Mark a task as completed. Optionally add a completion summary to metadata.',
      parameters: z.object({
        id: z.string().describe('Task ID'),
        summary: z.string().optional().describe('Completion summary'),
      }),
      execute: async ({ id, summary }) => {
        const metadata = summary ? { completionSummary: summary } : undefined;
        const task = store.update(id, { status: 'completed', metadata });
        // Cleanup ephemeral tasks
        store.cleanup();
        return task;
      },
    }),

    defineTool<
      { id: string },
      { canStart: boolean; blockedBy: string[] }
    >({
      name: 'check_dependencies',
      operation: 'task',
      description: 'Check if a task can start (all dependencies completed).',
      parameters: z.object({
        id: z.string().describe('Task ID'),
      }),
      execute: async ({ id }) => {
        const task = store.get(id);
        if (!task) throw new Error(`Task '${id}' not found`);
        const blockedBy = task.dependencies.filter((depId) => {
          const dep = store.get(depId);
          return !dep || dep.status !== 'completed';
        });
        return { canStart: blockedBy.length === 0, blockedBy };
      },
    }),
  ]);
}
