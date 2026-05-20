/**
 * TaskTools — Task management with priorities, dependencies, and ephemeral tasks.
 */

import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';

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

export function taskToolSet(): ToolSet {
  const store = new TaskStore();

  return new ToolSet('task', [
    defineTool<
      { name: string; description: string; priority?: string; dependencies?: string[]; ephemeral?: boolean; metadata?: Record<string, unknown> },
      Task
    >({
      name: 'create_task',
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
      { id: string; status?: string; priority?: string; description?: string; metadata?: Record<string, unknown> },
      Task
    >({
      name: 'update_task',
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
