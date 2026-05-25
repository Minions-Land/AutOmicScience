import type { TaskManager } from './TaskManager.js';
import type { TaskSpec } from './TaskSpec.js';
import type { TaskStatus, TaskResult } from './TaskStatus.js';

interface InternalTask {
  status: TaskStatus;
  promise: Promise<TaskResult>;
  abortController: AbortController;
  timeoutHandle?: ReturnType<typeof setTimeout>;
}

/**
 * Working in-memory task manager implementation.
 * Executes tasks as Promises with optional timeout support.
 */
export class InMemoryTaskManager implements TaskManager {
  private tasks = new Map<string, InternalTask>();
  private nextId = 1;

  /** Submit a new task for background execution. Returns the task id. */
  async submit(spec: TaskSpec): Promise<string> {
    const id = `task_${this.nextId++}`;
    const abortController = new AbortController();

    const status: TaskStatus = {
      id,
      name: spec.name,
      state: 'pending',
      type: spec.type,
      description: spec.description,
      metadata: spec.metadata,
      progress: [],
    };

    const promise = this.execute(id, spec, abortController);
    const internal: InternalTask = { status, promise, abortController };

    if (spec.timeout && spec.timeout > 0) {
      internal.timeoutHandle = setTimeout(() => {
        abortController.abort();
      }, spec.timeout);
    }

    this.tasks.set(id, internal);
    return id;
  }

  /** Get the current status of a task by id. */
  async status(id: string): Promise<TaskStatus> {
    const task = this.tasks.get(id);
    if (!task) throw new Error(`Task not found: ${id}`);
    return { ...task.status };
  }

  /** Cancel a running or pending task. */
  async cancel(id: string): Promise<void> {
    const task = this.tasks.get(id);
    if (!task) throw new Error(`Task not found: ${id}`);
    if (task.status.state === 'pending' || task.status.state === 'running') {
      task.abortController.abort();
      task.status.state = 'cancelled';
      task.status.completedAt = Date.now();
      if (task.timeoutHandle) clearTimeout(task.timeoutHandle);
    }
  }

  /** List all tasks and their statuses. */
  async list(): Promise<TaskStatus[]> {
    return [...this.tasks.values()].map((t) => ({ ...t.status }));
  }

  /** Wait for a task to complete and return its result. */
  async onComplete(id: string): Promise<TaskResult> {
    const task = this.tasks.get(id);
    if (!task) throw new Error(`Task not found: ${id}`);
    return task.promise;
  }

  private async execute(
    id: string,
    spec: TaskSpec,
    abortController: AbortController,
  ): Promise<TaskResult> {
    // Wait a microtask so the task is registered before execution starts
    await Promise.resolve();
    const task = this.tasks.get(id);

    if (!task || abortController.signal.aborted) {
      const s = task?.status;
      if (s && s.state === 'pending') {
        s.state = 'cancelled';
        s.completedAt = Date.now();
      }
      return {
        id,
        name: spec.name,
        state: 'cancelled',
        duration: 0,
      };
    }

    task.status.state = 'running';
    task.status.startedAt = Date.now();

    try {
      const result = await spec.fn({
        signal: abortController.signal,
        reportProgress: (progress) => {
          task.status.progress = [...(task.status.progress ?? []), progress];
        },
      });

      if (abortController.signal.aborted) {
        task.status.state = 'cancelled';
        task.status.completedAt = Date.now();
        if (task.timeoutHandle) clearTimeout(task.timeoutHandle);
        return {
          id,
          name: spec.name,
          state: 'cancelled',
          duration: Date.now() - (task.status.startedAt ?? Date.now()),
        };
      }

      task.status.state = 'completed';
      task.status.result = result;
      task.status.completedAt = Date.now();
      if (task.timeoutHandle) clearTimeout(task.timeoutHandle);

      return {
        id,
        name: spec.name,
        state: 'completed',
        result,
        duration: task.status.completedAt - (task.status.startedAt ?? task.status.completedAt),
      };
    } catch (err) {
      task.status.state = 'failed';
      task.status.error = err instanceof Error ? err.message : String(err);
      task.status.completedAt = Date.now();
      if (task.timeoutHandle) clearTimeout(task.timeoutHandle);

      return {
        id,
        name: spec.name,
        state: 'failed',
        error: task.status.error,
        duration: task.status.completedAt - (task.status.startedAt ?? task.status.completedAt),
      };
    }
  }
}
