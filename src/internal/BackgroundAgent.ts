import type { AgentEvent } from '../types.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export type BgTaskPriority = 'low' | 'normal' | 'high' | 'critical';

export type BgTaskStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

export interface BackgroundTask<T = unknown> {
  id: string;
  name: string;
  priority: BgTaskPriority;
  status: BgTaskStatus;
  progress: number;
  createdAt: number;
  startedAt?: number;
  completedAt?: number;
  result?: T;
  error?: string;
}

export interface TaskOptions {
  name?: string;
  priority?: BgTaskPriority;
  /** If true, cache the result for subsequent calls with the same name. */
  cache?: boolean;
  /** Abort signal for cancellation. */
  signal?: AbortSignal;
}

type TaskFn<T> = (report: (progress: number) => void) => Promise<T>;

// ── Priority ordering ─────────────────────────────────────────────────────────

const PRIORITY_ORDER: Record<BgTaskPriority, number> = {
  critical: 0,
  high: 1,
  normal: 2,
  low: 3,
};

// ── BackgroundAgent ───────────────────────────────────────────────────────────

/**
 * Runs agent tasks in the background (non-blocking).
 *
 * Features:
 * - Task queue with priority ordering
 * - Progress reporting via events
 * - Cancellation support via AbortSignal
 * - Result caching by task name
 */
export class BackgroundAgent {
  private queue: Array<{ task: BackgroundTask; fn: TaskFn<unknown>; resolve: (v: unknown) => void; reject: (e: unknown) => void }> = [];
  private running: Map<string, BackgroundTask> = new Map();
  private completed: Map<string, BackgroundTask> = new Map();
  private resultCache: Map<string, unknown> = new Map();
  private concurrency: number;
  private activeCount = 0;
  private listeners: Array<(event: AgentEvent) => void> = [];
  private idCounter = 0;

  constructor(concurrency = 3) {
    this.concurrency = concurrency;
  }

  /** Submit a task to run in the background. Returns the task handle. */
  submit<T>(fn: TaskFn<T>, opts: TaskOptions = {}): { task: BackgroundTask<T>; promise: Promise<T> } {
    const name = opts.name ?? `task-${++this.idCounter}`;

    // Check cache
    if (opts.cache && this.resultCache.has(name)) {
      const cached = this.resultCache.get(name) as T;
      const task: BackgroundTask<T> = {
        id: `bg-${Date.now()}-${this.idCounter}`,
        name,
        priority: opts.priority ?? 'normal',
        status: 'done',
        progress: 100,
        createdAt: Date.now(),
        completedAt: Date.now(),
        result: cached,
      };
      return { task, promise: Promise.resolve(cached) };
    }

    const task: BackgroundTask<T> = {
      id: `bg-${Date.now()}-${++this.idCounter}`,
      name,
      priority: opts.priority ?? 'normal',
      status: 'queued',
      progress: 0,
      createdAt: Date.now(),
    };

    // Handle cancellation
    if (opts.signal) {
      opts.signal.addEventListener('abort', () => {
        if (task.status === 'queued' || task.status === 'running') {
          task.status = 'cancelled';
          this.emit({ type: 'task_cancelled', data: { id: task.id, name: task.name } });
        }
      });
    }

    const promise = new Promise<T>((resolve, reject) => {
      this.queue.push({
        task: task as BackgroundTask,
        fn: fn as TaskFn<unknown>,
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      this.sortQueue();
      this.processQueue();
    });

    // Cache result if requested
    if (opts.cache) {
      promise.then((result) => {
        this.resultCache.set(name, result);
      }).catch(() => {});
    }

    return { task, promise };
  }

  /** Cancel a task by ID. */
  cancel(taskId: string): boolean {
    // Check queue
    const queueIdx = this.queue.findIndex((item) => item.task.id === taskId);
    if (queueIdx >= 0) {
      const item = this.queue[queueIdx];
      item.task.status = 'cancelled';
      this.queue.splice(queueIdx, 1);
      item.reject(new Error('Task cancelled'));
      this.emit({ type: 'task_cancelled', data: { id: taskId, name: item.task.name } });
      return true;
    }

    // Check running
    const running = this.running.get(taskId);
    if (running) {
      running.status = 'cancelled';
      this.emit({ type: 'task_cancelled', data: { id: taskId, name: running.name } });
      return true;
    }

    return false;
  }

  /** Get status of all tasks. */
  getTasks(): BackgroundTask[] {
    return [
      ...this.queue.map((item) => item.task),
      ...this.running.values(),
      ...this.completed.values(),
    ];
  }

  /** Get a specific task by ID. */
  getTask(taskId: string): BackgroundTask | undefined {
    const queued = this.queue.find((item) => item.task.id === taskId);
    if (queued) return queued.task;
    return this.running.get(taskId) ?? this.completed.get(taskId);
  }

  /** Subscribe to task events. */
  on(listener: (event: AgentEvent) => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  /** Clear the result cache. */
  clearCache(): void {
    this.resultCache.clear();
  }

  /** Clear completed task history. */
  clearCompleted(): void {
    this.completed.clear();
  }

  private sortQueue(): void {
    this.queue.sort(
      (a, b) => PRIORITY_ORDER[a.task.priority] - PRIORITY_ORDER[b.task.priority],
    );
  }

  private processQueue(): void {
    while (this.activeCount < this.concurrency && this.queue.length > 0) {
      const item = this.queue.shift()!;
      if (item.task.status === 'cancelled') {
        item.reject(new Error('Task cancelled'));
        continue;
      }
      this.executeTask(item);
    }
  }

  private async executeTask(item: typeof this.queue[number]): Promise<void> {
    const { task, fn, resolve, reject } = item;
    task.status = 'running';
    task.startedAt = Date.now();
    this.running.set(task.id, task);
    this.activeCount++;

    this.emit({ type: 'task_start', data: { id: task.id, name: task.name, priority: task.priority } });

    const reportProgress = (progress: number): void => {
      task.progress = Math.min(100, Math.max(0, progress));
      this.emit({ type: 'task_progress', data: { id: task.id, name: task.name, progress: task.progress } });
    };

    try {
      const result = await fn(reportProgress);
      if ((task as BackgroundTask).status === 'cancelled') {
        reject(new Error('Task cancelled'));
        return;
      }
      task.status = 'done';
      task.progress = 100;
      task.completedAt = Date.now();
      task.result = result;
      resolve(result);
      this.emit({ type: 'task_done', data: { id: task.id, name: task.name, result } });
    } catch (err) {
      if ((task as BackgroundTask).status !== 'cancelled') {
        task.status = 'failed';
        task.error = err instanceof Error ? err.message : String(err);
        task.completedAt = Date.now();
        reject(err);
        this.emit({ type: 'task_error', data: { id: task.id, name: task.name, error: task.error } });
      }
    } finally {
      this.running.delete(task.id);
      this.completed.set(task.id, task);
      this.activeCount--;
      this.processQueue();
    }
  }

  private emit(event: AgentEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch {
        // Don't let listener errors break the queue
      }
    }
  }
}
