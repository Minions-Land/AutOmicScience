import type { TaskSpec } from './TaskSpec.js';
import type { TaskStatus, TaskResult } from './TaskStatus.js';

/**
 * Manages background tasks — submit, monitor, cancel, and await completion.
 */
export interface TaskManager {
  /** Submit a new task for background execution. Returns the task id. */
  submit(task: TaskSpec): Promise<string>;
  /** Get the current status of a task by id. */
  status(id: string): Promise<TaskStatus>;
  /** Cancel a running or pending task. */
  cancel(id: string): Promise<void>;
  /** List all tasks and their statuses. */
  list(): Promise<TaskStatus[]>;
  /** Wait for a task to complete and return its result. */
  onComplete(id: string): Promise<TaskResult>;
}
