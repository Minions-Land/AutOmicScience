/**
 * The current status of a background task.
 */
export interface TaskStatus {
  /** Unique task identifier. */
  id: string;
  /** Human-readable task name. */
  name: string;
  /** Current state of the task. */
  state: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  /** The result value if completed. */
  result?: unknown;
  /** Error message if failed. */
  error?: string;
  /** Unix timestamp (ms) when the task started running. */
  startedAt?: number;
  /** Unix timestamp (ms) when the task completed/failed/was cancelled. */
  completedAt?: number;
}

/**
 * The final result of a completed task.
 */
export interface TaskResult {
  id: string;
  name: string;
  state: 'completed' | 'failed' | 'cancelled';
  result?: unknown;
  error?: string;
  duration: number;
}
