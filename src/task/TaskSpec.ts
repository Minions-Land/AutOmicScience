/**
 * Specification for a background task to be submitted to the TaskManager.
 */
export interface TaskSpec {
  /** Human-readable name for the task. */
  name: string;
  /** The async function to execute. */
  fn: () => Promise<unknown>;
  /** Optional timeout in milliseconds. If exceeded, the task is cancelled. */
  timeout?: number;
}
