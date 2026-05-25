/**
 * Specification for a background task to be submitted to the TaskManager.
 */
export interface TaskSpec {
  /** Human-readable name for the task. */
  name: string;
  /** Optional task type for routing/inspection. */
  type?: string;
  /** Optional task description. */
  description?: string;
  /** Optional metadata carried with task status snapshots. */
  metadata?: Record<string, unknown>;
  /** The async function to execute. */
  fn: (ctx: { signal: AbortSignal; reportProgress: (progress: unknown) => void }) => Promise<unknown>;
  /** Optional timeout in milliseconds. If exceeded, the task is cancelled. */
  timeout?: number;
}
