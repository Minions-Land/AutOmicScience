export type HookEvent =
  | 'agent:beforeRun'
  | 'agent:afterRun'
  | 'agent:error'
  | 'message'
  | 'tool:beforeCall'
  | 'tool:afterCall';

export interface HookPayloads {
  'agent:beforeRun': { input: unknown };
  'agent:afterRun': { result: string };
  'agent:error': { error: Error };
  message: { message: unknown };
  'tool:beforeCall': { name: string; args: unknown };
  'tool:afterCall': { name: string; result: unknown };
}

export type HookHandler<E extends HookEvent = HookEvent> = (
  payload: HookPayloads[E],
) => Promise<void> | void;

export class HookManager {
  private handlers = new Map<HookEvent, HookHandler[]>();

  on<E extends HookEvent>(event: E, handler: HookHandler<E>): this {
    const list = this.handlers.get(event) ?? [];
    list.push(handler as HookHandler);
    this.handlers.set(event, list);
    return this;
  }

  off<E extends HookEvent>(event: E, handler: HookHandler<E>): boolean {
    const list = this.handlers.get(event);
    if (!list) return false;
    const next = list.filter((h) => h !== handler);
    this.handlers.set(event, next);
    return next.length !== list.length;
  }

  async emit<E extends HookEvent>(event: E, payload: HookPayloads[E]): Promise<void> {
    for (const handler of this.handlers.get(event) ?? []) {
      await handler(payload as never);
    }
  }
}
