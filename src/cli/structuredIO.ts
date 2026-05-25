import readline from 'node:readline';
import type { Agent } from '../agent/index.js';
import type { AgentEvent } from '../types.js';

export interface StructuredRequest {
  id?: string;
  type: 'run' | 'cancel' | 'ping';
  input?: string;
}

export interface StructuredEvent {
  id?: string;
  type: string;
  data?: unknown;
}

export function writeStructuredEvent(event: StructuredEvent, output: NodeJS.WritableStream = process.stdout): void {
  output.write(`${JSON.stringify(event)}\n`);
}

export async function runStructuredAgent(agent: Agent, input: string, id?: string): Promise<void> {
  for await (const event of agent.run(input)) {
    writeStructuredEvent({ id, type: event.type, data: event.data });
  }
}

export async function startStructuredIO(agent: Agent): Promise<void> {
  const rl = readline.createInterface({ input: process.stdin, terminal: false });
  const current: { abort: AbortController | null } = { abort: null };

  for await (const line of rl) {
    if (!line.trim()) continue;
    let req: StructuredRequest;
    try {
      req = JSON.parse(line) as StructuredRequest;
    } catch (err) {
      writeStructuredEvent({ type: 'error', data: { message: 'invalid_json' } });
      continue;
    }

    if (req.type === 'ping') {
      writeStructuredEvent({ id: req.id, type: 'pong' });
      continue;
    }
    if (req.type === 'cancel') {
      current.abort?.abort();
      writeStructuredEvent({ id: req.id, type: 'cancelled' });
      continue;
    }
    if (req.type === 'run') {
      if (!req.input) {
        writeStructuredEvent({ id: req.id, type: 'error', data: { message: 'input_required' } });
        continue;
      }
      current.abort = new AbortController();
      try {
        for await (const event of agent.run(req.input, { signal: current.abort.signal })) {
          writeStructuredEvent({ id: req.id, type: event.type, data: event.data });
        }
      } catch (err) {
        writeStructuredEvent({ id: req.id, type: 'error', data: { message: (err as Error).message } });
      } finally {
        current.abort = null;
      }
    }
  }
}
