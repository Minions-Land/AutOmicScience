export type CommandKind = 'prompt' | 'local' | 'workflow';

export interface CommandContext {
  args: string;
  metadata?: Record<string, unknown>;
}

export interface RegisteredCommand {
  name: string;
  description: string;
  kind?: CommandKind;
  source?: string;
  handler: (ctx: CommandContext) => Promise<string | void> | string | void;
}

export class CommandRegistry {
  private commands = new Map<string, RegisteredCommand>();

  register(command: RegisteredCommand): this {
    this.commands.set(command.name, {
      ...command,
      kind: command.kind ?? 'local',
    });
    return this;
  }

  unregister(name: string): boolean {
    return this.commands.delete(name);
  }

  get(name: string): RegisteredCommand | undefined {
    return this.commands.get(name);
  }

  list(): RegisteredCommand[] {
    return [...this.commands.values()].sort((a, b) => a.name.localeCompare(b.name));
  }

  async run(name: string, args = '', metadata?: Record<string, unknown>): Promise<string | void> {
    const command = this.commands.get(name);
    if (!command) throw new Error(`Unknown command: ${name}`);
    return command.handler({ args, metadata });
  }
}
