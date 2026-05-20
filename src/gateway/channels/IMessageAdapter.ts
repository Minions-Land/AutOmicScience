import { spawn } from 'child_process';
import type { ChannelAdapter, ChannelBridge } from '../ChannelAdapter.js';
import type { ConversationRoute } from '../Route.js';

export class IMessageAdapter implements ChannelAdapter {
  readonly channel = 'imessage';
  private running = false;
  private bridge: ChannelBridge | null = null;
  private cliPath = 'imsg';
  private pollInterval: ReturnType<typeof setInterval> | null = null;
  private lastTs = Date.now();

  async start(config: Record<string, unknown>, bridge: ChannelBridge): Promise<void> {
    this.cliPath = (config.cliPath as string) ?? 'imsg';
    this.bridge = bridge;
    this.running = true;
    this.lastTs = Date.now();
    this.poll();
  }

  async stop(): Promise<void> {
    this.running = false;
    if (this.pollInterval) clearInterval(this.pollInterval);
  }

  async sendReply(route: ConversationRoute, text: string): Promise<void> {
    await this.runCli(['send', '--to', route.scopeId, '--text', text]);
  }

  private poll(): void {
    this.pollInterval = setInterval(async () => {
      if (!this.running) return;
      try {
        const output = await this.runCli(['recent', '--since', String(this.lastTs), '--json']);
        this.lastTs = Date.now();
        const messages = JSON.parse(output || '[]');
        for (const msg of messages) {
          if (msg.is_from_me) continue;
          await this.handleMessage(msg);
        }
      } catch { /* ignore poll errors */ }
    }, 5000);
  }

  private async handleMessage(msg: any): Promise<void> {
    const text = msg.text ?? '';
    if (!text) return;

    const chatId = msg.chat_id ?? msg.handle ?? '';
    const route: ConversationRoute = {
      channel: 'imessage',
      scopeType: msg.is_group ? 'group' : 'dm',
      scopeId: chatId,
      senderId: msg.sender ?? '',
    };

    if (text.startsWith('/')) {
      const cmdResult = await this.bridge!.handleCommand(route, text);
      if (cmdResult.handled && cmdResult.message) {
        await this.sendReply(route, cmdResult.message);
      }
      return;
    }

    const reply = await this.bridge!.handleMessage(route, {
      channelId: chatId,
      channelType: 'imessage',
      from: msg.sender ?? chatId,
      text,
      ts: msg.date ? new Date(msg.date).getTime() : Date.now(),
    });
    if (reply) await this.sendReply(route, reply);
  }

  private runCli(args: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
      const proc = spawn(this.cliPath, args, { timeout: 10000 });
      let stdout = '';
      let stderr = '';
      proc.stdout.on('data', (d) => { stdout += d; });
      proc.stderr.on('data', (d) => { stderr += d; });
      proc.on('close', (code) => {
        if (code === 0) resolve(stdout);
        else reject(new Error(`imsg exited ${code}: ${stderr}`));
      });
      proc.on('error', reject);
    });
  }
}
