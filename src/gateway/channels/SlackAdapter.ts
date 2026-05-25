import type { ChannelAdapter, ChannelBridge } from '../ChannelAdapter.js';
import type { ConversationRoute } from '../Route.js';

export class SlackAdapter implements ChannelAdapter {
  readonly channel = 'slack';
  private running = false;
  private bridge: ChannelBridge | null = null;
  private abortController: AbortController | null = null;

  async start(config: Record<string, unknown>, bridge: ChannelBridge): Promise<void> {
    const appToken = config.appToken as string;
    const botToken = config.botToken as string;
    if (!appToken || !botToken) throw new Error('Slack requires appToken and botToken');

    this.bridge = bridge;
    this.running = true;
    this.abortController = new AbortController();

    const WebSocket = (await import('ws')).default;
    const appsUrl = await this.getWssUrl(appToken);
    const ws = new WebSocket(appsUrl);

    ws.on('message', async (data: Buffer) => {
      try {
        const payload = JSON.parse(data.toString());
        if (payload.type === 'events_api') {
          const event = payload.payload?.event;
          if (event?.type === 'message' && !event.bot_id && event.text) {
            const route: ConversationRoute = {
              channel: 'slack',
              scopeType: event.channel_type === 'im' ? 'dm' : 'channel',
              scopeId: event.channel,
              threadId: event.thread_ts,
              senderId: event.user,
            };

            const text = event.text as string;
            if (text.startsWith('/')) {
              const cmdResult = await bridge.handleCommand(route, text);
              if (cmdResult.handled && cmdResult.message) {
                await this.postMessage(botToken, event.channel, cmdResult.message, event.thread_ts);
              }
            } else {
              const reply = await bridge.handleMessage(route, {
                channelId: event.channel,
                channelType: 'slack',
                from: event.user,
                text: event.text,
                ts: Date.now(),
                threadId: event.thread_ts,
                metadata: { slackTs: event.ts },
              });
              if (reply) {
                await this.postMessage(botToken, event.channel, reply, event.thread_ts ?? event.ts);
              }
            }
          }
          // Acknowledge the envelope
          ws.send(JSON.stringify({ envelope_id: payload.envelope_id }));
        }
      } catch {
        // ignore malformed messages
      }
    });

    ws.on('close', () => { this.running = false; });
    ws.on('error', () => { this.running = false; });
  }

  async stop(): Promise<void> {
    this.running = false;
    this.abortController?.abort();
  }

  async sendReply(route: ConversationRoute, text: string): Promise<void> {
    // Reply is sent inline during message handling
  }

  private async getWssUrl(appToken: string): Promise<string> {
    const resp = await fetch('https://slack.com/api/apps.connections.open', {
      method: 'POST',
      headers: { Authorization: `Bearer ${appToken}`, 'Content-Type': 'application/x-www-form-urlencoded' },
    });
    const data = await resp.json() as any;
    if (!data.ok) throw new Error(`Slack connection failed: ${data.error}`);
    return data.url;
  }

  private async postMessage(botToken: string, channel: string, text: string, threadTs?: string): Promise<void> {
    const body: Record<string, string> = { channel, text };
    if (threadTs) body.thread_ts = threadTs;
    await fetch('https://slack.com/api/chat.postMessage', {
      method: 'POST',
      headers: { Authorization: `Bearer ${botToken}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }
}
