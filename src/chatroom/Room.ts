export interface RoomMessage {
  from: string;
  subject: string;
  body: unknown;
  ts: number;
}

export abstract class Room {
  abstract readonly name: string;
  abstract publish(subject: string, body: unknown): Promise<void>;
  abstract subscribe(subject: string, handler: (msg: RoomMessage) => void | Promise<void>): Promise<() => Promise<void>>;
  abstract close(): Promise<void>;
}
