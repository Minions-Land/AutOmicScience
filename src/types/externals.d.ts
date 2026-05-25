declare module 'ws' {
  import { EventEmitter } from 'events';
  class WebSocket extends EventEmitter {
    constructor(url: string, opts?: any);
    send(data: string | Buffer): void;
    close(): void;
    on(event: string, listener: (...args: any[]) => void): this;
  }
  export default WebSocket;
}

declare module 'better-sqlite3' {
  interface Statement {
    run(...params: any[]): any;
    get(...params: any[]): any;
    all(...params: any[]): any[];
  }
  interface Database {
    prepare(sql: string): Statement;
    exec(sql: string): void;
    close(): void;
  }
  function BetterSqlite3(filename: string, options?: any): Database;
  export = BetterSqlite3;
}

declare module 'pg' {
  interface QueryResult {
    rows: any[];
    rowCount: number;
    fields: { name: string; dataTypeID: number }[];
  }
  class Client {
    constructor(config: any);
    connect(): Promise<void>;
    query(text: string, values?: any[]): Promise<QueryResult>;
    end(): Promise<void>;
  }
  class Pool {
    constructor(config: any);
    query(text: string, values?: any[]): Promise<QueryResult>;
    end(): Promise<void>;
  }
  export { Client, Pool, QueryResult };
}

declare module 'nats' {
  interface NatsConnection {
    publish(subject: string, data?: Uint8Array): void;
    subscribe(subject: string, opts?: any): any;
    drain(): Promise<void>;
    close(): Promise<void>;
    request(subject: string, data?: Uint8Array, opts?: any): Promise<any>;
    jetstream(): any;
  }
  function connect(opts?: { servers?: string | string[] }): Promise<NatsConnection>;
  function JSONCodec(): { encode(data: any): Uint8Array; decode(data: Uint8Array): any };
  function StringCodec(): { encode(data: string): Uint8Array; decode(data: Uint8Array): string };
  export { connect, JSONCodec, StringCodec, NatsConnection };
}

declare module 'sharp' {
  interface SharpInstance {
    metadata(): Promise<{ width?: number; height?: number; format?: string }>;
    resize(width: number, height: number, opts?: { fit?: string }): SharpInstance;
    toBuffer(): Promise<Buffer>;
  }
  function sharp(input: Buffer | string): SharpInstance;
  export default sharp;
}
