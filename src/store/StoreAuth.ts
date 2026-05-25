import { readFileSync, writeFileSync, existsSync, mkdirSync, chmodSync } from 'fs';
import { join, dirname } from 'path';
import { homedir } from 'os';

const AUTH_FILE = join(homedir(), '.aos', 'store_auth.json');

export interface AuthData {
  hubUrl: string;
  accessToken: string;
  username: string;
  userId: string;
}

export class StoreAuth {
  private data: Partial<AuthData> = {};

  constructor() {
    this.load();
  }

  private load(): void {
    if (!existsSync(AUTH_FILE)) return;
    try {
      this.data = JSON.parse(readFileSync(AUTH_FILE, 'utf-8'));
    } catch {
      this.data = {};
    }
  }

  save(hubUrl: string, accessToken: string, username: string, userId = ''): void {
    const dir = dirname(AUTH_FILE);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    this.data = { hubUrl, accessToken, username, userId };
    writeFileSync(AUTH_FILE, JSON.stringify(this.data, null, 2), 'utf-8');
    try { chmodSync(AUTH_FILE, 0o600); } catch { /* Windows */ }
  }

  clear(): void {
    this.data = {};
    try {
      const { unlinkSync } = require('fs');
      if (existsSync(AUTH_FILE)) unlinkSync(AUTH_FILE);
    } catch { /* ignore */ }
  }

  get token(): string | undefined { return this.data.accessToken; }
  get hubUrl(): string | undefined { return this.data.hubUrl; }
  get username(): string | undefined { return this.data.username; }
  get isLoggedIn(): boolean { return !!this.data.accessToken; }
}
