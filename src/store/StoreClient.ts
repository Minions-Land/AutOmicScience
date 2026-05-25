import { StoreAuth } from './StoreAuth.js';
import type { StoreEntry, StoreVersion } from './StoreEntry.js';

const DEFAULT_HUB_URL = 'https://store.aos.ai';

export class StoreClient {
  private readonly auth: StoreAuth;
  private readonly hubUrl: string;

  constructor(hubUrl?: string, auth?: StoreAuth) {
    this.auth = auth ?? new StoreAuth();
    this.hubUrl = (hubUrl ?? process.env.AOS_HUB_URL ?? this.auth.hubUrl ?? DEFAULT_HUB_URL).replace(/\/$/, '');
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.auth.token) h['Authorization'] = `Bearer ${this.auth.token}`;
    return h;
  }

  private checkAuth(): void {
    if (!this.auth.isLoggedIn) throw new Error('Not logged in. Run: aos store login');
  }

  async login(username: string, password: string): Promise<{ accessToken: string; user: any }> {
    const resp = await fetch(`${this.hubUrl}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ username, password }),
    });
    if (resp.status === 401) throw new Error('Login failed: incorrect credentials');
    if (!resp.ok) throw new Error(`Login failed: ${resp.statusText}`);
    const data = await resp.json() as any;
    this.auth.save(this.hubUrl, data.access_token, username, data.user?.id ?? '');
    return { accessToken: data.access_token, user: data.user };
  }

  async logout(): Promise<void> {
    this.auth.clear();
  }

  async search(opts: { q?: string; type?: string; category?: string; limit?: number; offset?: number } = {}): Promise<{ packages: StoreEntry[]; total: number }> {
    const params = new URLSearchParams();
    if (opts.q) params.set('q', opts.q);
    if (opts.type) params.set('type', opts.type);
    if (opts.category) params.set('category', opts.category);
    if (opts.limit) params.set('limit', String(opts.limit));
    if (opts.offset) params.set('offset', String(opts.offset));
    const resp = await fetch(`${this.hubUrl}/api/store/packages?${params}`);
    if (!resp.ok) throw new Error(`Search failed: ${resp.statusText}`);
    return resp.json() as any;
  }

  async getPackage(packageId: string): Promise<StoreEntry> {
    const resp = await fetch(`${this.hubUrl}/api/store/packages/${packageId}`);
    if (!resp.ok) throw new Error(`Package not found: ${packageId}`);
    return resp.json() as any;
  }

  async listVersions(packageId: string): Promise<StoreVersion[]> {
    const resp = await fetch(`${this.hubUrl}/api/store/packages/${packageId}/versions`);
    if (!resp.ok) throw new Error(`Failed to list versions: ${resp.statusText}`);
    const data = await resp.json() as any;
    return data.versions ?? data;
  }

  async download(packageId: string, version?: string): Promise<{ content: string; files?: Record<string, string> }> {
    const url = version
      ? `${this.hubUrl}/api/store/packages/${packageId}/download/${version}`
      : `${this.hubUrl}/api/store/packages/${packageId}/download`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Download failed: ${resp.statusText}`);
    return resp.json() as any;
  }

  async publish(data: Partial<StoreEntry> & { content: string }): Promise<StoreEntry> {
    this.checkAuth();
    const resp = await fetch(`${this.hubUrl}/api/store/packages`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify(data),
    });
    if (resp.status === 409) {
      const detail = ((await resp.json()) as any).detail ?? 'Name already taken';
      throw new Error(`Publish failed: ${detail}`);
    }
    if (!resp.ok) throw new Error(`Publish failed: ${resp.statusText}`);
    return resp.json() as any;
  }

  async publishVersion(packageId: string, data: { version: string; content: string; files?: Record<string, string>; changelog?: string }): Promise<StoreVersion> {
    this.checkAuth();
    const resp = await fetch(`${this.hubUrl}/api/store/packages/${packageId}/versions`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify(data),
    });
    if (!resp.ok) throw new Error(`Publish version failed: ${resp.statusText}`);
    return resp.json() as any;
  }

  async deletePackage(packageId: string): Promise<void> {
    this.checkAuth();
    const resp = await fetch(`${this.hubUrl}/api/store/packages/${packageId}`, {
      method: 'DELETE',
      headers: this.headers(),
    });
    if (!resp.ok) throw new Error(`Delete failed: ${resp.statusText}`);
  }

  async myPublished(): Promise<StoreEntry[]> {
    this.checkAuth();
    const resp = await fetch(`${this.hubUrl}/api/store/my/published`, { headers: this.headers() });
    if (!resp.ok) throw new Error(`Failed: ${resp.statusText}`);
    const data = await resp.json() as any;
    return data.packages ?? data;
  }

  async myInstalled(): Promise<StoreEntry[]> {
    this.checkAuth();
    const resp = await fetch(`${this.hubUrl}/api/store/my/installed`, { headers: this.headers() });
    if (!resp.ok) throw new Error(`Failed: ${resp.statusText}`);
    const data = await resp.json() as any;
    return data.packages ?? data;
  }

  async recordInstall(packageId: string, version: string): Promise<void> {
    this.checkAuth();
    await fetch(`${this.hubUrl}/api/store/my/installed`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({ package_id: packageId, version }),
    });
  }

  async recordUninstall(packageId: string): Promise<void> {
    this.checkAuth();
    await fetch(`${this.hubUrl}/api/store/my/installed/${packageId}`, {
      method: 'DELETE',
      headers: this.headers(),
    });
  }
}
