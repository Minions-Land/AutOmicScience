import type { Store } from './Store.js';
import type { StoreEntry } from './StoreEntry.js';
import { StoreClient } from './StoreClient.js';
import { PackageInstaller } from './PackageInstaller.js';
import type { PackageType } from './PackageInstaller.js';

export class RemoteStore implements Store {
  private readonly client: StoreClient;
  private readonly installer: PackageInstaller;

  constructor(hubUrl?: string) {
    this.client = new StoreClient(hubUrl);
    this.installer = new PackageInstaller();
  }

  async search(query: string, opts?: { type?: string; limit?: number }): Promise<StoreEntry[]> {
    const result = await this.client.search({ q: query, category: opts?.type, limit: opts?.limit ?? 20 });
    return result.packages;
  }

  async install(id: string, version?: string): Promise<string[]> {
    const downloaded = await this.client.download(id, version);
    const pkg = await this.client.getPackage(id);
    const written = this.installer.install(
      pkg.category as PackageType,
      pkg.name,
      downloaded.content,
      downloaded.files,
    );
    try { await this.client.recordInstall(id, version ?? pkg.version); } catch { /* best effort */ }
    return written;
  }

  async uninstall(id: string): Promise<string[]> {
    const pkg = await this.client.getPackage(id);
    const removed = this.installer.uninstall(pkg.category as PackageType, pkg.name);
    try { await this.client.recordUninstall(id); } catch { /* best effort */ }
    return removed;
  }

  async publish(entry: StoreEntry & { content: string }): Promise<void> {
    await this.client.publish(entry);
  }

  async list(category?: StoreEntry['category']): Promise<StoreEntry[]> {
    const result = await this.client.search({ category, limit: 100 });
    return result.packages;
  }

  async getPackage(id: string): Promise<StoreEntry | undefined> {
    try {
      return await this.client.getPackage(id);
    } catch {
      return undefined;
    }
  }

  getClient(): StoreClient {
    return this.client;
  }
}
