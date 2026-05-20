export interface StoreEntry {
  id: string;
  name: string;
  category: 'agent' | 'skill' | 'tool' | 'team';
  version: string;
  description: string;
  author: string;
  downloads?: number;
  createdAt?: string;
  updatedAt?: string;
  files?: Record<string, string>;
  tags?: string[];
}

export interface StoreVersion {
  version: string;
  content: string;
  files?: Record<string, string>;
  publishedAt: string;
  changelog?: string;
}
