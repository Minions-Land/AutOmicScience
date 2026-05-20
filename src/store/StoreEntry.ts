/**
 * A publishable/installable entry in the Novaeve store.
 */
export interface StoreEntry {
  /** Unique identifier (e.g. '@author/my-agent'). */
  id: string;
  /** Human-readable name. */
  name: string;
  /** Category of the entry. */
  category: 'agent' | 'skill' | 'tool' | 'team';
  /** Semver version string. */
  version: string;
  /** Short description of what this entry does. */
  description: string;
  /** Author name or handle. */
  author: string;
}
