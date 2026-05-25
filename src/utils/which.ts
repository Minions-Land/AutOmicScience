import { existsSync, statSync } from 'node:fs';
import { join } from 'node:path';

/**
 * Find an executable in PATH (synchronous, like Unix `which`).
 * Returns the full path or null if not found.
 */
export function which(name: string): string | null {
  const pathEnv = process.env.PATH ?? '';
  const dirs = pathEnv.split(process.platform === 'win32' ? ';' : ':');
  const extensions = process.platform === 'win32' ? ['.exe', '.cmd', '.bat', ''] : [''];

  for (const dir of dirs) {
    for (const ext of extensions) {
      const candidate = join(dir, name + ext);
      try {
        if (existsSync(candidate)) {
          const stat = statSync(candidate);
          // Check if it's a file and executable
          if (stat.isFile()) {
            return candidate;
          }
        }
      } catch {
        // Permission error or similar - skip
      }
    }
  }
  return null;
}
