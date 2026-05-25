import { cp, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const copies = [
  ['src/ui/assets', 'dist/ui/assets'],
  ['src/ui/aos', 'dist/ui/aos'],
];

for (const [from, to] of copies) {
  const source = path.join(repoRoot, from);
  const target = path.join(repoRoot, to);
  await rm(target, { recursive: true, force: true });
  await cp(source, target, { recursive: true });
}
