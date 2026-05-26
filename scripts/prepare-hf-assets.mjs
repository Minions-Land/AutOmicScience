#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, readdir, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifestRoot = path.join(repoRoot, 'hf-assets');
const owner = process.env.AOS_HF_OWNER || 'PoorOtterBob';
const withSha256 = process.argv.includes('--sha256');

const assetSets = [
  {
    key: 'foundation-models',
    repo: `${owner}/AutOmicScience-FoundationModels`,
    type: 'model',
    outputDir: path.join(manifestRoot, 'foundation-models'),
    sources: [
      {
        local: 'src/bridge/runtime/checkpoints/foundation_models',
        target: '.',
      },
    ],
    readme: `---
license: other
library_name: automic-science
tags:
- biology
- bioinformatics
- single-cell
- omics
- foundation-model
---

# AutOmicScience Foundation Models

Optional foundation-model checkpoints for AutOmicScience (AOS).

These assets are not required for the core CLI, UI, tests, or tiny Bio MAS smoke checks. Download them only for production Bio MAS workflows that need foundation-model adapters.

## Install

\`\`\`bash
hf download ${owner}/AutOmicScience-FoundationModels \\
  --repo-type model \\
  --local-dir src/bridge/runtime/checkpoints/foundation_models
\`\`\`

See \`README.md\` and \`HUGGINGFACE_ASSETS.md\` in the GitHub repository for environment variables and GPU safety notes.
`,
  },
  {
    key: 'reference',
    repo: `${owner}/AutOmicScience-Reference`,
    type: 'dataset',
    outputDir: path.join(manifestRoot, 'reference'),
    sources: [
      {
        local: 'src/bridge/runtime/data',
        target: 'data',
      },
      {
        local: 'src/bridge/runtime/external/SEA-AD',
        target: 'external/SEA-AD',
      },
      {
        local: 'src/bridge/runtime/vendor/foundation_model_based_mas/tools_layer/mcp_tools/UCE-main/data',
        target: 'vendor/foundation_model_based_mas/tools_layer/mcp_tools/UCE-main/data',
      },
    ],
    readme: `---
license: other
library_name: automic-science
tags:
- biology
- bioinformatics
- single-cell
- omics
- reference
---

# AutOmicScience Reference Assets

Optional biological reference, query, external SEA-AD, and UCE data assets for AutOmicScience (AOS).

These assets are not required for the core CLI, UI, tests, or tiny Bio MAS smoke checks. Download them only for Bio MAS workflows that need full reference collections.

## Install

\`\`\`bash
hf download ${owner}/AutOmicScience-Reference \\
  --repo-type dataset \\
  --local-dir src/bridge/runtime
\`\`\`

See \`README.md\` and \`HUGGINGFACE_ASSETS.md\` in the GitHub repository for environment variables and GPU safety notes.
`,
  },
];

async function pathExists(localPath) {
  try {
    await stat(localPath);
    return true;
  } catch {
    return false;
  }
}

async function fileSha256(localPath) {
  const hash = createHash('sha256');
  await new Promise((resolve, reject) => {
    createReadStream(localPath)
      .on('data', (chunk) => hash.update(chunk))
      .on('error', reject)
      .on('end', resolve);
  });
  return hash.digest('hex');
}

async function walkFiles(root, base = root) {
  const entries = await readdir(root, { withFileTypes: true });
  const rows = [];
  for (const entry of entries) {
    const fullPath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      rows.push(...await walkFiles(fullPath, base));
    } else if (entry.isFile()) {
      const info = await stat(fullPath);
      const rel = path.relative(base, fullPath).split(path.sep).join('/');
      rows.push({
        path: rel,
        bytes: info.size,
        sha256: withSha256 ? await fileSha256(fullPath) : undefined,
      });
    }
  }
  return rows;
}

function targetPath(prefix, filePath) {
  if (prefix === '.') {
    return filePath;
  }
  return `${prefix.replace(/\/$/, '')}/${filePath}`;
}

async function prepare() {
  await mkdir(manifestRoot, { recursive: true });

  for (const assetSet of assetSets) {
    await mkdir(assetSet.outputDir, { recursive: true });
    const rows = [];
    const missing = [];

    for (const source of assetSet.sources) {
      const sourceRoot = path.join(repoRoot, source.local);
      if (!(await pathExists(sourceRoot))) {
        missing.push(source.local);
        continue;
      }
      const files = await walkFiles(sourceRoot);
      for (const file of files) {
        rows.push({
          source_root: source.local,
          path_in_repo: targetPath(source.target, file.path),
          bytes: file.bytes,
          ...(file.sha256 ? { sha256: file.sha256 } : {}),
        });
      }
    }

    rows.sort((a, b) => a.path_in_repo.localeCompare(b.path_in_repo));
    const totalBytes = rows.reduce((sum, row) => sum + row.bytes, 0);
    const manifest = {
      repo_id: assetSet.repo,
      repo_type: assetSet.type,
      generated_at: new Date().toISOString(),
      sha256_included: withSha256,
      total_files: rows.length,
      total_bytes: totalBytes,
      missing_source_roots: missing,
      files: rows,
    };

    await writeFile(path.join(assetSet.outputDir, 'README.md'), assetSet.readme);
    await writeFile(path.join(assetSet.outputDir, 'asset-manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
    console.log(`${assetSet.key}: ${rows.length} files, ${totalBytes} bytes`);
    if (missing.length > 0) {
      console.warn(`${assetSet.key}: missing ${missing.join(', ')}`);
    }
  }
}

await prepare();
