#!/usr/bin/env node

import { access, constants } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const owner = process.env.AOS_HF_OWNER || 'PoorOtterBob';
const token = process.env.HF_TOKEN || process.env.HUGGINGFACE_HUB_TOKEN || '';
const dryRun = process.argv.includes('--dry-run');

const uploads = [
  {
    repo: `${owner}/AutOmicScience-FoundationModels`,
    type: 'model',
    source: 'src/bridge/runtime/checkpoints/foundation_models',
    target: '.',
    message: 'Upload AutOmicScience foundation model assets',
  },
  {
    repo: `${owner}/AutOmicScience-Reference`,
    type: 'dataset',
    source: 'src/bridge/runtime/data',
    target: 'data',
    message: 'Upload AutOmicScience reference and query assets',
  },
  {
    repo: `${owner}/AutOmicScience-Reference`,
    type: 'dataset',
    source: 'src/bridge/runtime/external/SEA-AD',
    target: 'external/SEA-AD',
    message: 'Upload AutOmicScience external SEA-AD assets',
  },
  {
    repo: `${owner}/AutOmicScience-Reference`,
    type: 'dataset',
    source: 'src/bridge/runtime/vendor/foundation_model_based_mas/tools_layer/mcp_tools/UCE-main/data',
    target: 'vendor/foundation_model_based_mas/tools_layer/mcp_tools/UCE-main/data',
    message: 'Upload AutOmicScience UCE reference data',
  },
];

function run(args) {
  const command = ['hf', ...args];
  console.log(command.join(' '));
  if (dryRun) {
    return;
  }
  const result = spawnSync(command[0], command.slice(1), { stdio: 'inherit' });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

async function exists(localPath) {
  try {
    await access(localPath, constants.R_OK);
    return true;
  } catch {
    return false;
  }
}

for (const upload of uploads) {
  const source = path.join(repoRoot, upload.source);
  if (!(await exists(source))) {
    console.warn(`Skipping missing asset path: ${upload.source}`);
    continue;
  }

  const args = [
    'upload',
    upload.repo,
    upload.source,
    upload.target,
    '--type',
    upload.type,
    '--commit-message',
    upload.message,
  ];
  if (token) {
    args.push('--token', token);
  }
  run(args);
}
