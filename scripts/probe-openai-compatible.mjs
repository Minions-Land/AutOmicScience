#!/usr/bin/env node
import { existsSync, readFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const env = {
  ...parseEnvFile(path.join(process.cwd(), '.env')),
  ...parseEnvFile(path.join(os.homedir(), '.aos', '.env')),
  ...process.env,
};

const apiKey = env.AOS_OPENAI_API_KEY || env.OPENAI_API_KEY;
const rawBaseUrl = env.AOS_OPENAI_BASE_URL || env.OPENAI_BASE_URL || env.NEWAPI_BASE_URL || 'https://api.openai.com/v1';
const baseUrl = normalizeBaseUrl(rawBaseUrl);
const models = parseList(process.argv.slice(2).join(',') || env.AOS_PROBE_MODELS || env.AOS_MODEL || 'gpt-5.5');

if (!apiKey) {
  console.error('OPENAI_API_KEY is required in env or ~/.aos/.env');
  process.exit(1);
}

console.log(`Base URL: ${baseUrl}`);
console.log(`API key: ${mask(apiKey)}`);
console.log(`Models: ${models.join(', ')}`);

await probeModels();
for (const model of models) {
  await safeProbe(() => probeChat({ model, stream: false, tools: false }));
  await safeProbe(() => probeChat({ model, stream: true, tools: false }));
  await safeProbe(() => probeChat({ model, stream: false, tools: true }));
  await safeProbe(() => probeChat({ model, stream: true, tools: true }));
}

async function probeModels() {
  const url = `${baseUrl}/models`;
  const response = await timedFetch(url, {
    headers: authHeaders(),
  });
  const text = await response.text();
  printResult('GET /models', response, summarize(text));
}

async function probeChat({ model, stream, tools }) {
  const body = {
    model,
    messages: [
      { role: 'system', content: 'You are a concise assistant.' },
      { role: 'user', content: `请用一句中文回复：${model} 连通性测试。` },
    ],
    stream,
    temperature: 0,
    max_tokens: 64,
    ...(tools ? {
      tools: [
        {
          type: 'function',
          function: {
            name: 'echo_probe',
            description: 'Return the probe text unchanged.',
            parameters: {
              type: 'object',
              properties: {
                text: { type: 'string' },
              },
              required: ['text'],
              additionalProperties: false,
            },
          },
        },
      ],
      tool_choice: 'auto',
    } : {}),
  };

  const response = await timedFetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      ...authHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  const label = `POST /chat/completions model=${model} stream=${stream} tools=${tools}`;
  printResult(label, response, summarize(text));
}

async function safeProbe(fn) {
  try {
    await fn();
  } catch (err) {
    console.log('\n=== request failed ===');
    console.log(err instanceof Error ? `${err.name}: ${err.message}` : String(err));
  }
}

function authHeaders() {
  return {
    Authorization: `Bearer ${apiKey}`,
  };
}

async function timedFetch(url, init) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 120000);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

function printResult(label, response, summary) {
  console.log(`\n=== ${label} ===`);
  console.log(`HTTP ${response.status} ${response.statusText}`);
  console.log(summary);
}

function summarize(text) {
  if (!text) return '(empty body)';
  if (text.startsWith('data:')) {
    const lines = text
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:') && line !== 'data: [DONE]')
      .slice(0, 8)
      .map((line) => line.slice(6));
    return lines.join('\n').slice(0, 1600);
  }
  try {
    const json = JSON.parse(text);
    return JSON.stringify(redact(json), null, 2).slice(0, 1600);
  } catch {
    return text.slice(0, 1600);
  }
}

function redact(value) {
  if (Array.isArray(value)) return value.map(redact);
  if (!value || typeof value !== 'object') return value;
  const next = {};
  for (const [key, item] of Object.entries(value)) {
    next[key] = /key|token|secret|authorization/i.test(key) && typeof item === 'string'
      ? mask(item)
      : redact(item);
  }
  return next;
}

function parseEnvFile(file) {
  if (!existsSync(file)) return {};
  const parsed = {};
  for (const line of readFileSync(file, 'utf8').split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const index = trimmed.indexOf('=');
    if (index === -1) continue;
    const key = trimmed.slice(0, index).trim();
    const value = trimmed.slice(index + 1).trim().replace(/^["']|["']$/g, '');
    parsed[key] = value;
  }
  return parsed;
}

function normalizeBaseUrl(value) {
  const trimmed = value.trim().replace(/\/+$/, '');
  return trimmed.endsWith('/v1') ? trimmed : `${trimmed}/v1`;
}

function parseList(value) {
  return value
    .split(/[,;]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function mask(value) {
  if (!value) return '';
  if (value.length <= 10) return `${value.slice(0, 2)}***`;
  return `${value.slice(0, 6)}...${value.slice(-4)}`;
}
