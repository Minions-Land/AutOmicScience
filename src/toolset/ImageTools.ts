import { z } from 'zod';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';
import { homedir } from 'os';
import { randomUUID } from 'crypto';
import { defineTool } from './Tool.js';
import type { Tool } from './Tool.js';

const IMAGE_MODEL_SHORTCUTS = ['openai', 'gemini'] as const;

const OPENAI_IMAGE_MODELS = [
  'gpt-image-2',
  'gpt-image-1',
  'gpt-image-1.5',
  'chatgpt-image-latest',
  'dall-e-3',
  'dall-e-2',
];

const GEMINI_IMAGE_MODELS = [
  'gemini/gemini-3-pro-image-preview',
  'gemini/gemini-3.1-flash-image-preview',
  'gemini/gemini-2.5-flash-image',
];

const MULTIMODAL_IMAGE_MODELS = new Set([
  'gemini/gemini-3-pro-image-preview',
  'gemini/gemini-3.1-flash-image-preview',
  'gemini/gemini-2.5-flash-image',
]);

const IMAGE_EDIT_MODELS = new Set([
  'gpt-image-2',
  'gpt-image-1',
  'gpt-image-1.5',
  'chatgpt-image-latest',
]);

interface ImageGenResult {
  success: boolean;
  images?: string[];
  error?: string;
  modelUsed?: string;
  text?: string;
}

function getImageStoreDir(): string {
  const dir = join(homedir(), '.aos', 'images');
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  return dir;
}

function saveBase64Image(chatId: string, dataUri: string): string {
  const dir = join(getImageStoreDir(), chatId);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

  const match = dataUri.match(/^data:image\/(\w+);base64,(.+)$/);
  if (!match) throw new Error('Invalid data URI');
  const ext = match[1] === 'jpeg' ? 'jpg' : match[1];
  const buffer = Buffer.from(match[2], 'base64');
  const filename = `${Date.now()}_${randomUUID().slice(0, 8)}.${ext}`;
  const filePath = join(dir, filename);
  writeFileSync(filePath, buffer);
  return filePath;
}

function detectProvider(model: string): 'openai' | 'gemini' | 'unknown' {
  if (model.startsWith('gemini/') || model.startsWith('gemini-')) return 'gemini';
  if (model.startsWith('gpt-') || model.startsWith('dall-e') || model.startsWith('chatgpt-')) return 'openai';
  return 'unknown';
}

function resolveModel(model?: string): { resolved: string; error?: string } {
  if (!model) {
    if (process.env.OPENAI_API_KEY) return { resolved: 'gpt-image-2' };
    if (process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY) return { resolved: 'gemini/gemini-3-pro-image-preview' };
    return { resolved: '', error: 'No image generation API key configured. Set OPENAI_API_KEY or GOOGLE_API_KEY.' };
  }
  const normalized = model.trim().toLowerCase();
  if (normalized === 'openai') return { resolved: OPENAI_IMAGE_MODELS[0] };
  if (normalized === 'gemini') return { resolved: GEMINI_IMAGE_MODELS[0] };
  return { resolved: model };
}

async function generateWithOpenAI(
  prompt: string,
  model: string,
  opts: { size?: string; quality?: string; n?: number; referenceImages?: string[] },
): Promise<ImageGenResult> {
  const apiKey = process.env.OPENAI_API_KEY;
  const baseUrl = (process.env.OPENAI_API_BASE ?? 'https://api.openai.com/v1').replace(/\/$/, '');
  if (!apiKey) return { success: false, error: 'OPENAI_API_KEY not set' };

  const isEdit = opts.referenceImages && opts.referenceImages.length > 0 && IMAGE_EDIT_MODELS.has(model);

  if (isEdit) {
    // Use image edit endpoint
    const FormData = (await import('node:buffer')).Buffer ? globalThis.FormData : (await import('formdata-node' as any)).FormData;
    const form = new FormData();
    form.append('model', model);
    form.append('prompt', prompt);
    form.append('size', opts.size ?? '1024x1024');
    form.append('n', String(opts.n ?? 1));
    form.append('response_format', 'b64_json');

    // Read reference images and append
    const { readFileSync: readFs } = await import('fs');
    for (const imgPath of opts.referenceImages!) {
      const resolved = imgPath.startsWith('file://') ? imgPath.slice(7) : imgPath;
      const buf = readFs(resolved);
      const blob = new Blob([buf], { type: 'image/png' });
      form.append('image[]', blob, 'reference.png');
    }

    const resp = await fetch(`${baseUrl}/images/edits`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${apiKey}` },
      body: form as any,
    });
    if (!resp.ok) {
      const err = await resp.text();
      return { success: false, error: `OpenAI image edit failed: ${err}` };
    }
    const data = await resp.json() as any;
    const chatId = `gen_${Date.now()}`;
    const images = (data.data ?? []).map((item: any) => {
      if (item.b64_json) return saveBase64Image(chatId, `data:image/png;base64,${item.b64_json}`);
      return item.url;
    });
    return { success: true, images, modelUsed: model };
  }

  // Standard image generation
  const body: Record<string, any> = {
    model,
    prompt,
    size: opts.size ?? '1024x1024',
    n: opts.n ?? 1,
    response_format: 'b64_json',
  };
  if (opts.quality) body.quality = opts.quality;

  const resp = await fetch(`${baseUrl}/images/generations`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const err = await resp.text();
    return { success: false, error: `OpenAI image generation failed: ${err}` };
  }
  const data = await resp.json() as any;
  const chatId = `gen_${Date.now()}`;
  const images = (data.data ?? []).map((item: any) => {
    if (item.b64_json) return saveBase64Image(chatId, `data:image/png;base64,${item.b64_json}`);
    return item.url;
  });
  return { success: true, images, modelUsed: model };
}

async function generateWithGemini(
  prompt: string,
  model: string,
  opts: { referenceImages?: string[]; aspectRatio?: string },
): Promise<ImageGenResult> {
  const apiKey = process.env.GOOGLE_API_KEY ?? process.env.GEMINI_API_KEY;
  if (!apiKey) return { success: false, error: 'GOOGLE_API_KEY or GEMINI_API_KEY not set' };

  const modelName = model.startsWith('gemini/') ? model.slice(7) : model;
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${apiKey}`;

  const parts: any[] = [{ text: prompt }];

  // Add reference images as inline data
  if (opts.referenceImages) {
    const { readFileSync: readFs } = await import('fs');
    for (const imgPath of opts.referenceImages) {
      const resolved = imgPath.startsWith('file://') ? imgPath.slice(7) : imgPath;
      const buf = readFs(resolved);
      const base64 = buf.toString('base64');
      parts.push({ inline_data: { mime_type: 'image/png', data: base64 } });
    }
  }

  const body: Record<string, any> = {
    contents: [{ parts }],
    generationConfig: {
      responseModalities: ['TEXT', 'IMAGE'],
    },
  };
  if (opts.aspectRatio) {
    body.generationConfig.imageGenerationConfig = { aspectRatio: opts.aspectRatio };
  }

  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const err = await resp.text();
    return { success: false, error: `Gemini image generation failed: ${err}` };
  }
  const data = await resp.json() as any;
  const chatId = `gen_${Date.now()}`;
  const images: string[] = [];
  let text = '';

  const candidates = data.candidates ?? [];
  for (const candidate of candidates) {
    for (const part of candidate.content?.parts ?? []) {
      if (part.inline_data) {
        const mimeType = part.inline_data.mime_type ?? 'image/png';
        const ext = mimeType.split('/')[1] ?? 'png';
        const uri = `data:${mimeType};base64,${part.inline_data.data}`;
        images.push(saveBase64Image(chatId, uri));
      } else if (part.text) {
        text += part.text;
      }
    }
  }

  return { success: true, images, text: text || undefined, modelUsed: model };
}

export function createImageTools(): Tool[] {
  return [
    defineTool<{
      prompt: string;
      reference_images?: string[];
      model?: string;
      size?: string;
      quality?: string;
      aspect_ratio?: string;
    }, ImageGenResult>({
      name: 'generate_image',
      description: 'Generate an image from a text description. Supports text-to-image and image editing with reference images. Use "openai" or "gemini" as model shortcuts, or specify a concrete model name.',
      parameters: z.object({
        prompt: z.string().describe('Detailed image description including subject, style, composition, colors, lighting'),
        reference_images: z.array(z.string()).optional().describe('File paths of reference images for style transfer or editing'),
        model: z.string().optional().describe('Model: "openai", "gemini", or concrete name like "gpt-image-2"'),
        size: z.string().optional().describe('Image size for OpenAI models: "1024x1024", "1536x1024", "1024x1536"'),
        quality: z.string().optional().describe('Quality for OpenAI: "low", "medium", "high"'),
        aspect_ratio: z.string().optional().describe('Aspect ratio for Gemini: "1:1", "16:9", "9:16"'),
      }),
      execute: async (args) => {
        const { resolved, error } = resolveModel(args.model);
        if (error) return { success: false, error };

        const provider = detectProvider(resolved);
        if (provider === 'gemini' || MULTIMODAL_IMAGE_MODELS.has(resolved)) {
          return generateWithGemini(resolved, resolved, {
            referenceImages: args.reference_images,
            aspectRatio: args.aspect_ratio,
          });
        }
        return generateWithOpenAI(args.prompt, resolved, {
          size: args.size,
          quality: args.quality,
          referenceImages: args.reference_images,
        });
      },
    }),

    defineTool<{ images: string[]; prompt: string; model?: string; size?: string }, ImageGenResult>({
      name: 'edit_image',
      description: 'Edit existing images using AI. Provide reference images and describe the desired changes.',
      parameters: z.object({
        images: z.array(z.string()).describe('File paths of images to edit'),
        prompt: z.string().describe('Description of desired edits'),
        model: z.string().optional().describe('Model to use (default: gpt-image-2)'),
        size: z.string().optional().describe('Output size'),
      }),
      execute: async (args) => {
        const model = args.model ?? 'gpt-image-2';
        const provider = detectProvider(model);
        if (provider === 'gemini' || MULTIMODAL_IMAGE_MODELS.has(model)) {
          return generateWithGemini(args.prompt, model, { referenceImages: args.images });
        }
        return generateWithOpenAI(args.prompt, model, {
          size: args.size,
          referenceImages: args.images,
        });
      },
    }),

    defineTool<{ prompt: string; style?: string; model?: string }, ImageGenResult>({
      name: 'generate_diagram',
      description: 'Generate a diagram or scientific figure. Specify panels, labels, arrows, layout, and visual hierarchy.',
      parameters: z.object({
        prompt: z.string().describe('Detailed diagram description with layout, labels, arrows, colors'),
        style: z.string().optional().describe('Visual style: "scientific", "flowchart", "architecture", "infographic"'),
        model: z.string().optional().describe('Model to use'),
      }),
      execute: async (args) => {
        const stylePrefix = args.style ? `Create a ${args.style} diagram: ` : 'Create a clear, professional diagram: ';
        const fullPrompt = stylePrefix + args.prompt + '. Use clean lines, readable labels, and professional colors.';
        const { resolved, error } = resolveModel(args.model);
        if (error) return { success: false, error };
        const provider = detectProvider(resolved);
        if (provider === 'gemini' || MULTIMODAL_IMAGE_MODELS.has(resolved)) {
          return generateWithGemini(fullPrompt, resolved, {});
        }
        return generateWithOpenAI(fullPrompt, resolved, { size: '1536x1024', quality: 'high' });
      },
    }),
  ];
}
