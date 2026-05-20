import { promises as fs } from 'node:fs';
import path from 'node:path';

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg']);
const IMAGE_URL_PATTERN = /\.(png|jpe?g|gif|webp|bmp|svg|tiff?)(\?.*)?$/i;
const BASE64_IMAGE_PATTERN = /^data:image\/(png|jpe?g|gif|webp|bmp|svg\+xml);base64,/;

/** Detect if a URL points to an image based on extension or content-type hints. */
export function isImageUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return IMAGE_URL_PATTERN.test(parsed.pathname);
  } catch {
    // Not a valid URL, check raw string
    return IMAGE_URL_PATTERN.test(url);
  }
}

/** Detect if a string is a base64-encoded image data URI. */
export function isBase64Image(data: string): boolean {
  return BASE64_IMAGE_PATTERN.test(data);
}

/** Detect MIME type from magic bytes in a buffer. */
export function detectImageMimeType(data: Buffer): string {
  if (data.length < 4) return 'application/octet-stream';

  // PNG: 89 50 4E 47
  if (data[0] === 0x89 && data[1] === 0x50 && data[2] === 0x4e && data[3] === 0x47) {
    return 'image/png';
  }
  // JPEG: FF D8 FF
  if (data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff) {
    return 'image/jpeg';
  }
  // GIF: 47 49 46 38
  if (data[0] === 0x47 && data[1] === 0x49 && data[2] === 0x46 && data[3] === 0x38) {
    return 'image/gif';
  }
  // WebP: 52 49 46 46 ... 57 45 42 50
  if (data[0] === 0x52 && data[1] === 0x49 && data[2] === 0x46 && data[3] === 0x46 && data.length >= 12) {
    if (data[8] === 0x57 && data[9] === 0x45 && data[10] === 0x42 && data[11] === 0x50) {
      return 'image/webp';
    }
  }
  // BMP: 42 4D
  if (data[0] === 0x42 && data[1] === 0x4d) {
    return 'image/bmp';
  }

  return 'application/octet-stream';
}

/**
 * Convert a local image file to a base64 data URI string.
 * Returns `data:<mime>;base64,<encoded>`.
 */
export async function imageToBase64(filePath: string): Promise<string> {
  const data = await fs.readFile(filePath);
  const mime = detectImageMimeType(data);
  const encoded = data.toString('base64');
  return `data:${mime};base64,${encoded}`;
}

/**
 * Check if an image buffer likely exceeds a max dimension and return it
 * (potentially resized). Without a native image library, this returns the
 * buffer as-is but logs a warning if the file size suggests a very large image.
 *
 * For actual resizing, integrate `sharp` or a similar library.
 */
export async function resizeImageIfNeeded(data: Buffer, maxDimension: number): Promise<Buffer> {
  // Without a native image processing library, we can only do a heuristic check.
  // A 1568x1568 RGBA PNG is ~9.4MB uncompressed. If the buffer is very large,
  // the caller should consider using `sharp` for real resizing.
  // For now, return as-is — this is a placeholder for when sharp is available.
  try {
    const sharp = await import('sharp');
    const metadata = await sharp.default(data).metadata();
    const width = metadata.width ?? 0;
    const height = metadata.height ?? 0;
    if (width > maxDimension || height > maxDimension) {
      return await sharp.default(data)
        .resize(maxDimension, maxDimension, { fit: 'inside' })
        .toBuffer();
    }
  } catch {
    // sharp not available — return original buffer
  }
  return data;
}

/** Get the image extension from a file path. */
export function getImageExtension(filePath: string): string | null {
  const ext = path.extname(filePath).toLowerCase();
  return IMAGE_EXTENSIONS.has(ext) ? ext : null;
}
