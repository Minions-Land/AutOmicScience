/**
 * Smart text truncation utilities.
 * Preserves meaningful content boundaries when cutting text.
 */

/**
 * Truncate text to maxLength, appending a suffix to indicate truncation.
 * Tries to break at a word boundary when possible.
 */
export function truncateText(text: string, maxLength: number, suffix = '...'): string {
  if (text.length <= maxLength) return text;

  const available = maxLength - suffix.length;
  if (available <= 0) return suffix.slice(0, maxLength);

  // Try to break at a word boundary
  let cutPoint = available;
  const lastSpace = text.lastIndexOf(' ', available);
  if (lastSpace > available * 0.7) {
    cutPoint = lastSpace;
  }

  return text.slice(0, cutPoint) + suffix;
}

/**
 * Truncate text keeping both the start and end, with an indicator in the middle.
 * Useful for showing context from both ends of a long string.
 */
export function truncateMiddle(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;

  const indicator = '\n...[truncated]...\n';
  const available = maxLength - indicator.length;
  if (available <= 0) return text.slice(0, maxLength);

  const half = Math.floor(available / 2);
  const head = text.slice(0, half);
  const tail = text.slice(-half);

  return head + indicator + tail;
}

/**
 * Truncate text by line count, keeping the first maxLines lines.
 * Appends a note about how many lines were omitted.
 */
export function truncateLines(text: string, maxLines: number): string {
  const lines = text.split('\n');
  if (lines.length <= maxLines) return text;

  const kept = lines.slice(0, maxLines);
  const omitted = lines.length - maxLines;
  kept.push(`\n... (${omitted} more line${omitted === 1 ? '' : 's'})`);
  return kept.join('\n');
}

/**
 * Truncate a string preserving head and tail with size info.
 * Matches PantheonOS truncate_string behavior.
 */
export function truncateString(content: string, maxLength: number): string {
  if (content.length <= maxLength) return content;

  const suffix = `\n[truncated ${(content.length - maxLength).toLocaleString()}/${content.length.toLocaleString()} chars]`;
  const available = maxLength - suffix.length - 20;

  if (available < 100) {
    const simpleMax = Math.max(0, maxLength - suffix.length);
    return content.slice(0, simpleMax) + suffix;
  }

  const half = Math.floor(available / 2);
  const head = content.slice(0, half);
  const tail = content.slice(-half);

  return `${head}\n\n...truncated...\n\n${tail}${suffix}`;
}
