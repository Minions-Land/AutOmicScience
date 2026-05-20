export function renderTemplate(template: string, vars: Record<string, string>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (_, key) => vars[key] ?? `{{${key}}}`);
}

export function extractTemplateVars(template: string): string[] {
  const matches = template.matchAll(/\{\{(\w+)\}\}/g);
  const vars = new Set<string>();
  for (const m of matches) vars.add(m[1]);
  return [...vars];
}

export function validateTemplate(template: string, vars: Record<string, string>): string[] {
  const required = extractTemplateVars(template);
  return required.filter((v) => !(v in vars));
}

export interface ParsedFrontmatter {
  meta: Record<string, any>;
  body: string;
}

export function parseFrontmatter(content: string): ParsedFrontmatter {
  const match = content.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) return { meta: {}, body: content };
  const meta: Record<string, any> = {};
  for (const line of match[1].split('\n')) {
    const idx = line.indexOf(':');
    if (idx < 0) continue;
    const key = line.slice(0, idx).trim();
    let value: any = line.slice(idx + 1).trim();
    if (value === 'true') value = true;
    else if (value === 'false') value = false;
    else if (/^\d+$/.test(value)) value = parseInt(value, 10);
    else if (value.startsWith('[')) try { value = JSON.parse(value); } catch { /* keep string */ }
    meta[key] = value;
  }
  return { meta, body: match[2] };
}

export function serializeFrontmatter(meta: Record<string, any>, body: string): string {
  const lines = Object.entries(meta).map(([k, v]) => {
    if (Array.isArray(v)) return `${k}: ${JSON.stringify(v)}`;
    return `${k}: ${v}`;
  });
  return `---\n${lines.join('\n')}\n---\n${body}`;
}
