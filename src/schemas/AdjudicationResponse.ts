import { z } from 'zod';

/**
 * AdjudicationResponse schema — what an LLM Adjudicator agent returns when
 * resolving low-consistency cell groups. `selected_label` must be one of
 * the allowed_labels supplied at runtime; the static schema can't enforce
 * that, so use `validateAllowedLabels` after parsing.
 */
export const GroupDecision = z.object({
  group_id: z.string().min(1),
  selected_label: z.string().min(1),
  confidence: z.number().min(0).max(1),
  rationale: z.string().default(''),
});
export type GroupDecision = z.infer<typeof GroupDecision>;

export const AdjudicationResponse = z.object({
  groups: z.array(GroupDecision).min(1),
});
export type AdjudicationResponse = z.infer<typeof AdjudicationResponse>;

/**
 * Returns invalid group_ids if any selected_label is outside the allowlist.
 */
export function validateAllowedLabels(
  resp: AdjudicationResponse,
  allowedLabels: readonly string[],
): { ok: boolean; invalid: { group_id: string; selected_label: string }[] } {
  const allowed = new Set(allowedLabels);
  const invalid: { group_id: string; selected_label: string }[] = [];
  for (const g of resp.groups) {
    if (!allowed.has(g.selected_label)) {
      invalid.push({ group_id: g.group_id, selected_label: g.selected_label });
    }
  }
  return { ok: invalid.length === 0, invalid };
}
