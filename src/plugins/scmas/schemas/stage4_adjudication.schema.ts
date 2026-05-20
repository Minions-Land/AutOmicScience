import { z } from 'zod';

/**
 * Stage-4 low-consistency cell adjudication schema.
 *
 * Mirrors the `response_contract` defined in
 * `_import_scMAS/CanChen_MAS/src/scmas/stage4/llm_adjudicator.py::adjudicate_low_consistency_cells`.
 *
 * `selected_label` MUST be one of the allowed_labels supplied at runtime
 * (we cannot enforce that in the static schema since the set is dynamic).
 * The deterministic Python reviewer enforces that bound.
 */
export const Stage4GroupDecision = z.object({
  group_id: z.string().min(1),
  selected_label: z.string().min(1),
  confidence: z.number().min(0).max(1),
  rationale: z.string().default(''),
});
export type Stage4GroupDecision = z.infer<typeof Stage4GroupDecision>;

export const Stage4AdjudicationResponse = z.object({
  groups: z.array(Stage4GroupDecision).min(1),
});
export type Stage4AdjudicationResponse = z.infer<typeof Stage4AdjudicationResponse>;

/**
 * Validate decisions against a runtime-known allowed_labels set.
 * Returns invalid group_ids if any selected_label is outside the allowlist.
 */
export function validateAllowedLabels(
  resp: Stage4AdjudicationResponse,
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
