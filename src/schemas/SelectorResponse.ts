import { z } from 'zod';

/**
 * Selector response schema — what an LLM Selector agent returns when
 * picking (source, model) execution pairs for an annotation query.
 *
 * The deterministic reviewer at the Python boundary enforces stricter
 * rules (uniqueness, family caps, exact length per top_k); this schema
 * validates structure only.
 */
export const SelectedPair = z.object({
  rank: z.number().int().positive(),
  model_id: z.string().min(1),
  source_id: z.string().min(1),
  rationale: z.string().min(1),
});
export type SelectedPair = z.infer<typeof SelectedPair>;

export const RejectedPairNote = z.object({
  model_id: z.string().min(1),
  source_id: z.string().min(1),
  reason: z.string().min(1),
});
export type RejectedPairNote = z.infer<typeof RejectedPairNote>;

export const SelectorResponse = z.object({
  thought_summary: z.string().default(''),
  selected_pairs: z.array(SelectedPair).min(1),
  rejected_pair_notes: z.array(RejectedPairNote).default([]),
  review_flags: z.array(z.string()).default([]),
});
export type SelectorResponse = z.infer<typeof SelectorResponse>;
