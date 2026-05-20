import { z } from 'zod';

/**
 * Schema for the Stage-2 selector LLM JSON output.
 *
 * Mirrors the `response_contract` defined in
 * `_import_scMAS/CanChen_MAS/src/scmas/stage2/selector.py::_render_llm_prompt`.
 *
 * The Python deterministic reviewer enforces stricter rules at runtime
 * (uniqueness of model_id, family caps, exact length per top_k). We keep
 * the zod schema permissive on those — the bridge / Python reviewer is
 * still the source of truth — but require the structural fields.
 */
export const Stage2SelectedPair = z.object({
  rank: z.number().int().positive(),
  model_id: z.string().min(1),
  source_id: z.string().min(1),
  rationale: z.string().min(1),
});
export type Stage2SelectedPair = z.infer<typeof Stage2SelectedPair>;

export const Stage2RejectedPairNote = z.object({
  model_id: z.string().min(1),
  source_id: z.string().min(1),
  reason: z.string().min(1),
});
export type Stage2RejectedPairNote = z.infer<typeof Stage2RejectedPairNote>;

export const Stage2SelectionResponse = z.object({
  thought_summary: z.string().default(''),
  selected_pairs: z.array(Stage2SelectedPair).min(1),
  rejected_pair_notes: z.array(Stage2RejectedPairNote).default([]),
  review_flags: z.array(z.string()).default([]),
});
export type Stage2SelectionResponse = z.infer<typeof Stage2SelectionResponse>;
