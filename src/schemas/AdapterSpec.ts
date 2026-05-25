import { z } from 'zod';

/**
 * AdapterSpec schema — the executable plan an LLM Adapter agent returns
 * for a single (source, model) pair. Actions are drawn from a fixed
 * allowlist; the deterministic reviewer at the Python boundary rejects
 * anything outside it.
 */
export const ALLOWED_ACTIONS = [
  'load_query_npz_kukanja',
  'load_query_h5ad',
  'load_reference_standard_bundle',
  'align_shared_genes',
  'write_raw_label_transfer_input',
  'invoke_raw_embedding_transfer',
  'invoke_postprocessor',
  'skip_with_reason',
] as const;

export type AllowedAction = (typeof ALLOWED_ACTIONS)[number];
export const AllowedActionEnum = z.enum(ALLOWED_ACTIONS);

export const AdapterAction = z
  .object({
    action_name: AllowedActionEnum,
  })
  .passthrough();
export type AdapterAction = z.infer<typeof AdapterAction>;

export const AdapterSpec = z
  .object({
    actions: z.array(AdapterAction).min(1),
    immutable_fields: z.record(z.unknown()).optional(),
    input_artifacts: z.record(z.unknown()).optional(),
    runtime_payload: z.record(z.unknown()).optional(),
  })
  .passthrough();
export type AdapterSpec = z.infer<typeof AdapterSpec>;
