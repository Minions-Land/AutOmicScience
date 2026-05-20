import { z } from 'zod';

/**
 * The exhaustive set of action names accepted by the Stage-3 deterministic
 * reviewer. Lifted verbatim from
 * `_import_scMAS/CanChen_MAS/src/scmas/stage3/adapter_executor.py::ALLOWED_ACTIONS`.
 *
 * Any AdapterSpec returned by the Stage-3 LLM agent that contains an
 * action_name outside this set will be rejected by the Python reviewer.
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

/**
 * AdapterSpec action object — kept loose because the Python reviewer
 * already enforces the strict shape. We require an action_name from the
 * allowed list and let the rest pass through.
 */
export const Stage3Action = z
  .object({
    action_name: AllowedActionEnum,
  })
  .passthrough();
export type Stage3Action = z.infer<typeof Stage3Action>;

/**
 * Stage-3 AdapterSpec response. The Python reviewer additionally checks
 * immutable_fields, input_artifacts, runtime_payload defaults and that
 * the action sequence matches the deterministic_draft. We mirror the
 * top-level structure here for early validation.
 */
export const Stage3AdapterSpec = z
  .object({
    actions: z.array(Stage3Action).min(1),
    immutable_fields: z.record(z.unknown()).optional(),
    input_artifacts: z.record(z.unknown()).optional(),
    runtime_payload: z.record(z.unknown()).optional(),
  })
  .passthrough();
export type Stage3AdapterSpec = z.infer<typeof Stage3AdapterSpec>;
