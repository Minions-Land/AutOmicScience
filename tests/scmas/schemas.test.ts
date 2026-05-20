import { describe, it, expect } from 'vitest';
import {
  Stage2SelectionResponse,
  Stage3AdapterSpec,
  ALLOWED_ACTIONS,
  Stage4AdjudicationResponse,
  validateAllowedLabels,
} from '../../src/plugins/scmas/schemas/index.js';

describe('Stage2SelectionResponse', () => {
  it('parses a minimal valid LLM response', () => {
    const parsed = Stage2SelectionResponse.parse({
      thought_summary: 'pick by gene fit',
      selected_pairs: [
        { rank: 1, model_id: 'geneformer_raw_knn', source_id: 'allen_mouse_xyz', rationale: 'shared genes' },
      ],
    });
    expect(parsed.selected_pairs).toHaveLength(1);
    expect(parsed.review_flags).toEqual([]);
    expect(parsed.rejected_pair_notes).toEqual([]);
  });

  it('rejects empty selected_pairs', () => {
    expect(() => Stage2SelectionResponse.parse({ selected_pairs: [] })).toThrow();
  });
});

describe('Stage3AdapterSpec', () => {
  it('accepts every action in the allowlist', () => {
    for (const action of ALLOWED_ACTIONS) {
      const parsed = Stage3AdapterSpec.parse({ actions: [{ action_name: action }] });
      expect(parsed.actions[0].action_name).toBe(action);
    }
  });

  it('rejects unknown action_name', () => {
    expect(() =>
      Stage3AdapterSpec.parse({ actions: [{ action_name: 'execute_arbitrary_code' }] }),
    ).toThrow();
  });
});

describe('Stage4AdjudicationResponse', () => {
  it('validates allowed_labels post-parse', () => {
    const parsed = Stage4AdjudicationResponse.parse({
      groups: [
        { group_id: 'g1', selected_label: 'Astro', confidence: 0.9, rationale: 'high vote' },
        { group_id: 'g2', selected_label: 'Wat', confidence: 0.3, rationale: '' },
      ],
    });
    const check = validateAllowedLabels(parsed, ['Astro', 'Micro', 'Unknown']);
    expect(check.ok).toBe(false);
    expect(check.invalid).toEqual([{ group_id: 'g2', selected_label: 'Wat' }]);
  });
});
