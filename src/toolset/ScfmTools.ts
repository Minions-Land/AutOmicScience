/**
 * ScfmTools — Single-cell foundation model tools.
 * All heavy compute runs via PythonBridge; this layer provides the typed
 * tool interface and model registry.
 */

import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';
import { runPython, type BridgeOptions } from '../bridge/PythonBridge.js';

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

export interface ScfmModelInfo {
  id: string;
  name: string;
  description: string;
  capabilities: string[];
  inputType: string;
  outputType: string;
  paperUrl?: string;
}

const MODEL_REGISTRY: ScfmModelInfo[] = [
  {
    id: 'geneformer',
    name: 'Geneformer',
    description: 'Transformer-based foundation model for single-cell transcriptomics. Pre-trained on ~30M cells.',
    capabilities: ['embeddings', 'cell_annotation', 'gene_network', 'perturbation_prediction'],
    inputType: 'h5ad (raw counts)',
    outputType: 'embeddings, predictions',
    paperUrl: 'https://doi.org/10.1038/s41586-023-06139-9',
  },
  {
    id: 'scgpt',
    name: 'scGPT',
    description: 'Generative pre-trained transformer for single-cell multi-omics. Supports gene expression and chromatin accessibility.',
    capabilities: ['embeddings', 'cell_annotation', 'batch_correction', 'gene_perturbation'],
    inputType: 'h5ad (raw counts)',
    outputType: 'embeddings, predictions',
    paperUrl: 'https://doi.org/10.1038/s41592-024-02201-0',
  },
  {
    id: 'uce',
    name: 'UCE (Universal Cell Embeddings)',
    description: 'Universal cell embedding model supporting cross-species and cross-tissue transfer.',
    capabilities: ['embeddings', 'cell_annotation', 'cross_species'],
    inputType: 'h5ad (raw counts)',
    outputType: 'embeddings',
  },
  {
    id: 'nicheformer',
    name: 'Nicheformer',
    description: 'Foundation model for spatial transcriptomics incorporating niche/neighborhood context.',
    capabilities: ['embeddings', 'cell_annotation', 'spatial_analysis'],
    inputType: 'h5ad (spatial)',
    outputType: 'embeddings, spatial predictions',
  },
  {
    id: 'scvi',
    name: 'scVI',
    description: 'Variational inference framework for single-cell gene expression data.',
    capabilities: ['embeddings', 'batch_correction', 'differential_expression', 'imputation'],
    inputType: 'h5ad (raw counts)',
    outputType: 'latent embeddings, corrected expression',
  },
  {
    id: 'scanvi',
    name: 'scANVI',
    description: 'Semi-supervised variant of scVI for cell type annotation with reference labels.',
    capabilities: ['cell_annotation', 'label_transfer', 'batch_correction'],
    inputType: 'h5ad (raw counts + labels)',
    outputType: 'cell type predictions, embeddings',
  },
];

// ---------------------------------------------------------------------------
// Toolset factory
// ---------------------------------------------------------------------------

export interface ScfmToolsOptions {
  /** PythonBridge options for model execution. */
  bridgeOptions?: BridgeOptions;
  /** Additional models to register. */
  extraModels?: ScfmModelInfo[];
}

export function scfmToolSet(opts: ScfmToolsOptions = {}): ToolSet {
  const bridgeOpts = opts.bridgeOptions ?? {};
  const registry = [...MODEL_REGISTRY, ...(opts.extraModels ?? [])];

  return new ToolSet('scfm', [
    // -----------------------------------------------------------------------
    // list_models
    // -----------------------------------------------------------------------
    defineTool<
      { capability?: string },
      { models: ScfmModelInfo[] }
    >({
      name: 'list_models',
      description:
        'List available single-cell foundation models. Optionally filter by capability ' +
        '(embeddings, cell_annotation, batch_correction, etc.).',
      parameters: z.object({
        capability: z.string().optional().describe('Filter by capability name'),
      }),
      execute: async ({ capability }) => {
        if (capability) {
          return { models: registry.filter((m) => m.capabilities.includes(capability)) };
        }
        return { models: registry };
      },
    }),

    // -----------------------------------------------------------------------
    // run_model
    // -----------------------------------------------------------------------
    defineTool<
      { modelId: string; inputPath: string; outputPath: string; params?: Record<string, unknown> },
      { ok: boolean; outputPath: string; stdout: string; stderr: string }
    >({
      name: 'run_model',
      description:
        'Run a foundation model on input data (h5ad file). The model is executed via ' +
        'the Python runtime. Returns the output file path.',
      parameters: z.object({
        modelId: z.string().describe('Model ID from list_models'),
        inputPath: z.string().describe('Path to input h5ad file'),
        outputPath: z.string().describe('Path for output file'),
        params: z.record(z.unknown()).optional().describe('Model-specific parameters'),
      }),
      execute: async ({ modelId, inputPath, outputPath, params }) => {
        const model = registry.find((m) => m.id === modelId);
        if (!model) {
          throw new Error(`Model '${modelId}' not found. Use list_models to see available models.`);
        }

        const paramsJson = JSON.stringify(params ?? {});
        const result = await runPython(
          'run-model',
          [
            ['--model', modelId],
            ['--input', inputPath],
            ['--output', outputPath],
            ['--params', paramsJson],
          ],
          bridgeOpts,
        );

        if (result.exitCode !== 0) {
          throw new Error(`Model execution failed (exit ${result.exitCode}): ${result.stderr}`);
        }

        return {
          ok: true,
          outputPath,
          stdout: result.stdout,
          stderr: result.stderr,
        };
      },
    }),

    // -----------------------------------------------------------------------
    // get_embeddings
    // -----------------------------------------------------------------------
    defineTool<
      { modelId: string; inputPath: string; outputPath: string; layer?: string; batchSize?: number },
      { ok: boolean; outputPath: string; numCells: number; embeddingDim: number; stdout: string }
    >({
      name: 'get_embeddings',
      description:
        'Extract cell embeddings from a foundation model. Outputs an h5ad file with ' +
        'embeddings stored in obsm.',
      parameters: z.object({
        modelId: z.string().describe('Model ID (must support "embeddings" capability)'),
        inputPath: z.string().describe('Path to input h5ad file'),
        outputPath: z.string().describe('Path for output h5ad with embeddings'),
        layer: z.string().optional().describe('AnnData layer to use (default: X)'),
        batchSize: z.number().int().positive().optional().describe('Batch size for inference'),
      }),
      execute: async ({ modelId, inputPath, outputPath, layer, batchSize }) => {
        const model = registry.find((m) => m.id === modelId);
        if (!model) throw new Error(`Model '${modelId}' not found.`);
        if (!model.capabilities.includes('embeddings')) {
          throw new Error(`Model '${modelId}' does not support embeddings.`);
        }

        const result = await runPython(
          'get-embeddings',
          [
            ['--model', modelId],
            ['--input', inputPath],
            ['--output', outputPath],
            ['--layer', layer ?? ''],
            ['--batch-size', batchSize ?? 64],
          ],
          bridgeOpts,
        );

        if (result.exitCode !== 0) {
          throw new Error(`Embedding extraction failed: ${result.stderr}`);
        }

        // Parse output for metadata
        const parsed = result.parsedJson as { num_cells?: number; embedding_dim?: number } | undefined;

        return {
          ok: true,
          outputPath,
          numCells: parsed?.num_cells ?? -1,
          embeddingDim: parsed?.embedding_dim ?? -1,
          stdout: result.stdout,
        };
      },
    }),

    // -----------------------------------------------------------------------
    // annotate_cells
    // -----------------------------------------------------------------------
    defineTool<
      { modelId: string; inputPath: string; outputPath: string; referenceDataset?: string; labelKey?: string },
      { ok: boolean; outputPath: string; numAnnotated: number; stdout: string }
    >({
      name: 'annotate_cells',
      description:
        'Run cell type annotation using a foundation model. Requires a model with ' +
        '"cell_annotation" capability. Optionally uses a reference dataset for label transfer.',
      parameters: z.object({
        modelId: z.string().describe('Model ID (must support "cell_annotation")'),
        inputPath: z.string().describe('Path to input h5ad file'),
        outputPath: z.string().describe('Path for annotated output h5ad'),
        referenceDataset: z.string().optional().describe('Path to reference h5ad with labels'),
        labelKey: z.string().optional().describe('obs column in reference with cell type labels'),
      }),
      execute: async ({ modelId, inputPath, outputPath, referenceDataset, labelKey }) => {
        const model = registry.find((m) => m.id === modelId);
        if (!model) throw new Error(`Model '${modelId}' not found.`);
        if (!model.capabilities.includes('cell_annotation')) {
          throw new Error(`Model '${modelId}' does not support cell_annotation.`);
        }

        const result = await runPython(
          'annotate-cells',
          [
            ['--model', modelId],
            ['--input', inputPath],
            ['--output', outputPath],
            ['--reference', referenceDataset ?? ''],
            ['--label-key', labelKey ?? 'cell_type'],
          ],
          bridgeOpts,
        );

        if (result.exitCode !== 0) {
          throw new Error(`Cell annotation failed: ${result.stderr}`);
        }

        const parsed = result.parsedJson as { num_annotated?: number } | undefined;

        return {
          ok: true,
          outputPath,
          numAnnotated: parsed?.num_annotated ?? -1,
          stdout: result.stdout,
        };
      },
    }),
  ]);
}
