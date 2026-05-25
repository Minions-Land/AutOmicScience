/**
 * BuiltinToolSets — Factory functions for all built-in toolsets.
 * Each function returns a configured ToolSet ready for agent use.
 */

// Re-export the original simple toolsets (kept for backward compatibility)
export { shellToolSet } from './ShellTools.js';
export { codeToolSet } from './CodeTools.js';
export { fileToolSet } from './FileTools.js';
export { pythonToolSet } from './PythonTools.js';
export { notebookToolSet } from './NotebookTools.js';
export { webToolSet } from './WebTools.js';
export { databaseToolSet } from './DatabaseTools.js';
export { taskToolSet } from './TaskTools.js';
export { fileTransferToolSet } from './FileTransferTools.js';
export { knowledgeToolSet } from './KnowledgeTools.js';
export { scfmToolSet } from './ScfmTools.js';
export { rToolSet } from './RTools.js';
export { juliaToolSet } from './JuliaTools.js';

// Re-export domain-specific toolsets
export { bioDataToolSet } from './BioDataTools.js';
export { bioMasToolSet } from './BioMasTools.js';
export { syntheticDataToolSet } from './SyntheticDataTools.js';
export { benchmarkToolSet } from './BenchmarkTools.js';
export { annotationStageToolSet } from './AnnotationStageTools.js';
export { evolutionToolSet } from './EvolutionTools.js';
export { agentToolSet } from './AgentTools.js';
export { skillToolSet } from './SkillTools.js';
export { ToolCatalog, toolSearchToolSet } from './ToolSearch.js';

import { ToolSet } from './ToolSet.js';
import { shellToolSet } from './ShellTools.js';
import { codeToolSet } from './CodeTools.js';
import { fileToolSet } from './FileTools.js';
import { pythonToolSet } from './PythonTools.js';
import { notebookToolSet } from './NotebookTools.js';
import { webToolSet } from './WebTools.js';
import { databaseToolSet } from './DatabaseTools.js';
import { taskToolSet } from './TaskTools.js';
import { fileTransferToolSet } from './FileTransferTools.js';
import { knowledgeToolSet } from './KnowledgeTools.js';
import { scfmToolSet } from './ScfmTools.js';
import { rToolSet } from './RTools.js';
import { juliaToolSet } from './JuliaTools.js';
import { bioDataToolSet } from './BioDataTools.js';
import { bioMasToolSet } from './BioMasTools.js';
import { syntheticDataToolSet } from './SyntheticDataTools.js';
import { benchmarkToolSet } from './BenchmarkTools.js';
import { annotationStageToolSet } from './AnnotationStageTools.js';
import { evolutionToolSet } from './EvolutionTools.js';
import { agentToolSet } from './AgentTools.js';
import { ToolCatalog, toolSearchToolSet } from './ToolSearch.js';
import { PermissionManager } from '../permissions/index.js';
import type { TaskManager } from '../task/index.js';

export interface DefaultToolSetOptions {
  /** Root directory for file/code operations. */
  rootDir?: string;
  /** Agent name (used for file transfer). */
  agentName?: string;
  /** Which toolset categories to include. Default: all. */
  include?: (
    | 'shell'
    | 'code'
    | 'file'
    | 'python'
    | 'notebook'
    | 'web'
    | 'database'
    | 'task'
    | 'file_transfer'
    | 'knowledge'
    | 'scfm'
    | 'bio_data'
    | 'bio_mas'
    | 'synthetic_data'
    | 'benchmark'
    | 'annotation_stage'
    | 'evolution'
    | 'r'
    | 'julia'
  )[];
  /** Toolset categories to exclude. */
  exclude?: string[];
  permissionManager?: PermissionManager;
  taskManager?: TaskManager;
}

/**
 * Create the default set of all built-in toolsets merged into one.
 * Use `include` or `exclude` to control which categories are active.
 */
export function createDefaultToolSet(opts: DefaultToolSetOptions = {}): ToolSet {
  const rootDir = opts.rootDir ?? process.cwd();
  const agentName = opts.agentName ?? 'agent';

  const all: Record<string, () => ToolSet> = {
    shell: () => shellToolSet({ cwd: rootDir, permissionManager: opts.permissionManager }),
    code: () => codeToolSet({ rootDir }),
    file: () => fileToolSet({ rootDir, permissionManager: opts.permissionManager }),
    python: () => pythonToolSet({ cwd: rootDir }),
    notebook: () => notebookToolSet({ notebookDir: rootDir }),
    web: () => webToolSet(),
    database: () => databaseToolSet(),
    task: () => taskToolSet({ manager: opts.taskManager }),
    file_transfer: () => fileTransferToolSet({ agentName }),
    knowledge: () => knowledgeToolSet({ rootDir }),
    scfm: () => scfmToolSet(),
    bio_data: () => bioDataToolSet(),
    bio_mas: () => bioMasToolSet(),
    synthetic_data: () => syntheticDataToolSet(),
    benchmark: () => benchmarkToolSet(),
    annotation_stage: () => annotationStageToolSet(),
    evolution: () => evolutionToolSet(),
    r: () => rToolSet({ cwd: rootDir }),
    julia: () => juliaToolSet({ cwd: rootDir }),
  };

  const include = opts.include ?? (Object.keys(all) as (keyof typeof all)[]);
  const exclude = new Set(opts.exclude ?? []);

  const merged = new ToolSet('default', [], { permissionManager: opts.permissionManager });
  for (const key of include) {
    if (exclude.has(key)) continue;
    const factory = all[key];
    if (factory) {
      merged.merge(factory());
    }
  }
  const catalog = new ToolCatalog();
  for (const key of Object.keys(all)) {
    catalog.register(key, all[key], { preload: include.includes(key as keyof typeof all) && !exclude.has(key) });
  }
  merged.merge(toolSearchToolSet(catalog));
  merged.merge(agentToolSet());

  return merged;
}
