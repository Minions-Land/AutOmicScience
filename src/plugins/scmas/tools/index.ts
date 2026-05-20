export * from './PythonBridge.js';
export { dataToolSet } from './DataTools.js';
export { scDesign3ToolSet } from './ScDesign3Tools.js';
export { evalToolSet } from './EvalTools.js';
export { stageToolSet } from './StageTools.js';

import { ToolSet } from '../../../toolset/ToolSet.js';
import { dataToolSet } from './DataTools.js';
import { scDesign3ToolSet } from './ScDesign3Tools.js';
import { evalToolSet } from './EvalTools.js';
import { stageToolSet } from './StageTools.js';
import type { BridgeOptions } from './PythonBridge.js';

/** Convenience: a single ToolSet containing every scMAS subprocess tool. */
export function scmasToolSet(opt: BridgeOptions = {}): ToolSet {
  return new ToolSet('scmas')
    .merge(dataToolSet(opt))
    .merge(scDesign3ToolSet(opt))
    .merge(evalToolSet(opt))
    .merge(stageToolSet(opt));
}
