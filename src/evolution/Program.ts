/** A program in the evolutionary search: a piece of code + metadata. */
export interface Program {
  id: string;
  code: string;
  generation: number;
  parentIds: string[];
  fitness?: number;
  metadata?: Record<string, unknown>;
}

export function newProgram(code: string, parentIds: string[] = [], generation = 0): Program {
  return {
    id: `prog_${Math.random().toString(36).slice(2, 10)}`,
    code,
    generation,
    parentIds,
  };
}
