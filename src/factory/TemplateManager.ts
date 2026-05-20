import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export interface AgentTemplate {
  name: string;
  model: string | string[];
  systemPrompt?: string;
  toolsets?: string[];
  skills?: string[];
  mcp?: { name: string; command?: string; args?: string[]; url?: string }[];
}

export interface TeamTemplate {
  name: string;
  pattern: 'sequential' | 'swarm' | 'coordinator';
  members: string[]; // agent template names
  coordinator?: string;
}

/**
 * Manages a `~/.novaeve/` directory holding agent/team templates as JSON.
 * Markdown skills can also live alongside under `skills/`.
 */
export class TemplateManager {
  public readonly root: string;

  constructor(root?: string) {
    this.root = root ?? path.join(os.homedir(), '.novaeve');
  }

  async init(): Promise<void> {
    await fs.mkdir(path.join(this.root, 'agents'), { recursive: true });
    await fs.mkdir(path.join(this.root, 'teams'), { recursive: true });
    await fs.mkdir(path.join(this.root, 'skills'), { recursive: true });
  }

  agentPath(name: string): string {
    return path.join(this.root, 'agents', `${name}.json`);
  }

  teamPath(name: string): string {
    return path.join(this.root, 'teams', `${name}.json`);
  }

  skillsDir(): string {
    return path.join(this.root, 'skills');
  }

  async saveAgent(t: AgentTemplate): Promise<void> {
    await this.init();
    await fs.writeFile(this.agentPath(t.name), JSON.stringify(t, null, 2));
  }

  async loadAgent(name: string): Promise<AgentTemplate> {
    return JSON.parse(await fs.readFile(this.agentPath(name), 'utf8')) as AgentTemplate;
  }

  async listAgents(): Promise<string[]> {
    try {
      const entries = await fs.readdir(path.join(this.root, 'agents'));
      return entries.filter((e) => e.endsWith('.json')).map((e) => e.replace(/\.json$/, ''));
    } catch {
      return [];
    }
  }

  async saveTeam(t: TeamTemplate): Promise<void> {
    await this.init();
    await fs.writeFile(this.teamPath(t.name), JSON.stringify(t, null, 2));
  }

  async loadTeam(name: string): Promise<TeamTemplate> {
    return JSON.parse(await fs.readFile(this.teamPath(name), 'utf8')) as TeamTemplate;
  }
}
