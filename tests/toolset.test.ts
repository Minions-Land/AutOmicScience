import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import { mkdtemp, readFile, rm, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { ToolCatalog, ToolSet, createDefaultToolSet, defineTool, tool, taskToolSet, toolSearchToolSet } from '../src/toolset/index.js';
import { FileCommandLoader, FilePermissionStore, PermissionManager, PluginLoader } from '../src/index.js';
import { FileSessionStore } from '../src/session/index.js';

describe('ToolSet', () => {
  it('registers and executes a tool', async () => {
    const ts = new ToolSet('t');
    ts.register(
      defineTool({
        name: 'add',
        description: 'add two numbers',
        parameters: z.object({ a: z.number(), b: z.number() }),
        execute: async ({ a, b }) => a + b,
      }),
    );
    expect(ts.size()).toBe(1);
    const res = await ts.execute('add', { a: 2, b: 3 }, { agentName: 't' });
    expect(res.content).toBe('5');
  });

  it('toOpenAITools() returns function-typed defs', () => {
    const ts = new ToolSet('t', [
      defineTool({
        name: 'noop',
        description: 'no op',
        parameters: z.object({}),
        execute: async () => 'ok',
      }),
    ]);
    const defs = ts.toOpenAITools();
    expect(defs).toHaveLength(1);
    expect(defs[0].type).toBe('function');
    expect(defs[0].function.name).toBe('noop');
  });

  it('serializes positive integer params without openapi boolean exclusiveMinimum', () => {
    const ts = new ToolSet('t', [
      defineTool({
        name: 'timed',
        description: 'timed',
        parameters: z.object({
          timeoutMs: z.number().int().positive().max(120000).optional(),
        }),
        execute: async () => 'ok',
      }),
    ]);
    const defs = ts.toOpenAITools();
    const params = defs[0].function.parameters as Record<string, unknown>;
    expect(params).not.toHaveProperty('$schema');
    expect((params.properties as Record<string, any>).timeoutMs).toMatchObject({
      type: 'integer',
      minimum: 1,
      maximum: 120000,
    });
    expect((params.properties as Record<string, any>).timeoutMs).not.toHaveProperty('exclusiveMinimum');
  });

  it('builds a ToolSet from a class with @tool decorators', async () => {
    class Calc {
      @tool('add', z.object({ a: z.number(), b: z.number() }))
      add(args: { a: number; b: number }) {
        return args.a + args.b;
      }
    }
    const ts = ToolSet.fromClass(new Calc());
    expect(ts.has('add')).toBe(true);
    const res = await ts.execute('add', { a: 1, b: 2 }, { agentName: 'x' });
    expect(res.content).toBe('3');
  });

  it('supports aliases, permission checks, and persisted large results', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'aos-toolset-'));
    const permissions = new PermissionManager({
      rules: [{ effect: 'deny', tool: 'blocked' }],
    });
    const ts = new ToolSet('t', [], {
      permissionManager: permissions,
      resultStorageDir: dir,
      defaultMaxResultSizeChars: 10,
    });
    ts.register(defineTool({
      name: 'long',
      aliases: ['LongAlias'],
      description: 'long output',
      parameters: z.object({}),
      execute: async () => 'x'.repeat(30),
    }));
    ts.register(defineTool({
      name: 'blocked',
      description: 'blocked output',
      parameters: z.object({}),
      execute: async () => 'nope',
    }));

    const denied = await ts.execute('blocked', {}, { agentName: 't' });
    expect(denied.content).toContain('not permitted');

    const res = await ts.execute('LongAlias', {}, { agentName: 't' });
    const payload = JSON.parse(res.content);
    expect(payload.truncated).toBe(true);
    expect(await readFile(payload.fullResultPath, 'utf-8')).toHaveLength(30);
    await rm(dir, { recursive: true, force: true });
  });

  it('runs and observes background tasks from task tools', async () => {
    const ts = taskToolSet();
    const created = await ts.execute('start_background_task', {
      name: 'bg',
      script: 'ctx.reportProgress("half"); return 42;',
    }, { agentName: 't' });
    const id = JSON.parse(created.content).id;
    const done = await ts.execute('wait_background_task', { id }, { agentName: 't' });
    expect(JSON.parse(done.content).result).toBe(42);
    const status = await ts.execute('get_background_task', { id }, { agentName: 't' });
    expect(JSON.parse(status.content).progress).toEqual(['half']);
  });

  it('loads non-UI plugins with skills, tools, and commands', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'aos-plugin-'));
    const pluginDir = path.join(root, 'sample');
    await mkdir(pluginDir, { recursive: true });
    await writeFile(path.join(pluginDir, 'plugin.json'), JSON.stringify({ name: 'sample', entry: 'index.js' }), 'utf-8');
    await writeFile(path.join(pluginDir, 'index.js'), `
      export const skills = [{ name: 's', description: 'skill', instructions: 'be useful' }];
      export const tools = [{ name: 'hello_tool', description: 'hello', parameters: {}, execute: async () => 'hi' }];
      export const commands = [{ name: 'hello', description: 'hello command', handler: () => 'hi' }];
    `, 'utf-8');

    const loader = new PluginLoader([root]);
    const plugin = await loader.load('sample');
    expect(plugin.skills[0].name).toBe('s');
    expect(plugin.tools[0].name).toBe('hello_tool');
    expect(plugin.commands[0].name).toBe('hello');
    await rm(root, { recursive: true, force: true });
  });

  it('exports and imports session bundles', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'aos-session-'));
    const store = new FileSessionStore(path.join(root, 'sessions'));
    await store.save('one', { chatId: 'chat1', messages: [{ role: 'user', content: 'hello' }] });
    await store.exportBundle('one', path.join(root, 'bundle'));
    const imported = await store.importBundle(path.join(root, 'bundle'), 'two');
    expect(imported).toBe('two');
    expect((await store.load('two'))?.messages?.[0].content).toBe('hello');
    await rm(root, { recursive: true, force: true });
  });

  it('persists permission modes and rules', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'aos-perm-'));
    const store = new FilePermissionStore(path.join(root, 'permissions.json'));
    const manager = new PermissionManager({ mode: 'plan' });
    manager.addRule({ effect: 'deny', tool: 'x' });
    await store.persistManager(manager);
    const loaded = await store.createManager();
    expect(loaded.getMode()).toBe('plan');
    expect(loaded.listRules()[0].tool).toBe('x');
    await rm(root, { recursive: true, force: true });
  });

  it('loads prompt commands from command files', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'aos-cmd-'));
    await writeFile(path.join(root, 'review.md'), '---\nname: review\ndescription: Review command\n---\nReview this: {{args}}', 'utf-8');
    const loader = new FileCommandLoader([root]);
    const commands = await loader.loadAll();
    expect(commands[0].name).toBe('review');
    expect(await commands[0].handler({ args: 'abc' })).toBe('Review this: abc');
    await rm(root, { recursive: true, force: true });
  });

  it('searches and loads deferred toolsets', async () => {
    const catalog = new ToolCatalog();
    catalog.register('math', () => new ToolSet('math', [
      defineTool({
        name: 'add_numbers',
        description: 'add two numbers',
        parameters: z.object({ a: z.number(), b: z.number() }),
        execute: async ({ a, b }) => a + b,
      }),
    ]));
    const searchTools = toolSearchToolSet(catalog);
    const before = JSON.parse((await searchTools.execute('search_tools', { query: 'math' }, {})).content);
    expect(before.results[0].deferred).toBe(true);
    await searchTools.execute('load_toolset', { toolset: 'math' }, {});
    const after = JSON.parse((await searchTools.execute('search_tools', { query: 'add' }, {})).content);
    expect(after.results[0].name).toBe('add_numbers');
  });

  it('exposes bio MAS tools through the default toolset', async () => {
    const ts = createDefaultToolSet({
      include: ['bio_mas', 'evolution'],
    });
    expect(ts.has('bio_mas_preflight')).toBe(true);
    expect(ts.has('bio_mas_create_tiny_demo')).toBe(true);
    expect(ts.has('bio_mas_run_tiny_demo')).toBe(true);
    expect(ts.has('evolution_capabilities')).toBe(true);
    expect(ts.has('evolution_run_smoke')).toBe(true);

    const planned = await ts.execute('bio_mas_plan_workflow', {
      goal: 'annotate a single-cell h5ad',
      inputKind: 'h5ad',
      hasRealData: false,
      hasFoundationWeights: false,
      allowLLMSelection: true,
      allowSyntheticTinyDemo: true,
    }, { agentName: 'test' });
    const payload = JSON.parse(planned.content);
    expect(payload.agents.length).toBeGreaterThanOrEqual(5);
    expect(payload.smokeTest.tool).toBe('bio_mas_run_tiny_demo');
    expect(payload.productionRequirements.join(' ')).toContain('Foundation checkpoints');

    const searched = await ts.execute('search_tools', { query: 'bio mas', limit: 5 }, { agentName: 'test' });
    const results = JSON.parse(searched.content).results;
    expect(results.some((item: { name: string }) => item.name === 'bio_mas_preflight')).toBe(true);

    const evolutionSearch = await ts.execute('search_tools', { query: 'genetic evolution', limit: 5 }, { agentName: 'test' });
    const evolutionResults = JSON.parse(evolutionSearch.content).results;
    expect(evolutionResults.some((item: { name: string }) => item.name === 'evolution_capabilities')).toBe(true);
  });
});
