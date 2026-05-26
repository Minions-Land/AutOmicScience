import { describe, it, expect, afterEach } from 'vitest';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { Agent } from '../src/agent/index.js';
import { NatsManager } from '../src/chatroom/index.js';
import { ToolSet } from '../src/toolset/index.js';
import type { ChatOptions, Message, ToolCall } from '../src/types.js';
import { DevServer } from '../src/ui/index.js';

const servers: DevServer[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) => server.stop()));
});

async function stopServer(server: DevServer): Promise<void> {
  const index = servers.indexOf(server);
  if (index >= 0) servers.splice(index, 1);
  await server.stop();
}

describe('DevServer UI', () => {
  it('serves the console and exposes migrated capabilities through APIs', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'aos-ui-'));
    await writeFile(path.join(root, 'AGENTS.md'), 'Always reply in chinese.', 'utf-8');
    const sessionsDir = path.join(root, 'sessions');
    const server = new DevServer({
      rootDir: root,
      sessionsDir,
      permissionsFile: path.join(root, 'permissions.json'),
      tasksFile: path.join(root, 'tasks.json'),
      pluginsFile: path.join(root, 'plugins.json'),
      issueDir: path.join(root, 'gitissues'),
    });
    servers.push(server);
    const port = 38111 + Math.floor(Math.random() * 1000);
    await server.start(port);
    const base = `http://127.0.0.1:${port}`;

    const html = await fetch(`${base}/`).then((res) => res.text());
    expect(html).toContain('AutOmicScience Console');
    expect(html).toContain('Permissions');
    expect(html).toContain('Conversation');
    expect(html).toContain('AutOmicScience');
    expect(html).toContain('Thinking and checking tools');

    const aosRedirect = await fetch(`${base}/aos`, { redirect: 'manual' });
    expect(aosRedirect.status).toBe(302);
    expect(aosRedirect.headers.get('location')).toBe('/aos/');

    const aosHtmlRes = await fetch(`${base}/aos/`);
    expect(aosHtmlRes.status).toBe(200);
    expect(aosHtmlRes.headers.get('content-type')).toContain('text/html');
    const aosHtml = await aosHtmlRes.text();
    expect(aosHtml).toContain('<div id="app"></div>');
    expect(aosHtml).toContain('/aos/assets/index-');
    expect(aosHtml).toContain("remoteApiOrigin = 'https://aos.local'");
    expect(aosHtml).toContain('rewriteApiUrl');
    expect(aosHtml).toContain("searchParams.set('service'");
    expect(aosHtml).toContain("searchParams.set('nats'");
    expect(aosHtml).toContain('location.replace');
    expect(aosHtml).toContain('/api/issues/report');
    expect(aosHtml).toContain('Issue reporting must never affect the AutOmicScience UI');

    const mainScript = aosHtml.match(/src="([^"]+index-[^"]+\.js(?:\?[^"]*)?)"/)?.[1];
    expect(mainScript).toBeTruthy();
    const aosJs = await fetch(`${base}${mainScript}`);
    expect(aosJs.status).toBe(200);
    expect(aosJs.headers.get('content-type')).toContain('application/javascript');

    const mainStyle = aosHtml.match(/href="([^"]+index-[^"]+\.css(?:\?[^"]*)?)"/)?.[1];
    expect(mainStyle).toBeTruthy();
    const aosCss = await fetch(`${base}${mainStyle}`);
    expect(aosCss.status).toBe(200);
    expect(aosCss.headers.get('content-type')).toContain('text/css');

    const aosRootAsset = await fetch(`${base}/assets/inter-latin-400-normal-C38fXH4l.woff2`);
    expect(aosRootAsset.status).toBe(200);
    expect(aosRootAsset.headers.get('content-type')).toContain('font/woff2');

    const initial = await fetchJson<{ permissions: { mode: string }; agent: { toolCount: number; tools: { name: string }[] } }>(`${base}/api/state`);
    expect(initial.permissions.mode).toBe('default');
    expect(initial.agent.toolCount).toBeGreaterThan(0);
    expect(initial.agent.tools.some((tool) => tool.name === 'evolution_capabilities')).toBe(true);
    expect(initial.agent.tools.some((tool) => tool.name === 'list_available_skills')).toBe(true);

    const mode = await fetchJson<{ mode: string }>(`${base}/api/permissions/mode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'plan' }),
    });
    expect(mode.mode).toBe('plan');

    await fetchJson(`${base}/api/permissions/rules`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rule: 'deny destructive' }),
    });
    const afterRule = await fetchJson<{ permissions: { rules: unknown[] } }>(`${base}/api/state`);
    expect(afterRule.permissions.rules).toHaveLength(1);

    const instructions = await fetchJson<{ files: { content: string }[] }>(`${base}/api/project-instructions`);
    expect(instructions.files[0].content).toContain('Always reply in chinese.');

    const saved = await fetchJson<{ name: string }>(`${base}/api/session/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'ui-session' }),
    });
    expect(saved.name).toBe('ui-session');
    const sessions = await fetchJson<{ sessions: string[] }>(`${base}/api/sessions`);
    expect(sessions.sessions).toContain('ui-session');

    await rm(root, { recursive: true, force: true });
  });

  it('exposes AutOmicScience-compatible store and chatroom HTTP APIs as a superset', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'aos-aos-http-'));
    const agent = new Agent({
      name: 'aos-test',
      model: 'test-model',
      provider: echoProvider(),
      toolset: new ToolSet('empty'),
    });
    const server = new DevServer({
      rootDir: root,
      agent,
      enableAOSCompat: false,
      sessionsDir: path.join(root, 'sessions'),
      permissionsFile: path.join(root, 'permissions.json'),
      tasksFile: path.join(root, 'tasks.json'),
      pluginsFile: path.join(root, 'plugins.json'),
      aosCompatDataDir: path.join(root, 'aos-compat'),
      issueDir: path.join(root, 'gitissues'),
    });
    servers.push(server);
    const port = 39211 + Math.floor(Math.random() * 1000);
    await server.start(port);
    const base = `http://127.0.0.1:${port}`;

    const ready = await fetchJson<any>(`${base}/api/aos/ready`);
    expect(ready.success).toBe(true);
    expect(ready.service_id).toHaveLength(64);

    const auth = await fetchJson<any>(`${base}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ username: 'tester', password: 'x' }),
    });
    expect(auth.access_token).toBe('aos-local-token');

    const listed = await fetchJson<any>(`${base}/api/store/packages?q=aos&limit=10`);
    expect(listed.packages.some((pkg: any) => pkg.name === 'aos-bio-mas')).toBe(true);

    const published = await fetchJson<any>(`${base}/api/store/packages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: 'compat-skill',
        type: 'skill',
        version: '1.2.3',
        description: 'Compatibility skill',
        content: '# Compat Skill',
      }),
    });
    expect(published.name).toBe('compat-skill');

    const versions = await fetchJson<any>(`${base}/api/store/packages/compat-skill/versions`);
    expect(versions.versions[0].version).toBe('1.2.3');

    const downloaded = await fetchJson<any>(`${base}/api/store/packages/compat-skill/download`);
    expect(downloaded.content).toContain('Compat Skill');

    const stats = await fetchJson<any>(`${base}/api/store/packages/stats`);
    expect(stats.total).toBeGreaterThanOrEqual(3);
    expect(stats.by_type.skill).toBeGreaterThanOrEqual(1);
    expect(Array.isArray(stats.by_category)).toBe(true);
    expect(stats.by_category.some((entry: any) => entry.name === 'bioinformatics' && entry.count > 0)).toBe(true);
    expect(stats.by_category_map.bioinformatics).toBeGreaterThanOrEqual(1);

    const installedStorePackage = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'install_store_package', parameters: { package_id: 'aos-bio-mas' } }),
    });
    expect(installedStorePackage.result).toMatchObject({
      success: true,
      id: 'aos-bio-mas',
      name: 'aos-bio-mas',
      type: 'skill',
      version: '1.0.0',
    });
    const installedStorePackages = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'get_installed_store_packages', parameters: {} }),
    });
    expect(installedStorePackages.result.installs['aos-bio-mas']).toMatchObject({
      name: 'aos-bio-mas',
      type: 'skill',
      version: '1.0.0',
    });

    const created = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'create_chat', parameters: { chat_name: 'AOS UI HTTP' } }),
    });
    expect(created.result.success).toBe(true);
    expect(created.result.chat_id).toBeTruthy();

    const chat = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'chat',
        parameters: {
          chat_id: created.result.chat_id,
          message: [{ role: 'user', content: 'hello through aos http' }],
        },
      }),
    });
    expect(chat.result.success).toBe(true);
    expect(chat.result.response).toContain('echo: hello through aos http');

    const messages = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'get_chat_messages', parameters: { chat_id: created.result.chat_id } }),
    });
    expect(messages.result.messages).toHaveLength(2);

    const frontendGeneratedChatId = '374c9814-6e0a-4194-a902-5cfc4818eb9a';
    const autoCreatedChat = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'chat',
        parameters: {
          chat_id: frontendGeneratedChatId,
          chat_name: 'Frontend Generated Chat',
          message: [{ role: 'user', content: 'created from frontend state' }],
        },
      }),
    });
    expect(autoCreatedChat.result.success).toBe(true);
    expect(autoCreatedChat.result.chat_id).toBe(frontendGeneratedChatId);
    const autoMessages = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'get_chat_messages', parameters: { chat_id: frontendGeneratedChatId } }),
    });
    expect(autoMessages.result.messages).toHaveLength(2);

    await writeFile(path.join(root, 'sample.txt'), 'alpha\nbeta\ngamma\n', 'utf-8');
    const files = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'proxy_toolset', parameters: { method_name: 'list_files', args: { sub_dir: '' }, toolset_name: 'file_manager' } }),
    });
    expect(files.result.success).toBe(true);
    expect(files.result.files.some((file: any) => file.name === 'sample.txt' && file.type === 'file')).toBe(true);
    const readFile = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'proxy_toolset',
        parameters: { method_name: 'read_file', args: { file_path: 'sample.txt', start_line: 2, end_line: 2 }, toolset_name: 'file_manager' },
      }),
    });
    expect(readFile.result.success).toBe(true);
    expect(readFile.result.content).toBe('beta');
    const wroteFile = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'proxy_toolset',
        parameters: { method_name: 'write_file', args: { file_path: 'nested/from-aos.txt', content: 'ok' }, toolset_name: 'file_manager' },
      }),
    });
    expect(wroteFile.result.success).toBe(true);
    const envFile = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'proxy_toolset',
        parameters: { method_name: 'read_file', args: { file_path: '~/.aos/.env' }, toolset_name: 'file_manager' },
      }),
    });
    expect(envFile.result.success).toBe(true);
    expect(envFile.result.content).toContain('AOS_MODEL=');
    const writeEnvFile = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'proxy_toolset',
        parameters: { method_name: 'write_file', args: { file_path: '~/.aos/.env', content: 'AOS_MODEL=test-model\n' }, toolset_name: 'file_manager' },
      }),
    });
    expect(writeEnvFile.result.success).toBe(true);
    const settingsFile = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'proxy_toolset',
        parameters: { method_name: 'read_file', args: { file_path: '.aos/settings.json' }, toolset_name: 'file_manager' },
      }),
    });
    expect(settingsFile.result.content).toContain('"app_name": "AutOmicScience"');
    const mcpFile = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'proxy_toolset',
        parameters: { method_name: 'read_file', args: { file_path: '.aos/mcp.json' }, toolset_name: 'file_manager' },
      }),
    });
    expect(mcpFile.result.content).toContain('"mcpServers"');
    const skillsIndex = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'proxy_toolset',
        parameters: { method_name: 'read_file', args: { file_path: '.aos/skills/SKILLS.md' }, toolset_name: 'file_manager' },
      }),
    });
    expect(skillsIndex.result.content).toContain('AutOmicScience Skills');

    const writeAgent = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'write_template_file',
        parameters: {
          file_path: 'agents/variant-caller.md',
          content: {
            id: 'variant-caller',
            name: 'Variant Caller',
            description: 'Calls and annotates variants',
            instructions: 'Run variant workflows with AutOmicScience tools.',
            model: 'normal',
            toolsets: ['aos_default'],
          },
        },
      }),
    });
    expect(writeAgent.result).toMatchObject({ success: true, operation: 'create', type: 'agent', id: 'variant-caller' });
    const listAgents = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'list_template_files', parameters: { file_type: 'agents' } }),
    });
    expect(listAgents.result.files.some((file: any) => file.path === 'agents/variant-caller.md')).toBe(true);
    const readAgent = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'read_template_file', parameters: { file_path: 'agents/variant-caller.md' } }),
    });
    expect(readAgent.result.success).toBe(true);
    expect(readAgent.result.type).toBe('agent');
    expect(readAgent.result.content.instructions).toContain('AutOmicScience');

    const writeTeam = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'write_template_file',
        parameters: {
          file_path: 'teams/genomics-lab.md',
          content: {
            id: 'genomics-lab',
            name: 'Genomics Lab',
            description: 'A local genomics team',
            agents: [{ id: 'variant-caller', name: 'Variant Caller' }],
            tags: ['genomics'],
          },
        },
      }),
    });
    expect(writeTeam.result).toMatchObject({ success: true, operation: 'create', type: 'team', id: 'genomics-lab' });
    const readResolvedTeam = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'read_template_file', parameters: { file_path: 'teams/genomics-lab.md', resolve_refs: true } }),
    });
    expect(readResolvedTeam.result.success).toBe(true);
    expect(readResolvedTeam.result.type).toBe('team');
    expect(readResolvedTeam.result.content.agents[0].instructions).toContain('AutOmicScience');

    const validated = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'validate_template', parameters: { template: readResolvedTeam.result.content } }),
    });
    expect(validated.result.success).toBe(true);
    expect(validated.result.required_toolsets).toContain('aos_default');

    const writeSkill = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'write_template_file',
        parameters: {
          file_path: 'skills/qc-skill.md',
          content: {
            id: 'qc-skill',
            name: 'QC Skill',
            description: 'Quality control workflow notes',
            content: '# QC Skill\n\nRun tiny checks before full analysis.',
            tags: ['qc'],
          },
        },
      }),
    });
    expect(writeSkill.result).toMatchObject({ success: true, operation: 'create', type: 'skill', id: 'qc-skill' });
    const listAllTemplates = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'list_template_files', parameters: { file_type: 'all' } }),
    });
    expect(listAllTemplates.result.files.some((file: any) => file.path === 'teams/genomics-lab.md')).toBe(true);
    expect(listAllTemplates.result.files.some((file: any) => file.path === 'agents/variant-caller.md')).toBe(true);
    expect(listAllTemplates.result.files.some((file: any) => file.path === 'skills/qc-skill.md')).toBe(true);

    const createdSkillFile = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'proxy_toolset',
        parameters: {
          method_name: 'write_file',
          args: { file_path: '.aos/skills/frontend-created/SKILL.md', content: '# Frontend Created\n\nDescribe this skill.', overwrite: false },
          toolset_name: 'file_manager',
        },
      }),
    });
    expect(createdSkillFile.result.success).toBe(true);
    const readSkillFile = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'proxy_toolset',
        parameters: { method_name: 'read_file', args: { file_path: '.aos/skills/frontend-created/SKILL.md' }, toolset_name: 'file_manager' },
      }),
    });
    expect(readSkillFile.result.content).toContain('Frontend Created');

    const deleteTeam = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'delete_template_file', parameters: { file_path: 'teams/genomics-lab.md' } }),
    });
    expect(deleteTeam.result.success).toBe(true);
    const teamsAfterDelete = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'list_template_files', parameters: { file_type: 'teams' } }),
    });
    expect(teamsAfterDelete.result.files.some((file: any) => file.path === 'teams/genomics-lab.md')).toBe(false);

    const toolAgent = new Agent({
      name: 'aos-tool-message-test',
      model: 'test-model',
      provider: toolProvider(),
      toolset: new ToolSet('empty'),
    });
    const toolServer = new DevServer({
      rootDir: root,
      agent: toolAgent,
      enableAOSCompat: false,
      sessionsDir: path.join(root, 'tool-sessions'),
      permissionsFile: path.join(root, 'tool-permissions.json'),
      tasksFile: path.join(root, 'tool-tasks.json'),
      pluginsFile: path.join(root, 'tool-plugins.json'),
      aosCompatDataDir: path.join(root, 'tool-aos-compat'),
    });
    servers.push(toolServer);
    const toolPort = port + 2000;
    await toolServer.start(toolPort);
    const toolBase = `http://127.0.0.1:${toolPort}`;
    const toolCreated = await fetchJson<any>(`${toolBase}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'create_chat', parameters: { chat_name: 'AOS UI Tool Calls' } }),
    });
    await fetchJson<any>(`${toolBase}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'chat',
        parameters: {
          chat_id: toolCreated.result.chat_id,
          message: [{ role: 'user', content: 'make a tool call' }],
        },
      }),
    });
    const toolMessages = await fetchJson<any>(`${toolBase}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'get_chat_messages', parameters: { chat_id: toolCreated.result.chat_id } }),
    });
    const assistantWithTool = toolMessages.result.messages.find((msg: any) => msg.role === 'assistant' && msg.tool_calls);
    expect(assistantWithTool.tool_calls[0].function.name).toBe('bio_mas_preflight');
    expect(assistantWithTool.tool_calls[0].function.arguments).toContain('mode');

    const slashCreated = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'create_chat', parameters: { chat_name: 'AOS UI Slash' } }),
    });
    const slash = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'chat',
        parameters: {
          chat_id: slashCreated.result.chat_id,
          message: [{ role: 'user', content: [{ type: 'text', text: '/status' }] }],
        },
      }),
    });
    expect(slash.result.success).toBe(true);
    expect(slash.result.response).toContain('"chat_id"');
    const slashMessages = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'get_chat_messages', parameters: { chat_id: slashCreated.result.chat_id } }),
    });
    expect(slashMessages.result.messages).toHaveLength(2);
    expect(slashMessages.result.messages[1].content).toContain('"running"');
    expect(slashMessages.result.messages[1].text).toContain('"running"');

    const context = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'update_chat_context',
        parameters: { chat_id: slashCreated.result.chat_id, context: { assay: 'single-cell' } },
      }),
    });
    expect(context.result.context.assay).toBe('single-cell');

    const workspaceMode = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'set_chat_workspace_mode',
        parameters: { chat_id: slashCreated.result.chat_id, workspace_mode: 'isolated' },
      }),
    });
    expect(workspaceMode.result.success).toBe(true);
    expect(workspaceMode.result.workspace_path).toBeTruthy();

    const gateway = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'list_gateway_channels', parameters: {} }),
    });
    expect(gateway.result.channels).toEqual([]);

    const exported = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'export_chat', parameters: { chat_id: slashCreated.result.chat_id } }),
    });
    expect(exported.result.success).toBe(true);
    expect(exported.result.bundle_path).toContain('chat.json');

    const reverted = await fetchJson<any>(`${base}/api/aos/rpc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method: 'revert_to_message',
        parameters: { chat_id: slashCreated.result.chat_id, message_id: slashMessages.result.messages[1].id },
      }),
    });
    expect(reverted.result.success).toBe(true);

    await rm(root, { recursive: true, force: true });
  });

  it('serves real AutOmicScience NATS RPC when nats-server is available', async () => {
    if (!new NatsManager().checkBinaryAvailable().available) return;
    const root = await mkdtemp(path.join(os.tmpdir(), 'aos-aos-nats-'));
    const agent = new Agent({
      name: 'aos-test',
      model: 'test-model',
      provider: echoProvider(),
      toolset: new ToolSet('empty'),
    });
    const server = new DevServer({
      rootDir: root,
      agent,
      enableAOSCompat: true,
      aosServiceIdHash: 'vitest-aos',
      sessionsDir: path.join(root, 'sessions'),
      permissionsFile: path.join(root, 'permissions.json'),
      tasksFile: path.join(root, 'tasks.json'),
      pluginsFile: path.join(root, 'plugins.json'),
      aosCompatDataDir: path.join(root, 'aos-compat'),
    });
    servers.push(server);
    const port = 40211 + Math.floor(Math.random() * 1000);
    await server.start(port);
    const base = `http://127.0.0.1:${port}`;
    const ready = await fetchJson<any>(`${base}/api/aos/ready`);
    expect(ready.nats.running).toBe(true);

    const mod: any = await import('nats');
    const { connect, JSONCodec } = mod;
    const nc = await connect({ servers: ready.nats.tcpUrl });
    const codec = JSONCodec();
    try {
      const ping = await nc.request(
        ready.service_subject,
        codec.encode({ method: '_ping', parameters: {} }),
        { timeout: 5000 },
      );
      expect(codec.decode(ping.data).result.status).toBe('ok');

      const created = await nc.request(
        ready.service_subject,
        codec.encode({ method: 'create_chat', parameters: { chat_name: 'AOS UI NATS' } }),
        { timeout: 5000 },
      );
      const createdResult = codec.decode(created.data).result;
      expect(createdResult.chat_id).toBeTruthy();

      const events: any[] = [];
      const sub = nc.subscribe(`aos.stream.chat_${createdResult.chat_id}`);
      const streamDone = (async () => {
        for await (const msg of sub) {
          const payload = codec.decode(msg.data);
          events.push(payload);
          if (payload.data?.type === 'chat_finished') {
            sub.unsubscribe();
            return events;
          }
        }
        return events;
      })();

      const chat = await nc.request(
        ready.service_subject,
        codec.encode({
          method: 'chat',
          parameters: {
            chat_id: createdResult.chat_id,
            message: [{ role: 'user', content: 'hello through aos nats' }],
          },
        }),
        { timeout: 10000 },
      );
      expect(codec.decode(chat.data).result).toMatchObject({ success: true, message: 'Chat started' });
      const streamEvents = await Promise.race([
        streamDone,
        new Promise<any[]>((resolve) => setTimeout(() => resolve(events), 5000)),
      ]);
      const chunk = streamEvents.find((event) => event.data?.type === 'chunk')?.data?.chunk;
      expect(chunk?.content).toContain('echo: hello through aos nats');
      expect(chunk?.message_id).toBeTruthy();
      expect(chunk?.chunk_index).toBe(1);
      const finalMessage = streamEvents.find((event) => event.data?.type === 'step_message')?.data?.step_message;
      expect(finalMessage?.id).toBe(chunk.message_id);
      expect(finalMessage?.role).toBe('assistant');
      expect(finalMessage?.text).toContain('echo: hello through aos nats');
      expect(finalMessage?.agent_name).toBe('AOS');
      expect(streamEvents.some((event) => event.data?.type === 'chat_finished')).toBe(true);
    } finally {
      await nc.drain();
      await stopServer(server);
      await rm(root, { recursive: true, force: true });
    }
  }, 30000);
});

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const json = await res.json();
  if (!res.ok) throw new Error(JSON.stringify(json));
  return json as T;
}

function echoProvider() {
  return {
    async *chat(messages: Message[], _opts: ChatOptions) {
      const last = [...messages].reverse().find((msg) => msg.role === 'user');
      const content = typeof last?.content === 'string' ? last.content : JSON.stringify(last?.content ?? '');
      yield { type: 'text' as const, text: `echo: ${content}` };
    },
  };
}

function toolProvider() {
  let called = false;
  return {
    async *chat(_messages: Message[], _opts: ChatOptions) {
      if (!called) {
        called = true;
        yield {
          type: 'tool_call' as const,
          toolCall: {
            id: 'call_bio_mas_preflight',
            name: 'bio_mas_preflight',
            arguments: { mode: 'summary' },
          } satisfies ToolCall,
        };
        return;
      }
      yield { type: 'text' as const, text: 'preflight summarized' };
    },
  };
}
