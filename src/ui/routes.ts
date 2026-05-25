export interface RouteHandler {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  description: string;
  handler: (req: RouteRequest) => Promise<RouteResponse>;
}

export interface RouteRequest {
  params: Record<string, string>;
  query: Record<string, string>;
  body: unknown;
}

export interface RouteResponse {
  status: number;
  headers?: Record<string, string>;
  body: unknown;
}

function notImplemented(): Promise<RouteResponse> {
  return Promise.resolve({ status: 501, body: { status: 'not_implemented' } });
}

export const chatRoute: RouteHandler = {
  method: 'POST',
  path: '/api/chat',
  description: 'Run the active AOS agent and return AgentEvents.',
  handler: notImplemented,
};

export const listAgentsRoute: RouteHandler = {
  method: 'GET',
  path: '/api/agents',
  description: 'List registered agents. Superseded by /api/state for the built-in console.',
  handler: notImplemented,
};

export const runAgentRoute: RouteHandler = {
  method: 'POST',
  path: '/api/agents/:name/run',
  description: 'Invoke a named agent. Superseded by /api/chat for the built-in console.',
  handler: notImplemented,
};

export const stateRoute: RouteHandler = {
  method: 'GET',
  path: '/api/state',
  description: 'Return UI state: agent snapshot, permissions, plugins, commands, tasks, sessions, hooks.',
  handler: notImplemented,
};

export const permissionModeRoute: RouteHandler = {
  method: 'POST',
  path: '/api/permissions/mode',
  description: 'Set permission mode: default, plan, auto, bypassPermissions.',
  handler: notImplemented,
};

export const permissionRuleRoute: RouteHandler = {
  method: 'POST',
  path: '/api/permissions/rules',
  description: 'Add an allow/deny/ask permission rule.',
  handler: notImplemented,
};

export const pluginLoadRoute: RouteHandler = {
  method: 'POST',
  path: '/api/plugins/load',
  description: 'Load a local plugin and expose its tools, skills, commands, and hooks.',
  handler: notImplemented,
};

export const tasksRoute: RouteHandler = {
  method: 'GET',
  path: '/api/tasks',
  description: 'List background tasks from the shared task manager.',
  handler: notImplemented,
};

export const sessionsRoute: RouteHandler = {
  method: 'GET',
  path: '/api/sessions',
  description: 'List saved sessions.',
  handler: notImplemented,
};

export const saveSessionRoute: RouteHandler = {
  method: 'POST',
  path: '/api/session/save',
  description: 'Save the active UI conversation to the session store.',
  handler: notImplemented,
};

export const projectInstructionsRoute: RouteHandler = {
  method: 'GET',
  path: '/api/project-instructions',
  description: 'Return AGENTS/AOS/AUTOMICSCIENCE project instruction files discovered for the workspace.',
  handler: notImplemented,
};

export const aosReadyRoute: RouteHandler = {
  method: 'GET',
  path: '/api/aos/ready',
  description: 'Return AutOmicScience frontend compatibility status, NATS URLs, and service id.',
  handler: notImplemented,
};

export const aosRpcRoute: RouteHandler = {
  method: 'POST',
  path: '/api/aos/rpc',
  description: 'Invoke a AutOmicScience ChatRoom-compatible RPC method over HTTP.',
  handler: notImplemented,
};

export const storePackagesRoute: RouteHandler = {
  method: 'GET',
  path: '/api/store/packages',
  description: 'AutOmicScience-compatible package store search/list API.',
  handler: notImplemented,
};

export const storePackageStatsRoute: RouteHandler = {
  method: 'GET',
  path: '/api/store/packages/stats',
  description: 'AutOmicScience-compatible package store statistics API.',
  handler: notImplemented,
};

export const storePackageRoute: RouteHandler = {
  method: 'GET',
  path: '/api/store/packages/:id',
  description: 'AutOmicScience-compatible package detail API.',
  handler: notImplemented,
};

export const storePackageDownloadRoute: RouteHandler = {
  method: 'GET',
  path: '/api/store/packages/:id/download',
  description: 'AutOmicScience-compatible package download API.',
  handler: notImplemented,
};

export const chatroomRoute: RouteHandler = {
  method: 'POST',
  path: '/api/chatroom/:method',
  description: 'HTTP bridge for AutOmicScience ChatRoom-compatible methods.',
  handler: notImplemented,
};

export const routes: RouteHandler[] = [
  chatRoute,
  listAgentsRoute,
  runAgentRoute,
  stateRoute,
  permissionModeRoute,
  permissionRuleRoute,
  pluginLoadRoute,
  tasksRoute,
  sessionsRoute,
  saveSessionRoute,
  projectInstructionsRoute,
  aosReadyRoute,
  aosRpcRoute,
  storePackagesRoute,
  storePackageStatsRoute,
  storePackageRoute,
  storePackageDownloadRoute,
  chatroomRoute,
];
