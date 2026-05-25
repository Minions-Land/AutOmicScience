/**
 * DatabaseTools — SQL query execution, schema introspection, and data insertion.
 * Supports SQLite (via better-sqlite3) and PostgreSQL (via pg) with graceful
 * fallback when drivers are not installed.
 */

import { z } from 'zod';
import { defineTool } from './Tool.js';
import { ToolSet } from './ToolSet.js';

// ---------------------------------------------------------------------------
// Database connection abstraction
// ---------------------------------------------------------------------------

export type DbDriver = 'sqlite' | 'postgres';

interface DbConnection {
  driver: DbDriver;
  /** For SQLite: file path. For Postgres: connection string. */
  connectionString: string;
}

/** Execute a query and return rows. Dynamically imports the driver. */
async function executeQuery(
  conn: DbConnection,
  sql: string,
  params: unknown[] = [],
): Promise<{ rows: Record<string, unknown>[]; rowCount: number; fields: string[] }> {
  if (conn.driver === 'sqlite') {
    return executeSqlite(conn.connectionString, sql, params);
  } else {
    return executePostgres(conn.connectionString, sql, params);
  }
}

async function executeSqlite(
  dbPath: string,
  sql: string,
  params: unknown[],
): Promise<{ rows: Record<string, unknown>[]; rowCount: number; fields: string[] }> {
  // Dynamic import — better-sqlite3 is optional
  let Database: any;
  try {
    Database = (await import('better-sqlite3')).default;
  } catch {
    throw new Error(
      'better-sqlite3 is not installed. Run: npm install better-sqlite3',
    );
  }

  const db = new Database(dbPath, { readonly: false });
  try {
    const isSelect = /^\s*(SELECT|PRAGMA|EXPLAIN|WITH)/i.test(sql);
    if (isSelect) {
      const stmt = db.prepare(sql);
      const rows = stmt.all(...params) as Record<string, unknown>[];
      const fields = rows.length > 0 ? Object.keys(rows[0]) : [];
      return { rows, rowCount: rows.length, fields };
    } else {
      const stmt = db.prepare(sql);
      const info = stmt.run(...params);
      return { rows: [], rowCount: info.changes, fields: [] };
    }
  } finally {
    db.close();
  }
}

async function executePostgres(
  connectionString: string,
  sql: string,
  params: unknown[],
): Promise<{ rows: Record<string, unknown>[]; rowCount: number; fields: string[] }> {
  let pg: any;
  try {
    pg = await import('pg');
  } catch {
    throw new Error('pg is not installed. Run: npm install pg');
  }

  const client = new pg.Client({ connectionString });
  await client.connect();
  try {
    const result = await client.query(sql, params);
    const fields = result.fields?.map((f: any) => f.name) ?? [];
    return {
      rows: result.rows ?? [],
      rowCount: result.rowCount ?? 0,
      fields,
    };
  } finally {
    await client.end();
  }
}

// ---------------------------------------------------------------------------
// Toolset factory
// ---------------------------------------------------------------------------

export interface DatabaseToolsOptions {
  /** Default database driver. */
  driver?: DbDriver;
  /** Default connection string (file path for SQLite, URI for Postgres). */
  connectionString?: string;
}

export function databaseToolSet(opts: DatabaseToolsOptions = {}): ToolSet {
  const defaultDriver = opts.driver ?? 'sqlite';
  const defaultConn = opts.connectionString ?? ':memory:';

  function getConnection(driver?: string, connectionString?: string): DbConnection {
    return {
      driver: (driver as DbDriver) ?? defaultDriver,
      connectionString: connectionString ?? defaultConn,
    };
  }

  return new ToolSet('database', [
    // -----------------------------------------------------------------------
    // query
    // -----------------------------------------------------------------------
    defineTool<
      { sql: string; params?: unknown[]; driver?: string; connectionString?: string },
      { rows: Record<string, unknown>[]; rowCount: number; fields: string[] }
    >({
      name: 'query',
      description:
        'Execute a SQL query and return results. Supports SELECT, INSERT, UPDATE, DELETE. ' +
        'Use parameterized queries ($1, $2 for Postgres; ? for SQLite) to prevent injection.',
      parameters: z.object({
        sql: z.string().describe('SQL query to execute'),
        params: z.array(z.unknown()).optional().describe('Query parameters'),
        driver: z.enum(['sqlite', 'postgres']).optional().describe('Database driver'),
        connectionString: z.string().optional().describe('Connection string override'),
      }),
      execute: async ({ sql, params, driver, connectionString }) => {
        const conn = getConnection(driver, connectionString);
        return executeQuery(conn, sql, params ?? []);
      },
    }),

    // -----------------------------------------------------------------------
    // list_tables
    // -----------------------------------------------------------------------
    defineTool<
      { driver?: string; connectionString?: string; schema?: string },
      { tables: { name: string; type: string }[] }
    >({
      name: 'list_tables',
      description: 'List all tables in the database.',
      parameters: z.object({
        driver: z.enum(['sqlite', 'postgres']).optional(),
        connectionString: z.string().optional(),
        schema: z.string().optional().default('public').describe('Schema name (Postgres only)'),
      }),
      execute: async ({ driver, connectionString, schema }) => {
        const conn = getConnection(driver, connectionString);
        let sql: string;
        let params: unknown[] = [];

        if (conn.driver === 'sqlite') {
          sql = "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name";
        } else {
          sql =
            "SELECT table_name as name, table_type as type FROM information_schema.tables WHERE table_schema = $1 ORDER BY table_name";
          params = [schema ?? 'public'];
        }

        const result = await executeQuery(conn, sql, params);
        return {
          tables: result.rows.map((r) => ({
            name: String(r.name ?? r.table_name ?? ''),
            type: String(r.type ?? r.table_type ?? 'table'),
          })),
        };
      },
    }),

    // -----------------------------------------------------------------------
    // describe_table
    // -----------------------------------------------------------------------
    defineTool<
      { table: string; driver?: string; connectionString?: string },
      { table: string; columns: { name: string; type: string; nullable: boolean; defaultValue: string | null }[] }
    >({
      name: 'describe_table',
      description: 'Get the schema (columns, types, constraints) of a table.',
      parameters: z.object({
        table: z.string().describe('Table name'),
        driver: z.enum(['sqlite', 'postgres']).optional(),
        connectionString: z.string().optional(),
      }),
      execute: async ({ table, driver, connectionString }) => {
        const conn = getConnection(driver, connectionString);
        let sql: string;
        let params: unknown[] = [];

        if (conn.driver === 'sqlite') {
          sql = `PRAGMA table_info("${table.replace(/"/g, '""')}")`;
        } else {
          sql = `SELECT column_name, data_type, is_nullable, column_default
                 FROM information_schema.columns
                 WHERE table_name = $1
                 ORDER BY ordinal_position`;
          params = [table];
        }

        const result = await executeQuery(conn, sql, params);

        const columns = result.rows.map((r) => {
          if (conn.driver === 'sqlite') {
            return {
              name: String(r.name ?? ''),
              type: String(r.type ?? ''),
              nullable: r.notnull === 0,
              defaultValue: r.dflt_value != null ? String(r.dflt_value) : null,
            };
          } else {
            return {
              name: String(r.column_name ?? ''),
              type: String(r.data_type ?? ''),
              nullable: r.is_nullable === 'YES',
              defaultValue: r.column_default != null ? String(r.column_default) : null,
            };
          }
        });

        return { table, columns };
      },
    }),

    // -----------------------------------------------------------------------
    // insert
    // -----------------------------------------------------------------------
    defineTool<
      { table: string; rows: Record<string, unknown>[]; driver?: string; connectionString?: string },
      { ok: boolean; insertedCount: number }
    >({
      name: 'insert',
      description: 'Insert one or more rows into a table.',
      parameters: z.object({
        table: z.string().describe('Table name'),
        rows: z.array(z.record(z.unknown())).min(1).describe('Array of row objects to insert'),
        driver: z.enum(['sqlite', 'postgres']).optional(),
        connectionString: z.string().optional(),
      }),
      execute: async ({ table, rows, driver, connectionString }) => {
        const conn = getConnection(driver, connectionString);
        let totalInserted = 0;

        for (const row of rows) {
          const columns = Object.keys(row);
          const values = Object.values(row);

          let sql: string;
          if (conn.driver === 'sqlite') {
            const placeholders = columns.map(() => '?').join(', ');
            const colNames = columns.map((c) => `"${c.replace(/"/g, '""')}"`).join(', ');
            sql = `INSERT INTO "${table.replace(/"/g, '""')}" (${colNames}) VALUES (${placeholders})`;
          } else {
            const placeholders = columns.map((_, i) => `$${i + 1}`).join(', ');
            const colNames = columns.map((c) => `"${c.replace(/"/g, '""')}"`).join(', ');
            sql = `INSERT INTO "${table.replace(/"/g, '""')}" (${colNames}) VALUES (${placeholders})`;
          }

          const result = await executeQuery(conn, sql, values);
          totalInserted += result.rowCount || 1;
        }

        return { ok: true, insertedCount: totalInserted };
      },
    }),
  ]);
}
