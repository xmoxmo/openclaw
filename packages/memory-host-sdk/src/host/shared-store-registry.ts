import { type DatabaseSync } from "node:sqlite";

/**
 * Master registry for shared memory stores.
 *
 * Tracks:
 *   1. Which directory hashes map to which store files (shared_stores)
 *   2. Which agents reference which stores (agent_source_refs)
 *
 * Future: when a single store grows too large, the registry can support
 * sharding by recording multiple store_path entries per dir_hash with
 * a shard_index and shard_count.
 */

const REGISTRY_TABLE = "shared_stores";
const AGENT_REFS_TABLE = "agent_source_refs";

export type StoreRecord = {
  dir_hash: string;
  store_path: string;
  file_count: number;
  chunk_count: number;
  agent_ids: string;
  created_at: number;
  updated_at: number;
};

export type AgentRefRecord = {
  agent_id: string;
  dir_hash: string;
  store_path: string;
  workspace_dir: string;
  extra_paths: string;
  created_at: number;
  updated_at: number;
};

export function ensureRegistrySchema(db: DatabaseSync): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS ${REGISTRY_TABLE} (
      dir_hash TEXT PRIMARY KEY,
      store_path TEXT NOT NULL,
      file_count INTEGER NOT NULL DEFAULT 0,
      chunk_count INTEGER NOT NULL DEFAULT 0,
      agent_ids TEXT NOT NULL DEFAULT '[]',
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL
    );
  `);
  db.exec(`
    CREATE TABLE IF NOT EXISTS ${AGENT_REFS_TABLE} (
      agent_id TEXT NOT NULL,
      dir_hash TEXT NOT NULL,
      store_path TEXT NOT NULL,
      workspace_dir TEXT NOT NULL DEFAULT '',
      extra_paths TEXT NOT NULL DEFAULT '[]',
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      PRIMARY KEY (agent_id, dir_hash)
    );
  `);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_agent_refs_store ON ${AGENT_REFS_TABLE}(dir_hash);`);
}

/**
 * Register an agent's reference to a shared store.
 * Creates or updates the agent→store mapping.
 */
export function registerAgentForStore(
  db: DatabaseSync,
  params: {
    agentId: string;
    dirHash: string;
    storePath: string;
    workspaceDir: string;
    extraPaths: string[];
  },
): void {
  const now = Date.now();
  const extraPathsJson = JSON.stringify(params.extraPaths);

  // Upsert agent ref
  db.prepare(
    `INSERT INTO ${AGENT_REFS_TABLE} (agent_id, dir_hash, store_path, workspace_dir, extra_paths, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(agent_id, dir_hash) DO UPDATE SET
       store_path=excluded.store_path,
       workspace_dir=excluded.workspace_dir,
       extra_paths=excluded.extra_paths,
       updated_at=excluded.updated_at`,
  ).run(
    params.agentId,
    params.dirHash,
    params.storePath,
    params.workspaceDir,
    extraPathsJson,
    now,
    now,
  );

  // Upsert store with agent added to the set
  const existing = db
    .prepare(`SELECT agent_ids FROM ${REGISTRY_TABLE} WHERE dir_hash = ?`)
    .get(params.dirHash) as { agent_ids: string } | undefined;

  let agentIds: string[];
  if (existing) {
    const parsed: string[] = JSON.parse(existing.agent_ids);
    if (!parsed.includes(params.agentId)) {
      agentIds = [...parsed, params.agentId].sort();
    } else {
      agentIds = parsed;
    }
  } else {
    agentIds = [params.agentId];
  }

  db.prepare(
    `INSERT INTO ${REGISTRY_TABLE} (dir_hash, store_path, agent_ids, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(dir_hash) DO UPDATE SET
       store_path=excluded.store_path,
       agent_ids=excluded.agent_ids,
       updated_at=excluded.updated_at`,
  ).run(params.dirHash, params.storePath, JSON.stringify(agentIds), now, now);
}

/**
 * Get all stores referenced by a given agent.
 */
export function getStoresForAgent(
  db: DatabaseSync,
  agentId: string,
): AgentRefRecord[] {
  return db
    .prepare(
      `SELECT * FROM ${AGENT_REFS_TABLE} WHERE agent_id = ? ORDER BY created_at ASC`,
    )
    .all(agentId) as AgentRefRecord[];
}

/**
 * Get all agents referencing a given store.
 */
export function getAgentsForStore(
  db: DatabaseSync,
  dirHash: string,
): AgentRefRecord[] {
  return db
    .prepare(
      `SELECT * FROM ${AGENT_REFS_TABLE} WHERE dir_hash = ? ORDER BY created_at ASC`,
    )
    .all(dirHash) as AgentRefRecord[];
}

/**
 * Remove an agent's reference from a store.
 * Cleans up the agent ref and removes the agent from the store's agent set.
 */
export function unregisterAgentFromStore(
  db: DatabaseSync,
  params: { agentId: string; dirHash: string },
): void {
  db.prepare(`DELETE FROM ${AGENT_REFS_TABLE} WHERE agent_id = ? AND dir_hash = ?`).run(
    params.agentId,
    params.dirHash,
  );

  // Update the store's agent_ids
  const existing = db
    .prepare(`SELECT agent_ids FROM ${REGISTRY_TABLE} WHERE dir_hash = ?`)
    .get(params.dirHash) as { agent_ids: string } | undefined;

  if (existing) {
    const parsed: string[] = JSON.parse(existing.agent_ids).filter(
      (id: string) => id !== params.agentId,
    );
    const now = Date.now();
    if (parsed.length === 0) {
      db.prepare(`DELETE FROM ${REGISTRY_TABLE} WHERE dir_hash = ?`).run(params.dirHash);
    } else {
      db.prepare(
        `UPDATE ${REGISTRY_TABLE} SET agent_ids = ?, updated_at = ? WHERE dir_hash = ?`,
      ).run(JSON.stringify(parsed), now, params.dirHash);
    }
  }
}

/**
 * Update file and chunk counts for a store.
 */
export function updateStoreCounts(
  db: DatabaseSync,
  params: { dirHash: string; fileCount: number; chunkCount: number },
): void {
  const now = Date.now();
  db.prepare(
    `UPDATE ${REGISTRY_TABLE} SET file_count = ?, chunk_count = ?, updated_at = ? WHERE dir_hash = ?`,
  ).run(params.fileCount, params.chunkCount, now, params.dirHash);
}

/**
 * Get a store record by its directory hash.
 */
export function getStoreByHash(
  db: DatabaseSync,
  dirHash: string,
): StoreRecord | undefined {
  return db
    .prepare(`SELECT * FROM ${REGISTRY_TABLE} WHERE dir_hash = ?`)
    .get(dirHash) as StoreRecord | undefined;
}

/**
 * List all stores in the registry.
 */
export function listAllStores(db: DatabaseSync): StoreRecord[] {
  return db
    .prepare(`SELECT * FROM ${REGISTRY_TABLE} ORDER BY created_at DESC`)
    .all() as StoreRecord[];
}
