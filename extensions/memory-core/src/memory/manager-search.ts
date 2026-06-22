import type { DatabaseSync } from "node:sqlite";
import { truncateUtf16Safe } from "openclaw/plugin-sdk/memory-core-host-engine-foundation";
import {
  cosineSimilarity,
  parseEmbedding,
} from "openclaw/plugin-sdk/memory-core-host-engine-storage";

const vectorToBlob = (embedding: number[]): Buffer =>
  Buffer.from(new Float32Array(embedding).buffer);
const FTS_QUERY_TOKEN_RE = /[\p{L}\p{N}_]+/gu;
const SHORT_CJK_TRIGRAM_RE = /[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\u3131-\u3163]/u;

export type SearchSource = string;

export type SearchRowResult = {
  id: string;
  path: string;
  startLine: number;
  endLine: number;
  score: number;
  snippet: string;
  source: SearchSource;
};

/**
 * Build a SQL fragment to filter chunks by agent ID.
 * Legacy chunks (empty agent_ids or NULL) are visible to all agents.
 * Agent-scoped chunks are only visible to their owning agent.
 */
export function buildAgentFilter(agentId?: string): { sql: string; params: string[] } {
  if (!agentId) {
    return { sql: "", params: [] };
  }
  // Safe string matching: convert JSON array to comma-separated list without quotes,
  // then use LIKE with comma-delimited agent ID to avoid substring collisions.
  // E.g. ["main","orion"] → ,main,orion, → LIKE '%,agentId,%'
  return {
    sql: ` AND (c.agent_ids = '[]' OR c.agent_ids IS NULL OR (',' || replace(substr(c.agent_ids, 2, length(c.agent_ids) - 2), '"', '') || ',') LIKE ?)`,
    params: [`%,${agentId},%`],
  };
}

function escapeLikePattern(term: string): string {
  return term.replaceAll("\\", "\\\\").replaceAll("%", "\\%").replaceAll("_", "\\_");
}

function buildMatchQueryFromTerms(terms: string[]): string | null {
  if (terms.length === 0) {
    return null;
  }
  const quoted = terms.map((term) => `"${term.replaceAll('"', "")}"`);
  return quoted.join(" AND ");
}

function planKeywordSearch(params: {
  query: string;
  ftsTokenizer?: "unicode61" | "trigram";
  buildFtsQuery: (raw: string) => string | null;
}): { matchQuery: string | null; substringTerms: string[] } {
  if (params.ftsTokenizer !== "trigram") {
    return {
      matchQuery: params.buildFtsQuery(params.query),
      substringTerms: [],
    };
  }

  const tokens =
    params.query
      .match(FTS_QUERY_TOKEN_RE)
      ?.map((token) => token.trim())
      .filter(Boolean) ?? [];
  if (tokens.length === 0) {
    return { matchQuery: null, substringTerms: [] };
  }

  const matchTerms: string[] = [];
  const substringTerms: string[] = [];
  for (const token of tokens) {
    if (SHORT_CJK_TRIGRAM_RE.test(token) && Array.from(token).length < 3) {
      substringTerms.push(token);
      continue;
    }
    matchTerms.push(token);
  }

  return {
    matchQuery: buildMatchQueryFromTerms(matchTerms),
    substringTerms,
  };
}

export async function searchVector(params: {
  db: DatabaseSync;
  vectorTable: string;
  providerModel: string;
  queryVec: number[];
  limit: number;
  snippetMaxChars: number;
  ensureVectorReady: (dimensions: number) => Promise<boolean>;
  sourceFilterVec: { sql: string; params: SearchSource[] };
  sourceFilterChunks: { sql: string; params: SearchSource[] };
  agentId?: string;
}): Promise<SearchRowResult[]> {
  if (params.queryVec.length === 0 || params.limit <= 0) {
    return [];
  }
  const agentFilter = buildAgentFilter(params.agentId);
  if (await params.ensureVectorReady(params.queryVec.length)) {
    const rows = params.db
      .prepare(
        `SELECT c.id, c.path, c.start_line, c.end_line, c.text,\n` +
          `       c.source,\n` +
          `       vec_distance_cosine(v.embedding, ?) AS dist\n` +
          `  FROM ${params.vectorTable} v\n` +
          `  JOIN chunks c ON c.id = v.id\n` +
          ` WHERE c.model = ?${params.sourceFilterVec.sql}${agentFilter.sql}\n` +
          ` ORDER BY dist ASC\n` +
          ` LIMIT ?`,
      )
      .all(
        vectorToBlob(params.queryVec),
        params.providerModel,
        ...params.sourceFilterVec.params,
        ...agentFilter.params,
        params.limit,
      ) as Array<{
      id: string;
      path: string;
      start_line: number;
      end_line: number;
      text: string;
      source: SearchSource;
      dist: number;
    }>;
    return rows.map((row) => ({
      id: row.id,
      path: row.path,
      startLine: row.start_line,
      endLine: row.end_line,
      score: 1 - row.dist,
      snippet: truncateUtf16Safe(row.text, params.snippetMaxChars),
      source: row.source,
    }));
  }

  const candidates = listChunks({
    db: params.db,
    providerModel: params.providerModel,
    sourceFilter: params.sourceFilterChunks,
    agentId: params.agentId,
  });
  const scored = candidates
    .map((chunk) => ({
      chunk,
      score: cosineSimilarity(params.queryVec, chunk.embedding),
    }))
    .filter((entry) => Number.isFinite(entry.score));
  return scored
    .toSorted((a, b) => b.score - a.score)
    .slice(0, params.limit)
    .map((entry) => ({
      id: entry.chunk.id,
      path: entry.chunk.path,
      startLine: entry.chunk.startLine,
      endLine: entry.chunk.endLine,
      score: entry.score,
      snippet: truncateUtf16Safe(entry.chunk.text, params.snippetMaxChars),
      source: entry.chunk.source,
    }));
}

export function listChunks(params: {
  db: DatabaseSync;
  providerModel: string;
  sourceFilter: { sql: string; params: SearchSource[] };
  agentId?: string;
}): Array<{
  id: string;
  path: string;
  startLine: number;
  endLine: number;
  text: string;
  embedding: number[];
  source: SearchSource;
}> {
  const agentFilter = buildAgentFilter(params.agentId);
  const rows = params.db
    .prepare(
      `SELECT c.id, c.path, c.start_line, c.end_line, c.text, c.embedding, c.source\n` +
        `  FROM chunks c\n` +
        ` WHERE c.model = ?${params.sourceFilter.sql}${agentFilter.sql}`,
    )
    .all(params.providerModel, ...params.sourceFilter.params, ...agentFilter.params) as Array<{
    id: string;
    path: string;
    start_line: number;
    end_line: number;
    text: string;
    embedding: string;
    source: SearchSource;
  }>;

  return rows.map((row) => ({
    id: row.id,
    path: row.path,
    startLine: row.start_line,
    endLine: row.end_line,
    text: row.text,
    embedding: parseEmbedding(row.embedding),
    source: row.source,
  }));
}

export async function searchKeyword(params: {
  db: DatabaseSync;
  ftsTable: string;
  providerModel: string | undefined;
  query: string;
  ftsTokenizer?: "unicode61" | "trigram";
  limit: number;
  snippetMaxChars: number;
  sourceFilter: { sql: string; params: SearchSource[] };
  buildFtsQuery: (raw: string) => string | null;
  bm25RankToScore: (rank: number) => number;
  agentId?: string;
}): Promise<Array<SearchRowResult & { textScore: number }>> {
  if (params.limit <= 0) {
    return [];
  }
  const plan = planKeywordSearch({
    query: params.query,
    ftsTokenizer: params.ftsTokenizer,
    buildFtsQuery: params.buildFtsQuery,
  });
  if (!plan.matchQuery && plan.substringTerms.length === 0) {
    return [];
  }

  // When providerModel is undefined (FTS-only mode), search all models
  const modelClause = params.providerModel ? " AND c.model = ?" : "";
  const modelParams = params.providerModel ? [params.providerModel] : [];
  const substringClause = plan.substringTerms.map(() => " AND c.text LIKE ? ESCAPE '\\'").join("");
  const substringParams = plan.substringTerms.map((term) => `%${escapeLikePattern(term)}%`);
  const whereClause = plan.matchQuery
    ? `${params.ftsTable} MATCH ?${substringClause}${modelClause}${params.sourceFilter.sql}`
    : `1=1${substringClause}${modelClause}${params.sourceFilter.sql}`;
  const queryParams = [
    ...(plan.matchQuery ? [plan.matchQuery] : []),
    ...substringParams,
    ...modelParams,
    ...params.sourceFilter.params,
    params.limit,
  ];
  const rankExpression = plan.matchQuery ? `bm25(${params.ftsTable})` : "0";

  // Filter by agent when searching a shared store
  const agentFilter = buildAgentFilter(params.agentId);
  const rows = params.db
    .prepare(
      `SELECT ${params.ftsTable}.id, ${params.ftsTable}.path, ${params.ftsTable}.source, ${params.ftsTable}.start_line, ${params.ftsTable}.end_line, ${params.ftsTable}.text,\n` +
        `       ${rankExpression} AS rank\n` +
        `  FROM ${params.ftsTable}\n` +
        `  JOIN chunks c ON c.id = ${params.ftsTable}.id\n` +
        ` WHERE ${whereClause}${agentFilter.sql}\n` +
        ` ORDER BY rank ASC\n` +
        ` LIMIT ?`,
    )
    .all(...queryParams, ...agentFilter.params) as Array<{
    id: string;
    path: string;
    source: SearchSource;
    start_line: number;
    end_line: number;
    text: string;
    rank: number;
  }>;

  return rows.map((row) => {
    const textScore = plan.matchQuery ? params.bm25RankToScore(row.rank) : 1;
    return {
      id: row.id,
      path: row.path,
      startLine: row.start_line,
      endLine: row.end_line,
      score: textScore,
      textScore,
      snippet: truncateUtf16Safe(row.text, params.snippetMaxChars),
      source: row.source,
    };
  });
}
