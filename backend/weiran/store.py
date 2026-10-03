"""本地知识存储：SQLite 单文件 + FTS5 全文检索 + 向量列。

替换上游硬锁的 Zep Cloud。理由不是「Zep 不好」，而是**评委要能一条命令跑起来**：
一个 .db 文件可以随仓库分发、可以 diff、可以删掉重建，不需要任何服务端。

## 关于中文全文检索的三个实测结论

这些是在本机 SQLite 3.45.1 上**实测**出来的，不是推测：

1. FTS5 的默认分词器 `unicode61` 不切分中文，整段会被当成一个 token，
   中文检索完全不可用。必须用 `tokenize='trigram'`。
2. `trigram` 分词器**对长度 < 3 的查询串静默返回 0 命中**——不报错，只是给错答案。
   「就业」这种最核心的两字词会直接失效。这是最危险的一类 bug，必须在调用层
   兜住：短查询走 `LIKE`。
3. 查询串里的标点会破坏 MATCH 语法（`78.6` 直接抛 `syntax error`）。
   把整串当短语并转义双引号可以解决。

## 关于向量的规模抉择

语料是 5 份种子材料，切下来几百个 chunk 量级。这个规模下：

- 不需要 ANN 索引，全量暴力算余弦即可（numpy 一次矩阵乘法的事）；
- 向量检索的召回**不可能**超过词法检索，因为语料太小、同义表述太少。

所以本轮**只实现词法与向量两条独立通路，刻意不做融合（RRF）**。
在几百个 chunk 上做 RRF 是增加复杂度而不产生可测量的收益。
本条取舍与向量通路的接线状态一并记在《开源及第三方资源使用清单》第二节。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

# trigram 分词器要求查询串至少 3 个字符。低于此长度改走 LIKE。
_TRIGRAM_MIN = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- 情景：时间线上的一段，对应种子材料里的一个阶段
CREATE TABLE IF NOT EXISTS episodes (
    id         TEXT PRIMARY KEY,
    phase_id   TEXT,
    day        INTEGER,
    title      TEXT NOT NULL,
    summary    TEXT,
    source     TEXT,               -- 材料编号，如 SM-01
    created_at TEXT NOT NULL
);

-- 实体：人、机构、文档、事件、指标
CREATE TABLE IF NOT EXISTS entities (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    type             TEXT NOT NULL,
    aliases          TEXT,         -- JSON array
    attrs            TEXT,         -- JSON object
    first_seen_phase TEXT,
    created_at       TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name_type
    ON entities(name, type);

CREATE TABLE IF NOT EXISTS edges (
    id         TEXT PRIMARY KEY,
    src_id     TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    dst_id     TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation   TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    episode_id TEXT REFERENCES episodes(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(relation);

-- 检索单元。embedding 存 float32 的裸字节，写入前已 L2 归一化，
-- 因此检索时余弦相似度退化为点积。
CREATE TABLE IF NOT EXISTS chunks (
    id         TEXT PRIMARY KEY,
    episode_id TEXT REFERENCES episodes(id) ON DELETE CASCADE,
    source     TEXT,
    ordinal    INTEGER NOT NULL DEFAULT 0,
    text       TEXT NOT NULL,
    embedding  BLOB,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_episode ON chunks(episode_id);

-- 独立 FTS5 表（不用 external content）。
-- 用 content='chunks' 可以省一份文本存储，但 rowid 必须手工维护对齐，
-- 多一个失败模式；我们的语料只有几百个 chunk，重复存一份文本的代价可以忽略。
-- chunk_id 标 UNINDEXED，只作为回连 chunks 的外键，不参与匹配。
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    chunk_id UNINDEXED,
    tokenize='trigram'
);
"""


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


# -- 连接与建表 ------------------------------------------------------------


def connect(db_path: str | Path) -> sqlite3.Connection:
    """打开（必要时创建）数据库，并保证 schema 就绪。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # WAL：读写并发时更稳，且崩溃后不易损坏
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """建表。可重复调用。

    Raises:
        RuntimeError: 当前 SQLite 不含 FTS5。与其在后面对着一个空结果集发呆，
            不如在这里直接说清楚。
    """
    try:
        conn.executescript("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x);")
        conn.execute("DROP TABLE IF EXISTS _fts5_probe;")
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "当前 Python 的 SQLite 未编译 FTS5 支持，全文检索不可用。"
            f"原始错误：{exc}"
        ) from exc

    conn.executescript(_SCHEMA)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


# -- 写入 ------------------------------------------------------------------


def upsert_episode(
    conn: sqlite3.Connection,
    *,
    title: str,
    phase_id: str | None = None,
    day: int | None = None,
    summary: str | None = None,
    source: str | None = None,
    episode_id: str | None = None,
) -> str:
    eid = episode_id or new_id("ep_")
    conn.execute(
        """INSERT INTO episodes(id, phase_id, day, title, summary, source, created_at)
           VALUES(?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET
             phase_id=excluded.phase_id, day=excluded.day, title=excluded.title,
             summary=excluded.summary, source=excluded.source""",
        (eid, phase_id, day, title, summary, source, _now()),
    )
    conn.commit()
    return eid


def upsert_entity(
    conn: sqlite3.Connection,
    *,
    name: str,
    type: str,
    aliases: list[str] | None = None,
    attrs: dict | None = None,
    first_seen_phase: str | None = None,
    entity_id: str | None = None,
) -> str:
    """按 (name, type) 去重。已存在则合并 aliases / attrs，不覆盖已有的非空字段。"""
    row = conn.execute(
        "SELECT id, aliases, attrs FROM entities WHERE name=? AND type=?", (name, type)
    ).fetchone()

    if row is None:
        eid = entity_id or new_id("en_")
        conn.execute(
            """INSERT INTO entities(id, name, type, aliases, attrs,
                                    first_seen_phase, created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (
                eid, name, type,
                json.dumps(aliases or [], ensure_ascii=False),
                json.dumps(attrs or {}, ensure_ascii=False),
                first_seen_phase, _now(),
            ),
        )
        conn.commit()
        return eid

    eid = row["id"]
    old_aliases = set(json.loads(row["aliases"] or "[]"))
    merged_aliases = sorted(old_aliases | set(aliases or []))
    merged_attrs = {**json.loads(row["attrs"] or "{}"), **(attrs or {})}
    conn.execute(
        "UPDATE entities SET aliases=?, attrs=?, first_seen_phase=COALESCE(first_seen_phase, ?) "
        "WHERE id=?",
        (
            json.dumps(merged_aliases, ensure_ascii=False),
            json.dumps(merged_attrs, ensure_ascii=False),
            first_seen_phase, eid,
        ),
    )
    conn.commit()
    return eid


def insert_edge(
    conn: sqlite3.Connection,
    *,
    src_id: str,
    dst_id: str,
    relation: str,
    confidence: float = 1.0,
    episode_id: str | None = None,
) -> str:
    eid = new_id("ed_")
    conn.execute(
        """INSERT INTO edges(id, src_id, dst_id, relation, confidence,
                             episode_id, created_at)
           VALUES(?,?,?,?,?,?,?)""",
        (eid, src_id, dst_id, relation, confidence, episode_id, _now()),
    )
    conn.commit()
    return eid


def insert_chunk(
    conn: sqlite3.Connection,
    *,
    text: str,
    episode_id: str | None = None,
    source: str | None = None,
    ordinal: int = 0,
    embedding: np.ndarray | None = None,
    chunk_id: str | None = None,
) -> str:
    """写入一个检索单元，并同步全文索引。

    embedding 会被 L2 归一化后以 float32 存储。
    """
    cid = chunk_id or new_id("ck_")
    blob = None
    if embedding is not None:
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        blob = vec.tobytes()

    conn.execute(
        """INSERT INTO chunks(id, episode_id, source, ordinal, text,
                              embedding, created_at)
           VALUES(?,?,?,?,?,?,?)""",
        (cid, episode_id, source, ordinal, text, blob, _now()),
    )
    conn.execute(
        "INSERT INTO chunks_fts(text, chunk_id) VALUES(?, ?)", (text, cid)
    )
    conn.commit()
    return cid


# -- 检索 ------------------------------------------------------------------


def _fts_phrase(query: str) -> str:
    """把查询串整个当成一个短语。

    不转义的话，`78.6` 会因句点被 FTS5 解析成语法错误。
    内部的双引号翻倍是 FTS5 的转义规则。
    """
    return '"' + query.replace('"', '""') + '"'


def search_lexical(
    conn: sqlite3.Connection, query: str, *, limit: int = 10
) -> list[sqlite3.Row]:
    """词法检索。

    长度 < 3 走 LIKE —— 见模块开头第 2 条实测结论。
    短查询在 trigram 下会静默返回空，这是不能接受的。
    """
    q = query.strip()
    if not q:
        return []

    if len(q) < _TRIGRAM_MIN:
        return conn.execute(
            """SELECT c.*, NULL AS score FROM chunks c
               WHERE c.text LIKE ?
               ORDER BY c.ordinal LIMIT ?""",
            (f"%{q}%", limit),
        ).fetchall()

    try:
        return conn.execute(
            """SELECT c.*, bm25(chunks_fts) AS score
               FROM chunks_fts
               JOIN chunks c ON c.id = chunks_fts.chunk_id
               WHERE chunks_fts MATCH ?
               ORDER BY score
               LIMIT ?""",
            (_fts_phrase(q), limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # 转义后仍可能有极端输入触发解析错误。降级到 LIKE 而不是让管线崩掉。
        return conn.execute(
            """SELECT c.*, NULL AS score FROM chunks c
               WHERE c.text LIKE ?
               ORDER BY c.ordinal LIMIT ?""",
            (f"%{q}%", limit),
        ).fetchall()


def search_vector(
    conn: sqlite3.Connection, query_vec: np.ndarray, *, limit: int = 10
) -> list[tuple[sqlite3.Row, float]]:
    """向量检索。

    全量暴力计算。在几百个 chunk 的规模下，这比引入 ANN 索引更快也更简单。
    存储时已归一化，故点积即余弦相似度。
    """
    rows = conn.execute(
        "SELECT * FROM chunks WHERE embedding IS NOT NULL"
    ).fetchall()
    if not rows:
        return []

    q = np.asarray(query_vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(q))
    if norm == 0:
        return []
    q = q / norm

    matrix = np.vstack(
        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    )
    if matrix.shape[1] != q.shape[0]:
        raise ValueError(
            f"向量维度不一致：库内 {matrix.shape[1]}，查询 {q.shape[0]}。"
            "换嵌入模型后需要重建索引。"
        )

    sims = matrix @ q
    order = np.argsort(-sims)[:limit]
    return [(rows[i], float(sims[i])) for i in order]


def stats(conn: sqlite3.Connection) -> dict[str, int]:
    """库内计数。用于自检与进度报告。"""
    return {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in ("episodes", "entities", "edges", "chunks")
    }
