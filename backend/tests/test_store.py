"""存储层测试。

刻意写成「普通 assert + 函数」，不依赖 pytest 的 fixture / 参数化：

    python backend/tests/test_store.py        # 直接跑
    pytest backend/tests/                     # 也可以用 pytest 跑

理由和 LLM 客户端只用 requests 一样——复现路径上少一个依赖就少一个失败点。

其中 test_short_query_* 与 test_punctuated_query_* 是**回归测试**：
它们对应的 bug 是「静默返回错误结果」和「抛解析异常」，
不写测试的话下次换 SQLite 版本一定会复发。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weiran import store  # noqa: E402
from weiran.scenario import split_markdown  # noqa: E402


def _fresh_conn():
    """每个测试一个独立的临时库，避免相互污染。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return store.connect(tmp.name)


# -- 建表与基本读写 --------------------------------------------------------


def test_schema_creates_expected_tables():
    conn = _fresh_conn()
    names = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }
    for expected in ("meta", "episodes", "entities", "edges", "chunks", "chunks_fts"):
        assert expected in names, f"缺表 {expected}"
    version = conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()[0]
    assert version == str(store.SCHEMA_VERSION)


def test_entity_upsert_dedupes_and_merges_aliases():
    conn = _fresh_conn()
    a = store.upsert_entity(conn, name="陈默", type="person", aliases=["陈老师"])
    b = store.upsert_entity(conn, name="陈默", type="person", aliases=["辅导员陈"])
    assert a == b, "同名同类型应复用同一实体"
    assert conn.execute("SELECT count(*) FROM entities").fetchone()[0] == 1

    import json

    aliases = json.loads(
        conn.execute("SELECT aliases FROM entities WHERE id=?", (a,)).fetchone()[0]
    )
    assert set(aliases) == {"陈老师", "辅导员陈"}, f"别名未合并：{aliases}"


def test_episode_and_edge_roundtrip():
    conn = _fresh_conn()
    ep = store.upsert_episode(conn, title="P3 删帖争议", phase_id="P3", day=5)
    a = store.upsert_entity(conn, name="吴海燕", type="person")
    b = store.upsert_entity(conn, name="云溪大学", type="org")
    store.insert_edge(conn, src_id=a, dst_id=b, relation="属于", episode_id=ep)

    assert store.stats(conn) == {
        "episodes": 1, "entities": 2, "edges": 1, "chunks": 0
    }


# -- 全文检索：两个回归测试 ------------------------------------------------


def test_short_query_returns_hits_not_silent_zero():
    """回归：trigram 分词器对 < 3 字的查询静默返回 0 命中。

    「就业」是本场景最核心的两字词。这条测试断言它必须能查到东西。
    如果哪天有人把 search_lexical 里的短查询分支删掉，这条会立刻红。
    """
    conn = _fresh_conn()
    store.insert_chunk(conn, text="毕业生就业质量报告显示落实率为78.6%")

    hits = store.search_lexical(conn, "就业")
    assert len(hits) == 1, "2 字查询必须走 LIKE 兜底，否则静默丢结果"

    # 对照：确认这不是因为语料里根本没有——3 字同义查询也应该命中
    assert len(store.search_lexical(conn, "毕业生")) == 1


def test_punctuated_query_does_not_raise():
    """回归：查询串里的标点会破坏 FTS5 的 MATCH 语法。"""
    conn = _fresh_conn()
    store.insert_chunk(conn, text="2025 届落实率为 78.6%，较上年下降 13.7 个百分点")

    for q in ("78.6", "78.6%", "-13.7", "（公开）"):
        try:
            store.search_lexical(conn, q)
        except Exception as exc:  # noqa: BLE001 - 目的就是断言不抛
            raise AssertionError(f"查询 {q!r} 不应抛异常：{exc}") from exc


def test_absent_term_returns_empty():
    """确认兜底逻辑没有引入误召回。"""
    conn = _fresh_conn()
    store.insert_chunk(conn, text="毕业生就业质量报告")
    assert store.search_lexical(conn, "量子纠缠") == []
    assert store.search_lexical(conn, "   ") == []


def test_fts_finds_multi_char_phrase():
    conn = _fresh_conn()
    store.insert_chunk(conn, text="辅导员被要求劝导学生删除帖子")
    hits = store.search_lexical(conn, "删除帖子")
    assert len(hits) == 1


# -- 向量检索 --------------------------------------------------------------


def test_vector_search_ranks_by_similarity():
    conn = _fresh_conn()
    store.insert_chunk(conn, text="甲", embedding=np.array([1.0, 0.0, 0.0]))
    store.insert_chunk(conn, text="乙", embedding=np.array([0.0, 1.0, 0.0]))
    store.insert_chunk(conn, text="丙", embedding=np.array([0.9, 0.1, 0.0]))

    results = store.search_vector(conn, np.array([1.0, 0.0, 0.0]), limit=3)
    assert [r[0]["text"] for r in results][:2] == ["甲", "丙"]
    assert results[0][1] > results[1][1] > results[2][1]

    # 写入时已归一化，故自身相似度应为 1
    assert abs(results[0][1] - 1.0) < 1e-6


def test_vector_dimension_mismatch_raises():
    conn = _fresh_conn()
    store.insert_chunk(conn, text="甲", embedding=np.array([1.0, 0.0, 0.0]))
    try:
        store.search_vector(conn, np.array([1.0, 0.0]))
    except ValueError as exc:
        assert "维度不一致" in str(exc)
    else:
        raise AssertionError("维度不一致时必须报错，否则会算出无意义的相似度")


def test_vector_search_skips_chunks_without_embedding():
    conn = _fresh_conn()
    store.insert_chunk(conn, text="无向量")
    assert store.search_vector(conn, np.array([1.0, 0.0])) == []


# -- 切分 ------------------------------------------------------------------


def test_split_markdown_splits_on_h2():
    text = "# 标题\n前言\n\n## 一节\n甲\n\n## 二节\n乙"
    chunks = split_markdown(text)
    assert len(chunks) == 3
    assert chunks[1].startswith("## 一节")
    assert chunks[2].startswith("## 二节")


def test_split_markdown_respects_max_size():
    body = "\n\n".join("段落" + str(i) * 200 for i in range(6))
    chunks = split_markdown(f"## 大节\n{body}")
    assert len(chunks) > 1, "超长小节必须被继续切分"
    for c in chunks:
        # 允许单段落自身超限（无法再切），但装箱后的块不应远超上限
        assert len(c) <= 1600, f"块过大：{len(c)}"


def test_split_markdown_keeps_short_section_intact():
    """短小节不应被腰斩——种子材料里一条完整的对话记录是有意义的检索单元。

    用例形状对齐真实种子材料：h1 标题 + 表格前言，然后才是 `## 材料 X-Y`。
    """
    text = (
        "# 种子材料 03 · P3 删帖争议\n\n| 项 | 值 |\n\n"
        "## 材料 3-B\n> 来源：某处\n\n```\n[09:14] 辅导员：在吗\n```"
    )
    chunks = split_markdown(text)
    assert len(chunks) == 2, f"应为「前言 + 3-B」两节，实际 {len(chunks)}"
    assert chunks[1].startswith("## 材料 3-B")
    assert "09:14" in chunks[1], "对话记录不应被拆散"
    assert "09:14" not in chunks[0]


# -- 端到端：装载真实场景 --------------------------------------------------


def test_load_real_scenario():
    repo_root = Path(__file__).resolve().parents[2]
    scenario = repo_root / "benchmark" / "scenarios" / "employment_trust_crisis"
    if not scenario.is_dir():
        return  # 场景不在时跳过，不让测试因为缺数据而失败

    conn = _fresh_conn()
    from weiran.scenario import load

    load(conn, scenario, reset=True)
    counts = store.stats(conn)

    assert counts["episodes"] == 5, f"应有 5 个阶段，实际 {counts['episodes']}"
    assert counts["entities"] == 27, f"应有 27 个角色，实际 {counts['entities']}"
    assert counts["chunks"] > 20, "种子材料应切出足够多的检索单元"

    # 真实查询必须能在装载后的库里命中
    assert store.search_lexical(conn, "灵活就业"), "核心术语查不到东西"
    assert store.search_lexical(conn, "辅导员"), "关键角色群体查不到东西"

    # 幂等：重复装载不应产生重复实体
    load(conn, scenario, reset=True)
    assert store.stats(conn)["entities"] == 27


# -- 简易 runner -----------------------------------------------------------


def _run() -> int:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - len(failed)}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
