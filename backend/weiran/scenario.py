"""场景装载：把种子材料读进本地知识库。

这一层**不调用任何 LLM**，因此可以离线跑、可以进 CI、可以作为复现的起点。
LLM 抽取（实体识别、关系抽取）是在此之上的增补，不是前提。

装载三样东西：
  - episodes：从 reference_data.json 的阶段定义
  - chunks  ：从种子材料的 markdown 正文，按 `##` 小节切分
  - entities：从 reference_data.json 的角色表（金标角色，确定性来源）

用法：
    python -m weiran.scenario --scenario benchmark/scenarios/employment_trust_crisis
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import store

# 种子材料按文件名前缀映射到阶段。01 -> P1，以此类推。
_MATERIAL_PHASE_RE = re.compile(r"^(\d{2})_")

# 超过这个长度的段落再往下切。中文一个字约等于一个 token 的量级，
# 800 字对检索粒度来说偏长，但保留完整语境对简报生成更有用，取折中。
_MAX_CHUNK_CHARS = 800


def _phase_from_filename(path: Path) -> str | None:
    m = _MATERIAL_PHASE_RE.match(path.name)
    if not m:
        return None
    return f"P{int(m.group(1))}"


def split_markdown(text: str) -> list[str]:
    """按 `##` 小节切分，过长的小节再按空行分段。

    种子材料的结构本身就很规整（每份由若干条「材料 X-Y」组成），
    顺着它切比按固定窗口切更自然，也不会把一条完整的对话记录腰斩。
    """
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current:
                sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    chunks: list[str] = []
    for sec in sections:
        if not sec:
            continue
        if len(sec) <= _MAX_CHUNK_CHARS:
            chunks.append(sec)
            continue
        # 超长小节：按空行切成段，再贪心装箱
        buf = ""
        for para in re.split(r"\n\s*\n", sec):
            para = para.strip()
            if not para:
                continue
            if buf and len(buf) + len(para) > _MAX_CHUNK_CHARS:
                chunks.append(buf.strip())
                buf = para
            else:
                buf = f"{buf}\n\n{para}" if buf else para
        if buf.strip():
            chunks.append(buf.strip())
    return chunks


def load(conn, scenario_dir: str | Path, *, reset: bool = False) -> dict[str, int]:
    """把一个场景装载进库。

    Args:
        reset: 先清空 chuck / episode / entity / edge 表。默认 False，可重复装载。

    Returns:
        各表的写入计数。
    """
    root = Path(scenario_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"场景目录不存在：{root}")

    ref_path = root / "reference_data.json"
    if not ref_path.is_file():
        raise FileNotFoundError(f"缺少金标文件：{ref_path}")
    ref = json.loads(ref_path.read_text(encoding="utf-8"))

    if reset:
        # 顺序要紧：chunks_fts 是独立表，不会被级联删掉，必须显式清
        for sql in (
            "DELETE FROM edges", "DELETE FROM chunks", "DELETE FROM chunks_fts",
            "DELETE FROM entities", "DELETE FROM episodes",
        ):
            conn.execute(sql)
        conn.commit()

    counters = {"episodes": 0, "chunks": 0, "entities": 0}

    # -- episodes：阶段 --------------------------------------------------
    phase_to_episode: dict[str, str] = {}
    for ph in ref.get("phases", []):
        eid = store.upsert_episode(
            conn,
            title=f"{ph['id']} {ph['label']}",
            phase_id=ph["id"],
            day=ph.get("day"),
            summary=ph.get("trigger"),
            source="reference_data.json",
        )
        phase_to_episode[ph["id"]] = eid
        counters["episodes"] += 1

    # -- chunks：种子材料正文 --------------------------------------------
    seed_dir = root / "seed_materials"
    if seed_dir.is_dir():
        for md in sorted(seed_dir.glob("*.md")):
            phase = _phase_from_filename(md)
            episode_id = phase_to_episode.get(phase) if phase else None
            body = md.read_text(encoding="utf-8")
            for i, chunk in enumerate(split_markdown(body)):
                store.insert_chunk(
                    conn,
                    text=chunk,
                    episode_id=episode_id,
                    source=md.name,
                    ordinal=i,
                )
                counters["chunks"] += 1

    # -- entities：金标角色 ------------------------------------------------
    for actor in ref.get("actors", []):
        attrs = {
            "group": actor.get("group"),
            "role": actor.get("role"),
            "stance": actor.get("stance"),
            "knows": actor.get("knows", []),
            "hidden": actor.get("hidden", []),
        }
        store.upsert_entity(
            conn,
            name=actor["name"],
            type="person",
            attrs=attrs,
            entity_id=actor["id"],  # 金标 id 直接作为主键，便于断言按 id 对齐
        )
        counters["entities"] += 1

    return counters


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="把场景种子材料装载进本地知识库")
    ap.add_argument(
        "--scenario",
        default="benchmark/scenarios/employment_trust_crisis",
        help="场景目录（相对仓库根目录）",
    )
    ap.add_argument("--db", default=None, help="数据库路径，默认 data/weiran.db")
    ap.add_argument("--reset", action="store_true", help="装载前清空数据表")
    args = ap.parse_args(argv)

    from .config import REPO_ROOT

    db_path = Path(args.db) if args.db else REPO_ROOT / "data" / "weiran.db"
    scenario_dir = Path(args.scenario)
    if not scenario_dir.is_absolute():
        scenario_dir = REPO_ROOT / scenario_dir

    conn = store.connect(db_path)
    counters = load(conn, scenario_dir, reset=args.reset)
    total = store.stats(conn)

    print(f"数据库    {db_path}")
    print(f"场景      {scenario_dir}")
    print(f"本次写入  episodes={counters['episodes']} "
          f"chunks={counters['chunks']} entities={counters['entities']}")
    print(f"库内合计  {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
