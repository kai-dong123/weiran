"""多智能体推演驱动：把角色画像跑成一轮轮的 agent 动作。

**为什么自己写驱动，而不是整份搬上游的 `run_parallel_simulation.py`：**

上游那份 1699 行里，真正与 OASIS 交互的部分不到三分之一，其余是给
「Web 端实时下发命令」准备的：IPC 命令目录轮询、env_status 心跳、
等待命令循环、双平台 asyncio 并行、信号处理。本项目是**批式推演**——
跑完一轮拿到动作、喂给世界状态引擎、再跑下一轮——那套常驻机制是纯粹的
负重。（上游那套的用途是让人在网页上对正在跑的 agent 做访谈，
本项目保留 `ManualAction` 做**事件注入**，效果相同而实现简单得多。）

自己写还有一个更硬的理由：**世界状态引擎必须逐轮拿到动作**。
上游把动作写进 OASIS 自己的 SQLite 就不再管了，而我们要在每轮之间
读出来、算状态、再把状态回注给下一轮的 agent。这个循环是自研核心，
夹在别人的驱动里会很别扭。

**复用的部分如实标注**：可用动作列表（TWITTER_ACTIONS / REDDIT_ACTIONS）
取自上游，profile 格式取自上游的 `test_profile_format.py`。见
`docs/开源及第三方资源使用清单.md`。

用法：
    cd backend
    python -m weiran.simulate --agents 3 --rounds 2 --inspect-db   # 最小冒烟
    python -m weiran.simulate --rounds 10
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import REPO_ROOT, ConfigError, load_config
from .perception import (
    Knowledge,
    KnowledgeError,
    active_phase_by_round,
    inject_round_context,
    knowledge_cutoff_by_round,
    install_injection,
    load_knowledge,
    load_phases,
    phase_schedule,
)
from .profiles import DEFAULT_SCENARIO
from .stance import StanceClassifier
from .world_state import DIMENSIONS, WorldState, WorldStateEngine

DEFAULT_OUT = "data/simulation"

# 可用动作。**取自上游** run_parallel_simulation.py:178-202（AGPL-3.0）。
# 不含 INTERVIEW —— 它只能由 ManualAction 触发，本项目用它做事件注入。
TWITTER_ACTIONS_NAMES = (
    "CREATE_POST", "LIKE_POST", "REPOST", "FOLLOW", "DO_NOTHING", "QUOTE_POST",
)
REDDIT_ACTIONS_NAMES = (
    "LIKE_POST", "DISLIKE_POST", "CREATE_POST", "CREATE_COMMENT",
    "LIKE_COMMENT", "DISLIKE_COMMENT", "SEARCH_POSTS", "SEARCH_USER",
    "TREND", "REFRESH", "DO_NOTHING", "FOLLOW", "MUTE",
)


@dataclass
class RoundLog:
    """一轮的产出。世界状态引擎的输入就是它。"""

    index: int
    seconds: float = 0.0
    llm_seconds: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # 本轮的调用失败次数与最慢单次。**这两个字段存在的理由就是上面那段注释**：
    # 「第二轮 409.6s」那次故障里，能区分「模型慢」与「端点在限流」的只有它们。
    # `errors` 不是它们 —— 那是 `env.step` 抛的异常数，与 LLM 调用无关。
    failures: int = 0
    slowest: float = 0.0
    actions: list[dict] = field(default_factory=list)
    errors: int = 0
    # 世界状态：本轮识别出的行为构成，以及推进后的六维取值。
    behaviors: list[str] = field(default_factory=list)
    state: dict[str, float] = field(default_factory=dict)
    # 感知层。**这份记录是必须的**：「注入没生效」与「注入生效了但没差别」
    # 在六维曲线上一模一样，只有这里能把两者分开。曲线平的时候第一件事
    # 就是看它 —— 若注入次数是 0，问题在接线，不在模型。
    phase_id: str = ""                  # 本轮触发的阶段（P1..P5），无则空
    injected: dict[str, str] = field(default_factory=dict)   # actor_id -> 块摘要
    injection_sample: str = ""          # 首个非空块全文，供人眼核对措辞


@dataclass
class SimulationResult:
    rounds: list[RoundLog] = field(default_factory=list)
    db_path: Path | None = None
    total_seconds: float = 0.0
    stance_summary: str = ""
    # 运行口径。**落盘时必须一起写出去** —— 压缩模式下的曲线与 round=day
    # 不可比（激励项按步累加，见 days_per_round 的注释），一份不知道自己
    # 是压缩过的结果文件，日后必然被当成 round=day 的结果引用。
    days_per_round: float = 1.0
    compressed: bool = False
    phases_on: bool = True
    knowledge_on: bool = True
    feedback_on: bool = True

    @property
    def total_actions(self) -> int:
        return sum(len(r.actions) for r in self.rounds)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

# camel 的默认上下文上限是 999,999,999（模型不被识别时的兜底值）。
# 我们显式给一个：**这个值同时是两个东西**，见 build_model 的说明。
DEFAULT_MAX_TOKENS = 16384


def build_model(
    config,
    *,
    thinking: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float | None = None,
    temperature: float | None = None,
    seed: int | None = None,
):
    """按本项目配置造一个 camel 模型后端。

    **三处非默认设置，都是实测逼出来的，不是照文档写的：**

    1. `extra_body={"thinking": {"type": "disabled"}}`
       本项目所用模型默认是推理模型，思维链占输出 token 的 92%~96%。
       关掉之后输出降到约 1/13。**必须走 `extra_body`** —— camel 会把
       `model_config_dict` 原样 `**` 展开给 `client.chat.completions.create()`，
       而 openai SDK 不认识 `thinking` 这个关键字，直接传会
       `TypeError: unexpected keyword argument`。

    2. **`max_tokens` 在 camel 里是一人身兼两职的，这一点文档没写。**
       `BaseModelBackend.token_limit` 的实现是：

           return self.model_config_dict.get("max_tokens") or self.model_type.token_limit

       而 `token_limit` 又被 `ScoreBasedContextCreator` 拿去当**上下文窗口上限**。
       于是 `max_tokens` 既是发给 API 的**输出上限**，又是 camel 截断
       agent 对话记忆的**上下文上限**。

       我一开始把它设成 2048（想让输出有界），结果每轮日志都在刷：

           WARNING: Context truncation performed: before=6950, after=2047, limit=2048

       —— agent 的记忆被砍到 2048 token，profile 加上几条帖子就撑爆了。
       推演会退化成「失忆的 agent 各说各话」。

       所以这个值必须**按上下文来定**，不能按输出定。取 16384：
       实测单轮 prompt 约 6000–7000 token，留一倍余量；
       同时 16384 仍是单次输出的上界，不至于跑飞。

    3. camel 不认识 `deepseek-flash`（不在它的模型枚举里），因此
       `model_type.token_limit` 落到兜底值 999,999,999 —— 不显式给值
       就等于没有上下文上限，也就没有账单上限。
    """
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    # camel 通过这两个环境变量找端点，不认我们自己的命名。
    os.environ["OPENAI_API_KEY"] = config.llm.api_key
    os.environ["OPENAI_API_BASE_URL"] = config.llm.base_url

    cfg: dict = {"max_tokens": max_tokens}
    if config.llm.send_thinking_param:
        cfg["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}

    # 采样控制。**这两项是给「演示可重复」用的，不是给「模型更聪明」用的。**
    #
    # 实测过一件事，值得记在这里：六维曲线**无法端到端复现**，
    # 原因不在世界状态引擎（它给定行为序列是完全确定的，单测已锁），
    # 而在**上游** —— agent 每跑一次说的话都不一样，于是喂给引擎的行为序列
    # 每次都不一样。缓存分类标签只能保证「同一句话给同一标签」，
    # 保证不了「同一场推演说出同一句话」。
    #
    # 压 temperature + 固定 seed 能显著收窄这个抖动，代价是 agent 说话变单调
    # （实测低温度下会出现多个 agent 发出雷同帖文）。**这是一个真实的取舍，
    # 不是免费的**，所以默认值不动，由调用方显式选。
    #
    # 另外：`seed` 只被服务端「尽力」遵守，OpenAI 兼容端点不保证逐位可复现。
    # 所以正式的可复现性方案仍然是「多 seed 跑 3 次报均值与方差」，
    # 见 进度.md 第 5 步，而不是指望 seed 一劳永逸。
    if temperature is not None:
        cfg["temperature"] = temperature
    if seed is not None:
        cfg["seed"] = seed

    kwargs: dict = {}
    if timeout is not None:
        kwargs["timeout"] = timeout

    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=config.llm.model,
        model_config_dict=cfg,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# LLM 调用观测
#
# 为什么必须自己插桩：camel 不暴露调用次数与耗时。而本项目要知道的正是
# 「一次推演多少次调用、每轮多久、有没有慢调用」—— 这个数字决定演示规模。
#
# 起因是一次真实故障：一次 3 agent 的冒烟里，第一轮 2.2s、**第二轮 409.6s**，
# 而同样输入的另一次运行第二轮只要 5.4s。差异巨大却没有任何日志线索，
# 因为 camel 把重试与超时都吞在内部。**看不见的等待最贵** —— 它会让人
# 以为是模型慢，实际可能是端点在限流。
# ---------------------------------------------------------------------------

@dataclass
class CallStats:
    """LLM 调用统计。同步与异步两条路都记。"""

    calls: int = 0
    seconds: float = 0.0
    failures: int = 0
    slowest: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # 每次调用的耗时。`slowest` 是**最大值，做不了差分**，所以想按轮报
    # 「本轮最慢一次」就必须留下逐次样本。代价是每次调用一个 float（可忽略），
    # 收益是那个 409s 的故障下次能直接定位到轮次。
    samples: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return self.seconds / self.calls if self.calls else 0.0

    def summary(self) -> str:
        return (
            f"{self.calls} 次调用 / {self.seconds:.1f}s"
            f"（均值 {self.mean:.1f}s，最慢 {self.slowest:.1f}s，失败 {self.failures}）"
            f" · token 入 {self.prompt_tokens} / 出 {self.completion_tokens}"
        )


# 要包的方法名。四条都包：非结构化与结构化（response_format）各一对，
# 少包一条就会在 OASIS 换用结构化输出时**静默**报 0 次调用。
_CALL_METHODS = (
    "_request_chat_completion",
    "_arequest_chat_completion",
    "_request_parse",
    "_arequest_parse",
)


def instrument_model(model) -> CallStats:
    """给 camel 模型**实例所属的类**插上计数与计时。

    **必须按实例的类来，不能按 `OpenAICompatibleModel` 猜。**
    我第一版就是猜的，结果跑完报「0 次调用 / 0.0s」，还触发了
    「墙钟远超 LLM 时间」的告警 —— 而真相是 ModelFactory 对
    `ModelPlatformType.OPENAI` 返回的是 `OpenAIModel`，
    **不是** `OpenAICompatibleModel`（两者都继承 BaseModelBackend，
    但各自实现了一套 `_request_*`）。包装打在了没人调用的类上。

    教训记在这里：插桩本身也会骗人。所以 `summary()` 与
    「0 次调用」都被当成异常信号对待，而不是当成结论。

    幂等：同一个类重复包装不会叠加。
    """
    cls = type(model)
    if getattr(cls, "_weiran_instrumented", False):
        return getattr(cls, "_weiran_stats")

    stats = CallStats()

    def _record(el: float, result) -> None:
        stats.seconds += el
        stats.slowest = max(stats.slowest, el)
        stats.samples.append(el)
        # 流式响应没有 usage，结构化解析的返回也不是 ChatCompletion。
        usage = getattr(result, "usage", None)
        if usage is not None:
            stats.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            stats.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def _wrap(name: str) -> bool:
        original = getattr(cls, name, None)
        if original is None:
            return False

        if asyncio.iscoroutinefunction(original):
            async def wrapper(self, *a, **kw):
                t0 = time.monotonic()
                stats.calls += 1
                try:
                    result = await original(self, *a, **kw)
                except BaseException:
                    stats.failures += 1
                    # 失败的调用**也要记耗时**。超时/重试那一次往往正是最慢的
                    # 一次（409s 那次就可能落在这一支），只记成功的会让
                    # 「本轮最慢」系统性偏小 —— 而它存在的意义就是抓这种。
                    stats.samples.append(time.monotonic() - t0)
                    raise
                else:
                    _record(time.monotonic() - t0, result)
                    return result
        else:
            def wrapper(self, *a, **kw):
                t0 = time.monotonic()
                stats.calls += 1
                try:
                    result = original(self, *a, **kw)
                except BaseException:
                    stats.failures += 1
                    stats.samples.append(time.monotonic() - t0)
                    raise
                else:
                    _record(time.monotonic() - t0, result)
                    return result

        wrapper.__name__ = name
        setattr(cls, name, wrapper)
        return True

    wrapped = [n for n in _CALL_METHODS if _wrap(n)]
    if not wrapped:
        raise RuntimeError(
            f"{cls.__name__} 上找不到任何可插桩的方法（{_CALL_METHODS}）——"
            " camel 版本可能变了，插桩已失效，此时报出的 0 次调用是假的"
        )

    cls._weiran_instrumented = True
    cls._weiran_stats = stats
    return stats


# ---------------------------------------------------------------------------
# 动作读取
# ---------------------------------------------------------------------------

# OASIS 把动作写进自己的 SQLite。表名与列名属于**上游 OASIS 的内部实现**，
# 不是稳定接口 —— 所以这里读表失败不抛异常，而是退化成「本轮无动作」，
# 并把原因记下来。一次读不到不该让整场推演崩掉。
_ACTION_TABLES = ("twitter_action", "reddit_action", "action", "trace")


def read_actions(db_path: Path, *, since_rowid: int = 0) -> tuple[list[dict], int]:
    """从 OASIS 的库里读出新增动作。

    Returns:
        (动作列表, 最大 rowid)。读不出来时返回 ([], since_rowid)。
    """
    if not db_path.is_file():
        return [], since_rowid

    out: list[dict] = []
    max_rowid = since_rowid
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return [], since_rowid

    try:
        conn.row_factory = sqlite3.Row
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        target = next((t for t in _ACTION_TABLES if t in tables), None)
        if target is None:
            return [], since_rowid

        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({target})").fetchall()}
        if "rowid" not in cols:
            pass  # rowid 是隐式列，PRAGMA 里可能不出现

        for row in conn.execute(
            f"SELECT rowid AS _rid, * FROM {target} WHERE rowid > ? ORDER BY rowid",
            (since_rowid,),
        ).fetchall():
            d = dict(row)
            max_rowid = max(max_rowid, int(d.pop("_rid", 0)))
            out.append(d)
    except sqlite3.Error:
        return [], since_rowid
    finally:
        conn.close()

    return out, max_rowid


def load_posts(db_path: Path) -> dict[int, dict]:
    """读 `post` 表，返回 post_id -> 行。用于把动作还原成「它传的是什么话」。"""
    if not db_path.is_file():
        return {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}
    try:
        conn.row_factory = sqlite3.Row
        return {int(r["post_id"]): dict(r)
                for r in conn.execute("SELECT * FROM post").fetchall()}
    except sqlite3.Error:
        return {}
    finally:
        conn.close()


# OASIS 动作 -> 世界状态引擎的行为类型。**这里有一个必须踩过才知道的坑：**
#
# `quote_post` 在 `post` 表里落两列：`content` 是**被引用帖子的原文**
# （原样抄了一份），`quote_content` 才是**这个 agent 自己说的话**。
# 实测确认：post_id 2/3/4 的 `content` 全都是第 1 条种子事件的一字不差副本。
#
# 所以引用帖必须读 `quote_content`。若读 `content`，那么**每一条引用帖都会
# 被 classify() 归成与种子事件同一个类型** —— 六维曲线会立刻退化成「一条平线」，
# 而且看上去毫不出错。这是个静默错误，值得专门写在这里。
#
# `like_post` 也计入：在一个校园舆情时间线上，「点赞一条指控」不是沉默，
# 它抬的是那条指控的声势。计入后它按**被赞帖子的类型**参与构成占比，
# 于是「同情者的赞」与「辟谣者的赞」会推向相反的方向 —— 这正是我们要的。
_ACTION_TEXT_SOURCE: dict[str, str] = {
    "create_post": "content",
    "quote_post": "quote_content",
}

#: 不产生世界状态信号的动作。刷新与潜水不改变任何人的判断。
_SILENT_ACTIONS = frozenset({"refresh", "do_nothing", "sign_up", "follow", "mute"})


def texts_from_actions(actions: list[dict], posts: dict[int, dict]) -> list[str]:
    """把一轮的 OASIS 动作还原成「它在传什么话」。

    返回的是**文本**而不是行为类型：归类那一步交给 `stance.StanceClassifier`
    （关键词表读不懂 agent 写的自由文本，理由见该模块开头）。
    这里只负责最有把握的一件事 —— 找出每条动作真正对应的那句话。
    """
    out: list[str] = []
    for a in actions:
        name = a.get("action", "")
        if name in _SILENT_ACTIONS:
            continue
        info = a.get("info")
        if isinstance(info, str):
            try:
                info = json.loads(info)
            except (ValueError, TypeError):
                info = {}
        info = info or {}

        # 找出这条动作指向的帖子，再决定读哪一列。
        pid = info.get("post_id") or info.get("new_post_id")
        row = posts.get(int(pid)) if isinstance(pid, int) else None
        if row is None:
            continue

        col = _ACTION_TEXT_SOURCE.get(name)
        text = ""
        if col:
            text = row.get(col) or ""
        # 转帖（repost）落库时 `content` 是空串，只有 `original_post_id` 指向原文
        # （实测：post_id 8 的 content=''、original_post_id=2）。
        # 所以这里要顺着 original_post_id 找回被转的那条，
        # 否则「转帖」会被静默丢掉 —— 而转帖恰恰是 amplification 最强的信号。
        if not text and row.get("original_post_id") is not None:
            src = posts.get(int(row["original_post_id"]))
            if src is not None:
                text = (src.get("quote_content") or src.get("content") or "")
        # 引用帖若没留下自己的话（模型有时返回空），退回原文 ——
        # 至少算「他把这条传出去了」，而不是凭空丢掉一次动作。
        if not text:
            text = row.get("content") or ""
        if text.strip():
            out.append(text)
    return out


def dump_schema(db_path: Path) -> str:
    """把库结构打出来。第一次接 OASIS 时必须先看这个 ——
    表名和列名没有文档，只能实测。"""
    if not db_path.is_file():
        return f"（库不存在：{db_path}）"
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        lines = []
        for (name, sql) in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall():
            n = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            lines.append(f"\n【{name}】 {n} 行")
            lines.append(f"  {sql}")
        return "\n".join(lines) if lines else "（无表）"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 推演
# ---------------------------------------------------------------------------

async def _run(
    config,
    out_dir: Path,
    *,
    rounds: int,
    agents: int | None,
    platform: str,
    thinking: bool,
    max_tokens: int,
    seed_text: str = "",
    world_state: bool = True,
    scenario_dir: Path | None = None,
    phases_on: bool = True,
    knowledge_on: bool = True,
    feedback_on: bool = True,
    llm=None,
    stance_cache: Path | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> SimulationResult:
    import oasis
    from oasis import ActionType, LLMAction, ManualAction, generate_reddit_agent_graph, generate_twitter_agent_graph

    is_twitter = platform == "twitter"
    names = TWITTER_ACTIONS_NAMES if is_twitter else REDDIT_ACTIONS_NAMES
    actions = [getattr(ActionType, n) for n in names]

    profile_file = out_dir / ("twitter_profiles.csv" if is_twitter else "reddit_profiles.json")
    if not profile_file.is_file():
        raise FileNotFoundError(
            f"缺少 {profile_file.name} —— 先跑 python -m weiran.profiles"
        )

    db_path = out_dir / f"{platform}_simulation.db"
    if db_path.exists():
        db_path.unlink()

    model = build_model(config, thinking=thinking, max_tokens=max_tokens,
                        temperature=temperature, seed=seed)

    # 注入插槽必须在建图**之前**装上 —— 只留一个安装时间点，好排查。
    # 类级补丁对已存在实例也生效，但两个安装点会让「到底装上了没有」变难回答。
    if feedback_on or knowledge_on:
        install_injection()

    gen = generate_twitter_agent_graph if is_twitter else generate_reddit_agent_graph
    graph = await gen(profile_path=str(profile_file), model=model, available_actions=actions)

    agents_list = [a for _, a in graph.get_agents()]
    if agents is not None:
        agents_list = agents_list[:agents]

    # 知情映射。**在建图之后、花钱之前加载** —— 画像文件与 actor_knowledge.json
    # 不同源是真缺陷，要在打第一发 LLM 之前就炸，而不是跑到第 3 轮才发现。
    knowledge: Knowledge | None = None
    if knowledge_on:
        knowledge = load_knowledge(out_dir, scenario_dir)
        missing = [a.social_agent_id for a in agents_list
                   if a.social_agent_id not in knowledge.by_user_id]
        if missing:
            raise KnowledgeError(
                f"这些 agent 的 social_agent_id 在 actor_knowledge.json 里没有"
                f"对应角色：{missing}\n"
                "说明 profile 文件与知情映射不同源（一个重跑过、另一个没有），"
                "先重跑 python -m weiran.profiles。"
                "\n（注意：反查走 social_agent_id，不是 user_info.user_name —— "
                "后者实测为 None）"
            )

    env = oasis.make(
        agent_graph=graph,
        platform=oasis.DefaultPlatformType.TWITTER if is_twitter
                 else oasis.DefaultPlatformType.REDDIT,
        database_path=str(db_path),
        semaphore=10,   # 限并发，防止打爆端点
    )
    await env.reset()

    stats = instrument_model(model)

    result = SimulationResult(db_path=db_path)
    started = time.monotonic()
    result.knowledge_on = knowledge_on
    result.feedback_on = feedback_on
    # `result.phases_on` 在阶段表那段之后才赋值 —— 那里可能会因为轮数不足
    # 把 phases_on 降级成 False，在这里赋值会让落盘的 meta 声称「阶段开着」，
    # 而实际一轮都没注入。落盘口径错了比不落盘更坏。

    # **reset 也会往 trace 里写。** OASIS 在 reset 阶段为每个 agent 记一条
    # `sign_up`，只有真正 step 出来的才是本轮动作。第一次跑 3 个 agent 的
    # 冒烟时，轮 0 报「动作 30 条」—— 27 条是 reset 的 sign_up（画像文件里
    # 一共 27 个角色），只有 3 条是这一轮的动作。虚高 10 倍。
    #
    # 对世界状态引擎来说这不是小事：它会把这 27 条当成「本轮舆情动作」，
    # 于是第 0 轮的状态曲线是纯噪声。所以这里先把 reset 的水位读掉。
    # 代价是多一次读库，收益是「轮 0 的动作数与其他轮可比」。
    _, last_rowid = read_actions(db_path)

    # 事件注入。**没有它，推演是空的** —— 这不是猜测，是实测：
    # 3 个 agent 跑 3 轮，9 次动作全是 do_nothing，`post` 表 0 行。
    # 时间线上什么也没有，agent「什么都不做」是完全理性的选择，
    # 于是世界状态引擎拿到的是一条平线。
    #
    # 事件来源有两条，**阶段表优先**：
    #
    #   - **阶段表（默认）**：从金标 `reference_data.json` 的 `phases` 段读
    #     `day` 与 `trigger`，把 day 映射到轮号。本场景为
    #     P1/D0、P2/D0+2、P3/D0+5、P4/D0+8、P5/D0+14。
    #     **注入的是场景作者编写的事件标题（trigger），不是抓取内容。**
    #     已核对：`seed_materials/0N_*.md` 是带元数据表格的多段文档
    #     （材料 3-A 通知 / 3-B 截图…），**不是帖子形态**，整份当
    #     `CREATE_POST` 注入反而不真实。
    #   - **`--seed-text`（退化路径）**：没有可用阶段表时用，只在第 0 轮发一条。
    #
    # **走「事件进世界」，不走「直接改状态」。** `ws_engine.step(extra=...)`
    # 已经存在、注释也写着「手动注入事件」，但本项目刻意**不用**它：事件必须
    # 以一条帖子的形式进入时间线，六维的变化要**经由 agent 的反应**发生。
    # 直接给维度加激励等于「我们知道该涨多少」，正是本项目批评过的
    # 「看着答案推过程」。（`extra` 留作日后「外部施加 vs 社会传播」的消融对照。）
    #
    # 事件由 `ManualAction` 发出、**不经过 LLM**，所以执行者拿不到感知注入 ——
    # 这是对的：它在制造事件，不需要被告知态势。免得日后被当成 bug。
    schedule: dict[int, tuple] = {}
    phase_by_round: dict[int, str] = {}
    #: 轮号 -> 「已知可以到第几天」。`--no-phases` 时恒为 0：连删帖通知这个
    #: 事件都不注入，agent 当然不该知道它 —— 事实集合与事件集合必须同步，
    #: 否则「只关事件不关事实」会造出一个场景里根本没发生过的知情。
    knowledge_cutoff: dict[int, float] = {}
    dpr, compressed = 1.0, False
    if phases_on and scenario_dir is not None and rounds < 2:
        # 1 轮没法把 5 个阶段映射上去（至少需要「起点 + 一个后续」）。
        # **降级要出声**：默默关掉一个默认开启的开关，正是本项目专门猎杀的
        # 「静默失效」——用户会以为阶段注入跑了，只是没效果。
        print("  ⚠️ 只有 1 轮，阶段无法映射到轮号 —— **本次不注入阶段事件**。"
              "要阶段就 --rounds ≥2，要明确关掉就 --no-phases。")
        phases_on = False
    if phases_on and scenario_dir is not None:
        phase_list = load_phases(scenario_dir)
        schedule, dpr, compressed = phase_schedule(phase_list, rounds)
        # 每一轮归属到「当时生效的最近一个阶段」，只用于给引擎识别出的事件打标签。
        phase_by_round = active_phase_by_round(phase_list, rounds, dpr)
        # 知情范围也要按轮裁：第 0 轮不该知道 P3 才下发的删帖通知。
        # 只关掉 --phases 而没有这一条时，agent 会拿到「阶段事件不注入、
        # 但相关事实已经知道」的组合 —— 比两者都开更古怪。
        knowledge_cutoff = knowledge_cutoff_by_round(
            phase_list, rounds, dpr, schedule)

    if compressed:
        # 压缩**不是「精度低一点」**：弛豫项 `exp(-k·dt)` 可以复合，但激励项
        # 按步累加 —— 同样 6 天，拆成 6 步会激励 6 次、合成 1 步只激励 1 次。
        # 两条曲线不可比，所以要在日志与落盘里都标出来。
        full = max(p.day for p in phase_list) + 1
        print(f"  ⚠️ 轮数 {rounds} 少于 {full}，已压缩：每轮代表 {dpr:.2f} 天。\n"
              f"     压缩曲线与 round=day（--rounds {full}）**不可比**，"
              "只适合省钱冒烟。")

    seeder = (agents_list[0]
              if (seed_text and agents_list and not schedule) else None)
    if seed_text and schedule:
        print("  （同时给了 --seed-text 与阶段表：**阶段表优先**，种子文本本次不用）")

    # 口径在这里才定稿 —— 上面可能刚把 phases_on 降级成 False。
    result.phases_on = phases_on
    result.days_per_round = dpr
    result.compressed = compressed

    # 世界状态引擎接进逐轮循环 —— **这一步之前，六维曲线只在离线重放上跑过**。
    # 每一轮：读 trace → 还原行为类型 → 推进六维 → 存进 RoundLog。
    # dt=1.0：把 OASIS 的一轮当作一天。这是本项目的**建模约定，不是事实**，
    # 如实标注在这里（OASIS 的沙箱时钟另有自己的步进）。
    ws_engine = WorldStateEngine()
    ws_state = WorldState.baseline(ws_engine.params)
    # 归类走 stance（LLM + 磁盘缓存），不走 world_state.classify ——
    # 关键词表读不懂 agent 的自由文本，实测 12 条里错 2 条、10 条落默认类。
    # 详见 weiran/stance.py 开头。
    stance = StanceClassifier(
        llm=llm,
        cache_path=stance_cache if stance_cache is not None
        else out_dir / "stance_cache.json",
    )

    prev_state: dict[str, float] | None = None
    for i in range(rounds):
        t0 = time.monotonic()
        before = (stats.calls, stats.seconds,
                  stats.prompt_tokens, stats.completion_tokens,
                  stats.failures, len(stats.samples))
        entry = RoundLog(index=i)

        step: dict = {a: LLMAction() for a in agents_list}
        events = schedule.get(i, ())
        if events and agents_list:
            # 压缩模式下同一轮会落进多个阶段，**全部保留**（由不同 agent 各发
            # 一条），不合并、不丢弃 —— 丢掉一个阶段等于悄悄改掉场景。
            # 只有一个 agent 时它们会叠在同一条帖子里，这是压缩模式的已知代价。
            by_agent: dict = {}
            for k, ph in enumerate(events):
                a = agents_list[k % len(agents_list)]
                prev_txt = by_agent.get(a)
                by_agent[a] = (ph.trigger if prev_txt is None
                               else f"{prev_txt}\n\n{ph.trigger}")
            for a, txt in by_agent.items():
                step[a] = ManualAction(ActionType.CREATE_POST, {"content": txt})
            entry.phase_id = ",".join(p.phase_id for p in events)
        elif i == 0 and seeder is not None:
            step[seeder] = ManualAction(ActionType.CREATE_POST,
                                        {"content": seed_text})

        # ---- 闭环的另一半：把当前态势与该角色的知情范围回注给每个 agent ----
        # **必须在 env.step 之前** —— step 内部 asyncio.gather 一发起就读这些属性。
        #
        # `state` 是本轮**开始时**的六维取值，`prev` 是上一轮开始时的，所以
        # 方向词描述的是「上一轮里变了多少」，而不是「距离基线有多远」。
        cur_state = ws_state.as_dict() if world_state else {}
        if agents_list:
            blocks = inject_round_context(
                agents_list, round_index=i, state=cur_state, prev=prev_state,
                knowledge=knowledge,
                known_upto_day=knowledge_cutoff.get(i, 0.0),
                feedback=feedback_on,
                knowledge_on=knowledge_on,
            )
            nonempty = {k: v for k, v in blocks.items() if v}
            # 存摘要而非全文：27 agent × 15 轮 × 每条几百字会把落盘 JSON 撑成
            # 几百 KB，而这里唯一的用途是「证明注入逐轮在变、且各人不同」。
            # 全文留一份样本供人眼核对措辞。
            entry.injected = {k: hashlib.sha1(v.encode("utf-8")).hexdigest()[:12]
                              for k, v in nonempty.items()}
            entry.injection_sample = next(iter(nonempty.values()), "")
        prev_state = cur_state or None

        try:
            await env.step(step)
        except Exception as exc:  # noqa: BLE001
            entry.errors += 1
            print(f"  轮 {i}: step 失败 {type(exc).__name__}: {str(exc)[:200]}",
                  file=sys.stderr)

        new, last_rowid = read_actions(db_path, since_rowid=last_rowid)
        entry.actions = new
        entry.seconds = time.monotonic() - t0
        entry.calls = stats.calls - before[0]
        entry.llm_seconds = round(stats.seconds - before[1], 2)
        entry.prompt_tokens = stats.prompt_tokens - before[2]
        entry.completion_tokens = stats.completion_tokens - before[3]
        entry.failures = stats.failures - before[4]
        # 最慢一次取本轮样本的最大值 —— `stats.slowest` 是全程最大值，差分不出来。
        entry.slowest = round(max(stats.samples[before[5]:], default=0.0), 2)

        # 世界状态推进。**跑在这一轮的动作上，而不是素材上。**
        action_texts = texts_from_actions(new, load_posts(db_path))
        entry.behaviors = stance.classify_many(action_texts)
        if world_state:
            # 阶段标签有两个来源，**这两个是不同的问句**，压缩模式下会不一致：
            #   - `entry.phase_id`：这一轮**注入**了哪个阶段的事件（只有少数轮有）
            #   - `phase_by_round`：这一轮**处在**哪个阶段（每一轮都有）
            # 轮=天（--rounds 15）时两者恒等，已在测试里钉住。压缩时可能差一个
            # ——例如 7 轮（每轮 2.33 天）的第 2 轮跨 day 4.67~7.0，注入的是
            # day 5 的 P3，而起点仍在 P2 的地盘里。两个说法都对，但同一个日志里
            # 冒出两个「阶段」会让人以为是 bug，所以这里**优先用注入的那个**：
            # 本轮的引擎事件就是随这个事件发生的，标成 P3 与日志一致。
            engine_phase = (entry.phase_id.split(",")[0] if entry.phase_id
                            else phase_by_round.get(i) or f"R{i}")
            advance = ws_engine.step(ws_state, entry.behaviors, dt=dpr,
                                     phase_id=engine_phase)
            ws_state = advance.state_after
            entry.state = ws_state.as_dict()

        result.rounds.append(entry)

        # 把「LLM 实际花了多少」与「墙钟花了多少」分开报。
        # 两者差得远，说明时间花在了 LLM 之外（等待、锁、重试）。
        tok = (f"  token {entry.prompt_tokens}+{entry.completion_tokens}"
               if entry.prompt_tokens or entry.completion_tokens else "")
        # 慢与失败都要在**当轮**就看见，事后再翻日志已经晚了 —— 409s 那次
        # 就是因为在终端上只看到「LLM 5.4s」而实际墙钟 409s，无从定位。
        slow = f"  最慢一次 {entry.slowest:5.1f}s" if entry.slowest else ""
        fail = f"  ⚠️ 失败 {entry.failures} 次" if entry.failures else ""
        print(f"  轮 {i}: 墙钟 {entry.seconds:6.1f}s  "
              f"LLM {entry.llm_seconds:6.1f}s / {entry.calls} 次{slow}{fail}  "
              f"动作 {len(new)} 条{tok}"
              + (f"  错误 {entry.errors}" if entry.errors else "")
              + (f"  [阶段 {entry.phase_id}]" if entry.phase_id else ""))
        # 感知层的可见证据。**这行是「闭环到底闭没闭上」的唯一现场答案** ——
        # 六维曲线平的时候，第一件事是看这里是不是 0；若不是 0，问题在模型，
        # 若是 0，问题在接线。
        if agents_list:
            perf = f"         感知 注入 {len(entry.injected)}/{len(agents_list)} 个 agent"
            if entry.injection_sample:
                perf += ("  · 样本「"
                         + entry.injection_sample.replace("\n", " / ")[:56] + "…」")
            print(perf)
        if entry.behaviors:
            kinds = Counter(entry.behaviors)
            print("         行为 " + " ".join(
                f"{k}×{v}" for k, v in kinds.most_common()))
        if entry.state:
            print("         状态 " + "  ".join(
                f"{d[:4]}={entry.state[d]:.3f}" for d in DIMENSIONS))

    result.total_seconds = time.monotonic() - started
    # 缓存必须在收尾前落盘：它是产物的一部分，下次同输入复跑就靠它
    # 保证六维曲线一模一样（见 stance.py 开头）。
    stance.save()
    result.stance_summary = stance.summary()
    await env.close()
    return result


def main(argv: list[str] | None = None) -> int:
    from .config import ensure_console_encoding
    ensure_console_encoding()

    ap = argparse.ArgumentParser(description="多智能体推演驱动")
    ap.add_argument("--out", default=DEFAULT_OUT)
    # 默认值留给 config（.env 的 OASIS_DEFAULT_MAX_ROUNDS / OASIS_MAX_AGENTS）——
    # 那两个旋钮此前没有任何消费方，是死的。用 None 作哨兵，读到 config 后再填。
    ap.add_argument("--rounds", type=int, default=None,
                    help="推演轮数。默认取 .env 的 OASIS_DEFAULT_MAX_ROUNDS")
    ap.add_argument("--agents", type=int, default=None,
                    help="只用前 N 个 agent（冒烟用）。默认取 .env 的 OASIS_MAX_AGENTS")
    ap.add_argument("--platform", choices=("twitter", "reddit"), default="twitter")
    ap.add_argument("--thinking", action="store_true",
                    help="开启模型推理（默认关闭：实测输出 token 增至约 12 倍）")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="**同时**是输出上限与 camel 的上下文上限，见 build_model")
    ap.add_argument("--inspect-db", action="store_true",
                    help="结束后打印 OASIS 的库结构（首次接入时用）")
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO,
                    help="场景目录（读 phases 与 facts）。与 profiles 用同一个默认值")
    ap.add_argument("--seed-text", default="",
                    help="退化路径：只发一条种子事件、只在第 0 轮。"
                         "**默认不用** —— 有阶段表时阶段表优先；"
                         "只有在 --no-phases 或场景无 phases 段时才生效")
    ap.add_argument("--no-world-state", action="store_true",
                    help="只跑引擎，不推进六维状态（对照用）")
    ap.add_argument("--no-phases", action="store_true",
                    help="不注入阶段事件（对照用）。此时时间线只剩 --seed-text")
    ap.add_argument("--no-knowledge", action="store_true",
                    help="不回注知情范围（消融对照用）。差异化感知的另一半")
    ap.add_argument("--no-feedback", action="store_true",
                    help="不回注六维态势（消融对照用）。闭环的另一半")
    ap.add_argument("--keyword-stance", action="store_true",
                    help="归类退回关键词表（不发 LLM）。用于离线复跑："
                         "缓存已存在时两者结果相同，缓存缺失时关键词表读不准")
    ap.add_argument("--no-stance-cache", action="store_true",
                    help="不落盘归类缓存（调试用；会让归类不透明）")
    ap.add_argument("--temperature", type=float, default=None,
                    help="采样温度。压低能减少跨次抖动，代价是 agent 说话变单调。"
                         "默认不传，用端点默认值")
    ap.add_argument("--seed", type=int, default=None,
                    help="采样种子。服务端只「尽力」遵守，不保证逐位可复现")
    args = ap.parse_args(argv)

    try:
        config = load_config(require_llm=True)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    scenario_dir = Path(args.scenario)
    if not scenario_dir.is_absolute():
        scenario_dir = REPO_ROOT / scenario_dir

    # 命令行显式传参优先；没传才回落到 .env 的旋钮。
    # 回落时**把来源标出来** —— 「我明明设了 5 个 agent，怎么跑了 27 个」
    # 这种问题只能靠日志回答。
    rounds = args.rounds if args.rounds is not None else config.simulation.max_rounds
    agents = args.agents if args.agents is not None else config.simulation.max_agents
    src = []
    if args.rounds is None:
        src.append(f"轮数取自 .env OASIS_DEFAULT_MAX_ROUNDS={rounds}")
    if args.agents is None:
        src.append(f"agent 数取自 .env OASIS_MAX_AGENTS={agents}")

    print(f"模型 {config.llm.model} · 推理={'开' if args.thinking else '关'} · "
          f"平台={args.platform} · 轮数={rounds} · agent={agents}")
    for line in src:
        print(f"  （{line}；命令行显式传参会覆盖它）")
    # 把「这一轮哪些开关是开的」摆在最前面。默认全开是有意为之（闭环本该默认
    # 闭上），但同一条命令今天的输出与昨天**不同** —— 不写出来，旧结论与新结果
    # 对不上时没人知道为什么。
    print("  闭环：阶段事件={} · 态势回注={} · 知情回注={}".format(
        "关" if args.no_phases else "开",
        "关" if args.no_feedback else "开",
        "关" if args.no_knowledge else "开"))

    # 归类用独立的客户端：它要的是稳定与便宜，不需要推理。
    classifier_llm = None
    if not args.no_world_state and not args.keyword_stance:
        from .llm import LLMClient
        classifier_llm = LLMClient(config.llm)

    result = asyncio.run(_run(
        config, out_dir,
        rounds=rounds, agents=agents, platform=args.platform,
        thinking=args.thinking, max_tokens=args.max_tokens,
        seed_text=args.seed_text,
        world_state=not args.no_world_state,
        scenario_dir=scenario_dir,
        phases_on=not args.no_phases,
        knowledge_on=not args.no_knowledge,
        feedback_on=not args.no_feedback,
        llm=classifier_llm,
        stance_cache=None if args.no_stance_cache else out_dir / "stance_cache.json",
        temperature=args.temperature,
        seed=args.seed,
    ))

    total_calls = sum(r.calls for r in result.rounds)
    total_llm = sum(r.llm_seconds for r in result.rounds)
    total_in = sum(r.prompt_tokens for r in result.rounds)
    total_out = sum(r.completion_tokens for r in result.rounds)
    print(f"\n完成：{len(result.rounds)} 轮，动作 {result.total_actions} 条，"
          f"墙钟 {result.total_seconds:.1f}s")
    # 注意：这里的秒数是**各次调用耗时之和**，并发跑时会超过墙钟
    # （3 个 agent 并行，墙钟 5.0s 而合计 8.9s）。所以不报百分比 ——
    # 报「占墙钟 176%」只会让人以为哪儿算错了。
    print(f"LLM：{total_calls} 次调用，耗时合计 {total_llm:.1f}s"
          f"（均 {total_llm / total_calls:.1f}s/次）"
          f" · token 入 {total_in} / 出 {total_out}" if total_calls else
          "LLM：0 次调用")
    if total_calls == 0:
        # 别把「插桩没生效」误读成「模型很快」。
        print("  ⚠️ 一次调用都没记到 —— 插桩可能没打在实际被调用的类上，"
              "这个耗时数字不可当结论")
    elif result.total_seconds > 3 * max(total_llm, 1.0):
        # 墙钟远大于 LLM 时间，时间花在了别处，而不是「模型慢」。
        print("  ⚠️ 墙钟远超 LLM 时间，差额花在等待/调度/重试上，值得查")

    # 注入侧的同类检查：**0 次注入要当异常信号**。理由同上 ——
    # 「注入没生效」与「注入生效了但没差别」在六维曲线上一模一样。
    #
    # 「期望有注入」不是「开关没全关」：`--no-world-state` 时态势是空的，
    # 就算 `feedback` 开着也渲染不出内容。少了这一条，
    # `--no-world-state --no-knowledge` 会天天报假警。
    expect_injection = (not args.no_knowledge) or (
        not args.no_feedback and not args.no_world_state)
    total_inj = sum(len(r.injected) for r in result.rounds)
    if expect_injection and total_inj == 0:
        print("  ⚠️ 一次注入都没发生 —— 感知层没接上。"
              "此时任何「曲线没变化」的结论都不成立")
    elif total_inj:
        print(f"感知：累计注入 {total_inj} 次"
              f"（{len(result.rounds)} 轮，每轮最多 {agents} 个 agent）")

    out_file = out_dir / f"{args.platform}_rounds.json"
    out_file.write_text(
        json.dumps(
            {
                # 口径写在文件里，而不是只留在终端上。一份不知道自己是不是
                # 压缩过的结果，日后必然被当成 round=day 的结果引用。
                "meta": {
                    "rounds": len(result.rounds),
                    "agents": agents,
                    "platform": args.platform,
                    "seed_text": args.seed_text,
                    "days_per_round": result.days_per_round,
                    "compressed": result.compressed,
                    "comparable_to_round_day": not result.compressed,
                    "phases_on": result.phases_on,
                    "knowledge_on": result.knowledge_on,
                    "feedback_on": result.feedback_on,
                    "world_state": not args.no_world_state,
                    "total_seconds": round(result.total_seconds, 2),
                    "total_actions": result.total_actions,
                },
                "rounds": [
                    {"index": r.index, "seconds": round(r.seconds, 2),
                     "llm_seconds": r.llm_seconds, "calls": r.calls,
                     "prompt_tokens": r.prompt_tokens,
                     "completion_tokens": r.completion_tokens,
                     "failures": r.failures, "slowest": r.slowest,
                     "phase_id": r.phase_id,
                     "injected": r.injected,
                     "injection_sample": r.injection_sample,
                     "behaviors": r.behaviors, "state": r.state,
                     "actions": r.actions, "errors": r.errors}
                    for r in result.rounds
                ],
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    if result.stance_summary:
        print(f"归类：{result.stance_summary}")
    print(f"逐轮动作 → {out_file}")

    if args.inspect_db and result.db_path:
        print("\n" + "=" * 70)
        print("OASIS 库结构")
        print("=" * 70)
        print(dump_schema(result.db_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
