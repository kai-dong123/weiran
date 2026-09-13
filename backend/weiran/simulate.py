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
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import REPO_ROOT, ConfigError, load_config
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
    actions: list[dict] = field(default_factory=list)
    errors: int = 0
    # 世界状态：本轮识别出的行为构成，以及推进后的六维取值。
    behaviors: list[str] = field(default_factory=list)
    state: dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationResult:
    rounds: list[RoundLog] = field(default_factory=list)
    db_path: Path | None = None
    total_seconds: float = 0.0
    stance_summary: str = ""

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

    gen = generate_twitter_agent_graph if is_twitter else generate_reddit_agent_graph
    graph = await gen(profile_path=str(profile_file), model=model, available_actions=actions)

    agents_list = [a for _, a in graph.get_agents()]
    if agents is not None:
        agents_list = agents_list[:agents]

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
    # 所以本场景把《就业质量报告》公开这一**种子事件**在第 0 轮注入：
    # 由第一个 agent 走 ManualAction（不经过 LLM，事件内容是给定的），
    # 其余 agent 照常走 LLMAction 去反应。从第 1 轮起全员走 LLM ——
    # 注入的是「起点」，之后怎么演化仍然由 agent 自己决定。
    seeder = agents_list[0] if (seed_text and agents_list) else None

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

    for i in range(rounds):
        t0 = time.monotonic()
        before = (stats.calls, stats.seconds,
                  stats.prompt_tokens, stats.completion_tokens)
        entry = RoundLog(index=i)

        step: dict = {a: LLMAction() for a in agents_list}
        if i == 0 and seeder is not None:
            step[seeder] = ManualAction(
                ActionType.CREATE_POST, {"content": seed_text}
            )
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

        # 世界状态推进。**跑在这一轮的动作上，而不是素材上。**
        action_texts = texts_from_actions(new, load_posts(db_path))
        entry.behaviors = stance.classify_many(action_texts)
        if world_state:
            step = ws_engine.step(ws_state, entry.behaviors, dt=1.0,
                                  phase_id=f"R{i}")
            ws_state = step.state_after
            entry.state = ws_state.as_dict()

        result.rounds.append(entry)

        # 把「LLM 实际花了多少」与「墙钟花了多少」分开报。
        # 两者差得远，说明时间花在了 LLM 之外（等待、锁、重试）。
        tok = (f"  token {entry.prompt_tokens}+{entry.completion_tokens}"
               if entry.prompt_tokens or entry.completion_tokens else "")
        print(f"  轮 {i}: 墙钟 {entry.seconds:6.1f}s  "
              f"LLM {entry.llm_seconds:6.1f}s / {entry.calls} 次  "
              f"动作 {len(new)} 条{tok}"
              + (f"  错误 {entry.errors}" if entry.errors else ""))
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
    ap = argparse.ArgumentParser(description="多智能体推演驱动")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--agents", type=int, default=None,
                    help="只用前 N 个 agent（冒烟用）")
    ap.add_argument("--platform", choices=("twitter", "reddit"), default="twitter")
    ap.add_argument("--thinking", action="store_true",
                    help="开启模型推理（默认关闭：实测输出 token 增至约 12 倍）")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="**同时**是输出上限与 camel 的上下文上限，见 build_model")
    ap.add_argument("--inspect-db", action="store_true",
                    help="结束后打印 OASIS 的库结构（首次接入时用）")
    ap.add_argument("--seed-text", default="",
                    help="第 0 轮注入的种子事件全文（不注入则时间线为空，"
                         "agent 会全部 do_nothing）")
    ap.add_argument("--no-world-state", action="store_true",
                    help="只跑引擎，不推进六维状态（对照用）")
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

    print(f"模型 {config.llm.model} · 推理={'开' if args.thinking else '关'} · "
          f"平台={args.platform} · 轮数={args.rounds} · "
          f"agent={args.agents if args.agents else '全部'}")

    # 归类用独立的客户端：它要的是稳定与便宜，不需要推理。
    classifier_llm = None
    if not args.no_world_state and not args.keyword_stance:
        from .llm import LLMClient
        classifier_llm = LLMClient(config.llm)

    result = asyncio.run(_run(
        config, out_dir,
        rounds=args.rounds, agents=args.agents, platform=args.platform,
        thinking=args.thinking, max_tokens=args.max_tokens,
        seed_text=args.seed_text,
        world_state=not args.no_world_state,
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

    out_file = out_dir / f"{args.platform}_rounds.json"
    out_file.write_text(
        json.dumps(
            [{"index": r.index, "seconds": round(r.seconds, 2),
              "llm_seconds": r.llm_seconds, "calls": r.calls,
              "prompt_tokens": r.prompt_tokens,
              "completion_tokens": r.completion_tokens,
              "behaviors": r.behaviors, "state": r.state,
              "actions": r.actions, "errors": r.errors} for r in result.rounds],
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
