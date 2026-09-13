"""差异化感知层：把「当前态势」与「这个角色知道什么」拼成一段注入文本。

**纪律与 `world_state.py` 一致：不含 LLM、不含随机数。**
同样的输入必然给出同样的注入块 —— 否则「感知」就成了第二个随机源，
而它本来是用来解释随机性的。

本模块补的是 `simulate.py:14` 那句设计意图的后半截：
「读出来、算状态、**再把状态回注给下一轮的 agent**」。此前前半截做了、
后半截只是注释。注入点选在 `SocialAgent.astep` 上，理由见
`install_injection()` 的注释（`perform_action_by_llm` 没有形参，
没有可插入的位置）。

**注入的是「态势」，不是「答案」。** 两条硬纪律：

  1. `render_state` **只给档位与方向，不给数值**。把「恐慌 0.72」告诉 agent，
     下一步它就会把 0.72 抄进帖文里 —— 数字成了回声，六维曲线变成
     「我们喂进去什么就读出来什么」。档位（「恐慌偏高」）能驱动行为，
     数值只会驱动复述。
  2. 注入走 user message，**绝不碰 `agent.system_message`**。
     `oasis/social_agent/agent.py:176`/`:215` 拿 `"# RESPONSE FORMAT"`
     做切分标记，一旦将来接上带该标记的自定义模板，塞在 system 侧的注入
     会被整段静默切掉。`build_block` 里的子串断言就是这道禁令的廉价保险。

**关于 AS-9（辅导员角色冲突）的一处更正。** 原先设想过「`hidden` 那段措辞
是 `metric_notes.role_conflict` 的唯一来源」。**这不成立**：本场景里辅导员
A07/A08/A09 的 `hidden` 全是空的 —— 他们的冲突来自「知道 F9/F10（被要求
劝删）却在公开场合安抚」，也就是 `knows` 里的事实与**公开立场**之间的落差，
不是 `hidden`。所以 `render_knowledge` 提供的是素材，不是指标；
AS-9 只能从真实仿真的产出里读，不能从本模块推断。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .world_state import DIMENSION_ZH, DIMENSIONS

#: 挂在 agent 实例上的「本轮要注入的文本」。空串 = 本轮不注入。
#:
#: **为什么挂实例属性而不是闭包变量**：`env.step` 用
#: `asyncio.gather(*tasks)` 并发跑全部 agent（`oasis/environment/env.py:193`），
#: 闭包变量是进程级的，会被并发串掉。挂实例天然按 agent 隔离。
INJECT_ATTR = "_weiran_injection"

#: 注入文本里**绝不许可**出现的子串。见模块 docstring 第 2 条纪律。
FORBIDDEN = "# RESPONSE FORMAT"


class KnowledgeError(RuntimeError):
    """知情映射与金标对不上。消息里必须说清是哪个 id、哪个角色。"""


# ---------------------------------------------------------------------------
# 档位与方向
# ---------------------------------------------------------------------------

# 档位分界。**按绝对取值分档，不按「相对基线」分档** ——
# 相对基线在小基线上会失真：attention 基线 0.05，涨到 0.10 就算「翻倍」，
# 但那在语义上仍然是「几乎没人讨论」。绝对分档在这六个维度上读起来都是对的：
# 常态下「关注很低、稳定很高、信任偏高」，符合一所正常运转的大学的样子。
TIERS: tuple[tuple[float, str], ...] = (
    (0.15, "很低"), (0.35, "偏低"), (0.65, "中等"), (0.85, "偏高"), (1.01, "很高"),
)

#: 方向词按维度分别给，不用「上升/下降」一刀切。
#: 「信任在上升」和「恐慌在上升」的情感方向相反，而「极化在上升」听起来像好事
#: —— 一刀切会把这三件事写成同一个句式，agent 读不出差别。
#:
#: **元组顺序是 (上升词, 下降词)，不许写反。** 写反了不会报错，只会让 agent
#: 听到与事实相反的话 —— 信任在回升时告诉它「信任在下滑」。本模块的测试
#: 逐维钉住了顺序，因为这是「读起来通顺但其实反了」的高发处。
DIRECTION_ZH: dict[str, tuple[str, str]] = {
    "attention":    ("升温", "降温"),
    "panic":        ("加重", "缓解"),
    "trust":        ("回升", "下滑"),
    "polarization": ("扩大", "收窄"),
    "risk":         ("抬头", "回落"),
    "stability":    ("企稳", "动摇"),
}

#: 方向的分辨率。低于此的变动不写进文案 —— 免得 agent 对四舍五入级别的
#: 抖动做反应。这是**渲染约定**，不是引擎参数：引擎照旧按浮点数演化。
DIRECTION_EPS = 0.02


def tier_of(value: float) -> str:
    """把 [0,1] 的取值翻成一个档位词。"""
    for upper, word in TIERS:
        if value < upper:
            return word
    return TIERS[-1][1]


def render_state(state: dict[str, float],
                 prev: dict[str, float] | None = None) -> str:
    """六维状态 -> 一句中文态势。**不含任何数值。**

    Args:
        state: 本轮开始时的六维取值。
        prev: 上一轮的取值。给了才写方向；第 0 轮没有上一轮，只写档位。
    """
    parts: list[str] = []
    for d in DIMENSIONS:
        if d not in state:
            continue
        seg = f"{DIMENSION_ZH[d]}{tier_of(state[d])}"
        if prev is not None and d in prev:
            delta = state[d] - prev[d]
            if delta >= DIRECTION_EPS:
                seg += f"、{DIRECTION_ZH[d][0]}中"
            elif delta <= -DIRECTION_EPS:
                seg += f"、{DIRECTION_ZH[d][1]}中"
        parts.append(seg)
    return "；".join(parts)


def render_knowledge(entry: "ActorKnowledge") -> str:
    """「你已知晓」+「你知道但暂未公开」两段。**不含任何数值之外的判断。**

    第二段的措辞刻意点出「说出它可能有代价」，但不指示怎么办 ——
    取舍留给 agent。这是 `metric_notes.role_conflict` 想要的落差能在
    文本层被表达出来的前提（指标本身仍只能从仿真产出里读，见模块 docstring）。
    """
    segs: list[str] = []
    if entry.knows:
        segs.append(
            "你已知晓的事实（未列出的，你并不知道）：\n"
            + "\n".join(f"- {t}" for t in entry.knows)
        )
    if entry.hidden:
        segs.append(
            "你知道、但尚未公开的事实（说出它们可能给你带来麻烦，"
            "公开还是回避由你自己决定）：\n"
            + "\n".join(f"- {t}" for t in entry.hidden)
        )
    return "\n\n".join(segs)


def build_block(
    round_index: int,
    state: dict[str, float],
    prev: dict[str, float] | None = None,
    entry: "ActorKnowledge | None" = None,
    *,
    feedback: bool = True,
    knowledge: bool = True,
) -> str:
    """合成递给某个 agent 的注入块。空串表示本轮不注入。

    Args:
        feedback: 关掉则不带态势（消融对照用）。
        knowledge: 关掉则不带知情范围（消融对照用）。
    """
    segs: list[str] = []
    if feedback and state:
        head = f"[第 {round_index} 轮开始时的校园舆情态势]"
        segs.append(f"{head}\n{render_state(state, prev)}")
    if knowledge and entry is not None:
        seg = render_knowledge(entry)
        if seg:
            segs.append(seg)
    text = "\n\n".join(segs)
    if FORBIDDEN in text:
        raise KnowledgeError(
            f"注入文本里出现了 {FORBIDDEN!r} —— 它被 OASIS 用作切分标记，"
            "出现在这里说明有一段事实正文混进了不该有的内容"
        )
    return text


# ---------------------------------------------------------------------------
# 知情映射
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActorKnowledge:
    """一个角色的知情范围。事实已由 id 还原成正文。"""

    actor_id: str
    user_id: int
    name: str
    group: str
    knows: tuple[str, ...] = ()
    hidden: tuple[str, ...] = ()


@dataclass(frozen=True)
class Knowledge:
    """整个场景的知情映射。"""

    by_actor: dict[str, ActorKnowledge] = field(default_factory=dict)
    #: user_id -> actor_id。**这是 agent 反查的唯一正确路径** ——
    #: `social_agent_id` == CSV 行号 == `user_id`（profiles.py 写盘时按行序），
    #: 而 `agent.user_info.user_name` 是 `None`、`agent.agent_id` 是 uuid4。
    by_user_id: dict[int, str] = field(default_factory=dict)


def load_knowledge(out_dir: str | Path, scenario_dir: str | Path) -> Knowledge:
    """读 `actor_knowledge.json` + `reference_data.json`，join 成正文形式。

    悬空 fact id（在 `knows`/`hidden` 里但 `facts` 表里没有）**直接报错**，
    不静默跳过：注入一个裸露的 `F99` 进 prompt，agent 只会当成乱码，
    而没有任何地方会提示这件事发生过。
    """
    out_dir = Path(out_dir)
    scenario_dir = Path(scenario_dir)

    k_path = out_dir / "actor_knowledge.json"
    r_path = scenario_dir / "reference_data.json"
    for p in (k_path, r_path):
        if not p.is_file():
            raise KnowledgeError(
                f"缺少 {p} —— "
                + ("先跑 python -m weiran.profiles" if p == k_path
                   else "检查 --scenario 指向的场景目录")
            )

    raw_k = json.loads(k_path.read_text(encoding="utf-8"))
    raw_r = json.loads(r_path.read_text(encoding="utf-8"))

    fact_text = {f["id"]: f["text"] for f in raw_r.get("facts", [])}
    actors_raw = raw_k.get("actors", {})

    by_actor: dict[str, ActorKnowledge] = {}
    by_user_id: dict[int, str] = {}
    dangling: list[str] = []

    for actor_id, v in actors_raw.items():
        def _texts(ids: list[str]) -> tuple[str, ...]:
            out = []
            for fid in ids:
                if fid not in fact_text:
                    dangling.append(f"{actor_id}.{fid}")
                    continue
                out.append(fact_text[fid])
            return tuple(out)

        entry = ActorKnowledge(
            actor_id=actor_id,
            user_id=int(v["user_id"]),
            name=str(v.get("name", "")),
            group=str(v.get("group", "")),
            knows=_texts(list(v.get("knows", []))),
            hidden=_texts(list(v.get("hidden", []))),
        )
        by_actor[actor_id] = entry
        by_user_id[entry.user_id] = actor_id

    if dangling:
        raise KnowledgeError(
            "以下 fact id 在 actor_knowledge.json 里有、但 reference_data.json "
            "的 facts 表里没有：\n  " + "\n  ".join(sorted(dangling))
            + "\n（两份文件必须同源：profiles.py 从金标生成前者）"
        )
    if not by_actor:
        raise KnowledgeError(f"{k_path} 里一个角色都没有 —— 文件被写空了？")

    return Knowledge(by_actor=by_actor, by_user_id=by_user_id)


# ---------------------------------------------------------------------------
# 阶段事件 -> 轮号
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase:
    """金标里的一个阶段。`trigger` 是**场景作者编写的事件标题**，不是抓取内容。"""

    phase_id: str
    day: int
    trigger: str
    label: str = ""
    is_turning_point: bool = False


def load_phases(scenario_dir: str | Path) -> list[Phase]:
    """读 `reference_data.json` 的 phases 段。"""
    p = Path(scenario_dir) / "reference_data.json"
    if not p.is_file():
        raise KnowledgeError(f"缺少 {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    phases = [
        Phase(
            phase_id=str(x["id"]),
            day=int(x["day"]),
            trigger=str(x["trigger"]),
            label=str(x.get("label", "")),
            is_turning_point=bool(x.get("is_turning_point", False)),
        )
        for x in raw.get("phases", [])
    ]
    if not phases:
        raise KnowledgeError(f"{p} 里没有 phases 段")
    return sorted(phases, key=lambda x: x.day)


def days_per_round(phases: list[Phase], rounds: int) -> tuple[float, bool]:
    """给定总轮数，算出每轮代表几天，以及是否处于压缩模式。

    引擎的建模约定是「一轮 = 一天」（`simulate.py:551`），半衰期也按天定义。
    所以**首选 `rounds >= 末阶段 day + 1`**（本场景 = 15），此时返回 (1.0, False)，
    曲线与引擎的约定一致。

    轮数不够就必须压缩：让最后一轮正好落在最后一个阶段那天。
    **压缩后的曲线与 round=day 不可比**，原因不是「精度低一点」：
    弛豫项是 `exp(-k·dt)`，可以复合；但**激励项按步累加** ——
    同样 6 天，拆成 6 步会激励 6 次、合成 1 步只激励 1 次，两者不是
    同一件事的粗细版本，是两条不同的曲线。调用方必须把 compressed 标出来。

    Returns:
        (每轮天数, 是否压缩)
    """
    span = max(p.day for p in phases)
    if rounds >= span + 1:
        return 1.0, False
    if rounds < 2:
        raise KnowledgeError(
            f"轮数 {rounds} 太少 —— 至少 2 轮才能覆盖 day 0..{span}"
        )
    return span / (rounds - 1), True


def phase_schedule(
    phases: list[Phase], rounds: int
) -> tuple[dict[int, tuple[Phase, ...]], float, bool]:
    """阶段表 -> {轮号: 该轮要注入的事件}。

    压缩模式下多个阶段会落到同一轮，**全部保留**（同一轮由不同 agent
    各发一条），不合并、不丢弃 —— 丢掉一个阶段等于悄悄改掉场景。

    Returns:
        (轮号 -> 事件元组, 每轮天数, 是否压缩)
    """
    dpr, compressed = days_per_round(phases, rounds)
    schedule: dict[int, list[Phase]] = {}
    for p in phases:
        r = round(p.day / dpr)
        if r >= rounds:
            r = rounds - 1          # 浮点边界兜底，不丢阶段
        schedule.setdefault(r, []).append(p)
    return {r: tuple(v) for r, v in sorted(schedule.items())}, dpr, compressed


def active_phase_by_round(
    phases: list[Phase], rounds: int, dpr: float
) -> dict[int, str]:
    """每一轮归属到「当时生效的最近一个阶段」的 id；没有则空串。

    与 `phase_schedule` 是两件事，别混：`phase_schedule` 回答「这一轮要**注入**
    哪个事件」（少数轮才有），这里回答「这一轮**处在**哪个阶段」（每一轮都有）。
    轮=天时 15 轮里只有 5 轮有事件，但 15 轮都该有阶段标签 —— 否则曲线没法
    按阶段切片，也没法回答「P3 之后那几轮到底发生了什么」。

    **两者在压缩模式下会不一致，这是性质不是 bug。** 判据不同：
    `phase_schedule` 按「离哪个阶段最近」取整（事件落在最近的那轮），
    这里按「已经走到了哪」取上界（`p.day <= r*dpr`）。例如 7 轮覆盖 14 天
    （每轮 2.33 天）时，第 2 轮跨 day 4.67~7.0 —— 它**注入**的是 day 5 的 P3，
    但它的起点仍在 P2 的地盘里，所以这里返回 P2。轮=天时 `dpr == 1.0`、
    两边恒等（`test_round_equals_day_when_rounds_are_enough` 那一档钉住了）。

    **输入可以乱序**（内部会排一次），因为这里唯一的正确性来源是
    「day 最小的那个最近阶段」。
    """
    ordered = sorted(phases, key=lambda p: p.day)
    out: dict[int, str] = {}
    for r in range(rounds):
        day = r * dpr
        # 容差：压缩模式下 r*dpr 会带上浮点尾巴（例如 3×2.333…=7.000000000001），
        # 而阶段 day 是整数。不留容差会把「正好落在阶段当天」的那一轮
        # 判成上一个阶段。
        active = [p for p in ordered if p.day <= day + 1e-9]
        out[r] = active[-1].phase_id if active else ""
    return out


# ---------------------------------------------------------------------------
# 注入（唯一需要 OASIS 的部分，全部惰性导入 —— 好让本模块能离线被测试）
# ---------------------------------------------------------------------------

def install_injection() -> bool:
    """给 `SocialAgent.astep` 装上「本轮注入」的插槽。幂等。

    **必须在 `generate_twitter_agent_graph(...)` 之前调用**
    （`simulate.py:508`）—— 类级补丁对已存在实例也生效，但只留一个
    安装时间点更好排查。

    **为什么包 `astep` 而不是 `perform_action_by_llm`**：
    `perform_action_by_llm(self)` 没有参数（`oasis/social_agent/agent.py:125`），
    user message 是在它**函数体内**拼好再传给 `self.astep(user_msg)` 的
    （`agent.py:128-134`、`:139`），包装层没有可用于插入的形参 ——
    要在这层注入只能连它那 25 行函数体一起抄。
    而全包 grep `.astep(` **只有一处命中**（`agent.py:139`），
    即 OASIS 里 `astep` 只被 `perform_action_by_llm` 调用，
    所以两层的覆盖调用面完全相同 —— 包 `astep` 零函数体复制。

    Returns:
        True 表示本次装上；False 表示之前已装过（重复调用安全）。

    Raises:
        RuntimeError: 目标方法不在。**必须响** —— 插桩静默失效会让
            「注入没生效」看起来和「注入生效了但没差别」一模一样，
            照 `simulate.py` 里 `instrument_model` 的先例。
    """
    from camel.messages import BaseMessage
    from oasis.social_agent.agent import SocialAgent

    if getattr(SocialAgent, "_weiran_injection_installed", False):
        return False
    if not hasattr(SocialAgent, "astep"):
        raise RuntimeError(
            "SocialAgent 上没有 astep —— 仿真内核换版本了（本项目锁 "
            "camel-oasis==0.2.5 / camel-ai==0.2.78）。注入点必须重新确认，"
            "不要静默跳过注入。"
        )

    original = SocialAgent.astep

    async def astep_with_injection(self, input_message, response_format=None):
        text: str = getattr(self, INJECT_ATTR, "")
        if text:
            if isinstance(input_message, BaseMessage):
                # role_name 保留原值（OASIS 传的是 "User"）。camel 在 astep
                # 内部会把 role_at_backend 强制成 USER，role_name 只影响显示。
                input_message = BaseMessage.make_user_message(
                    role_name=input_message.role_name or "User",
                    content=f"{text}\n\n{input_message.content}",
                )
            else:
                input_message = f"{text}\n\n{input_message}"
        return await original(self, input_message, response_format)

    astep_with_injection.__name__ = "astep"
    SocialAgent.astep = astep_with_injection
    SocialAgent._weiran_injection_installed = True
    return True


def inject_round_context(
    agents: list,
    *,
    round_index: int,
    state: dict[str, float],
    prev: dict[str, float] | None,
    knowledge: Knowledge | None,
    feedback: bool = True,
    knowledge_on: bool = True,
) -> dict[str, str]:
    """轮首调用：把本轮注入文本挂到每个 agent 实例上。

    **必须在 `await env.step(...)` 之前调用** —— `env.step` 内部
    `asyncio.gather` 一发起就会读这些属性。

    Returns:
        {actor_id: 注入块}。空块也在里面（值 ""），好让调用方能把
        「注入了空串」与「压根没走这条路径」区分开。

    Raises:
        KnowledgeError: 某个 agent 在知情映射里找不到对应角色。
            这意味着画像文件与 `actor_knowledge.json` 不同源，是真缺陷。
    """
    blocks: dict[str, str] = {}
    for agent in agents:
        uid = getattr(agent, "social_agent_id", None)
        actor_id = knowledge.by_user_id.get(uid) if knowledge is not None else None
        if knowledge is not None and actor_id is None:
            raise KnowledgeError(
                f"agent social_agent_id={uid!r} 在 actor_knowledge.json 里"
                "找不到对应角色 —— 画像文件与知情映射不同源，"
                "先重跑 python -m weiran.profiles"
            )
        entry = knowledge.by_actor[actor_id] if actor_id else None
        block = build_block(
            round_index, state, prev, entry,
            feedback=feedback, knowledge=knowledge_on,
        )
        setattr(agent, INJECT_ATTR, block)
        blocks[actor_id or f"uid={uid}"] = block
    return blocks
