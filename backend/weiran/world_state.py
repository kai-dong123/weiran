"""世界状态引擎 —— 六维状态的演化、事件检测与反事实推演。

这是本项目的核心自研部分，也是「未然」两个字的兑现处：
如果只能事后描述舆情，那叫报告；能提前算出「再走一步会怎样」，才叫推演。

## 模型形式

六个维度（关注 / 恐慌 / 信任 / 极化 / 风险 / 稳定）构成一个**带耦合的弛豫系统**。
每一步对每个维度做三件事：

    step 1  激励   state += gain_d * tanh( Σ 行为权重 / SCALE )       被本轮行为推离平衡
    step 2  弛豫   state  = base_d + (state - base_d) * exp(-k_d * dt) 向基线回落
    step 3  耦合   state += Σ_j W[j][d] * (state_j - base_j)          维度间相互牵动

step 2 是全部结论的来源。**每个维度有各自的弛豫速率 k**：

    k 大 = 快速回落（关注、恐慌 —— 新闻周期）
    k 小 = 缓慢恢复（信任 —— 机构信任的典型特征）

这条不对称不是为了让结论好看而设的，它是「信任」与「情绪」在机制上的差别：
情绪随事件起落，信任是对长期行为记录的推断，需要新的行为记录才能改写。
模型的全部主张都从这里长出来。

## 参数从哪来（重要）

参数按**通用原则**设定，并逐条写明了理由（见 DIMENSION_PARAMS / EXCITATION 的注释）。
它们**没有**针对 benchmark 里的金标调过。

这是刻意的：金标来自场景叙事，模型来自机制假设，二者独立。
然后我们测「一个独立参数化的模型能否重现这段叙事」——
吻合是证据，不吻合是发现。反过来调参直到对上，就是自欺。

## 确定性

本模块**不含任何 LLM 调用、不含随机数**。同样的输入必得同样的输出。
这是「评测确定性」那一半的支点：仿真可以随机，评分不能随机。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DIMENSIONS: tuple[str, ...] = (
    "attention", "panic", "trust", "polarization", "risk", "stability",
)

DIMENSION_ZH: dict[str, str] = {
    "attention": "关注", "panic": "恐慌", "trust": "信任",
    "polarization": "极化", "risk": "风险", "stability": "稳定",
}


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DimensionParams:
    """单个维度的基线、弛豫半衰期与激励增益。

    Attributes:
        baseline: 事件发生前该维度的常态值。
        half_life: 偏离基线后，缺口缩小一半所需的天数。
        gain: 激励增益。单轮行为最多能把该维度推离基线多少。

    为什么用**半衰期**而不是衰减率 k 作参数：
    半衰期可以直接说出理由（「信任的恢复半衰期约两周」），
    k 不行 —— 0.02 这个数字本身不承载任何可检验的含义，
    写下来之后没人能判断它是对是错。参数化的方式决定了模型能不能被讨论。
    """

    baseline: float
    half_life: float
    gain: float

    @property
    def k(self) -> float:
        """弛豫速率（每天），由半衰期换算。"""
        return math.log(2.0) / self.half_life


# 基线取「一所正常运转的大学」的常态：
#   关注低、恐慌低、极化低、风险低、稳定高、信任中上。
# 信任给 0.70 而不是 0.95 —— 高校信任本就带着一点审视，
# 且若给满，后面任何下降都会显得像模型人造的。
#
# 半衰期的取值理由，逐条写：
#   attention     1.5天  校园话题的自然周期：起来一两天，退潮两三天
#   panic         2.0天  情绪起落快，但比纯注意力慢（背后有真实的焦虑撑着）
#   risk         10.0天  风险感不会因为没人讨论就消失
#   trust        14.0天  ★ 机构信任的恢复半衰期约两周 ——
#                        注意这是**恢复**半衰期。信任的下坠远比上浮快，
#                        见下方 ASYMMETRY 的说明。
#   stability    20.0天  秩序扰动恢复得最慢
#   polarization 30.0天  立场一旦分开，靠时间几乎不会自动合拢
DIMENSION_PARAMS: dict[str, DimensionParams] = {
    "attention":    DimensionParams(baseline=0.05, half_life=1.5,  gain=0.55),
    "panic":        DimensionParams(baseline=0.10, half_life=2.0,  gain=0.50),
    "trust":        DimensionParams(baseline=0.70, half_life=14.0, gain=0.45),
    "polarization": DimensionParams(baseline=0.10, half_life=30.0, gain=0.60),
    "risk":         DimensionParams(baseline=0.10, half_life=10.0, gain=0.60),
    "stability":    DimensionParams(baseline=0.85, half_life=20.0, gain=0.45),
}

# 下坠与恢复的不对称倍率。
#
# 这是模型里第二条、也是更关键的一条不对称：半衰期只描述「缺口缩小」的速度，
# 但信任的**下坠**并不遵守恢复半衰期 —— 负面行为记录写入得极快，
# 而抹掉它极慢。所以对 trust / stability / polarization 三个维度，
# 向**不利方向**的激励按其自然幅度生效，向**有利方向**的激励要打这个折扣。
#
# 折扣值 0.45 的含义：一次正面行为，只抵得上同等强度负面行为的不到一半。
# 这不是拟合出来的 —— 若不设这个折扣，模型会预测「道歉一次就能把信任拉回来」，
# 而那与任何一次真实舆情的观察都不符。
ASYMMETRY: dict[str, float] = {
    "trust": 0.45,
    "stability": 0.55,
    "polarization": 0.50,
}

#: 各维度的「有利方向」符号。+1 表示越高越好，-1 表示越低越好。
#: 只对 ASYMMETRY 里列出的维度有意义，其余维度两个方向对称。
FAVOURABLE_SIGN: dict[str, int] = {
    "trust": +1,
    "stability": +1,
    "polarization": -1,
}

# 维度间耦合：W[src][dst] = src 偏离基线一个单位时，dst 受到的牵动。
# 只保留机制上说得通的边，不做全连接——全连接等于给自己发一张
# 「想怎么解释都行」的许可证。
COUPLING: dict[str, dict[str, float]] = {
    # 关注度越高，被放大的机会越多，风险越大
    "attention":    {"risk": 0.10},
    # 恐慌直接抬升风险感，并侵蚀稳定
    "panic":        {"risk": 0.15, "stability": -0.10},
    # 信任是对冲：信任高时，同样的事不那么可怕
    "trust":        {"panic": -0.20, "risk": -0.15},
    # 极化同时打击稳定与信任（「你们」和「我们」一旦分开，就没什么共同事实了）
    "polarization": {"stability": -0.20, "trust": -0.10},
    # 风险感反过来加剧极化
    "risk":         {"polarization": 0.10},
}

# 行为类型 -> 各维度的激励权重。
#
# 符号与相对大小都是按机制定的，不是为了凑曲线：
#   suppression（压制）对信任的负向权重最大（-0.22），是其余行为的近两倍。
#   理由是它同时违反两条规范：「你有事瞒着我」和「你对自己人动手」。
#   单条违反可以解释，两条一起就没法解释了 —— 这正是 P3 成为转折点的原因。
EXCITATION: dict[str, dict[str, float]] = {
    "disclosure":    {"attention": 0.10, "panic": -0.06, "trust": 0.09,
                      "polarization": -0.04, "risk": -0.06, "stability": 0.06},
    "suppression":   {"attention": 0.14, "panic": 0.10, "trust": -0.22,
                      "polarization": 0.18, "risk": 0.15, "stability": -0.12},
    "contradiction": {"attention": 0.08, "panic": 0.06, "trust": -0.12,
                      "polarization": 0.06, "risk": 0.05},
    "testimony":     {"attention": 0.10, "panic": 0.05, "trust": -0.05,
                      "polarization": 0.03, "risk": 0.03},
    "amplification": {"attention": 0.20, "panic": 0.10, "trust": -0.05,
                      "polarization": 0.10, "risk": 0.12},
    "reassurance":   {"attention": 0.02, "panic": -0.03, "trust": -0.03},
    "discussion":    {"attention": 0.04, "panic": 0.02, "trust": -0.01},
}

# 激励表里单项权重绝对值的上界（= |suppression 对 trust 的 -0.22|）。
# 用它把「本轮话语构成」归一化到 [-1, 1]。改动激励表时若出现更大的权重，
# 这个常量必须同步更新 —— 测试里有断言守着。
REF_WEIGHT = 0.22

# 声势的饱和参考条数。本轮行为条数达到这个量级时，volume 项接近饱和。
# 取 8 的依据：离线重放一轮约 5–6 条材料，仿真一轮约数十条 agent 行为，
# 取 8 让前者处于半饱和、后者接近饱和，量级上能对接。
# **这是一个标定量，不是从数据里估出来的**，如实记在这里。
VOLUME_REF = 8.0


# ---------------------------------------------------------------------------
# 行为分类
# ---------------------------------------------------------------------------

# 按优先级排列：**先匹配到的类型胜出**。
# 优先级不是随意排的：压制的信号最强也最不该被误判成「讨论」，
# 所以放最前；reassurance 最弱，放最后。
KIND_PRIORITY: tuple[str, ...] = (
    "suppression", "disclosure", "contradiction",
    "amplification", "testimony", "reassurance", "discussion",
)

KIND_KEYWORDS: dict[str, tuple[str, ...]] = {
    "suppression": ("删帖", "删除", "劝导删除", "撤回", "不要讨论", "注意言行",
                    "禁止", "封禁", "施压", "要求删", "别在网上", "不接受采访",
                    "不转发", "不评论"),
    # ⚠️ disclosure 的词表刻意收窄过。
    # 早期版本含「公开」「说明」「发布」「回应」等泛用词，结果是每份材料都命中——
    # 因为这些词在**叙述性文本**里到处都是（「公开性」「校方发布说明」）。
    # 一个关键词类如果命中率超过一半，它就没有在分类，只是在计数。
    # 现在只保留**必然指向权威方主动披露**的具体词。
    "disclosure": ("公布", "承认", "道歉", "致歉", "更正", "出席",
                   "直播", "当面致歉", "完整文字实录", "逐专业列出"),
    "contradiction": ("不符", "矛盾", "失实", "对不上", "并未", "并没有",
                      "未接到", "说好的", "实际上", "翻了一遍", "对质"),
    "amplification": ("转载", "转发", "阅读量", "热搜", "媒体", "记者",
                      "报道", "自媒体", "话题"),
    "testimony": ("我是", "我们那届", "现身", "亲历", "我自己", "当年",
                  "被算进去", "我也被"),
    "reassurance": ("安抚", "请放心", "不用担心", "没有影响", "暂不回应",
                    "可控", "稳住", "做好记录"),
    "discussion": ("讨论", "问一下", "怎么看", "有没有人", "理性"),
}


def classify(text: str) -> str:
    """把一段文本归到一种行为类型。

    优先级顺序见 KIND_PRIORITY。**这是刻意的单标签设计**：
    多标签会让一条行为同时贡献多个方向，权重的可解释性立刻消失，
    而本模块的全部价值就在于「每个数字都能追溯到一条理由」。

    分类是确定性关键词匹配，不含 LLM。理由同模块开头：
    演化的数字必须可复现，LLM 只应出现在定性层（事件命名、简报撰写）。
    """
    for kind in KIND_PRIORITY:
        if any(kw in text for kw in KIND_KEYWORDS[kind]):
            return kind
    return "discussion"


# ---------------------------------------------------------------------------
# 状态容器
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    """某一时刻的六维状态。值域 [0, 1]。"""

    values: dict[str, float] = field(default_factory=dict)

    @classmethod
    def baseline(cls, params: dict[str, DimensionParams] | None = None) -> WorldState:
        p = params or DIMENSION_PARAMS
        return cls({d: p[d].baseline for d in DIMENSIONS})

    def as_dict(self) -> dict[str, float]:
        return {d: round(self.values[d], 4) for d in DIMENSIONS}

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def delta_from(self, other: WorldState) -> dict[str, float]:
        return {d: self.values[d] - other.values[d] for d in DIMENSIONS}


@dataclass
class DetectedEvent:
    """从状态变化中检出的一个事件。"""

    phase_id: str
    kind: str
    dimension: str
    magnitude: float
    message: str


@dataclass
class RoundResult:
    """一步演化的完整过程。中间量全部保留，便于解释与绘图。"""

    phase_id: str
    dt: float
    state_before: WorldState
    state_after: WorldState
    excitation: dict[str, float]
    relaxation: dict[str, float]
    coupling: dict[str, float]
    behavior_counts: dict[str, int]
    events: list[DetectedEvent] = field(default_factory=list)

    def deltas(self) -> dict[str, float]:
        # 调 delta_from 而不是重抄一遍公式：两处各写一遍，早晚会有一处改了
        # 另一处没改。此前这里确实是抄的，`delta_from` 因此成了死代码。
        return self.state_after.delta_from(self.state_before)


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------

class WorldStateEngine:
    """六维状态引擎。

    Args:
        params: 维度参数。默认用 DIMENSION_PARAMS。
        coupling: 耦合矩阵。默认用 COUPLING。
        excitation: 行为激励表。默认用 EXCITATION。
    """

    def __init__(
        self,
        params: dict[str, DimensionParams] | None = None,
        coupling: dict[str, dict[str, float]] | None = None,
        excitation: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.params = params or DIMENSION_PARAMS
        self.coupling = coupling if coupling is not None else COUPLING
        self.excitation = excitation or EXCITATION

    # -- 单步 --------------------------------------------------------------

    def step(
        self,
        state: WorldState,
        behaviors: list[str],
        *,
        dt: float,
        phase_id: str = "",
        extra: dict[str, float] | None = None,
    ) -> RoundResult:
        """推进一步。

        Args:
            state: 推进前的状态。
            behaviors: 本轮的行为类型列表（通常由 classify() 产出）。
            dt: 距上一步的天数。弛豫量由它决定。
            phase_id: 阶段标识，写进检出的事件里。
            extra: 额外激励，直接叠加到激励量上（用于手动注入事件）。
        """
        counts: dict[str, int] = {}
        for b in behaviors:
            counts[b] = counts.get(b, 0) + 1

        # --- step 1 激励 ---
        # 用**构成占比**而不是条数来算激励，理由是可移植性：
        # 离线重放一次只喂进来几条材料，仿真一轮可能喂进来几百条 agent 行为，
        # 按条数算，两者相差几十倍，同一个模型没法同时适配。
        # 按占比算，状态取决于「哪种话语占了上风」，与喂进来的绝对条数解耦。
        #
        # 条数并非完全不计 —— 它通过 volume 项以饱和的方式参与（见下），
        # 所以「同样构成下，声势更大者影响更大」仍然成立，只是不会线性放大。
        n_total = max(len(behaviors), 1)
        raw = {d: 0.0 for d in DIMENSIONS}
        for kind, n in counts.items():
            frac = n / n_total
            for dim, w in self.excitation.get(kind, {}).items():
                raw[dim] += w * frac
        for dim, v in (extra or {}).items():
            raw[dim] = raw.get(dim, 0.0) + v

        volume = math.tanh(n_total / VOLUME_REF)

        excitation: dict[str, float] = {}
        for d in DIMENSIONS:
            # REF_WEIGHT 是激励表里单项权重绝对值的上界。
            # 除以它，是把 raw 归一化到 [-1, 1]：
            # +1 表示「本轮话语完全由最有利于该维度的行为构成」。
            norm = _clamp(raw[d] / REF_WEIGHT, -1.0, 1.0)
            e = self.params[d].gain * volume * norm
            # 有利方向的激励打折扣 —— 见 ASYMMETRY 的说明。
            # 「有利」对每个维度不是同一个符号：信任与稳定是越高越好，
            # 极化是越低越好。所以先统一到「有利方向」，再判断。
            if d in ASYMMETRY and e * FAVOURABLE_SIGN.get(d, 1) > 0:
                e *= ASYMMETRY[d]
            excitation[d] = e

        excited = {d: state.values[d] + excitation[d] for d in DIMENSIONS}

        # --- step 2 弛豫 ---
        # 每个维度按各自的半衰期向基线回落。这一步是整个模型的题眼。
        relaxation = {
            d: (self.params[d].baseline - excited[d])
            * (1.0 - math.exp(-self.params[d].k * dt))
            for d in DIMENSIONS
        }
        relaxed = {d: excited[d] + relaxation[d] for d in DIMENSIONS}

        # --- step 3 耦合 ---
        # 用弛豫后的状态算，避免同一步内的激励被耦合二次放大。
        coupling = {d: 0.0 for d in DIMENSIONS}
        for src, targets in self.coupling.items():
            dev = relaxed[src] - self.params[src].baseline
            if dev == 0.0:
                continue
            for dst, w in targets.items():
                coupling[dst] += w * dev
        final = {
            d: _clamp(relaxed[d] + coupling[d], 0.0, 1.0) for d in DIMENSIONS
        }

        after = WorldState(final)
        before = WorldState(dict(state.values))
        events = self.detect_events(before, after, counts, phase_id=phase_id, dt=dt)

        return RoundResult(
            phase_id=phase_id,
            dt=dt,
            state_before=before,
            state_after=after,
            excitation=excitation,
            relaxation=relaxation,
            coupling=coupling,
            behavior_counts=counts,
            events=events,
        )

    # -- 事件检测 ----------------------------------------------------------

    #: 风险越过此值即视为需要预警。0.70 的取法：基线 0.10、饱和 1.0，
    #: 越过 0.70 意味着已经离开「正常波动」区间很远了。
    RISK_ALERT = 0.70

    #: 单步信任跌幅超过此值即视为骤降（而非自然回落）。
    TRUST_DROP = 0.06

    def detect_events(
        self,
        before: WorldState,
        after: WorldState,
        counts: dict[str, int],
        *,
        phase_id: str = "",
        dt: float = 1.0,
    ) -> list[DetectedEvent]:
        """从一步的状态变化里检出值得报告的事件。

        只报**可判定**的事件：阈值穿越、单步骤变、特定行为出现。
        不做「模型觉得这很重要」这类无法证伪的判断。
        """
        events: list[DetectedEvent] = []
        d = {k: after.values[k] - before.values[k] for k in DIMENSIONS}

        if before["risk"] < self.RISK_ALERT <= after["risk"]:
            events.append(DetectedEvent(
                phase_id, "risk_alert", "risk", after["risk"],
                f"风险越过预警线 {self.RISK_ALERT:.2f}（当前 {after['risk']:.3f}）",
            ))

        # 骤降按天归一 —— 否则同样大的跌幅，拖 6 天和拖 2 天会被判得一样重
        trust_rate = d["trust"] / max(dt, 1e-9)
        if trust_rate < -self.TRUST_DROP:
            events.append(DetectedEvent(
                phase_id, "trust_drop", "trust", d["trust"],
                f"信任单日跌幅 {-trust_rate:.3f}，超过阈值 {self.TRUST_DROP}",
            ))

        if d["polarization"] / max(dt, 1e-9) > 0.05:
            events.append(DetectedEvent(
                phase_id, "polarization_surge", "polarization", d["polarization"],
                f"极化加速上升（+{d['polarization']:.3f}）",
            ))

        if counts.get("suppression"):
            events.append(DetectedEvent(
                phase_id, "suppression_detected", "trust",
                float(counts["suppression"]),
                f"检出压制类行为 {counts['suppression']} 条 —— 议题可能从「事实争议」"
                f"转为「动机争议」",
            ))

        if before["stability"] >= 0.60 > after["stability"]:
            events.append(DetectedEvent(
                phase_id, "stability_break", "stability", after["stability"],
                f"稳定跌破 0.60（当前 {after['stability']:.3f}）",
            ))

        events.extend(self._detect_inflection(phase_id, before, after))
        return events

    def _detect_inflection(
        self, phase_id: str, before: WorldState, after: WorldState
    ) -> list[DetectedEvent]:
        """检出「叙事转向」：信任在掉，但关注在退。

        这是最值得预警的一类状态 —— 注意力退潮后剩下的不是平静，
        而是沉淀下来的不信任。热搜会掉，信任不会自己回来。
        """
        if (
            after["trust"] < before["trust"]
            and after["attention"] < before["attention"]
            and after["trust"] < self.params["trust"].baseline - 0.15
        ):
            return [DetectedEvent(
                phase_id, "narrative_settling", "trust", after["trust"],
                "关注度回落但信任继续下探 —— 议题正在沉淀为长期不信任，"
                "而非自然消散",
            )]
        return []

    # -- 反事实 ------------------------------------------------------------

    def counterfactual(
        self,
        state: WorldState,
        *,
        intervention: dict[str, float],
        behaviors_after: dict[str, list[str]],
        dts: dict[str, float],
    ) -> dict[str, WorldState]:
        """在给定状态上施加一次干预，再往后推演，返回与原路径的对照。

        这是「未然」的实际产出：不是告诉你现在多糟，
        而是告诉你「现在做 X，两周后会不一样」。

        Args:
            intervention: 直接叠加到**当前这一步**的激励，如 {"disclosure": ...}。
            behaviors_after: 阶段 id -> 该阶段的行为列表。
            dts: 阶段 id -> 该阶段的天数。

        Returns:
            阶段 id -> 该阶段结束时的状态。
        """
        extra = {d: intervention.get(d, 0.0) for d in DIMENSIONS}
        cur = WorldState(dict(state.values))
        timeline: dict[str, WorldState] = {}
        first = True
        for pid, behaviors in behaviors_after.items():
            result = self.step(
                cur, behaviors, dt=dts.get(pid, 1.0), phase_id=pid,
                extra=extra if first else None,
            )
            cur = result.state_after
            timeline[pid] = cur
            first = False
        return timeline

    def two_branch_futures(
        self,
        state: WorldState,
        *,
        behaviors_after: dict[str, list[str]],
        dts: dict[str, float],
        action: str = "disclosure",
        scale: float = 2.5,
        drop: str = "suppression",
    ) -> tuple[dict[str, WorldState], dict[str, WorldState]]:
        """「现在换成 X 会怎样」的两个分支 —— **只报一个数会假装它比实际更确定**。

        为什么是两个而不是一个：一次强公开之后，后续的行为本身会不会跟着变，
        模型答不了。所以给两端，并写明两端各自的假设：

          A **激励版**：后续行为不变，只在这一步叠加一次强公开。
            **保守下界** —— 它假设公开了也照样会有人去压。
          B **机制版**：公开之后那件不可逆的事（`drop`，默认劝删）不再发生。
            **乐观上界** —— 它假设「主动说清」真的能消掉「需要压」的动机。

        **`behaviors_after` 必须把「做决定的那个阶段」自己作为第一个键。**
        `counterfactual()` 只把 `extra` 施加在第一步（见那里的 `first` 标志），
        所以若从下一个阶段起算，干预就整整晚了一个阶段 —— 而曲线照样平滑、
        不报错、也不抛异常。

        （这不是假设：`validate.py` 的 AS-10 原先正好踩这个坑 —— 断言文字说
        「假设 P2 主动公开」，实际干预落在 P3，实测把收益从 +0.1219 夸大成
        +0.1740，差了 43%。两个分支都仍然「通过」，所以不看数字根本发现不了。）

        Args:
            action: 用作干预的行为类型，默认 `disclosure`。
            scale: 干预强度的倍数。**2.5 是本项目沿用的约定值，不是标定出来的
                系数** —— 它让 `disclosure` 在 `attention` 与 `trust` 上顶到
                归一化上限（`REF_WEIGHT`），于是这两个维度的**激励项**对再加大
                倍数不敏感（`trust` 仍会经由耦合变化，实测 K 从 2.5 加到 10
                时 trust 终值 0.7424 → 0.7706）。
            drop: 机制版里被剔除的行为类型。

        Returns:
            `(branch_a, branch_b)`，均为「阶段 id -> 该阶段末状态」。
        """
        table = self.excitation[action]
        intervention = {d: table.get(d, 0.0) * scale for d in DIMENSIONS}
        branch_a = self.counterfactual(
            state, intervention=intervention,
            behaviors_after=behaviors_after, dts=dts,
        )
        suppressed = {
            pid: [b for b in behaviors if b != drop]
            for pid, behaviors in behaviors_after.items()
        }
        branch_b = self.counterfactual(
            state, intervention=intervention,
            behaviors_after=suppressed, dts=dts,
        )
        return branch_a, branch_b

    # -- 整段推演 ----------------------------------------------------------

    def run(
        self,
        phases: list[dict],
        *,
        start: WorldState | None = None,
    ) -> list[RoundResult]:
        """按阶段顺序推演整段。

        Args:
            phases: 形如 [{"id": "P1", "dt": 1.0, "behaviors": ["discussion", ...]}]。
                行为列表已经过 classify() 归好类。
            start: 起始状态，默认从基线开始。
        """
        cur = start or WorldState.baseline(self.params)
        out: list[RoundResult] = []
        for ph in phases:
            result = self.step(
                cur,
                ph.get("behaviors", []),
                dt=ph.get("dt", 1.0),
                phase_id=ph.get("id", ""),
                extra=ph.get("extra"),
            )
            cur = result.state_after
            out.append(result)
        return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ---------------------------------------------------------------------------
# 从种子材料构造推演输入
# ---------------------------------------------------------------------------

def sections_from_material(text: str) -> list[str]:
    """把一份种子材料切成若干「材料 X-Y」小节，并剥掉元数据行。

    切分粒度的选择是有理由的：**一个 `##` 小节是一份独立的材料物**
    （一份通稿、一段群聊记录、一张截图转录），而不是一个段落。

    早期版本按段落切，结果是论坛摘要里 40 条回帖各算一次行为，
    而「学工部劝删通知」只有一段 —— 文本量淹没了信号量，
    等于暗含「写得多 = 影响大」这个明显错误的假设。

    `>` 行（来源、发布时间）与 `|` 行（表格）是**文档格式**，不是行为。
    不剥掉的话，每份材料开头的「公开性 | 通稿公开」都会命中
    `disclosure` 的「公开」关键词，把分类器变成格式识别器。
    """
    import re

    out: list[str] = []
    # parts[0] 是前言（h1 标题 + 元数据表），整块丢弃
    for part in re.split(r"(?m)^##\s+", text)[1:]:
        body = "\n".join(
            ln for ln in part.splitlines()
            if not ln.strip().startswith((">", "|"))
        ).strip()
        if body:
            out.append(body)
    return out


def behaviors_from_material(text: str) -> list[str]:
    """把一份种子材料转成行为列表：每个 `## 材料 X-Y` 小节计一个行为。

    **已知局限（写下来，不遮掩）**：关键词分类器分不清
    「实施压制」与「讨论压制」。03 里学生转发劝删截图的那一节，
    会被归成 `suppression` —— 它确实是关于压制的，但行为主体是学生不是校方。
    两者对状态的作用方向恰好一致（都压低信任、抬高极化），
    因此数值影响有限，但机制解释上是不准确的。
    彻底解决需要区分行为主体，那要等仿真产出带 agent 角色的 actions.jsonl。
    """
    return [classify(s) for s in sections_from_material(text)]
