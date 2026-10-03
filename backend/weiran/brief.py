"""决策简报：把一次推演产出变成一份**可比较的处置方案**对照。

**不含 LLM、不含随机数 —— 同输入同输出，可进 CI。**
（这条要和它读的那条曲线分开说，两句话同屏出现时极易被读成矛盾：
简报的**生成**是确定的；简报**读的那条曲线**是不可端到端复现的。
金标 `metric_notes.scoring_determinism` 说的正是这件事。）

## 这个模块在补什么

README 承诺三件事：重放舆论演化、逐轮追踪六维状态、**并在你真正开口之前
给出可比较的处置方案**。前两件此前已实现，第三件一直没有产出方 —— 而构件
早就齐了，缺的只是「消费者」这个角色：

  - `WorldStateEngine.counterfactual()` 自称「这是「未然」的实际产出」，
    此前只被 `validate.py` 的 AS-10 用过一次；
  - 金标 `decision_windows`（5 个决策窗口 + 当时的选择 + 哪个是拐点）
    在此之前**全仓零引用**；
  - `perception.Phase.label`（作者手写的「引爆／口径质疑／删帖争议／
    外部介入／收束」）读进了对象但没人用；
  - `RoundResult.deltas()`、`CallStats.summary()` 是完全没人调用的死代码
    （`WorldState.delta_from()` 也在其列 —— `deltas()` 当时是**把它的公式
    重抄了一遍**而不是调它，所以两者都死。现在 `deltas()` 改为调用它，
    一行同时消掉了死代码与公式重复：这种重复迟早会有一处改了另一处没改）。

## 三件它**不**做的事

1. **不预测。** 分支反事实假设「后续行为不变」；现实里一旦真的公开，
   agent 的反应本身就会变，而那正是模型答不了的部分。所以 A／B 两版
   **不是区间，是两个模型假设下的两个点**，不构成任何保证。
2. **不评分。** 不引用 `reference_shape`（金标自己声明它不是观测值、
   不是评分基准），不引用 `validate.py` 的 MAE。装载金标时**主动剥掉**
   `reference_shape`，让它物理上进不来。
3. **不给建议。** 只说「在该模型内部，两条路径的高低如何」，不说
   「应当采取 X」。正文里禁用的词由 `check_brief()` 机检拦下。

用法：
    cd backend
    python -m weiran.brief                       # 读 data/simulation/twitter_rounds.json
    python -m weiran.brief --stdout              # 顺带把全文打到终端
    python -m weiran.brief --intervention-scale 1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import REPO_ROOT, ensure_console_encoding
from .perception import Phase, active_phase_by_round
from .profiles import DEFAULT_SCENARIO
# `_SILENT_ACTIONS` **必须从 `simulate` 拿，不能在这里重抄一份** ——
# 「哪些动作算话、哪些不算」是仿真的口径，简报只是它的读者。抄一份出来，
# 两边就会各自演化，而下面那个「话语动作数」正是靠它算的。
from .simulate import _SILENT_ACTIONS
from .world_state import (
    DIMENSIONS,
    DIMENSION_ZH,
    REF_WEIGHT,
    VOLUME_REF,
    WorldState,
    WorldStateEngine,
)

DEFAULT_ROUNDS_FILE = REPO_ROOT / "data" / "simulation" / "twitter_rounds.json"
# 默认与**它依据的那次运行**放在一起（`data/simulation/`），而不是另起一个
# 目录：简报离开它依据的那份产出就没有意义，两者分开放必然日后对不上。
DEFAULT_REPORT_DIR = REPO_ROOT / "data" / "simulation"

# 干预强度的默认倍数。**这是沿用的约定值，不是标定出来的系数** ——
# 与 `validate.py` 同口径。两个模块对同一件事说两个数，比说错一个数更糟。
DEFAULT_SCALE = 2.5


class BriefError(RuntimeError):
    """简报做不下去时要**出声**。

    与 `perception.KnowledgeError` 同一条纪律：宁可在生成时炸，
    也不要交出一份悄悄少了内容的简报。
    """


# ---------------------------------------------------------------------------
# 干预口径：这是**本模块自己写的假设**，必须让读者一眼看到
# ---------------------------------------------------------------------------
#
# 金标 `decision_windows` 只有 `window`（问什么）与 `actual_choice`（当时怎么做的），
# **没有「另一种做法」字段** —— 替代方案在模型里怎么表达，是这份简报的假设，
# 不是金标的事实。所以这张表要连同理由一起印进正文，不能只留在代码里。
WINDOW_INTERVENTIONS: dict[str, tuple[str, str]] = {
    "P1": ("disclosure", "「主动说明」在本模型的激励表里对应一次强公开"),
    "P2": ("disclosure", "「公开分专业数据」对应一次强公开"),
    "P3": ("disclosure", "「承认劝删做法不当」对应一次强公开（回应而非压制）"),
    "P4": ("disclosure", "「校长出席、一次给到位」对应一次强公开"),
    "P5": ("disclosure", "「公布长效机制」对应一次强公开"),
}


# ---------------------------------------------------------------------------
# 口径标记与机检
# ---------------------------------------------------------------------------
#
# 每条声明都做成**短标记**：机检要求标记出现在**同一节正文里**，
# 而不是「文档某处提过」。声明写在别处等于没写。
MARKERS = {
    "nonreproducible": "[不可端到端复现]",
    "compressed": "[压缩模式·与轮=天不可比]",
    "model_internal": "[模型内对照·非预测]",
    "assumption": "[干预口径为本文自设假设]",
    "degraded": "[降级项]",
    # **这个标记无条件出现**，与其他几个不同 —— 因为「没截断」是一条
    # 必须主动说出来的结论，而不是一个可以省略的缺省。
    "context": "[上下文压力·已监测]",
    # 同理无条件出现：**每一轮的六维状态都是「几条行为算出来的」**，
    # 这个分母不是一个可以省略的细节 —— 见 `_evidence_block()`。
    "evidence": "[证据量·逐轮]",
}

DISCLAIMERS = {
    "nonreproducible": (
        "**端到端不可复现。** agent 每次说的话都不一样，缓存只能保证「同一句话给"
        "同一个标签」，保证不了「同一场推演说同一句话」。所以任何一次运行的曲线都"
        "只是**一个样本**，正式口径应当是「多 seed 跑 N 次、报均值与方差」，"
        "而不是「跑一次就准」。"
    ),
    "compressed": (
        "**本轮运行是压缩模式。** 每轮代表多于一天时，六维曲线与「轮 = 天」"
        "**不可比** —— 弛豫项 `exp(-k·dt)` 可以复合，但激励项按步累加："
        "同样 7 天拆成 7 步会激励 7 次、合成 1 步只激励 1 次。两条曲线不是"
        "同一条曲线的粗细版本，是两条不同的曲线。"
    ),
    "model_internal": (
        "**反事实不是预测。** 分支假设「后续行为不变」，而现实中一旦真的公开，"
        "agent 的反应本身就会变 —— 那正是模型答不了的部分。A／B 两版不是区间，"
        "是两个模型假设下的两个点，**不构成上下界保证**。表里的差异只表示在该模型"
        "内部两条路径的高低，不表示现实中的处置效果；而且本模型对有利方向的激励"
        "打了折扣（`ASYMMETRY`，trust 0.45），所以正值本身也是被折扣过的。"
    ),
    "assumption": (
        "**替代做法怎么表达，是这份简报自设的假设，不是金标的事实。** "
        "金标只在 `decision_windows` 里给了「当时怎么做」，没给「还能怎么做」。"
    ),
    "degraded": (
        "**降级项意味着下面某些数字不能按字面读。** 逐条列在文末。"
    ),
}

# 正文不得出现的词。**这是「不得把注入生效说成效果变好」的落地。**
_BANNED = (
    "准确率", "预测精度", "模型准确", "改善", "恶化", "提升", "证明",
    "建议采取", "应当立即", "推荐方案", "效果最好", "最优解",
)

_NUMBER = re.compile(r"\d+\.\d+")


# ---------------------------------------------------------------------------
# 装载
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """一份受版本控制的文本产出的指纹 —— **按文本算，不按字节**。

    `read_bytes()` 是个只有**别人 clone 下来**才撞得到的坑：`.gitattributes`
    的 `eol=lf` 会在检出时把行尾统一成 LF，而本机的工作区是 CRLF。同一个文件
    在两个人的工作区里**字节不同、文本相同**，于是按字节算出来的指纹钉住的
    是「谁的行尾配置」，不是「这份产出有没有变」—— 照 README 走一遍的人，
    会先看到一条指向**不存在的问题**的红。

    按文本算之后，LF 与 CRLF 两个副本给出同一个指纹。这不是把判据放松了：
    行尾是 git 的合法产物，不是内容的改动。展示层与金标对照表都引这一个
    函数（`viewer._sha256` 就是从这里取的）—— 同一件东西抄两份，迟早有一份
    会漏改。
    """
    return hashlib.sha256(
        path.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()[:16]


def _rel(path: Path) -> str:
    """相对仓库根的路径（正斜杠）；不在仓库内则原样返回。

    指纹里记**相对路径**而不是绝对路径：绝对路径把「本机目录」焊进了入库产物，
    换台机器、或仓库换个位置，产物就和产生它的那次运行对不上 —— 而这份产物的
    全部意义就是「我看到的和它说的是同一件事」。相对路径让它在任何克隆位置
    都能自我复现。
    """
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_rounds(path: Path) -> dict:
    """读一份推演产出，并记下它的指纹。

    指纹用 `sha256` 而**不是** `mtime` —— `mtime` 会让「同输入同输出」这条
    性质在文件被 touch 之后失效，而这两件事必须能分开。
    """
    if not path.is_file():
        raise BriefError(
            f"找不到推演产出：{path}\n"
            f"  先跑一次：python -m weiran.simulate --agents 3 --rounds 3"
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in ("meta", "rounds") if k not in doc]
    if missing:
        raise BriefError(
            f"{path.name} 缺少 {missing} —— 这不是本项目的推演产出格式。"
            f"（早期版本曾把整个文件写成裸列表，那种文件在这里会被拦下。）"
        )
    if not doc["rounds"]:
        raise BriefError(f"{path.name} 的 rounds 是空的，没有可对照的轮次。")
    doc["_provenance"] = {
        "rounds_file": _rel(path),
        "rounds_sha256": _sha256(path),
    }
    return doc


def load_gold(scenario_dir: Path) -> dict:
    """读金标，并**主动剥掉 `reference_shape`**。

    金标 `gold_status.what_is_NOT_solid` 原文：`reference_shape` 是「手写的
    预期，不是观测值」，任何以它为分母的「准确率」都是自说自话。这里不是
    「记得别引用它」，而是让它在物理上进不来 —— 剥掉之后，模板里就算手滑
    也拿不到这个字段。
    """
    path = scenario_dir / "reference_data.json"
    if not path.is_file():
        raise BriefError(f"找不到金标：{path}")
    ref = json.loads(path.read_text(encoding="utf-8"))

    for ph in ref.get("phases", []):
        ph.pop("reference_shape", None)
    ref["_provenance"] = {
        "scenario_dir": _rel(scenario_dir),
        "scenario_file": _rel(path),
        "scenario_sha256": _sha256(path),
    }
    return ref


# ---------------------------------------------------------------------------
# 纯函数：覆盖映射、重放、反事实
# ---------------------------------------------------------------------------

def phase_ids_of(round_entry: dict) -> list[str]:
    """一轮的 `phase_id` 可能是 `"P3,P4"` —— 压缩模式把两个阶段挤进了一轮。"""
    raw = (round_entry.get("phase_id") or "").strip()
    return [p.strip() for p in raw.split(",") if p.strip()]


def resolve_phase_label(i: int, round_entry: dict,
                        phase_by_round: dict[int, str]) -> str:
    """这一轮该标哪个阶段。

    **这是两个不同的问句**（`simulate.py` 里那段注释说的是同一件事）：
      - `round_entry["phase_id"]`：这一轮**注入**了哪个阶段的事件（只有少数轮有）
      - `phase_by_round[i]`：这一轮**处在**哪个阶段（每一轮都有）

    优先用注入的那个 —— 实跑时引擎标的就是它；没有才回落到「处于」。

    **回落到 `f"R{i}"` 是错的**（这里以前就是这么写的）：那等于把轮号当阶段名
    填进「阶段」列，简报里于是出现 `| R1 | R1 |` 这种行；而同一份缺失在逐轮小节
    里被渲染成 `—`，同一件事两套说法。轮号是**编出来**的标签，宁可空着。

    空串表示「无法判定」，由渲染层统一显示成 `—`。
    """
    injected = phase_ids_of(round_entry)
    return injected[0] if injected else phase_by_round.get(i, "")


def window_rounds(rounds: list[dict], windows: list[dict]) -> dict[str, list[int]]:
    """窗口阶段 -> 含该阶段的轮号列表。空列表 = **本次运行未覆盖**。"""
    out: dict[str, list[int]] = {}
    for w in windows:
        pid = w["phase"]
        out[pid] = [i for i, r in enumerate(rounds) if pid in phase_ids_of(r)]
    return out


def replay(engine: WorldStateEngine, rounds: list[dict], dpr: float,
           phase_by_round: dict[int, str] | None = None) -> list:
    """按 `behaviors` 重放整段，拿回 `RoundResult`。

    **为什么不直接从 JSON 相邻两轮相减**：JSON 只存 4 位小数，而且拿不到
    `events` 与激励／弛豫／耦合三个分量。重放拿得到全部，代价是要验证
    「重放确实复现了记录」（见 `replay_deviation`）。

    `phase_by_round` 只影响标签，不影响任何数值 —— 阶段名不参与激励、弛豫
    与耦合的计算。所以补上它不会让 `replay_deviation` 变化。
    """
    pbr = phase_by_round or {}
    cur = WorldState.baseline(engine.params)
    out = []
    for i, rd in enumerate(rounds):
        res = engine.step(cur, rd.get("behaviors", []), dt=dpr,
                          phase_id=resolve_phase_label(i, rd, pbr))
        cur = res.state_after
        out.append(res)
    return out


def replay_deviation(results: list, rounds: list[dict]) -> float:
    """重放结果与产出记录的最大逐维偏差。

    产出只存 4 位小数，所以理论下限约 `5e-5`。**明显超过 `1e-4` 说明这份
    产出不是本引擎按当前参数产生的**（换过参数、换过版本、或被人手改过）
    —— 那是个必须报出来的降级项，不是四舍五入误差。
    """
    worst = 0.0
    for res, rd in zip(results, rounds):
        for d in DIMENSIONS:
            worst = max(worst, abs(res.state_after[d] - rd.get("state", {}).get(d, 0.0)))
    return worst


def fork_state(rounds: list[dict], engine: WorldStateEngine, r: int) -> WorldState:
    """分叉状态 = 窗口所在轮的**前一轮结束时**。

    语义是「该阶段的事件照旧发生，只是处置不同」，所以分叉发生在该阶段
    **开始之前**。`r == 0` 时没有前一轮，以基线为分叉点（调用方要标注）。
    """
    if r <= 0:
        return WorldState.baseline(engine.params)
    return WorldState(dict(rounds[r - 1]["state"]))


def clamp_report(engine: WorldStateEngine, action: str, scale: float) -> list[dict]:
    """干预在各维度上的归一化激励，以及**是否被上限截断**。

    这条必须报出来：`scale=2.5` 时 `attention` 与 `trust` 的归一化激励已经
    超过 1 被截断，于是这两个维度的**激励项**对再加大倍数不敏感。读者看到
    「A 版信任高 0.03」时有权知道那个数是在一个已经饱和的推杆下得到的。
    """
    table = engine.excitation[action]
    out = []
    for d in DIMENSIONS:
        raw = table.get(d, 0.0) * scale
        norm = raw / REF_WEIGHT
        out.append({
            "dimension": d, "zh": DIMENSION_ZH[d],
            "raw": round(raw, 4), "normalized": round(norm, 4),
            "clamped": abs(norm) > 1.0,
        })
    return out


@dataclass
class WindowRow:
    """一个决策窗口在**本次运行**里的样子。"""
    phase: str
    label: str
    trigger: str
    question: str
    actual_choice: str
    is_turning_point: bool
    gold_day: int
    round_index: int | None = None
    fork_day: float | None = None
    day_skew: float | None = None
    coverage: str = "uncovered"          # uncovered | shared | exclusive
    shared_with: list[str] = field(default_factory=list)
    actual_from_fork: float | None = None
    branch_a: float | None = None
    branch_b: float | None = None
    delta_a: float | None = None
    delta_b: float | None = None
    note: str = ""

    @property
    def covered(self) -> bool:
        return self.round_index is not None


def build_windows(
    rounds: list[dict],
    phases: list,
    windows: list[dict],
    engine: WorldStateEngine,
    dpr: float,
    *,
    scale: float,
) -> tuple[list[WindowRow], dict]:
    """窗口表 + 覆盖统计。每一行都带**三态覆盖**，未覆盖的行也照样出现。"""
    mapping = window_rounds(rounds, windows)
    by_phase = {p.phase_id: p for p in phases}
    n = len(rounds)
    rows: list[WindowRow] = []

    for w in windows:
        pid = w["phase"]
        if pid not in WINDOW_INTERVENTIONS:
            raise BriefError(
                f"金标的 decision_windows 里有「{pid}」，但 WINDOW_INTERVENTIONS "
                f"里没有对应的干预口径。**不能静默跳过** —— 那会让这一行"
                f"无声消失。请补一行并写清理由。"
            )
        action, _ = WINDOW_INTERVENTIONS[pid]
        ph = by_phase.get(pid)
        row = WindowRow(
            phase=pid,
            label=ph.label if ph else "",
            trigger=ph.trigger if ph else "",
            question=w.get("window", ""),
            actual_choice=w.get("actual_choice", ""),
            is_turning_point=bool(w.get("is_turning_point")),
            gold_day=int(ph.day) if ph else 0,
        )
        hits = mapping.get(pid) or []
        if not hits:
            row.coverage = "uncovered"
            row.note = ("本次运行未覆盖此窗口 —— 该阶段在本轮运行的任何一轮里都"
                        "没有出现（多半是没开阶段注入）。")
            rows.append(row)
            continue

        r = hits[0]
        row.round_index = r
        row.fork_day = round(r * dpr, 4)
        row.day_skew = round(row.fork_day - row.gold_day, 4)
        siblings = [p for p in phase_ids_of(rounds[r]) if p != pid]
        if siblings:
            row.coverage = "shared"
            row.shared_with = siblings
        else:
            row.coverage = "exclusive"
        if r == 0:
            row.note = "分叉点为基线（该阶段是首轮，没有前一轮）"

        # 配对对照：**从同一个分叉点**再跑一条不施加干预的路径。
        # 不能拿产出的最后一轮当「实际」—— 那是从基线分叉出来的另一条路，
        # 与被比较的分支不同源，相减是方法论错误（compressed 模式下还会
        # 因为 JSON 只有 4 位小数而多出 1e-4 的假差异）。
        fut = {f"R{k}": rounds[k].get("behaviors", []) for k in range(r, n)}
        fdt = {f"R{k}": dpr for k in range(r, n)}
        st = fork_state(rounds, engine, r)
        last = f"R{n - 1}"
        paired = engine.counterfactual(st, intervention={},
                                      behaviors_after=fut, dts=fdt)
        a, b = engine.two_branch_futures(st, behaviors_after=fut, dts=fdt,
                                        action=action, scale=scale)
        row.actual_from_fork = round(paired[last]["trust"], 4)
        row.branch_a = round(a[last]["trust"], 4)
        row.delta_a = round(row.branch_a - row.actual_from_fork, 4)
        # 机制版只在拐点窗口上有对应物：它的定义是「那件不可逆的事没发生」，
        # 而非拐点窗口本就不涉及那件事。给每行都配 B 会让 4/5 行退化成 A==B，
        # 反而稀释 B 的信息量。
        if row.is_turning_point:
            row.branch_b = round(b[last]["trust"], 4)
            row.delta_b = round(row.branch_b - row.actual_from_fork, 4)
        else:
            row.note = (row.note + "；" if row.note else "") + \
                "本窗口不涉及劝删，不设机制版"
        rows.append(row)

    coverage = {
        "n_uncovered": sum(1 for x in rows if not x.covered),
        "n_shared": sum(1 for x in rows if x.coverage == "shared"),
        "by_phase": mapping,
    }
    return rows, coverage


def turning_point_block(engine: WorldStateEngine, results: list, gold: dict,
                        rows: list[WindowRow]) -> dict:
    """金标说「拐点在哪、什么信号」，运行说「实际看到什么」。

    两边**并列**报出，不合并成一个结论 —— 本次实测里它们并不一致
    （见 `signal_agrees_with_gold`），合并会把不一致掩盖掉。
    """
    tp = gold.get("turning_point", {})
    detected = []
    for i, res in enumerate(results):
        for e in res.events:
            detected.append({
                "round": i, "phase": res.phase_id, "kind": e.kind,
                "dimension": e.dimension,
                "magnitude": round(e.magnitude, 4),
                "message": e.message,
            })

    # 金标 `detectable_signal` 里唯一可机检的那条：极化增速超过关注增速。
    signals = []
    for i, res in enumerate(results):
        d = res.deltas()
        signals.append({
            "round": i,
            "phase": res.phase_id,
            "d_attention": round(d["attention"], 4),
            "d_polarization": round(d["polarization"], 4),
            "polarization_exceeds_attention": d["polarization"] > d["attention"],
        })
    first = next((s["round"] for s in signals
                  if s["polarization_exceeds_attention"]), None)
    gold_day = tp.get("day")

    return {
        "gold": {
            "phase": tp.get("phase", ""),
            "day": gold_day,
            "event": tp.get("event", ""),
            "why_it_matters": tp.get("why_it_matters", ""),
            "detectable_signal": tp.get("detectable_signal", ""),
            "provenance": "场景叙事，作者编写，**非模型产出**",
        },
        "detected_events": detected,
        "signal_series": signals,
        "first_round_polarization_exceeds_attention": first,
        "signal_agrees_with_gold": bool(
            first is not None and gold_day is not None
            and any(r.covered and r.phase == tp.get("phase")
                    and r.round_index == first for r in rows)
        ),
        # 角色冲突指标本轮**不报数**：`perception.py` 已明确它只能从真实仿真
        # 产出里读，而当前产出的 `injected` 只存了摘要。列出来并说明原因，
        # 而不是假装报了一个数。
        "unimplemented_signals": [{
            "name": "辅导员群体的角色冲突指标",
            "reason": ("需要逐角色「公开立场 vs 私下立场」的情感极性差；当前产出里 "
                       "`injected` 只存块摘要（sha1 前 12 位），不含立场文本。"
                       "宁可空着，也不报一个编出来的数。"),
        }],
    }


# ---------------------------------------------------------------------------
# 组装与渲染
# ---------------------------------------------------------------------------

def build_brief(
    rounds_doc: dict,
    gold: dict,
    *,
    scale: float = DEFAULT_SCALE,
    command: str = "",
) -> dict:
    """组装结构化简报。**最后一步强制做口径机检，违规就抛。**"""
    meta = rounds_doc["meta"]
    rounds = rounds_doc["rounds"]
    dpr = float(meta.get("days_per_round", 1.0))
    engine = WorldStateEngine()

    phases = _phases_from_gold(gold)
    if not phases:
        # 没有 phases，5 个窗口一个都映射不上、标签全是空串，但**渲染不会报错**
        # —— 交出去的会是一份看起来完整、实际每行都缺胳膊少腿的简报。
        # 所以这里出声。
        raise BriefError(
            f"金标 {gold.get('_provenance', {}).get('scenario_file', '')} 里"
            f"没有任何阶段（`phases` 为空）—— 决策窗口无从映射，简报无从生成。"
        )
    # 「这一轮处在哪个阶段」。产出 JSON 里**只存了注入的那个**（`phase_id`），
    # 注入只发生在少数轮上，所以多数轮在 JSON 里看不出自己属于哪个阶段。
    # 从金标阶段表 + `days_per_round` 现算，与实跑时 `simulate.py` 用的是
    # 同一个 `perception.active_phase_by_round` —— 不是重写一份等价逻辑。
    phase_by_round = active_phase_by_round(phases, len(rounds), dpr)
    results = replay(engine, rounds, dpr, phase_by_round)
    rows, coverage = build_windows(rounds, phases, gold.get("decision_windows", []),
                                   engine, dpr, scale=scale)

    per_round = []
    for i, (res, rd) in enumerate(zip(results, rounds)):
        behaviors = rd.get("behaviors", [])
        counts: dict[str, int] = {}
        for b in behaviors:
            counts[b] = counts.get(b, 0) + 1
        per_round.append({
            "index": i,
            # -- 证据量（见 `_evidence_block`）--------------------------------
            # `n_behaviors` **就是喂进引擎那个列表的长度**，也就是这一轮
            # 六维状态背后的全部证据。它此前没有被记录，于是「27 个 agent
            # 吵了 50 条」与「3 个 agent 说了 3 句话」在文件里长得一模一样。
            "n_behaviors": len(behaviors),
            "n_actions": len(rd.get("actions", []) or []),
            "n_talk_actions": sum(
                1 for a in (rd.get("actions", []) or [])
                if a.get("action") not in _SILENT_ACTIONS
            ),
            # 条数之外还要记**人数**。这一对是分开的两个量，而它们可以朝
            # 相反方向走：27 agent × 15 轮实测里，发声条数从 21 掉到 9、
            # 而**每一轮都有全部 27 个 agent 在动作** —— 少的是产出内容的
            # 人，不是在场的人。只看条数会把「在场但改成了潜水」读成
            # 「人少了 / 集体安静了」，那是两个不同的模型症状。
            "n_participants": len({a.get("user_id")
                                   for a in (rd.get("actions", []) or [])}),
            "n_speakers": len({a.get("user_id")
                               for a in (rd.get("actions", []) or [])
                               if a.get("action") not in _SILENT_ACTIONS}),
            # 引擎自己的声势权重。**不是本文新设的阈值** —— 直接复算
            # `WorldStateEngine.step()` 里那一行 `tanh(n / VOLUME_REF)`，
            # 所以它回答的是「这一轮的话语被允许推动状态多少」。
            "volume": round(math.tanh(len(behaviors) / VOLUME_REF), 4),
            # 恰好贴到 `_clamp(·, 0, 1)` 两端的维度。**无自由参数**：
            # 一个值恰好等于 0.0000 或 1.0000，等价于「未截断前已经越界」，
            # 即模型想走得更远而被夹住了 —— 那是边界产物，不是读数。
            "railed": [d for d in DIMENSIONS
                       if round(res.state_after[d], 4) in (0.0, 1.0)],
            "phase_id": rd.get("phase_id", ""),
            # 渲染用的阶段标签：注入优先、其次「处于」、都没有才是空（显示成 —）。
            # 与 `phase_id` 并存而不是替换它 —— 前者回答「这轮处在哪」，
            # 后者回答「这轮注入了什么」，两个都要能查到。
            "phase_label": resolve_phase_label(i, rd, phase_by_round),
            "behaviors": behaviors,
            "behavior_counts": counts,
            "state": {d: round(res.state_after[d], 4) for d in DIMENSIONS},
            "deltas": {d: round(v, 4) for d, v in res.deltas().items()},
            "excitation": {d: round(v, 4) for d, v in res.excitation.items()},
            "relaxation": {d: round(v, 4) for d, v in res.relaxation.items()},
            "coupling": {d: round(v, 4) for d, v in res.coupling.items()},
            "events": [{"kind": e.kind, "dimension": e.dimension,
                        "magnitude": round(e.magnitude, 4), "message": e.message}
                       for e in res.events],
            "injected_count": len(rd.get("injected", {}) or {}),
            "calls": rd.get("calls", 0),
            "llm_seconds": rd.get("llm_seconds", 0.0),
            "prompt_tokens": rd.get("prompt_tokens", 0),
            "completion_tokens": rd.get("completion_tokens", 0),
            "errors": rd.get("errors", 0),
            # 这两个字段是后加的：老产出里没有。**缺失要如实说「未记录」，
            # 不能当成 0** —— 一个编出来的 0 比一个空值危险得多。
            "failures": rd.get("failures"),
            "slowest": rd.get("slowest"),
            # 前缀缓存（逐轮）。同样是后加字段：老产出里没有 → None，
            # **不是 0**。0 与 None 在 JSON 里必须是两个不同的值。
            "cache_hit_tokens": rd.get("prompt_cache_hit_tokens"),
            "cache_miss_tokens": rd.get("prompt_cache_miss_tokens"),
        })

    deviation = replay_deviation(results, rounds)
    evidence = _evidence_block(per_round, [r.__dict__ for r in rows],
                               int(meta.get("agents", 0) or 0))
    degradations = _degradations(meta, rows, coverage, per_round, deviation, dpr,
                                 evidence=evidence)

    cost = _cost_block(per_round, meta)
    brief = {
        "provenance": {
            **rounds_doc.get("_provenance", {}),
            **gold.get("_provenance", {}),
            "command": command,
        },
        "meta": meta,
        "days_per_round": dpr,
        "coverage": coverage,
        "windows": [r.__dict__ for r in rows],
        "rounds": per_round,
        "turning_point": turning_point_block(engine, results, gold, rows),
        "clamp": clamp_report(engine, "disclosure", scale),
        "intervention_scale": scale,
        # 归一化上限本身也是个数，写进数据里 —— 否则它在正文里出现时就
        # 成了「没有来源的数字」，而那正是下面的机检要拦的东西。
        "ref_weight": REF_WEIGHT,
        "assumptions": {
            pid: {"action": a, "rationale": why}
            for pid, (a, why) in WINDOW_INTERVENTIONS.items()
        },
        "core_claim": gold.get("core_claim", {}),
        "metric_notes": gold.get("metric_notes", {}),
        "dimension_desc": {d["key"]: d["desc"] for d in gold.get("dimensions", [])},
        "cost": cost,
        "evidence": evidence,
        "replay_max_deviation": deviation,
        "degradations": degradations,
        "disclaimers": dict(DISCLAIMERS),
        "markers": dict(MARKERS),
    }
    missing = check_brief(brief, render_markdown(brief))
    if missing:
        raise BriefError(
            "口径机检未通过 —— 简报**不落盘**：\n  - " + "\n  - ".join(missing)
        )
    return brief


def _phases_from_gold(gold: dict) -> list:
    """从已装载的金标构造 `Phase` 列表（`load_phases` 读的是磁盘，这里已有 dict）。"""
    out = []
    for ph in gold.get("phases", []):
        out.append(Phase(
            phase_id=ph["id"], label=ph.get("label", ""),
            day=int(ph.get("day", 0)), trigger=ph.get("trigger", ""),
            is_turning_point=bool(ph.get("is_turning_point")),
        ))
    return out


def _visible_fraction(limit: int, worst_dropped: int) -> float | None:
    """最坏那一轮里，agent 还能看到自己记忆的多少 —— **上界，不是估计**。

    产出里只存了丢弃量（`dropped`），没存 camel 报的 `before` / `after`。
    但由 `before = after + dropped` 且 `after ≤ limit` 得

        after / before = after / (after + dropped) ≤ limit / (limit + dropped)

    所以这个数**只用产出里已有的两个字段**就能算出来，不重建、不引入参数。
    这是刻意选的：产出里另外两个数（`dropped`、`limit`）谁都会读，
    但「丢了多少 token」读者判断不出量级 —— 123 万对上 1.6 万的上限，
    是丢了一半还是丢了九成九，只有换成比例才看得出来。
    """
    if limit <= 0 or worst_dropped <= 0:
        return None
    return limit / (limit + worst_dropped)


def _visible_sentence(limit: int, worst_dropped: int) -> str:
    """那一句比例。**算不出来就返回空串，不许编一个数出来。**"""
    frac = _visible_fraction(limit, worst_dropped)
    if frac is None:
        return ""
    return (f"最坏的那一轮里，agent 能看到的记忆**不超过自己的 "
            f"{frac * 100:.2f}%**（丢弃量是窗口上限的 "
            f"{worst_dropped / limit:.0f} 倍），而且 camel 是**按打分**丢低分的"
            "—— 留下的不保证是最近的那一段。")


def _context_line(meta: dict) -> str:
    """上下文压力那一句。**三种状态必须分开说，不许合并。**

    - 记录到 0 次 → 断言「没发生」。**这是结论，不是缺省。**
    - 记录到 >0 次 → 说清丢了多少、哪些轮次受影响。
    - **压根没有这个字段** → 断言不了。旧产出就是这样，如实说「没测」，
      **不许默默按 0 处理** —— 那会把「不知道」伪装成「没问题」，
      正是本项目最防的那类静默失效。
    """
    if "truncations" not in meta:
        return ("**这一项本次没有记录。** 本产出生成于上下文监测接上之前，"
                "无法判断当时的 agent 是否已在失忆 —— "
                "**不要把它读成「没有截断」。**")
    limit = meta.get("context_limit", 0)
    n = meta.get("truncations", 0)
    if not n:
        return (f"**全程未发生记忆截断**（上限 {limit} token）。"
                "这是一条**主动断言不在场**，不是缺省：camel 在 agent 记忆超限时会"
                "静默丢弃旧记录，agent 失忆之后照样给出结论，六维曲线不会变难看、"
                "简报不会报错、测试也不会红。所以「没发生」必须被说出来 —— "
                "否则它与「没在数」在文件里长得一模一样。")
    return (f"**⚠️ 全程发生 {n} 次记忆截断**，单次最多丢弃 "
            f"{meta.get('truncation_worst_dropped', 0)} token（上限 {limit}）。"
            + _visible_sentence(limit, meta.get("truncation_worst_dropped", 0))
            + "被截断那些轮次的 agent 是在**丢失了历史记录**的状态下说话的 —— "
            "曲线照画、表格照出，全场只有这一句知道出过事。"
            "**这些轮次的数字要打折。**")


def _evidence_block(per_round: list[dict], windows: list[dict],
                    agents: int) -> dict:
    """每一轮的六维状态各自由多少条行为算出来 —— 连同贴界情况。

    **守的失效模式：状态看着有两个小数位、很精确，背后其实只有两三条行为。**

    这不是假设出来的风险。27 agent × 15 轮实测：轮 0 有 21 条归类行为，
    入库那遍轮 14 只剩 **9 条**（45 个动作里 20 个 refresh、16 个 do_nothing，
    只有 8 个 create_post）；**第一遍更低，只剩 3 条**。
    两遍差 3 倍，这个差本身也是结论 —— 这一档的读数还不稳定。
    而六维状态照样被算到小数点后四位。此前的产出里**没有任何字段能区分
    「50 条吵出来的状态」与「3 条说出来的状态」** —— 两者落盘之后一模一样。

    这里只做两件事，**两件都不引入新阈值**：

    1. 把证据量与引擎自己的声势权重 `tanh(n / VOLUME_REF)` 逐轮列出来。
       分母用的是引擎的常量，不是本文新定的数。
    2. 标出**恰好贴到上下界**的维度。`_clamp(·, 0, 1)` 是硬夹，
       一个值恰好等于 0.0000 或 1.0000 就等价于「夹之前已经越界」——
       那是边界产物而不是读数，且判定无需任何参数。
    """
    rows = [{
        "round": r["index"],
        "phase": r["phase_label"],
        "agents": agents,
        "n_actions": r["n_actions"],
        "n_talk_actions": r["n_talk_actions"],
        "n_participants": r["n_participants"],
        "n_speakers": r["n_speakers"],
        "n_behaviors": r["n_behaviors"],
        "volume": r["volume"],
        "railed": r["railed"],
    } for r in per_round]

    landed = sorted({w["round_index"] for w in windows
                     if w.get("round_index") is not None})
    window_rows = [r for r in rows if r["round"] in landed]
    observed = [r for r in window_rows if r["n_behaviors"] > 0]
    # 「证据最少的那一轮」：只在窗口落到的轮里挑，且**空列表不编造一个 0**
    # —— 没有窗口落到任何轮时，这里该说「无」，不是「最少 0 条」。
    leanest = min(observed, key=lambda r: r["n_behaviors"]) if observed else None
    # 「在场」与「产出」是两个量，可以朝相反方向走 —— 所以要分别汇总。
    # 判据是两个**严格比较**，不是新阈值：发声人数是否下降、在场人数是否恒定。
    presence = None
    p_min = min((r["n_participants"] for r in rows), default=0)
    p_max = max((r["n_participants"] for r in rows), default=0)
    # `p_max == 0` ⟹ 产出里**没有动作记录**（老产出、或夹具没给 actions）。
    # 那就什么都不说 —— 与 `leanest` 同一条纪律：宁可空着，也不报一个
    # 编出来的「在场 0 人」。
    if p_max > 0:
        presence = {
            "agents": agents,
            "first_round": rows[0]["round"],
            "last_round": rows[-1]["round"],
            "speakers_first": rows[0]["n_speakers"],
            "speakers_last": rows[-1]["n_speakers"],
            "participants_first": rows[0]["n_participants"],
            "participants_last": rows[-1]["n_participants"],
            "participants_min": p_min,
            "participants_max": p_max,
            # 在场人数**每一轮都一模一样**。这是严格比较，不是阈值 ——
            # 不设「波动小于 N 就算平稳」那种自由参数。实测 27 agent ×
            # 15 轮里 p_min=26、p_max=27（有一轮少一个人动作），所以
            # 这一条会是 False，正文改报区间而不是硬说「全程不变」。
            "participants_constant": p_min == p_max,
            "speakers_fell": rows[0]["n_speakers"] > rows[-1]["n_speakers"],
            # 「少的是产出内容的人、不是在场的人」这句话**只在两个量确实
            # 不同向时**才成立。判据是严格比较（发声的降幅 > 在场的降幅），
            # **不是阈值** —— 不设「波动小于 N 就算平稳」那种自由参数。
            # 少了这一条时，一段「人退出」的产出也会被套上这句解释。
            "speech_dropped_faster": (
                rows[0]["n_speakers"] - rows[-1]["n_speakers"]
                > rows[0]["n_participants"] - rows[-1]["n_participants"]),
        }
    return {
        "rows": rows,
        "presence": presence,
        "railed_rounds": [r["round"] for r in rows if r["railed"]],
        "railed_dims": sorted({d for r in rows for d in r["railed"]}),
        "window_rounds": landed,
        # 落在**零条行为**轮上的决策窗口。零条行为 ⟹ 引擎那一步的激励项
        # 恒为 0（`tanh(0/8)`），状态是纯弛豫外推 —— **那一轮没有观测**。
        "no_evidence_rounds": [r["round"] for r in window_rows
                               if r["n_behaviors"] == 0],
        "leanest": leanest,
        "min_n_behaviors": leanest["n_behaviors"] if leanest else None,
    }


def _evidence_sentence(ev: dict) -> str:
    """证据量那一段的正文。三种状态分开说，与 `_context_line` 同一纪律。"""
    rows = ev["rows"]
    if not rows:
        return "**本次产出没有可用的逐轮记录**，证据量无从统计。"
    total = sum(r["n_behaviors"] for r in rows)
    head = (f"本次推演共 {len(rows)} 轮，**喂进世界状态引擎的行为合计 {total} 条**；"
            f"表里的 `volume` 是引擎自己的声势权重 `tanh(n / {VOLUME_REF:g})`，"
            "即这一轮的话语被允许推动状态多少。")
    parts = [head]
    # 在场面 vs 产出面。**这一段存在的理由是一次实测**：27 agent × 15 轮里
    # 发声条数从 21 掉到 9，而每一轮都有全部 27 个 agent 在动作 ——
    # 只看条数会把它读成「人少了 / 集体安静了」，那是另一个模型症状。
    # 两种在场情形分开说，数字全部现算（不写死任何一次运行的值）。
    p = ev.get("presence")
    if p and p["agents"]:
        span = (f"R{p['first_round']} 的 {p['speakers_first']} 个到 "
                f"R{p['last_round']} 的 {p['speakers_last']} 个")
        if p["participants_constant"]:
            where = f"每一步都有 **{p['participants_min']} 个 agent 在动作**"
        else:
            where = (f"全程在 **{p['participants_min']}~{p['participants_max']}** "
                     f"之间（本次共 {p['agents']} 个 agent）")
        parts.append(f"**在场人数**：{where}。**发声人数**：从 {span}。")
        if p["speech_dropped_faster"]:
            parts.append(
                "**这两个数要分开看。** 本次它们不是同向的 —— 少的是"
                "**还在产出内容的人**，不是在场的人。把条数直接读成"
                "「人少了 / 集体安静了」，会把「在场但不再产出内容」与"
                "「人退出」两个不同的模型症状混成一个；而对引擎来说两者后果"
                "也不同：前者把输入的分母按下去（构成占比随之集中），"
                "后者才是真的没有话语。")
    if ev["no_evidence_rounds"]:
        rs = "、".join(f"R{i}" for i in ev["no_evidence_rounds"])
        parts.append(
            f"**⚠️ {rs} 是零条行为的轮，而且有决策窗口落在上面。** "
            "零条行为意味着引擎那一步的激励项恒为 0，状态只由弛豫与耦合外推 —— "
            "**那一轮根本没有观测**，落在它上面的窗口数字不是「测出来的」。")
    if ev["railed_rounds"]:
        rs = "、".join(f"R{i}" for i in ev["railed_rounds"])
        ds = "、".join(DIMENSION_ZH.get(d, d) for d in ev["railed_dims"])
        parts.append(
            f"**⚠️ {rs} 有维度恰好贴到上下界**（{ds}）。"
            "引擎末步是硬夹 `_clamp(·, 0, 1)`，恰好等于 0.0000 或 1.0000 "
            "等价于「夹之前已经越界」—— 这些数字是**边界产物**，"
            "不是模型算出来的读数；越靠后的轮越只反映「它撞墙了」。")
    lean = ev["leanest"]
    if lean is not None:
        parts.append(
            f"落在决策窗口上的轮里，证据最少的是 **R{lean['round']}**："
            f"{lean['n_actions']} 条动作、其中 {lean['n_talk_actions']} 条是发声"
            f"（其余是 refresh / do_nothing 一类不产生信号的），"
            f"最终 {lean['n_behaviors']} 条进入引擎、对 {lean['agents']} 个 agent。"
            "**窗口数字的可归因性上限就是这个分母。**")
    return "\n\n".join(parts)


def _degradations(meta, rows, coverage, per_round, deviation, dpr,
                  evidence=None) -> list[dict]:
    out: list[dict] = []
    # 上下文压力：三种状态都要报，其中两种是降级。
    if "truncations" not in meta:
        out.append({"kind": "no_context_record", "text":
                    "本产出**未记录上下文压力**（生成于该监测接上之前）—— "
                    "无法判断当时 agent 是否已在失忆，不要读成「没有截断」"})
    elif meta.get("truncations"):
        _lim = meta.get("context_limit", 0)
        _worst = meta.get("truncation_worst_dropped", 0)
        out.append({"kind": "truncated", "text":
                    f"全程 {meta['truncations']} 次记忆截断，单次最多丢弃 "
                    f"{_worst} token —— "
                    + _visible_sentence(_lim, _worst)
                    + "被截断轮次的 agent 在丢失历史的状态下说话，数字要打折"})
    if meta.get("truncation_unparsed"):
        out.append({"kind": "truncation_unparsed", "text":
                    f"有 {meta['truncation_unparsed']} 次截断警告**解析失败** —— "
                    f"camel 可能改了格式，本项监测已不可信，先修它再看结论"})
    # 证据量：两条**无自由参数**的降级。判据要么是「零」（激励项恒为 0），
    # 要么是引擎自己的夹逼边界，都不是本文新定的阈值。
    if evidence:
        if evidence["no_evidence_rounds"]:
            rs = "、".join(f"R{i}" for i in evidence["no_evidence_rounds"])
            out.append({"kind": "no_evidence_round", "text":
                        f"{rs} 是**零条行为**的轮，仍有决策窗口落在上面 —— "
                        f"该轮激励项恒为 0，状态是纯弛豫外推，不是测出来的"})
        if evidence["railed_rounds"]:
            rs = "、".join(f"R{i}" for i in evidence["railed_rounds"])
            ds = "、".join(DIMENSION_ZH.get(d, d)
                          for d in evidence["railed_dims"])
            out.append({"kind": "railed", "text":
                        f"{rs} 有维度**恰好贴到上下界**（{ds}）—— "
                        f"`_clamp(·, 0, 1)` 的边界产物，不是读数；"
                        f"这些轮次的曲线形状由夹逼决定，不由行为决定"})
    if meta.get("compressed"):
        out.append({"kind": "compressed", "text":
                    f"压缩模式：每轮代表 {dpr:.2f} 天，曲线与「轮 = 天」不可比"})
    if not meta.get("comparable_to_round_day", True):
        out.append({"kind": "not_comparable",
                    "text": "`comparable_to_round_day` 为假，不可与 --rounds 15 的结果并列"})
    if meta.get("agents", 0) < 5:
        out.append({"kind": "small_scale", "text":
                    f"仅 {meta.get('agents')} 个 agent —— 所有数值都是小规模"
                    f"敏感性测试，**不能当作金标规模（27 角色）下的结论**"})
    if coverage["n_uncovered"]:
        out.append({"kind": "uncovered",
                    "text": f"{coverage['n_uncovered']} 个决策窗口在本次运行里没有对应轮次"})
    if coverage["n_shared"]:
        out.append({"kind": "shared_round", "text":
                    f"{coverage['n_shared']} 个窗口与其他窗口共处同一轮 —— "
                    f"干预相同、分叉点相同，因此数值必然重复，**不可单独归因**"})
    if not meta.get("world_state", True):
        out.append({"kind": "no_world_state", "text": "本次运行未推进六维状态"})
    if meta.get("phases_on") is False:
        out.append({"kind": "no_phases", "text":
                    "本次运行未注入阶段事件，因此所有决策窗口都不会有对应轮次"})
    if deviation > 1e-4:
        out.append({"kind": "replay_mismatch", "text":
                    f"重放与产出记录的最大偏差 {deviation:.2e} 超过 1e-4 —— "
                    f"这份产出很可能不是本引擎按当前参数产生的"})
    calls = sum(r["calls"] for r in per_round)
    injected = sum(r["injected_count"] for r in per_round)
    if (meta.get("feedback_on") or meta.get("knowledge_on")) and calls and not injected:
        out.append({"kind": "no_injection", "text":
                    "开关声称回注开着，但逐轮注入次数为 0 —— 问题在接线，不在模型"})
    return out


def _cost_block(per_round: list[dict], meta: dict) -> dict:
    """成本汇总。

    **复用 `CallStats.summary()`，但不能假装它的默认值是真的。**
    `summary()` 会打印「失败 N」，而产出文件里**没有记录 LLM 调用失败次数**
    （`errors` 是 `env.step` 的异常数，不是 LLM 失败）。从 rounds 重建
    `CallStats` 会让它默认 0，于是打印出一条产出不出来的「失败 0」——
    这正是本项目要杀的那类静默失效。所以这里如实标注未记录。
    """
    from .simulate import CallStats

    stats = CallStats()
    for r in per_round:
        stats.calls += r["calls"]
        stats.seconds += r["llm_seconds"]
        stats.prompt_tokens += r["prompt_tokens"]
        stats.completion_tokens += r["completion_tokens"]
    tracked = [r for r in per_round if r["failures"] is not None]
    if tracked:
        stats.failures = sum(r["failures"] for r in tracked)
        stats.slowest = max((r["slowest"] or 0.0) for r in tracked)
        display = stats.summary()
    else:
        # **不能直接打印 `summary()`**：那两个字段没落盘时它会照默认值印出
        # 「最慢 0.0s，失败 0」—— 一个编出来的 0 比一个空值危险得多，
        # 而且它就贴在这条说明的上面一行，自相矛盾。所以这一段自己拼。
        display = (f"{stats.calls} 次调用 / {stats.seconds:.1f}s"
                   f"（均值 {stats.mean:.1f}s，最慢 **未记录**，失败 **未记录**）"
                   f" · token 入 {stats.prompt_tokens} / 出 {stats.completion_tokens}")
    return {
        "summary": display,
        "failures_recorded": bool(tracked),
        "note": ("调用失败次数与最慢单次来自产出文件逐轮记录。"
                 if tracked else
                 "调用失败次数与最慢单次**未记录在本产出文件里**"
                 "（这两个字段是本轮才加进落盘的，旧产出没有）。"
                 "生成这份简报时它们按「未记录」显示，不是 0。"),
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "paths": _path_block(per_round, meta),
    }


def _pct(hit: int, miss: int) -> float | None:
    """命中率，**以百分数存**（不是 0~1 的小数）。

    存成 87.5 而不是 0.875，是为了让正文能直接印 `87.5%` 而仍然满足
    `check_brief` 第 5 条「正文里的每个数值都能在结构化数据里找到来源」——
    那条检查按数值比，`0.875` 与印出来的 `87.5` 对不上。
    """
    seen = hit + miss
    if seen == 0:
        return None
    return round(100.0 * hit / seen, 1)


def _path_block(per_round: list[dict], meta: dict) -> dict:
    """两条调用路径的成本对照。

    本项目有两条路径：camel 驱动的 agent，和 `llm.LLMClient` 的直调
    （归类器每轮一次、画像构建每角色一次）。直调走 `requests`，不经过
    camel 的模型对象 —— 所以逐轮的 `calls` / `tokens` **一个数都不含它**。
    不把两条摆在一起，「钱花在哪了」这个问题就只能看一眼服务商账单。

    **旧产出没有 `meta.classifier_ledger`**（生成它的那次运行还没记这笔账），
    这时如实报「未记录」，**不许补 0** —— 0 是「跑了但没花钱」的形态，
    与「没记」是两件事。
    """
    camel_hits = [r for r in per_round if r.get("cache_hit_tokens") is not None]
    c_hit = sum(r["cache_hit_tokens"] for r in camel_hits)
    c_miss = sum(r["cache_miss_tokens"] for r in camel_hits)
    camel = {
        "calls": sum(r["calls"] for r in per_round),
        "prompt_tokens": sum(r["prompt_tokens"] for r in per_round),
        "completion_tokens": sum(r["completion_tokens"] for r in per_round),
        # **只要有任何一轮落了数就算「记了」**；缺的轮次不当 0 相加 ——
        # 那会让一个记了一半的产出看起来比一个完全没记的产出更准。
        "cache_recorded": bool(camel_hits),
        "cache_rounds": len(camel_hits),
        "cache_hit_tokens": c_hit,
        "cache_miss_tokens": c_miss,
        "cache_hit_pct": _pct(c_hit, c_miss),
    }

    led = meta.get("classifier_ledger")
    if not isinstance(led, dict):
        direct = {"recorded": False}
    else:
        d_hit = int(led.get("prompt_cache_hit_tokens") or 0)
        d_miss = int(led.get("prompt_cache_miss_tokens") or 0)
        d_rec = bool(led.get("cache_recorded"))
        direct = {
            "recorded": True,
            "calls": led.get("calls"),
            "failures": led.get("failures"),
            "prompt_tokens": led.get("prompt_tokens"),
            "completion_tokens": led.get("completion_tokens"),
            "cache_recorded": d_rec,
            "cache_hit_tokens": d_hit,
            "cache_miss_tokens": d_miss,
            "cache_hit_pct": _pct(d_hit, d_miss) if d_rec else None,
        }

    return {
        "camel": camel,
        "direct": direct,
        "note": ("直调路径的用量来自产出文件里的 `meta.classifier_ledger`。"
                 if direct["recorded"] else
                 "直调路径（归类器）的用量**未记录在本产出文件里** —— "
                 "生成它的那次运行还没把这笔账写进产物。这不是 0："
                 "这笔花费在服务商账单上看得见，在这份产物里看不见。"),
    }


def _path_sentence(p: dict) -> str:
    """把两条调用路径摆在同一句里。**未记录的那条不补 0，并说明为什么。**

    这一句是「为什么新路径看起来更花钱」的现场答案：旧产出里直调那条
    根本没被记账，于是读者只能看到 camel 那半张账。
    """
    c = p["camel"]
    d = p["direct"]
    line = (f"**两条调用路径**：camel 驱动的 agent 路径 {c['calls']} 次调用"
            f"（入 {c['prompt_tokens']} / 出 {c['completion_tokens']} tok）")
    if d["recorded"]:
        line += (f"；直调的归类器路径 {d['calls']} 次调用"
                 f"（入 {d['prompt_tokens']} / 出 {d['completion_tokens']} tok）")
    else:
        line += ("；直调的归类器路径**未记录在本产出里** —— 这不是 0，"
                 "生成它的那次运行还没把这笔账写进产物，"
                 "而这笔花费在服务商账单上看得见")
    line += ("。分开报的理由：直调走 `requests`、不经过 camel 的模型对象，"
             "逐轮表里的 calls 与 tokens **一个数都不含它**。")

    # 前缀缓存只在端点真的报过时才印数。**没报就不印 0%** —— 那等于替
    # 端点回答了一个它没回答的问题，而排查方向会被整个带反。
    if c["cache_recorded"] and c["cache_hit_pct"] is not None:
        line += (f" camel 路径的前缀缓存：命中 {c['cache_hit_tokens']} / "
                 f"未命中 {c['cache_miss_tokens']} tok"
                 f"（{c['cache_hit_pct']}%）—— 命中率决定的是单价，不是数量。")
    if d["recorded"] and d.get("cache_recorded") and d.get("cache_hit_pct") is not None:
        line += (f" 直调路径的前缀缓存：命中 {d['cache_hit_tokens']} / "
                 f"未命中 {d['cache_miss_tokens']} tok"
                 f"（{d['cache_hit_pct']}%）。")
    return line


def render_markdown(brief: dict) -> str:
    """**唯一的文本生成点。** 想改文案只改这里，口径机检跟着一起走。"""
    m = brief["meta"]
    L: list[str] = []
    w = L.append

    w("# 「未然」决策简报")
    w("")
    w(f"> 依据一次真实推演：**{m.get('agents')} 个 agent × {m.get('rounds')} 轮**"
      f"（平台 `{m.get('platform')}`），"
      f"共 {m.get('total_actions')} 条动作，墙钟 {m.get('total_seconds')}s。")
    w(f"> 每轮代表 **{brief['days_per_round']:.2f} 天**。"
      f"{MARKERS['model_internal']} {MARKERS['assumption']}")
    w("")

    # -- 【零】口径 -------------------------------------------------------
    w("## 【零】先读这一段，否则下面的数字会被误读")
    w("")
    w(f"生成侧：**本简报的生成是确定性的** —— 同输入同输出，可进 CI，"
      f"可被 `repro_check.py` 逐字校验。{MARKERS['assumption']}")
    w("")
    w(f"数据侧：**它读取的那条曲线不是。** {MARKERS['nonreproducible']}")
    w("")
    w(DISCLAIMERS["nonreproducible"])
    w("")
    if m.get("compressed"):
        w(f"{MARKERS['compressed']}")
        w("")
        w(DISCLAIMERS["compressed"])
        w("")
    w(f"{MARKERS['context']}")
    w("")
    w(_context_line(m))
    w("")
    w(f"{MARKERS['degraded']}")
    w("")
    for d in brief["degradations"]:
        w(f"- {d['text']}")
    w("")
    prov = brief["provenance"]
    w(f"来源指纹：`{Path(prov.get('rounds_file', '')).name}` "
      f"sha256:{prov.get('rounds_sha256', '')} ± "
      f"`{Path(prov.get('scenario_file', '')).name}` "
      f"sha256:{prov.get('scenario_sha256', '')}")
    w("")

    # -- 【一】窗口对照 ---------------------------------------------------
    w("## 【一】决策窗口对照")
    w("")
    w(f"{MARKERS['model_internal']} {MARKERS['assumption']} "
      f"差异列的数值来自**同一分叉点**上的配对推演，"
      f"不是与产出终值相减。干预倍数 `K = {brief['intervention_scale']}`。")
    w("")
    w("| 窗口 | 阶段 | 当时的做法 | 拐点 | 对应轮次 | 时点偏差 | "
      "配对实际 | A 激励版 | Δ(A−实际) | B 机制版 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in brief["windows"]:
        # **`is None` 而不是真值判断**：第 0 轮的 `round_index` 是 0，
        # 真值判断会把 P1/P2 两行误判成「未覆盖」——机检逮到过这一条。
        if r["round_index"] is None:
            w(f"| {r['question']} | {r['label']} | {r['actual_choice']} | "
              f"{'是' if r['is_turning_point'] else '否'} | — | — | — | — | — | — |")
            w(f"| ^ | | | | **本次运行未覆盖** | | | | | |")
            continue
        cov = f"R{r['round_index']}"
        if r["coverage"] == "shared":
            cov += f"（与 {'/'.join(r['shared_with'])} 同轮）"
        skew = f"{r['day_skew']:+.0f} 天"
        bb = f"{r['branch_b']:.4f}" if r["branch_b"] is not None else "—"
        db = f"{r['delta_b']:+.4f}" if r["delta_b"] is not None else "—"
        w(f"| {r['question']} | {r['label']} | {r['actual_choice']} | "
          f"{'是' if r['is_turning_point'] else '否'} | {cov} | {skew} | "
          f"{r['actual_from_fork']:.4f} | {r['branch_a']:.4f} | "
          f"Δ {r['delta_a']:+.4f} | {bb}{('  Δ ' + db) if r['delta_b'] is not None else ''} |")
    w("")
    w("「配对实际」= **从同一个分叉点**再跑一条不施加任何干预的路径。"
      "不能拿产出文件的终值当「实际」——那是从基线分叉出来的另一条路，"
      "与被比较的分支不同源。")
    w("")
    # **这句以前是写死的**：「本轮运行是压缩的，所以这个偏差最大到 2 天 ——
    # 全程只有 14 天」。跑出第一份非压缩产出（`--rounds 15`）之后，同一句话
    # 就在一份 `compressed=false` 的简报里声称自己压缩过。降级项那一节是按
    # `meta` 动态渲染的，这里却写死 —— 同一份简报里两处互相打脸。
    if brief["meta"].get("compressed"):
        _dpr = float(brief["meta"].get("days_per_round", 1.0))
        _n = brief["meta"].get("rounds", len(brief.get("per_round", [])))
        w("「时点偏差」= 本运行里分叉的那一天 − 金标里该阶段的第几天。"
          f"本轮运行是压缩的（每轮代表 {_dpr:.2f} 天、共 {_n} 轮），"
          "所以这个偏差可以不为零。")
    else:
        w("「时点偏差」= 本运行里分叉的那一天 − 金标里该阶段的第几天。"
          "本轮是「轮 = 天」，两者**应当恒等** —— 任何一行不为零都说明阶段"
          "映射有误，是缺陷而不是误差。")
    w("")
    if brief["coverage"]["n_shared"]:
        w("> **标着「同轮」的行，数字相同是必然的，不是巧合也不可单独归因。** "
          "它们在压缩后的同一条轮里，分叉点相同、干预相同，因此算出来就是同一个数。"
          "要在模型里把 P1 与 P2（或 P3 与 P4）分开，只能跑 `--rounds 15` 那一档。")
        w("")

    # -- 【一·附】每轮证据量 -----------------------------------------------
    # 放在窗口表**紧接着的下面**，而不是另起一节挪到文末：它回答的是
    # 「上面那张表的数字各自建立在几条行为上」，离开那张表就没有意义。
    ev = brief["evidence"]
    w(f"### 每轮证据量 —— 上面那些数字各自的分母 {MARKERS['evidence']}")
    w("")
    w(_evidence_sentence(ev))
    w("")
    w("| 轮 | 阶段 | 动作 | 在场 | 其中发声 | 发声的人 | 进入引擎 | `volume` | 贴界的维度 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in ev["rows"]:
        rail = "、".join(DIMENSION_ZH.get(d, d) for d in r["railed"]) or "—"
        w(f"| R{r['round']} | {r['phase'] or '—'} | {r['n_actions']} | "
          f"{r['n_participants']} | {r['n_talk_actions']} | {r['n_speakers']} | "
          f"{r['n_behaviors']} | {r['volume']:.3f} | {rail} |")
    w("")
    w("「其中发声」= 动作里不属于 `refresh` / `do_nothing` / `follow` / `mute` "
      "一类的条数 —— 与仿真用的是同一份名单（`simulate._SILENT_ACTIONS`）。"
      "**「进入引擎」才是分母**：状态是拿它算的，不是拿动作条数算的。")
    w("")
    w("「在场」与「发声的人」都是**去重后的人数**（`user_id` 计数），不是条数。"
      "两个数要分开看：**条数掉下来可以是「人少了」，也可以是「人在场但不再"
      "产出内容」** —— 两者对引擎的后果不同：后者把输入的分母按下去"
      "（构成占比随之集中），前者才是真的没有话语。哪个是本次的情形，"
      "看上面那一段，**不要从这张表自己猜**。")
    w("")

    # -- 【二】拐点 -------------------------------------------------------
    tp = brief["turning_point"]
    w("## 【二】拐点")
    w("")
    w(f"**场景叙事侧（作者编写，非模型产出）**：金标把拐点标在 "
      f"`{tp['gold']['phase']}`（第 {tp['gold']['day']} 天，事件：{tp['gold']['event']}）。")
    w("")
    w(tp["gold"]["why_it_matters"])
    w("")
    w(f"它给的可检信号：{tp['gold']['detectable_signal']}")
    w("")
    w("**模型产出侧**：")
    w("")
    if tp["detected_events"]:
        w("| 轮 | 阶段 | 检出事件 | 维度 | 幅度 |")
        w("|---|---|---|---|---|")
        for e in tp["detected_events"]:
            w(f"| R{e['round']} | {e['phase']} | `{e['kind']}` | "
              f"{DIMENSION_ZH.get(e['dimension'], e['dimension'])} | {e['magnitude']:.4f} |")
    else:
        w("本次运行没有检出任何事件。")
    w("")
    w("**逐轮的 Δ 极化 vs Δ 关注**（金标那条可机检信号）：")
    w("")
    w("| 轮 | 阶段 | Δ关注 | Δ极化 | 极化增速超过关注 |")
    w("|---|---|---|---|---|")
    for s in tp["signal_series"]:
        w(f"| R{s['round']} | {s['phase']} | {s['d_attention']:+.4f} | "
          f"{s['d_polarization']:+.4f} | "
          f"{'是' if s['polarization_exceeds_attention'] else '否'} |")
    w("")
    first = tp["first_round_polarization_exceeds_attention"]
    if first is None:
        w("本次运行里这个信号**从未成立**。")
    else:
        agrees = tp["signal_agrees_with_gold"]
        w(f"该信号**首次成立在 R{first}**。"
          + ("与金标所指的阶段一致。" if agrees else
             "**与金标所指的阶段不一致** —— 金标说拐点在 P3，"
             "而本运行里这个信号更早就成立了。两个结论都报出来，"
             "不合并成一个：这是模型侧与叙事侧的**不一致**，"
             "不是「拐点被复现了」。"))
    w("")
    w("**未实现的观测项**：")
    w("")
    for u in tp["unimplemented_signals"]:
        w(f"- `{u['name']}` —— {u['reason']}")
    w("")

    # -- 【三】逐轮 -------------------------------------------------------
    w("## 【三】逐轮状态与成本")
    w("")
    w(f"{MARKERS['model_internal']}")
    w("")
    w("每一格都是**重放**得到的（不是从产出文件相邻两轮相减），"
      "所以拿得到激励／弛豫／耦合三个分量 —— 每个数字都能追溯到一条理由。")
    w("")
    for r in brief["rounds"]:
        beh = "、".join(f"{k}×{v}" for k, v in r["behavior_counts"].items()) or "（无）"
        ev = "、".join(f"`{e['kind']}`" for e in r["events"]) or "无"
        w(f"### R{r['index']}　阶段 {r.get('phase_label') or '—'}")
        w("")
        w(f"- 行为构成：{beh}")
        w(f"- 六维：" + "　".join(
            f"{DIMENSION_ZH[d]} {r['state'][d]:.4f}" for d in DIMENSIONS))
        w(f"- 本轮 Δ：" + "　".join(
            f"{DIMENSION_ZH[d]} {r['deltas'][d]:+.4f}" for d in DIMENSIONS))
        w(f"- 三分量（以 trust 为例）：激励 {r['excitation']['trust']:+.4f} · "
          f"弛豫 {r['relaxation']['trust']:+.4f} · 耦合 {r['coupling']['trust']:+.4f}")
        w(f"- 检出事件：{ev}")
        w(f"- 闭环注入：{r['injected_count']} 个 agent 收到本轮的态势／知情块")
        w(f"- 成本：{r['calls']} 次调用 / {r['llm_seconds']:.1f}s · "
          f"token 入 {r['prompt_tokens']} / 出 {r['completion_tokens']}")
        w("")
    c = brief["cost"]
    w(f"**合计**：{c['summary']}")
    w("")
    w(c["note"])
    w("")
    w(_path_sentence(c["paths"]))
    w("")

    # -- 【四】能／不能 ---------------------------------------------------
    w("## 【四】这份简报能回答什么、不能回答什么")
    w("")
    w("**能**：说明反事实通路在**真实产出**上跑得通；两条路径可以在**同一个"
      "分叉点**上比较；并给出「在该模型内部」的高低。")
    w("")
    w("**不能**（三条，都不是谦虚，是限制）：")
    w("")
    # **这句以前是写死的**：「每轮只有 3~4 条行为，声势项只到半饱和」。
    # 那是 3 agent 那一档的实情，被写成了所有产出的通用声明。27 agent 实测
    # 第一轮就有 21 条行为（volume 0.998），同一句话就成了假话 —— 而它还是一句
    # **自我贬低**的假话，不会有任何人去核。现在按本产出的实测值生成。
    _ev = brief["evidence"]
    _lean = _ev["leanest"]
    if _lean is None:
        w(f"1. **规模**：{m.get('agents')} 个 agent。本次没有决策窗口落到任何轮上，"
          f"逐轮证据量见【一】的附表。")
    else:
        w(f"1. **规模**：{m.get('agents')} 个 agent。落到决策窗口上的轮里，"
          f"证据最少的是 R{_lean['round']} —— {_lean['n_behaviors']} 条行为、"
          f"对 {_lean['agents']} 个 agent。**窗口数字的可归因性上限就是这个分母**，"
          f"逐轮数字见【一】的附表。")
    w(f"2. **压缩**：每轮 {brief['days_per_round']:.2f} 天。激励项按步累加，"
      f"所以本次曲线与 `--rounds 15` 的曲线是**两条不同的曲线**，"
      f"不是同一条的粗细版本。")
    w("3. **反事实假设后续行为不变**，而现实里一旦公开，agent 的反应本身就会变。")
    w("")
    w(f"{MARKERS['model_internal']}")
    w("")

    # -- 附：干预口径 -----------------------------------------------------
    w("## 附：本简报用到的干预口径（**自设假设，不是金标事实**）")
    w("")
    w(f"{MARKERS['assumption']}")
    w("")
    w("| 阶段 | 替代做法在模型里怎么表达 | 理由 |")
    w("|---|---|---|")
    for pid, a in brief["assumptions"].items():
        w(f"| {pid} | `{a['action']}` × K | {a['rationale']} |")
    w("")
    w(f"K = {brief['intervention_scale']}。**这个倍数是从 `validate.py` 的 AS-10 "
      f"沿用下来的约定值，不是标定出来的系数。** 它让 `disclosure` 在部分维度上"
      f"顶到归一化上限（`REF_WEIGHT = {REF_WEIGHT}`），所以下面这张表要看：")
    w("")
    w("| 维度 | 原始激励 | 归一化 | 是否被上限截断 |")
    w("|---|---|---|---|")
    for c in brief["clamp"]:
        w(f"| {c['zh']} | {c['raw']:+.4f} | {c['normalized']:+.3f} | "
          f"{'**是**' if c['clamped'] else '否'} |")
    w("")
    clamped = [c["zh"] for c in brief["clamp"] if c["clamped"]]
    if clamped:
        w(f"**{'、'.join(clamped)} 的激励项已经饱和**，再加大 K 不会改变这两个维度"
          f"的**激励**（`trust` 仍会经由耦合变化）。所以表里的差异值是"
          f"「在一个已经推到顶的推杆下」得到的，不要读成「公开的边际效应」。")
    w("")
    w(f"核心论断（金标 `core_claim`）：{brief['core_claim'].get('statement', '')}")
    w("")
    return "\n".join(L) + "\n"


def _numeric_leaves(obj, out: list | None = None) -> list:
    """收集结构化简报里的所有数值（含嵌在字符串里的，如金标那段叙事）。"""
    out = [] if out is None else out
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(float(t) for t in _NUMBER.findall(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numeric_leaves(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numeric_leaves(v, out)
    return out


def check_brief(brief: dict, md: str) -> list[str]:
    """口径机检。返回违规清单，空列表 = 通过。

    **这些检查必须正反两面都测**（见 `tests/test_brief.py`）：一条恒真的检查
    比没有检查更危险 —— 它会让后来的人以为这里被守住了。
    """
    bad: list[str] = []
    m = brief["meta"]

    # 1. 条件化的声明：跟着 meta 走，不硬编码某一份产出。
    if any(r["calls"] for r in brief["rounds"]):
        if MARKERS["nonreproducible"] not in md:
            bad.append(f"有 LLM 调用却没出现 {MARKERS['nonreproducible']}")
    if m.get("compressed") and MARKERS["compressed"] not in md:
        bad.append(f"meta.compressed 为真却没出现 {MARKERS['compressed']}")
    if brief["days_per_round"] != 1.0:
        if f"{brief['days_per_round']:.2f} 天" not in md:
            bad.append("每轮天数没有写进正文（标了压缩却不说几天等于没说）")
    for key in ("model_internal", "assumption", "context", "evidence"):
        if MARKERS[key] not in md:
            bad.append(f"缺少必备标记 {MARKERS[key]}")
    # 上下文压力那一段必须是**三种状态之一**，且说出来的必须与实际相符。
    # 光检查标记在场不够 —— 一个永远印「未发生截断」的实现也能过那条检查，
    # 而那正是本项目最怕的「永远报平安」。
    ctx = _context_line(m)
    if ctx not in md:
        bad.append("上下文压力一段与实际记录不符（或根本没渲染）")
    if m.get("truncations") and "未发生记忆截断" in md:
        bad.append(f"meta 记录到 {m['truncations']} 次截断，正文却说「未发生」")
    # 证据量同上：**正反两面都要拦**。一个永远印「证据充足」的实现，
    # 与一个永远印「没截断」的实现，是同一个失效。
    ev = brief.get("evidence")
    if ev is None:
        bad.append("结构化简报里没有 evidence 段 —— 逐轮分母没被记录")
    else:
        if _evidence_sentence(ev) not in md:
            bad.append("证据量一段与实际记录不符（或根本没渲染）")
        if ev["no_evidence_rounds"]:
            for r in ev["no_evidence_rounds"]:
                if f"R{r}" not in md:
                    bad.append(f"R{r} 是零条行为的轮，正文里没有点名它")
        if ev["railed_rounds"] and "贴到上下界" not in md:
            bad.append(f"有 {len(ev['railed_rounds'])} 轮贴到夹逼边界，正文里没说")
        # 表格里的「进入引擎」列必须与结构化数据一致 —— 否则那张表可以
        # 印着好看的数字而 JSON 里是另一回事。
        for row in ev["rows"]:
            if f"| R{row['round']} | " not in md:
                bad.append(f"证据量表缺了 R{row['round']} 一行")
    if brief["degradations"] and MARKERS["degraded"] not in md:
        bad.append(f"有降级项却没出现 {MARKERS['degraded']}")

    # 2. 未覆盖**计数**，不是「提到过」就算。
    n_uncovered = brief["coverage"]["n_uncovered"]
    got = md.count("本次运行未覆盖")
    if got != n_uncovered:
        bad.append(f"未覆盖窗口 {n_uncovered} 个，但正文里只出现 {got} 次 —— "
                   f"未覆盖必须逐行标出，不能只说一句「有未覆盖」")

    # 3. 禁用词：不得把「注入生效」说成「效果变好」。
    for word in _BANNED:
        if word in md:
            bad.append(f"正文出现禁用词「{word}」")

    # 4. 不得引用 `reference_shape` 与 MAE（金标自己声明它俩不是基准）。
    for word in ("reference_shape", "MAE", "mae"):
        if word in md:
            bad.append(f"正文引用了 {word} —— 金标明确它不是观测值、不是评分基准")
    if "reference_shape" in json.dumps(brief, ensure_ascii=False):
        bad.append("结构化简报里混进了 reference_shape")

    # 5. 数字必须可溯源：正文里的每个数值都应在结构化数据里找得到。
    #    防的是「模板里手写死了一个数」——那种数没人会发现它已经不成立。
    #
    #    **必须按数值比，不能按字符串比**：正文是 `{:.4f}` 渲染的（`0.8110`），
    #    而 JSON 里的浮点 repr 是 `0.811`。第一版就是按字符串比的，结果
    #    14 个数字全报「找不到来源」——一个恒假的检查，和恒真的一样没用。
    nums = {abs(v) for v in _numeric_leaves(brief)}
    for tok in sorted(set(_NUMBER.findall(md))):
        val = float(tok)
        decimals = len(tok.partition(".")[2])
        if not any(round(v, decimals) == val for v in nums):
            bad.append(f"正文里的数值 {tok} 在结构化数据里找不到来源"
                       f"（模板里可能手写死了一个数）")
    return bad


def main(argv: list[str] | None = None) -> int:
    ensure_console_encoding()
    ap = argparse.ArgumentParser(description="决策简报：可比较的处置方案对照")
    ap.add_argument("--rounds-file", default=None,
                    help=f"推演产出（默认 {DEFAULT_ROUNDS_FILE.name}）")
    ap.add_argument("--scenario", default=None, help="场景目录（默认同 simulate）")
    ap.add_argument("--out", default=None, help="markdown 输出路径")
    ap.add_argument("--json", default=None, help="结构化输出路径")
    ap.add_argument("--intervention-scale", type=float, default=DEFAULT_SCALE,
                    help=f"干预倍数，默认 {DEFAULT_SCALE}（沿用 validate.py 的约定值）")
    ap.add_argument("--stdout", action="store_true", help="把全文也打到终端")
    args = ap.parse_args(argv)

    rounds_path = Path(args.rounds_file) if args.rounds_file else DEFAULT_ROUNDS_FILE
    scenario_dir = Path(args.scenario) if args.scenario else Path(DEFAULT_SCENARIO)
    if not rounds_path.is_absolute():
        rounds_path = REPO_ROOT / rounds_path
    if not scenario_dir.is_absolute():
        scenario_dir = REPO_ROOT / scenario_dir

    try:
        rounds_doc = load_rounds(rounds_path)
        gold = load_gold(scenario_dir)
        command = "python -m weiran.brief" + (
            f" --intervention-scale {args.intervention_scale}"
            if args.intervention_scale != DEFAULT_SCALE else "")
        brief = build_brief(rounds_doc, gold, scale=args.intervention_scale,
                            command=command)
    except BriefError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2

    md = render_markdown(brief)
    out = Path(args.out) if args.out else DEFAULT_REPORT_DIR / "brief.md"
    js = Path(args.json) if args.json else out.with_suffix(".json")
    for p in (out, js):
        if not p.is_absolute():
            p = REPO_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    js.write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.stdout:
        print(md)
    cov = brief["coverage"]
    print(f"已写入 {out}")
    print(f"  · 决策窗口 {len(brief['windows'])} 个"
          f"（未覆盖 {cov['n_uncovered']}，同轮共享 {cov['n_shared']}）"
          f" · 降级项 {len(brief['degradations'])}"
          f" · 重放偏差 {brief['replay_max_deviation']:.1e}"
          f" · 口径检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
