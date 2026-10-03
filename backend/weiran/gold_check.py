"""金标对照表：把金标自己写的 10 条断言，逐条拿到**一次真实产出**上核一遍。

**这不是分数。** 理由不在这份代码里，在金标自己的话里
（`reference_data.json → gold_status.what_is_NOT_solid`）：

> `reference_shape`（各阶段六维状态值）是**作者手写的强度预期**，不是观测值，
> 也不是评分基准。任何以它为分母算出的「准确率」都是自说自话。

所以这张表**不使用 `reference_shape`**（`load_gold` 已经把它从物理上剥掉了），
判据只吃两样东西：**产出里记下的六维曲线**，与**金标 `assertions` 自己写的
阈值与顺序**。它回答的是「金标声称会发生的事，这一次运行里发生了没有」，
不是「模型有多准」—— 后者需要一个基准，而这个项目没有、也不打算有一个。

它为什么不能进 `brief.md`：`check_brief` 第 4 条禁止正文出现 `reference_shape`
与 `MAE`，第 5 条要求正文里每个数值都能在结构化数据里找到来源。这张表是为
「逐条核对」设计的，形态与简报不同，所以是**独立产物**（`gold_check.json`
/ `gold_check.md`），由 `repro_check.py` 逐字锁定。

三种结果，**不可判定不是失败、也不是通过**：

  通过      判据成立，且没有被贴界/恒真污染
  否决      判据不成立
  不可判定   产出里没有可判的数 —— 每一条都写明「还缺什么才能判」

「贴界」为什么一票降级：`_clamp(·, 0, 1)` 把值夹在两端，贴到边界的读数是
**边界产物、不是读数** —— 曲线的形状由夹逼决定，不由行为决定。贴界对判据的
破坏是**双向的**：它既能让一条断言「通过得毫无内容」（AS-5），也能让一条断言
「否决得毫无内容」（AS-10）。

    python -m weiran.gold_check
    python -m weiran.gold_check --stdout
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .brief import (
    DEFAULT_ROUNDS_FILE,
    DEFAULT_SCALE,
    BriefError,
    _sha256,
    build_brief,
    load_gold,
    load_rounds,
)
from .config import REPO_ROOT, ensure_console_encoding
from .profiles import DEFAULT_SCENARIO

DEFAULT_JSON = REPO_ROOT / "data" / "simulation" / "gold_check.json"
DEFAULT_MD = REPO_ROOT / "data" / "simulation" / "gold_check.md"

#: 本表默认的生成命令。**它会印在产物正文里**（第五节「来源」），所以
#: `repro_check.py` 要整份逐字比对落盘样例时，必须拿**同一个字符串**去重算 ——
#: 两边命令不一致的话，比对失败的原因就只是那一行 provenance，而它会顶着
#: 「结论与代码不一致」的样子报出来。写成常量，让比对那边 import 而不是重抄。
DEFAULT_COMMAND = "python -m weiran.gold_check"

#: 三种结果。用常量而不是散落的字符串 —— 判别逻辑与渲染必须用同一套词，
#: 否则加一种结果时总有一处会被漏掉。
PASS_ = "通过"
FAIL_ = "否决"
UNDECIDED = "不可判定"

#: 贴界的一票降级措辞。同一条判据可能同时被两种原因污染，所以 `采信` 是
#: **一串理由**，不是一个布尔：写成一个布尔的话，「为什么不信」这件事就丢了。
NO_TRUST_RAIL = "贴界"
NO_TRUST_VACUOUS = "判据恒真"

#: 人对数据的解读。**每一条都必须带着自己的前提一起落盘，而且前提要机检。**
#:
#: 这一段原先是一个写死的字符串，而 `counts` 是算出来的 —— 两者可以对不上。
#: 具体会怎么坏：一旦相位缺陷被修好、AS-3 翻成通过，那句「AS-2/AS-3/AS-4/AS-8
#: 四条否决指向同一条相位缺陷」会**继续宣称 AS-3 否决**。那就是摘要层的恒真
#: 判据，与本表要防的病是同一个 —— 只不过它发生在人写的那句话里，不在求值
#: 函数里，所以贴界与恒真那两道闸都拦不住它。
#:
#: 修法不是把这句话删掉（它是有价值的判断），是**给它装上前提**：前提成立才
#: 输出，不成立就明说「本次运行不满足这条解读的前提，所以没有输出」。前提本身
#: 由 `_interpretations` 拿本次的实际 verdict 去核 —— 于是这句话不可能撒谎。
INTERPRETATIONS = (
    {
        "id": "shared_root_cause",
        "claim": (
            "**AS-2 / AS-3 / AS-4 / AS-8 四条否决指向同一条相位缺陷** —— "
            "模型的状态峰值落在 P3 之后一到两轮（见 进度.md 的相位定案）。"
            "不要读成四个独立问题：它们是同一个成因的四个落点。"
        ),
        "requires_ids_failed": ("AS-2", "AS-3", "AS-4", "AS-8"),
        "requires": "这四条在本次运行里都是「否决」",
    },
    {
        "id": "as10_is_another_matter",
        "claim": "AS-10 是另一件事（贴界 + 离线分支不是真反事实）。",
        "requires_ids_failed": ("AS-10",),
        "requires": "AS-10 在本次运行里是「否决」",
    },
)


def _num(pattern: str, check: str, what: str) -> float:
    """从金标 `check` 原文里**抠出阈值**，而不是在代码里再抄一遍。

    抄一遍等于给同一件事写两个数：金标改了阈值，代码不知道，于是这张表会
    拿着旧阈值去判新断言 —— 而它自己不会知道。抠不出来就抛，不猜。
    """
    m = re.search(pattern, check)
    if not m:
        raise BriefError(
            f"金标断言里找不到{what}：`{check}`\n"
            f"  本表不替金标补一个默认阈值 —— 补出来的数没人会发现它是编的。"
        )
    return float(m.group(1))


# ---------------------------------------------------------------------------
# 对齐口径：这一段曾经是本项目真实踩过的坑，所以写进产物
# ---------------------------------------------------------------------------

def alignment(rounds_doc: dict, gold: dict) -> dict:
    """阶段 → 轮号的对齐，**连同依据一起落盘**。

    坑在这里：金标 `phases[].day` 是 0 基的（0/2/5/8/14），而轮号在
    `--rounds 15` 时也是 0 基的；2026-09-14 那批表用的是 0 基、后来几批用
    1 基，两种编号混在同一个仓库里，同一个结论会被读成两个。所以：
    **本表一律用它自己的 0 基轮号 `index`**，并且把 `phase_id` 非空的那几个
    轮号与金标的天数并列印出来 —— 对不上就是对不上，看得见。
    """
    rounds = rounds_doc["rounds"]
    dpr = float(rounds_doc["meta"].get("days_per_round", 1.0))
    by_phase: dict[str, int] = {}
    for r in rounds:
        pid = (r.get("phase_id") or "").strip()
        if pid:
            # 同一阶段可能被注入多轮（压缩模式下同一轮落进多个阶段时会写成
            # "P1,P2"）。取**第一次**出现的轮号，与 `perception` 的口径一致。
            first = pid.split(",")[0].strip()
            by_phase.setdefault(first, r["index"])
    gold_days = {ph["id"]: int(ph.get("day", 0)) for ph in gold.get("phases", [])}
    rows = []
    for pid, day in gold_days.items():
        idx = by_phase.get(pid)
        rows.append({
            "phase": pid,
            "round_index": idx,
            "gold_day": day,
            "match": (idx == day) if (idx is not None and dpr == 1.0) else None,
        })
    return {
        "days_per_round": dpr,
        "index_base": 0,
        "rows": rows,
        "by_phase": by_phase,
        "rule": (
            "阶段 → 轮号取自产出里 `rounds[*].phase_id` **非空**的那几轮"
            "（`phase_id` 记录的是「这一轮注入了哪个阶段」，只有少数轮有）。"
            "本表一律用产出的 0 基 `index`；`gold_day` 是金标 `phases[].day`。"
            "两者在 `days_per_round = 1.0`（轮 = 天）时应逐位相等 —— "
            "`match` 列为 null 表示本产出不是轮=天，天与轮不可直接比。"
        ),
        "note_1based": (
            "**别与 1 基的 R 号混用。** 2026-09-14 那批表是 0 基（R0~R14），"
            "后来的日志用 1 基（R1~R15）。本表全部 0 基，如需对照 1 基加 1。"
        ),
    }


def _railed_cells(rounds_doc: dict) -> set[tuple[str, int]]:
    """贴到 `_clamp(·, 0, 1)` 两端的 **(维, 轮) 单元格**。**无自由参数。**

    一个值恰好等于 0.0000 或 1.0000，等价于「未截断前已经越界」—— 那是夹逼
    的产物。判据与 `brief.py` 的 `evidence.railed_rounds` 同一条：这里现算，
    是为了让本表不依赖某一份简报是否已经生成过。

    **这是单元格，不是两个集合。** 下面 `_tainted` 的注释记着一次真实的误判：
    把 (维集合 × 轮集合) 的乘积当成单元格，会把干净的操作数判成贴界。
    """
    return {(d, r["index"])
            for r in rounds_doc["rounds"]
            for d, v in r["state"].items()
            if round(v, 4) in (0.0, 1.0)}


def _railed_view(cells: set[tuple[str, int]]) -> tuple[list[int], list[str]]:
    """把单元格摊成给人看的两张清单（轮 / 维）。**只用于展示，不用于判定。**"""
    return (sorted({i for _, i in cells}),
            sorted({d for d, _ in cells}))


def _tainted(operands: list[tuple[str, int]],
             railed_cells: set[tuple[str, int]]) -> list[dict]:
    """判据读过的那几个 (维, 轮) 里，有几个**那一格本身**踩在贴界上。

    **按「读到的那一格」判，不按「整条断言沾没沾到贴界轮」判。** 后者会把
    几乎所有断言都染红：P5 那一轮（0 基 14）是贴界轮，而阶段级判据都要读它，
    于是整张表只剩「不采信」三个字，等于什么也没说。

    **也不按「两个全局集合的乘积」判。** 这里原先写的是
    `idx in railed_rounds and dim in railed_dims` —— 两条各自成立就染红，
    于是「R14 因为 trust 贴界」＋「polarization 因为 R11~R13 贴界」合起来，
    会把 `polarization@R14`（那一格实测 0.50，干净）判成贴界。它**比上面的
    文档规则更严**，而且严错了方向：一个干净的操作数被判不可采信。
    这是靠 `weiran/selfcheck.py` 的第三条反例（把相位三条改到及格线）翻出来的
    —— 那一格里 P5 的极化被压到 0.50，而 R14 仍因别的维度在贴界轮名单里。
    规则与实现不一致这件事本身，比它造成的误判更值得记下来。
    """
    return [{"dimension": dim, "round": idx, "value": None}
            for dim, idx in operands if (dim, idx) in railed_cells]


# ---------------------------------------------------------------------------
# 逐条断言
# ---------------------------------------------------------------------------

def _argmax(states: dict[str, dict], dim: str) -> tuple[str, float, list[str]]:
    """取最大值的阶段。**并列时把并列的都返回**，不靠顺序偷偷定一个。"""
    vals = {pid: st[dim] for pid, st in states.items()}
    top = max(vals.values())
    ties = sorted(pid for pid, v in vals.items() if abs(v - top) < 1e-12)
    return ties[0], top, ties


def _fmt(vals: dict[str, float], dim: str, order: list[str]) -> str:
    return " ".join(f"{pid}={vals[pid][dim]:.4f}" for pid in order if pid in vals)


def evaluate(rounds_doc: dict, gold: dict, brief: dict) -> dict:
    """核心：把 10 条断言逐条求值。"""
    rounds = rounds_doc["rounds"]
    align = alignment(rounds_doc, gold)
    by_phase = align["by_phase"]
    # 只对**有阶段标签的那几轮**做阶段级判据 —— 这是金标说法的唯一落点。
    order = [pid for pid in ("P1", "P2", "P3", "P4", "P5") if pid in by_phase]
    states = {pid: rounds[by_phase[pid]]["state"] for pid in order}
    railed_cells = _railed_cells(rounds_doc)
    per_round = brief["rounds"]

    def op(dim: str, pid: str) -> tuple[str, int]:
        return (dim, by_phase[pid])

    out: list[dict] = []

    def add(aid, verdict, values, detail, operands, *, trusted=True,
            no_trust=(), undecidable=None, to_fix=None, extra=None):
        """组装一条结果。**采信与结果是两个字段** —— 一条「通过」可以不可采信。"""
        hits = _tainted(operands, railed_cells)
        reasons = list(no_trust)
        if hits and NO_TRUST_RAIL not in reasons:
            reasons.append(NO_TRUST_RAIL)
        if not trusted and not reasons:
            reasons.append("其它")
        a = gold["assertions_by_id"][aid]
        row = {
            "id": aid, "type": a["type"], "claim": a["claim"], "check": a["check"],
            "verdict": verdict,
            # **「不可判定」不谈采信。** 用 None 而不是 True/False：一条判不出来
            # 的断言既不算「可采信」也不算「不可采信」，把它记成 False 会让
            # 「不采信」这个数虚高，记成 True 又等于说它可信 —— 两个都不对。
            "trusted": None if verdict == UNDECIDED else not reasons,
            "no_trust_because": [] if verdict == UNDECIDED else reasons,
            "values": values,
            "detail": detail,
            "tainted_by_railing": hits,
            "undecidable_because": undecidable,
            "to_make_decidable": to_fix,
        }
        if extra:
            row.update(extra)
        out.append(row)
        return row

    def need_phases(aid: str, *needs: str, what: str) -> bool:
        """这几条阶段级判据的**前置**：它要读的那几个阶段，这一轮里有没有落点。

        返回 True 表示可以往下判；缺的阶段当场记成 **`不可判定`**，并带上
        「缺什么、怎么才能判」。

        **这不是防御式编程。** 阶段落点是「这一轮注入了哪个阶段」
        （`rounds[*].phase_id` 非空的那几轮）：跑一次 `--no-phases`，或者轮数
        不够那个阶段该出现的位置，它就**根本没有**。原先这些判据直接写
        `states["P1"]`，于是整张表当场 `KeyError` 死掉 —— 十条断言一条都读不到。
        而缺读数折成「否决」正是本仓库在别处反复拦的那件事（AS-7/AS-9 就是
        这么处理的）：**读不到是「没测到」，不是「没通过」。**
        """
        gone = [p for p in needs if p not in by_phase]
        if not gone:
            return True
        add(aid, UNDECIDED, {},
            f"这一轮里 {'、'.join(gone)} 没有落点（`rounds[*].phase_id` 里找不到"
            f"它们）—— 这条判据要读的是{what}，量不存在。"
            "**这是「没测到」，不是「没通过」**，所以既不判通过也不判否决。",
            [], undecidable=f"缺阶段落点：{'、'.join(gone)}",
            to_fix=(
                "跑一次真正注入阶段事件的推演（去掉 `--no-phases`），并让轮数覆盖"
                "到这个阶段该出现的位置（金标 `phases[].day` 就是那个位置）。"
            ))
        return False

    # -- AS-1 方向 --------------------------------------------------------
    if need_phases("AS-1", "P1", "P2", "P3", what="P1/P2/P3 三个阶段的 trust"):
        t1, t2, t3 = (states["P1"]["trust"], states["P2"]["trust"],
                      states["P3"]["trust"])
        add("AS-1", PASS_ if (t3 < t2 < t1) else FAIL_,
            {"trust(P1)": round(t1, 4), "trust(P2)": round(t2, 4),
             "trust(P3)": round(t3, 4)},
            f"{t3:.4f} < {t2:.4f} < {t1:.4f}" if t3 < t2 < t1
            else f"不成立：{t3:.4f} / {t2:.4f} / {t1:.4f}",
            [op("trust", "P1"), op("trust", "P2"), op("trust", "P3")])

    # -- AS-2 / AS-4 次序 -------------------------------------------------
    argmax_note = (
        "`argmax_phase` 有两个口径：只在 5 个**阶段轮**上取，或在全部 15 轮上取。"
        "本表按「阶段轮」取，因为金标写的是阶段（P3），而阶段只在被注入的那一轮"
        "上有落点；另一个口径的读数同时印出来供核对。"
    )
    for aid, dim in (("AS-2", "attention"), ("AS-4", "polarization")):
        if not need_phases(aid, "P3", what=f"P3 上的 {dim}"):
            continue
        if len(order) < 2:
            # 只落下一个阶段轮时 argmax 恒等于它 —— 这条判据在这份数据上
            # **恒真**，而恒真的通过没有任何信息量（本表在 AS-5 上已经写过
            # 一次这件事）。判「不可判定」而不是「通过」。
            add(aid, UNDECIDED, {},
                f"这一轮里只有 {order[0]} 一个阶段有落点 —— argmax 恒等于它，"
                "这条判据在这份数据上恒真，通过与不通过都没有内容。",
                [], undecidable="有落点的阶段少于两个，argmax 无从比较",
                to_fix="把轮数放够，让金标 `phases[].day` 的 5 个位置都落在推演里。")
            continue
        pid, top, ties = _argmax(states, dim)
        all_pid, all_top, _ = _argmax({str(r["index"]): r["state"] for r in rounds}, dim)
        ok = (pid == "P3")
        # 剔掉贴界轮的变体：**剔的是「这一维在这一轮贴界」的那些轮**，不是
        # 「这一轮有任何一维贴界」的那些轮。后者对 attention 尤其错：R14 的
        # attention 从未触界，却因为同轮的 trust/risk/stability 贴界而被剔掉，
        # 于是给出一个「剔掉后 argmax 变成谁」的假变体。没有可剔的就不给变体
        # —— 一个空的「剔掉后还是老样子」比没有更坏，它看起来像做过检验。
        clean = {p: s for p, s in states.items()
                 if (dim, by_phase[p]) not in railed_cells}
        alt = None
        if clean and len(clean) != len(states):
            alt_pid, alt_top, _ = _argmax(clean, dim)
            alt = {"scope": f"剔掉 `{dim}` 自己贴界的阶段轮",
                   "argmax_phase": alt_pid, "value": round(alt_top, 4),
                   "rounds_used": [by_phase[p] for p in clean]}
        operands = [op(dim, p) for p in order]
        add(aid, PASS_ if ok else FAIL_,
            {f"{dim}(P3)": round(states["P3"][dim], 4),
             f"argmax_{dim}": round(top, 4),
             f"argmax_phase": pid,
             "argmax_all_15_rounds": {"phase": all_pid, "value": round(all_top, 4)}},
            f"argmax = {pid}（{top:.4f}）；P3 是 {states['P3'][dim]:.4f}。"
            + (f" 并列：{ties}" if len(ties) > 1 else "")
            + (f" 剔掉贴界轮后 argmax = {alt['argmax_phase']}（{alt['value']:.4f}）"
               if alt else ""),
            operands, extra={"argmax_rule": argmax_note, "variant": alt})

    # -- AS-3 阈值 --------------------------------------------------------
    if need_phases("AS-3", "P3", what="P3 上的 risk"):
        thr3 = _num(r">=\s*(\d+\.\d+)", gold["assertions_by_id"]["AS-3"]["check"],
                    "风险阈值")
        r3 = states["P3"]["risk"]
        add("AS-3", PASS_ if r3 >= thr3 else FAIL_,
            {"risk(P3)": round(r3, 4), "threshold": thr3},
            f"risk(P3) = {r3:.4f}，金标阈值 {thr3:.2f} —— 差 {thr3 - r3:.4f}",
            [op("risk", "P3")], extra={"threshold_from": "金标 check 原文"})

    # -- AS-5 不对称 ------------------------------------------------------
    if need_phases("AS-5", "P3", "P5", what="P3/P5 上的 attention 与 trust"):
        a3, a5 = states["P3"]["attention"], states["P5"]["attention"]
        t3, t5 = states["P3"]["trust"], states["P5"]["trust"]
        left = a3 - a5
        right = 3 * (t5 - t3)
        # **这一条要自己报「恒真」。** 右端 = 3×(trust(P5)−trust(P3))：信任若是
        # 下降的，右端为负，而左端（关注回落量）几乎总是正的 —— 于是判据自动
        # 成立，与它所声称要测的「不对称」无关。判据恒真时，通过没有信息量。
        vacuous = right <= 0
        add("AS-5", PASS_ if left > right else FAIL_,
            {"left": round(left, 4), "right": round(right, 4),
             "attention(P3)-attention(P5)": round(left, 4),
             "3*(trust(P5)-trust(P3))": round(right, 4)},
            f"左端 {left:.4f}，右端 {right:.4f}。"
            + (f" **右端为负（信任下降 {t5 - t3:+.4f}），这条判据在该数据上恒真** —— "
               f"它与「不对称」无关：把信任换成任何一个下降得更快的量，结论一样。"
               if vacuous else ""),
            [op("attention", "P3"), op("attention", "P5"),
             op("trust", "P3"), op("trust", "P5")],
            trusted=not vacuous,
            no_trust=(NO_TRUST_VACUOUS,) if vacuous else ())

    # -- AS-6 不可逆 ------------------------------------------------------
    if need_phases("AS-6", "P1", "P5", what="P1/P5 两端的 trust"):
        drop = _num(r"-\s*(\d+\.\d+)", gold["assertions_by_id"]["AS-6"]["check"],
                    "回落幅度")
        t1, t5 = states["P1"]["trust"], states["P5"]["trust"]
        add("AS-6", PASS_ if t5 < t1 - drop else FAIL_,
            {"trust(P5)": round(t5, 4), "trust(P1)": round(t1, 4),
             "trust(P1)-0.10": round(t1 - drop, 4)},
            f"trust(P5) = {t5:.4f}，基线减 {drop:.2f} 是 {t1 - drop:.4f}",
            [op("trust", "P5"), op("trust", "P1")])

    # -- AS-7 事件顺序：不可判定 ------------------------------------------
    add("AS-7", UNDECIDED, {},
        "产出里**没有「叙事事件 → 检出信号」的映射**。金标 `event_order` 是 17 条"
        "有序叙事事件，产出有逐轮行为分类与引擎自报的事件（`suppression_detected`"
        " 等），但没有任何字段把两者连起来 —— 「劝删被检出」这句话在产出里无法机检。",
        [],
        undecidable="缺两样：(a) 产出侧的事件 id，(b) 一条能表达「排在某两者之间」的判据",
        to_fix=(
            "① 产出：`injected` 现在只存「actor → 块文本的 sha1 前 12 位」，"
            "注入事件时把它的事件 id 一并落盘，事件顺序才可机检。"
            "② 金标：`check` 原文写的是 `seq(劝删) in (seq(口径质疑), seq(家长介入))`，"
            "这是「属于其中某一个」，而 `claim` 说的是「排在两者之间」—— "
            "表达式与声称**不等价**，按字面求值永远不会成立（除非三者序号相等）。"
            "**本表不替金标改写判据**：改写判据就是替金标做决定。"
        ))

    # -- AS-8 预警时点：本表给出定义后**可判** -----------------------------
    alert_rounds = [r["index"] for r in per_round
                    for e in r["events"] if e["kind"] == "risk_alert"]
    p2_idx = by_phase.get("P2")
    if not need_phases("AS-8", "P2", what="金标要求的那条线（P2 那一轮）"):
        pass                    # 没有 P2 就没法比「晚了多少轮」，见 need_phases
    elif not alert_rounds:
        add("AS-8", UNDECIDED, {},
            "产出全程没有任何 `risk_alert` —— 按本表的定义，「从未预警」，"
            "但本表**不把「从未发生」记成「否决」**：先把定义写清再判。",
            [], undecidable="产出里没有 `warning_issued_at` 这个字段",
            to_fix="在金标里写死「预警」的判据，产出里落一个字段")
    else:
        first = alert_rounds[0]
        add("AS-8", PASS_ if first <= p2_idx else FAIL_,
            {"warning_issued_at": first,
             "P2_round": p2_idx,
             "risk_alert_rounds": alert_rounds},
            f"首个 `risk_alert` 在轮 {first}（0 基），金标要求 ≤ P2 那一轮（{p2_idx}）"
            f" —— 晚了 {first - p2_idx} 轮。"
            "**注意别把这条与另一条混起来**：产出在 P2（轮 2）确实有一条 "
            "`polarization_surge`，而金标 `turning_point.detectable_signal` 说的"
            "正是「极化增速超过关注增速」；但 AS-8 判的是 `risk` **预警线**，不是"
            "这条信号。把两者混起来，一条否决会被读成通过。",
            [("risk", first - 1), ("risk", first)],
            extra={
                "definition_used": (
                    "**本表的定义**（金标只给了 `warning_issued_at` 一个名字，"
                    "没给判据）：产出里首个 `risk_alert` 事件所在的轮。依据是引擎"
                    "自己把这条事件的消息写成「风险越过预警线 0.70」"
                    "（`world_state.WorldStateEngine.RISK_ALERT = 0.70`，判据是"
                    "**穿越** `before < 0.70 <= after`）—— 这是产出里唯一自称"
                    "「预警」的量。**换一个定义可以改变结论**，所以定义写在这里。"
                ),
                "threshold_crossed": f"{gold['assertions_by_id']['AS-3']['check']} 用的是同一个 0.70",
            })

    # -- AS-9 角色冲突：不可判定 ------------------------------------------
    unimpl = brief["turning_point"]["unimplemented_signals"]
    add("AS-9", UNDECIDED, {},
        "该指标**在代码里根本没有实现**，不是「这次没测到」。产出自己声明了这件事："
        + "；".join(f"「{u['name']}」—— {u['reason']}" for u in unimpl),
        [],
        undecidable="缺逐条行为 → 角色的归属（产出里没有这个映射），以及该指标本身的实现",
        to_fix=(
            "需要逐角色「公开立场 vs 私下立场」的极性差。要判它，先得让产出落盘"
            "「哪一条行为出自哪个角色、立场是什么」—— 现在只有 actor id 与行为类别。"
            "**在实现出来之前，这一条不许被算成通过，也不许被算成失败。**"
        ))

    # -- AS-10 反事实分支：可判，但**两边都贴在 0.0 下界** --------------------
    rows = brief["windows"]
    row_p2 = next((w for w in rows if w["phase"] == "P2"), None)
    if row_p2 is None or row_p2.get("branch_a") is None:
        add("AS-10", UNDECIDED, {},
            "简报的窗口表里没有 P2 那一行，或它没有离线分支读数。",
            [], undecidable="缺 P2 窗口的离线分支读数")
    else:
        br, act = row_p2["branch_a"], row_p2["actual_from_fork"]
        add("AS-10", PASS_ if br > act else FAIL_,
            {"trust_branch(P5)": br, "trust_actual(P5)": act,
             "delta": round(br - act, 4)},
            f"分支 {br:.4f} 对配对实际 {act:.4f} —— **两边都贴在 0.0 下界**，"
            f"判据是 {br:.4f} > {act:.4f}。这条否决与它可能的「通过」一样没有内容："
            f"贴界把两个数都压到地板上，比较结果由夹逼决定。"
            "这一条与 AS-5 / AS-6 正好构成对偶 —— 那两条因贴界而「通过得毫无内容」，"
            "这一条因贴界而「否决得毫无内容」。**贴界对判据的破坏是双向的。**",
            [("trust", by_phase.get("P5", len(rounds) - 1))],
            extra={
                "branch_source": (
                    "`branch_a` / `actual_from_fork` 来自 `brief.build_windows` 的"
                    "离线分支：同一个分叉点、同一批后续行为，只把那一轮的动作换成"
                    "干预版。**agent 的反应不变** —— 所以它不是一次真正的反事实"
                    "重跑，金标自己的 `note` 也写着「此项需要运行干预分支」。"
                ),
                "gold_note": gold["assertions_by_id"]["AS-10"].get("note", ""),
            })

    # 按断言号排序再交出去。**求值顺序与阅读顺序必须一致** —— 上面是按
    # 判据类型分组的（先两条 argmax，再阈值），不排的话表里的条目会跳号，
    # 而跳号的表最容易被读成「漏了几条」。
    out.sort(key=lambda r: int(r["id"].split("-")[1]))
    return {
        "alignment": align,
        "railing": {
            "railed_cells": sorted(f"{d}@R{i}" for d, i in railed_cells),
            "railed_rounds": _railed_view(railed_cells)[0],
            "railed_dims": _railed_view(railed_cells)[1],
            "rule": (
                "一个值恰好等于 0.0000 或 1.0000 即为贴界（`_clamp(·, 0, 1)` 的"
                "夹逼产物）。**判据读过的那个 (维, 轮) 单元格只要本身踩在贴界上，"
                "该条即「不采信」** —— 不是「结果作废」，是「这个结果不能当证据」。"
                "**按单元格判，不按「沾没沾到贴界轮」判，也不按「轮集合 × 维集合」"
                "的乘积判** —— 后两者会把干净的操作数判成贴界，把整张表染红。"
                "`railed_rounds` / `railed_dims` 是两张给人看的汇总清单，"
                "**不参与判定**；判定只看 `railed_cells`。"
            ),
        },
        "assertions": out,
    }


def _interpretations(rows: list[dict]) -> list[dict]:
    """按**本次运行的实际结果**决定每条解读讲不讲、以及为什么不讲。

    这是 `INTERPRETATIONS` 那一段注释里那个修法的落点：解读不再是一句无条件的
    断言，而是一句**带前提的**断言，前提由这里机检。
    """
    verdict = {r["id"]: r["verdict"] for r in rows}
    out = []
    for spec in INTERPRETATIONS:
        observed = {aid: verdict.get(aid, "缺失")
                    for aid in spec["requires_ids_failed"]}
        missing = [aid for aid, v in observed.items() if v != FAIL_]
        applies = not missing
        out.append({
            "id": spec["id"],
            # 前提不成立时 `claim` 为 None，**不是空串** —— 空串会被渲染成
            # 一行空白，看起来像「这条解读没话说」；None 才逼着渲染层交代原因。
            "claim": spec["claim"] if applies else None,
            "requires": spec["requires"],
            "observed": observed,
            "applies": applies,
            "not_applicable_because": None if applies else (
                "本次运行里 " + "、".join(f"{aid} 是「{observed[aid]}」" for aid in missing)
                + " —— 该解读的前提不成立，所以没有输出"),
        })
    return out


def summarize(rows: list[dict]) -> dict:
    counts = {PASS_: 0, FAIL_: 0, UNDECIDED: 0}
    for r in rows:
        counts[r["verdict"]] += 1
    judged = [r for r in rows if r["verdict"] != UNDECIDED]
    trusted = [r["id"] for r in judged if r["trusted"]]
    return {
        "counts": counts,
        "total": len(rows),
        # 分母只算「判出来了的」那些 —— 不可判定的条目本来就不谈采信。
        "judged": len(judged),
        "trusted_ids": trusted,
        "not_trusted": [
            {"id": r["id"], "verdict": r["verdict"],
             "because": "、".join(r["no_trust_because"])}
            for r in judged if not r["trusted"]
        ],
        "interpretations": _interpretations(rows),
    }


# ---------------------------------------------------------------------------
# 文字
# ---------------------------------------------------------------------------

def render_markdown(doc: dict) -> str:
    L: list[str] = []
    w = L.append
    m = doc

    w("# 金标对照表")
    w("")
    w("> **这不是分数。** 它不含「准确率」「得分」「命中率」这类量，也不该被算成"
      "任何这类量。")
    w("")
    w(f"{m['not_a_score']['statement']}")
    w("")
    w(f"出处：`{m['not_a_score']['source']}`。")
    w("")

    # -- 读数按什么对齐 ---------------------------------------------------
    a = m["alignment"]
    w("## 一、先说对齐口径（这一步错过一次，所以写进产物）")
    w("")
    w(a["rule"])
    w("")
    w("| 阶段 | 本表轮号（0 基） | 金标 `day` | 逐位相等 |")
    w("|---|---|---|---|")
    for r in a["rows"]:
        mt = "—" if r["match"] is None else ("是" if r["match"] else "**否**")
        w(f"| {r['phase']} | {r['round_index']} | {r['gold_day']} | {mt} |")
    w("")
    w(a["note_1based"])
    w("")

    ra = m["railing"]
    w("## 二、贴界：读数的前提，不是附注")
    w("")
    w(ra["rule"])
    w("")
    w(f"本次运行的贴界轮：**{ra['railed_rounds']}**；贴界维度："
      f"**{', '.join(ra['railed_dims'])}**。")
    w("")

    w("## 三、逐条")
    w("")
    w("| 断言 | 金标 check 原文 | 读数 | 结果 | 采信 | 贴界 |")
    w("|---|---|---|---|---|---|")
    for r in m["assertions"]:
        vals = "；".join(f"{k}={v}" for k, v in r["values"].items()
                        if not isinstance(v, (dict, list)))
        tv = ("—" if r["trusted"] is None else
              ("是" if r["trusted"] else "**否**"))
        ra_ = "—" if not r["tainted_by_railing"] else \
            "、".join(f"{h['dimension']}@R{h['round']}" for h in r["tainted_by_railing"])
        w(f"| {r['id']} | `{r['check']}` | {vals or '—'} | **{r['verdict']}** | "
          f"{tv} | {ra_} |")
    w("")

    s = m["summary"]
    w(f"**合计：{s['total']} 条 —— 通过 {s['counts'][PASS_]} · "
      f"否决 {s['counts'][FAIL_]} · 不可判定 {s['counts'][UNDECIDED]}。**"
      f"其中**判得出来**的 {s['judged']} 条，而判得出来的里面只有 "
      f"**{len(s['trusted_ids'])} 条可采信**（{', '.join(s['trusted_ids']) or '无'}）—— "
      "其余的结果都不作数：不是「结果反了」，是「这个结果不能当证据」。")
    w("")
    # 解读**带着前提一起印**：前提成立才印那句话，不成立就明说它为什么没被输出。
    # 直接印 `claim`（而不是先 `if applies` 判断再印）会让前提失效时这句话
    # 静默变成 None、渲染出一行空白 —— 那正是这一段要修的病。
    for it in s["interpretations"]:
        if it["applies"]:
            w(it["claim"])
        else:
            w(f"（解读 `{it['id']}` 未输出：{it['not_applicable_because']}）")
        w("")
    if s["not_trusted"]:
        w("**不采信的条目**（不是「结果作废」，是「不能当证据」）：")
        w("")
        for n in s["not_trusted"]:
            w(f"- `{n['id']}`（{n['verdict']}）—— {n['because']}")
        w("")
    w("**不可判定的条目，一条都不许算成通过或失败**（它们是这张表最该补的地方）：")
    w("")
    for r in m["assertions"]:
        if r["verdict"] == UNDECIDED:
            w(f"- `{r['id']}` —— {r['undecidable_because']}")
    w("")

    w("## 四、逐条说明")
    w("")
    for r in m["assertions"]:
        w(f"### {r['id']} · {r['type']} · {r['verdict']}"
          + ("（不采信）" if r["trusted"] is False else ""))
        w("")
        w(f"- 声称：{r['claim']}")
        w(f"- 金标判据：`{r['check']}`")
        if r["values"]:
            for k, v in r["values"].items():
                if isinstance(v, (dict, list)):
                    w(f"- {k}：`{json.dumps(v, ensure_ascii=False)}`")
                else:
                    w(f"- {k}：{v}")
        w(f"- 读数：{r['detail']}")
        if r["tainted_by_railing"]:
            hits = "、".join(f"{h['dimension']}@R{h['round']}"
                            for h in r["tainted_by_railing"])
            w(f"- 受贴界影响：**是** —— 判据读过 {hits}")
        else:
            w("- 受贴界影响：否")
        if r["trusted"] is False:
            w(f"- 不采信的原因：{'、'.join(r['no_trust_because'])}")
        for key in ("argmax_rule", "variant", "threshold_from", "definition_used",
                    "threshold_crossed", "branch_source", "gold_note"):
            if r.get(key):
                v = r[key]
                if isinstance(v, dict):
                    w(f"- {key}：`{json.dumps(v, ensure_ascii=False)}`")
                else:
                    w(f"- {key}：{v}")
        if r["undecidable_because"]:
            w(f"- **为什么不可判定**：{r['undecidable_because']}")
        if r["to_make_decidable"]:
            w(f"- **还缺什么才能判**：{r['to_make_decidable']}")
        w("")

    w("## 五、这份表的来源")
    w("")
    p = m["_provenance"]
    w(f"- 推演产出：`{Path(p['rounds_file']).name}` "
      f"sha256:{p['rounds_sha256']}（{m['meta']['agents']} agent × "
      f"{m['meta']['rounds']} 轮）")
    w(f"- 金标：`{Path(p['scenario_file']).name}` sha256:{p['scenario_sha256']}")
    w(f"- 命令：`{p['command']}`")
    w(f"- 复现：`python -m weiran.gold_check`（确定性，同输入同输出）")
    w("")
    w(f"复现校验：`replay_max_deviation = {m['replay_max_deviation']}`"
      "（>1e-4 说明这份曲线不是本引擎产生的，那样整张表都不成立）")
    w("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------

def build_gold_check(rounds_doc: dict, gold: dict, *, scale: float = DEFAULT_SCALE,
                     command: str = "") -> dict:
    """组装对照表。**最后一步把结论与前提一起落盘。**"""
    assertions_by_id = {a["id"]: a for a in gold.get("assertions", [])}
    if not assertions_by_id:
        raise BriefError(
            "金标里没有任何 `assertions` —— 没有可对照的条目，这张表是空的。"
            "（空表看起来像「全部通过」，所以这里出声。）")
    # 「这不是分数」这句话**必须由金标自己说**，不能由本表代说。
    # 金标要是没写这条声明，本表就只剩自己的信用背书 —— 那正是这个项目
    # 一直在拒绝的姿态。所以缺了就抛，不生成一张无出处的表。
    if not str(gold.get("gold_status", {}).get("what_is_NOT_solid", "")).strip():
        raise BriefError(
            "金标里没有 `gold_status.what_is_NOT_solid` —— 这张表开头的"
            "「这不是分数」就失去了出处。\n"
            "  本表不替金标写这句话：**代它说，就是替它背书**。"
            "先把它补进 reference_data.json 的 gold_status。")

    gold = {**gold, "assertions_by_id": assertions_by_id}
    brief = build_brief(rounds_doc, gold, scale=scale, command=command)
    result = evaluate(rounds_doc, gold, brief)
    summary = summarize(result["assertions"])
    meta = rounds_doc["meta"]

    return {
        "not_a_score": {
            "statement": (
                "本表逐条核对金标自己写的断言「在这**一次**运行里发生了没有」，"
                "**不是模型有多准**。它没有分母，也没有被比较的基准 —— "
                "所以它既不能被加总成一个「得分」，也不能与另一次运行的读数"
                "比出「提高了多少」。"
            ),
            "why": gold["gold_status"]["what_is_NOT_solid"],
            "why_it_matters": gold.get("gold_status", {}).get("why_it_matters", ""),
            "source": "benchmark/scenarios/*/reference_data.json → gold_status",
            "excluded_from_judgement": (
                "`reference_shape`（作者手写的强度预期）**不参与任何判据** —— "
                "`load_gold` 在读取时已把它剥掉。判据只吃产出记的六维曲线与"
                "金标 assertions 自己写的阈值/顺序。"
            ),
        },
        "meta": {
            "agents": meta.get("agents"), "rounds": meta.get("rounds"),
            "platform": meta.get("platform"),
            "days_per_round": meta.get("days_per_round"),
            "compressed": meta.get("compressed"),
        },
        "replay_max_deviation": brief["replay_max_deviation"],
        "degradations": brief["degradations"],
        "alignment": result["alignment"],
        "railing": result["railing"],
        "assertions": result["assertions"],
        "summary": summary,
        "_provenance": {
            **rounds_doc.get("_provenance", {}),
            **gold.get("_provenance", {}),
            "command": command,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ensure_console_encoding()
    ap = argparse.ArgumentParser(
        description="金标对照表：逐条核对金标断言（不是分数）")
    ap.add_argument("--rounds-file", default=None,
                    help=f"推演产出（默认 {DEFAULT_ROUNDS_FILE.name}）")
    ap.add_argument("--scenario", default=None, help="场景目录（默认同 simulate）")
    ap.add_argument("--out", default=None, help="markdown 输出路径")
    ap.add_argument("--json", default=None, help="结构化输出路径")
    ap.add_argument("--intervention-scale", type=float, default=DEFAULT_SCALE,
                    help=f"离线分支的干预倍数，默认 {DEFAULT_SCALE}（沿用 validate.py）")
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
        command = DEFAULT_COMMAND + (
            f" --intervention-scale {args.intervention_scale}"
            if args.intervention_scale != DEFAULT_SCALE else "")
        doc = build_gold_check(rounds_doc, gold, scale=args.intervention_scale,
                              command=command)
    except BriefError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2

    md = render_markdown(doc)
    out_md = Path(args.out) if args.out else DEFAULT_MD
    out_js = Path(args.json) if args.json else DEFAULT_JSON
    out_js.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    out_md.write_text(md, encoding="utf-8")

    s = doc["summary"]
    print(f"已写出 {out_md}")
    print(f"  {s['total']} 条：通过 {s['counts'][PASS_]} · "
          f"否决 {s['counts'][FAIL_]} · 不可判定 {s['counts'][UNDECIDED]}"
          f"（其中不采信 {len(s['not_trusted'])} 条）")
    print(f"  贴界轮 {doc['railing']['railed_rounds']}")
    if args.stdout:
        print()
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
