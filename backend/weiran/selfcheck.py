"""装置的自证：喂它反例，看它会不会被喂倒。

**这个模块存在的理由，是现有那 22 条测试守不住的一件事。**

`test_gold_check.py` 写得很硬 —— 契约、贴界、出处、恒真，四条线都守住了。
但它的断言**全部跑在同一份真产出上**（`_load()` 缓存的那一份）。于是有一个
致命的推论：

    把 `evaluate()` 整个换成一张硬编码的 10 行表，全套测试仍然全绿。

具体到本装置声称的五个输出格（通过·采信 / 通过·不采信 / 否决·采信 /
否决·不采信 / 不可判定）上，「**否决 → 通过**」这个方向**一次都没被演示过**：
现有 5 条否决全部是「真产出上碰巧是红的」，没有任何一条被证明过「如果数据
改好了它会变绿」。（反方向只有一条：`_unrailed` 让 AS-5 由通过翻否决。）

这与 `repro_check.py` 里那句核心教训是同一个病，只是发生在装置层：

    只断言「参数在场」的检查，是这个项目最该防的那种恒真检查。

所以本模块要做的事只有一件：**让「这张表是算出来的」变成可执行的证词 ——
代价是必须喂它反例。**

## 为什么不违反「不另造合成输入」那条立场

`test_gold_check.py` 开头写着：数据一律取入库的真产出与真金标，不另造合成
输入 —— 因为「合成数据永远比真数据整齐，而这张表全部的意义就是能否吃下真
数据」。那条立场是对的，本模块**不反对它，是补它的另一半**：

    真数据  → 测「能不能吃下真实复杂度」（已有，22 条）
    反例    → 测「会不会被喂倒」（缺，本模块）

而且本模块的每一个反例**都不是另造一份数据**，而是**在真产出上做单点受控
扰动**：保住真数据的复杂度，一次只改一个变量。这在方法上比另造合成数据更
严 —— 改一个地方而结论跟着变，才说明那个地方真的被判据读着。

## 每个反例必须同时说清两件事

    该动的动了    （否则测不到区分力）
    不该动的没动  （否则「把 10 条全翻过来」的实现也能过）

第二件是本模块唯一真正的工程约束，写法是 `untouched`：它断言那一组断言的
**读数逐字不变**。而 `values` 正好是判据读过的那几个操作数的指纹 ——
所以「不该动」这件事是**机检**的，不是靠人记得。

`expect` / `untouched` / `expect_trusted` 三者必须**覆盖全部断言**：覆盖不全
的反例是在挑软柿子，跑得再绿也没用。少了哪条，运行时就报出来。

## 改动层：反例改的必须是它声明的那一层

三个可改层，对应装置的三类输入：

    产出   `rounds[*].state`   记录下来的六维曲线  → 判据的左手边
    金标   `assertions[*].check` 阈值与顺序原文     → 判据的右手边
    简报   `brief` 的事件与窗口  → 判据读的中间产物

**声明改动层不是分类学，是约束**：一个「什么都能改」的反例证明不了任何事 ——
它改坏了数据也照样能凑出预期的结论。

三层里有一条容易踩的实测事实（`brief.py:535`）：**简报的曲线与事件是从
`behaviors` 重放算出来的，不是从记录的 `state` 读的**。所以改「产出」那一层
的曲线**动不了 AS-8** —— 它的 `risk_alert` 来自重放。要动摇 AS-8，只能改简报。

## 五格必须都能被触发

装置声明自己有五个输出格。一个只会输出「通过」的装置是恒真装置，一个只会
输出「否决」的装置是噪声装置 —— **能落进每一个格，才叫有区分力**。所以最后
有一条元断言：把全部反例跑出来的 (verdict, trusted) 收集起来，五格必须都
出现过。少一格，说明有一个格是死代码。

用法：
    cd backend
    python -m weiran.selfcheck
    python -m weiran.selfcheck --verbose   # 每个反例逐条打印它对全部断言的表态
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass, field
from typing import Callable

from . import gold_check as G
from .config import REPO_ROOT, ensure_console_encoding

COMMAND = "python -m weiran.selfcheck"

#: `untouched` 里的哨兵：展开成「全部断言」。恒等反例用它 ——
#: 「什么都不改」这个反例的完整表述就是「每一条的读数都不许动」。
ALL_ASSERTIONS = "*"

#: 装置声称的五个输出格。全部都要在反例里出现过。
GRIDS = (
    (G.PASS_, True), (G.PASS_, False),
    (G.FAIL_, True), (G.FAIL_, False),
    (G.UNDECIDED, None),
)

#: `Case.layer` 的闭集。**必须校验**：`apply` 的返回值决定要不要重算简报，
#: 而重算与否只看 `touched & {"产出", "金标"}` —— 所以把「产出」写成「产物」，
#: 结果不是报错，是**简报不重算**：反例读到的是改动之前的中间产物，于是它
#: 静默退化成一个恒等反例，还会报告「通过」。一个错别字就能把反例的效力
#: 拿掉，且全绿。这与本装置要防的那些静默失效是同一类，所以在入口拦掉。
LAYERS = ("产出", "金标", "简报")

#: 恒等反例的改动层：它什么都不改，所以不在这三个里。
IDENTITY_LAYER = "—"


# ---------------------------------------------------------------------------
# 反例的表示
# ---------------------------------------------------------------------------

@dataclass
class Case:
    """一个反例：一处受控扰动 + 它对全部断言的表态。

    `apply(ctx)` 就地改 `ctx` 并返回**它改动的层**（`{"产出"}` / `{"金标"}` /
    `{"简报"}`）。返回值不是装饰：运行时会照着它决定要不要重算简报 ——
    改产出或金标就得重算，只改简报就不许重算（重算会把改动抹掉，于是反例
    静默变成一个恒等反例，还报告「通过」）。
    """
    layer: str
    what: str
    apply: Callable[[dict], set[str]]
    expect: dict[str, str] = field(default_factory=dict)
    untouched: tuple[str, ...] = ()
    expect_trusted: dict[str, bool | None] = field(default_factory=dict)
    expect_interpretations: dict[str, bool] = field(default_factory=dict)


# -- 扰动器 ---------------------------------------------------------------

def _phase_rounds(ctx: dict) -> dict[str, int]:
    return G.alignment(ctx["rd"], ctx["gold"])["by_phase"]


def _set_round(ctx: dict, i: int, **values) -> None:
    """改**某一轮**记录下来的六维曲线。

    只改 `state`，不碰 `behaviors`。两者的不一致会被本装置自己的
    `replay_max_deviation` 报出来（见运行输出里的那一列）—— 这是**故意的**：
    反例测的是「判据会不会跟着它的输入走」，不是「推演对不对」，所以改的就是
    判据读的那份数。把它藏在暗处才是问题，标出来就不是。
    """
    st = ctx["rd"]["rounds"][i]["state"]
    ctx["rd"]["rounds"][i]["state"] = {**st, **values}


def _set_phase(ctx: dict, phase: str, **values) -> int:
    i = _phase_rounds(ctx)[phase]
    _set_round(ctx, i, **values)
    return i


def _p2_window(ctx: dict) -> dict:
    row = next((w for w in ctx["brief"]["windows"] if w["phase"] == "P2"), None)
    if row is None or row.get("branch_a") is None:
        raise AssertionError(
            "简报的窗口表里没有 P2 那一行，或它没有离线分支读数 —— "
            "这一条反例的前提不成立（换一个前提成立的反例，别把断言改松）")
    return row


def _alert_events(ctx: dict) -> list[dict]:
    out = []
    for r in ctx["brief"]["rounds"]:
        for e in r["events"]:
            if e["kind"] == "risk_alert":
                out.append(e)
    return out


# ---------------------------------------------------------------------------
# 九个反例
# ---------------------------------------------------------------------------

def _case_identity() -> Case:
    return Case(
        layer=IDENTITY_LAYER,
        what="恒等：什么都不改（扰动器本身不许有副作用）",
        apply=lambda ctx: set(),
        untouched=(ALL_ASSERTIONS,),
    )


def _raise_trust_p2(ctx: dict) -> set[str]:
    # 基准 trust(P1)=0.6843 / P2=0.6191 / P3=0.4448（单调下坠，AS-1 通过）。
    # 只把 P2 抬到 P1 之上 → 次序断裂。**只碰一个数**。
    _set_phase(ctx, "P2", trust=0.90)
    return {"产出"}


def _phase_three_pass(ctx: dict) -> set[str]:
    # 把相位那三条的读数一次改到及格线上：attention 峰移到 P3、极化峰移到 P3、
    # risk(P3) 过 0.70。P5 的极化原本贴死在 1.0000（贴界），得一并压下来 ——
    # 否则它在相位轮里永远是 argmax，P3 翻不了身。
    _set_phase(ctx, "P3", attention=0.95, polarization=0.90, risk=0.80)
    _set_phase(ctx, "P5", polarization=0.50)
    return {"产出"}


def _lower_gold_threshold(ctx: dict) -> set[str]:
    # **只动金标，产出一个字不改。** 阈值从 0.70 降到 0.10，AS-3 必然翻通过。
    # 这一条是「阈值真的来自金标原文」的证词 —— 它防的正是 `_num` 那条注释
    # 警告的事：代码里偷偷重抄一个数，金标改了它不知道。
    ctx["gold"]["assertions_by_id"]["AS-3"]["check"] = "risk(P3) >= 0.10"
    return {"金标"}


def _unrail_p5(ctx: dict) -> set[str]:
    # R14 是贴界轮（trust=0.0 / polarization=1.0 / risk=1.0 / stability=0.0）。
    # 把它摘下来：四条降级的断言应当恢复可采信，而**结论一条都不许翻** ——
    # 唯一例外是 AS-5：trust(P5) 抬到 P3 之上后右端转正，恒真解除、判据
    # 真的不成立了。那一翻正是「恒真检测器不是装饰」的证据。
    _set_round(ctx, 14, trust=0.55, polarization=0.70, risk=0.60, stability=0.30)
    return {"产出"}


def _rail_attention(ctx: dict) -> set[str]:
    # 全仓库的测试都写着「attention 从未触界」。这里**让它触一次**：把
    # R14 的 attention 顶到 1.0000。要证明的是贴界检测**跟着值走**，不是
    # 照着一份写死的维度名单走 —— 否则一个只认识那四个维度的实现也能过。
    _set_round(ctx, 14, attention=1.0)
    return {"产出"}


def _alert_to_round_1(ctx: dict) -> set[str]:
    # AS-8 问的是「预警够不够早」。产出里首个 risk_alert 在轮 7，晚于 P2（轮 2）。
    # 把同一条事件挪到轮 1 —— 它必须翻成通过。
    evs = _alert_events(ctx)
    assert evs, "产出里没有 risk_alert，这一条反例的前提不成立"
    for r in ctx["brief"]["rounds"]:
        r["events"] = [e for e in r["events"] if e["kind"] != "risk_alert"]
    ctx["brief"]["rounds"][1]["events"] = list(evs[:1])
    return {"简报"}


def _alert_removed(ctx: dict) -> set[str]:
    # 反方向：一条预警都没有。**不许判成否决** —— 「从未发生」与「发生了但
    # 太晚」是两件事，把它们折成同一个结论，等于把「没测到」算成「没做到」。
    for r in ctx["brief"]["rounds"]:
        r["events"] = [e for e in r["events"] if e["kind"] != "risk_alert"]
    return {"简报"}


def _p2_branch_beats_actual(ctx: dict) -> set[str]:
    # AS-10 比的是 P2 窗口的离线分支与实际配对读数。基准里 branch_a < actual
    # （所以是否决）。把分支抬到实际之上 —— 它必须翻成通过，但仍然不可采信
    # （两边都贴在 0.0 下界的性质没变）。「结论翻、采信不翻」就是这一条要说的。
    row = _p2_window(ctx)
    row["branch_a"] = row["actual_from_fork"] + 0.05
    return {"简报"}


def _cases() -> list[Case]:
    return [
        _case_identity(),
        Case(
            layer="产出",
            what="单点抬 trust(P2)：AS-1 的单调下坠该被推翻",
            apply=_raise_trust_p2,
            expect={"AS-1": G.FAIL_, "AS-10": G.FAIL_},
            untouched=("AS-2", "AS-3", "AS-4", "AS-5", "AS-6",
                       "AS-7", "AS-8", "AS-9"),
        ),
        Case(
            layer="产出",
            what="相位三条改到及格：AS-2/AS-3/AS-4 该由否决翻为通过",
            apply=_phase_three_pass,
            expect={"AS-2": G.PASS_, "AS-3": G.PASS_, "AS-4": G.PASS_,
                    "AS-10": G.FAIL_},
            # AS-5 的读数会动（它读 attention(P3)），所以只在 expect 里表态。
            untouched=("AS-1", "AS-6", "AS-7", "AS-8", "AS-9"),
            expect_trusted={"AS-2": True, "AS-3": True, "AS-4": True, "AS-5": False},
            # 四条共享成因这个解读的前提在这里断了 —— 它**必须闭嘴**。
            expect_interpretations={"shared_root_cause": False},
        ),
        Case(
            layer="金标",
            what="只动金标：AS-3 阈值 0.70→0.10（产出一个字不改）",
            apply=_lower_gold_threshold,
            expect={"AS-3": G.PASS_},
            untouched=("AS-1", "AS-2", "AS-4", "AS-5", "AS-6", "AS-7",
                       "AS-8", "AS-9", "AS-10"),
            expect_trusted={"AS-3": True},
            expect_interpretations={"shared_root_cause": False},
        ),
        Case(
            layer="产出",
            what="摘掉 R14 的贴界：四条恢复可采信，结论一条不许翻（AS-5 除外）",
            apply=_unrail_p5,
            expect={"AS-1": G.PASS_, "AS-2": G.FAIL_, "AS-3": G.FAIL_,
                    "AS-4": G.FAIL_, "AS-5": G.FAIL_, "AS-6": G.PASS_,
                    "AS-8": G.FAIL_, "AS-10": G.FAIL_},
            untouched=("AS-7", "AS-9"),
            expect_trusted={"AS-1": True, "AS-2": True, "AS-3": True,
                            "AS-4": True, "AS-5": True, "AS-6": True,
                            "AS-8": True, "AS-10": True},
        ),
        Case(
            layer="产出",
            what="让 attention 也触界：贴界检测该跟着值走，不跟着写死的维度名单走",
            apply=_rail_attention,
            expect={"AS-2": G.FAIL_, "AS-4": G.FAIL_},
            untouched=("AS-1", "AS-3", "AS-6", "AS-7", "AS-8", "AS-9"),
            expect_trusted={"AS-2": False, "AS-3": True, "AS-4": False,
                            "AS-5": False, "AS-10": False},
        ),
        Case(
            layer="简报",
            what="预警挪到轮 1：AS-8 该翻为通过",
            apply=_alert_to_round_1,
            expect={"AS-8": G.PASS_},
            untouched=("AS-1", "AS-2", "AS-3", "AS-4", "AS-5", "AS-6",
                       "AS-7", "AS-9", "AS-10"),
            expect_trusted={"AS-8": True},
        ),
        Case(
            layer="简报",
            what="清空预警：AS-8 该判「不可判定」，不是「否决」",
            apply=_alert_removed,
            expect={"AS-8": G.UNDECIDED},
            untouched=("AS-1", "AS-2", "AS-3", "AS-4", "AS-5", "AS-6",
                       "AS-7", "AS-9", "AS-10"),
            expect_trusted={"AS-8": None},
        ),
        Case(
            layer="简报",
            what="P2 分支抬到实际之上：AS-10 该翻为通过，但仍不可采信",
            apply=_p2_branch_beats_actual,
            expect={"AS-10": G.PASS_},
            untouched=("AS-1", "AS-2", "AS-3", "AS-4", "AS-5", "AS-6",
                       "AS-7", "AS-8", "AS-9"),
            expect_trusted={"AS-10": False},
        ),
    ]


# ---------------------------------------------------------------------------
# 跑
# ---------------------------------------------------------------------------

def _prepare(rounds_file, scenario_dir) -> dict:
    rd = G.load_rounds(rounds_file)
    gold = G.load_gold(scenario_dir)
    gold["assertions_by_id"] = {a["id"]: a for a in gold["assertions"]}
    return {"rd": rd, "gold": gold}


def _run_case(case: Case, base: dict) -> dict:
    """跑一个反例：建上下文 → 扰 → （必要时重算简报） → 求值。

    **为什么不走 `build_gold_check`**：有几条反例要改简报本身（AS-8/AS-10），
    而 `build_gold_check` 会把简报整个重算一遍、把改动抹掉。所以这里手工串
    `build_brief` →（反例改）→ `evaluate` → `summarize` 这条链。

    代价是自证用的入口与产品入口不是同一个函数。这个代价由 `run()` 里那条
    **入口一致性检查**偿还：在没改过的数据上，这条链的输出必须与
    `build_gold_check` 逐字相同 —— 否则下面这些反例证明的是另一个东西。
    """
    ctx = {"rd": copy.deepcopy(base["rd"]), "gold": copy.deepcopy(base["gold"])}
    ctx["brief"] = G.build_brief(ctx["rd"], ctx["gold"], command=COMMAND)
    touched = case.apply(ctx)
    if touched & {"产出", "金标"}:
        # 改了判据的输入，简报必须重算 —— 否则判据读的是改动之前的中间产物，
        # 「改了产出而结论没变」会被误读成「判据没读那个数」。
        # 只声明改简报的反例**不许**重算，重算会把它的改动抹掉、静默退化成恒等反例。
        ctx["brief"] = G.build_brief(ctx["rd"], ctx["gold"], command=COMMAND)
    rows = G.evaluate(ctx["rd"], ctx["gold"], ctx["brief"])["assertions"]
    summary = G.summarize(rows)
    dev = G.build_brief(ctx["rd"], ctx["gold"],
                        command=COMMAND)["replay_max_deviation"]
    return {"ctx": ctx, "rows": {r["id"]: r for r in rows},
            "summary": summary, "deviation": dev, "touched": touched}


def run(rounds_file=None, scenario_dir=None, *,
        rounds_path=None, scenario_path=None,
        cases: list[Case] | None = None) -> dict:
    rf = rounds_path or rounds_file or G.DEFAULT_ROUNDS_FILE
    sd = scenario_path or scenario_dir or (REPO_ROOT / G.DEFAULT_SCENARIO)
    base = _prepare(rf, sd)

    # -- 入口一致性：自证用的那条链必须与产品入口等价 ----------------------
    prod = G.build_gold_check(copy.deepcopy(base["rd"]),
                              copy.deepcopy(base["gold"]), command=COMMAND)
    mine_rows = G.evaluate(copy.deepcopy(base["rd"]),
                           copy.deepcopy(base["gold"]),
                           G.build_brief(base["rd"], base["gold"], command=COMMAND)
                           )["assertions"]
    mine_sum = G.summarize(mine_rows)
    prod_rows = prod["assertions"]
    entry_ok = (
        [r["id"] for r in mine_rows] == [r["id"] for r in prod_rows]
        and [r["verdict"] for r in mine_rows] == [r["verdict"] for r in prod_rows]
        and [r["trusted"] for r in mine_rows] == [r["trusted"] for r in prod_rows]
        and mine_sum["counts"] == prod["summary"]["counts"]
        and mine_sum["trusted_ids"] == prod["summary"]["trusted_ids"]
        and [i["applies"] for i in mine_sum["interpretations"]]
        == [i["applies"] for i in prod["summary"]["interpretations"]]
    )

    base_run = _run_case(_case_identity(), base)
    base_rows = base_run["rows"]
    all_ids = sorted(base_rows, key=lambda s: int(s.split("-")[1]))
    known_interps = {s["id"] for s in G.INTERPRETATIONS}

    results, failures = [], []
    grids: set[tuple] = set()
    for n, case in enumerate(cases if cases is not None else _cases(), 1):
        # -- 覆盖检查：反例必须对**全部**断言表态 ---------------------------
        #    哨兵**先展开成全部 id**，再拿同一个集合去做覆盖检查和逐条比较。
        #    原先这两处各写各的：覆盖检查把哨兵展开成全部，而下面的比较循环写成
        #    `() if 哨兵 in untouched else untouched` —— 于是 `untouched=(*,)` 这条
        #    唯一的意思是「每条都不许动」，却被展开成**空集合**，一个字都没比。
        #    恒等反例正是唯一使用哨兵的那条，而它全部的职责就是「扰动器不许有
        #    副作用」：它只做了覆盖检查，而哨兵又让覆盖检查必然通过。**一个带
        #    副作用的扰动器会静默通过。** 这与本装置要防的静默失效是同一类，
        #    只是发生在 harness 自己身上 —— 所以这里只展开一次，两处共用。
        untouched_ids = (tuple(all_ids) if ALL_ASSERTIONS in case.untouched
                         else case.untouched)
        declared = set(case.expect) | set(case.expect_trusted) | set(untouched_ids)
        uncovered = sorted(set(all_ids) - declared)
        bogus = sorted(declared - set(all_ids))

        r = _run_case(case, base)
        rows = r["rows"]
        checks: list[str] = []

        # -- 改动层自报要先合法：不合法就谈不上「该不该重算简报」------------
        #    恒等反例的 layer 是 `—`（它什么都不改），单独放行。
        if case.layer not in LAYERS and case.layer != IDENTITY_LAYER:
            checks.append(f"改动层「{case.layer}」不在 {LAYERS} 里 —— "
                          f"这决定了简报重不重算，写错会静默退化")
        if not set(r["touched"]) <= set(LAYERS):
            checks.append(f"apply 自报改动的层是 {sorted(r['touched'])}，"
                          f"其中有不在 {LAYERS} 里的")
        claimed = set() if case.layer == IDENTITY_LAYER else {case.layer}
        if set(r["touched"]) != claimed:
            checks.append(f"apply 自报改了 {'、'.join(sorted(r['touched'])) or '（无）'}，"
                          f"与 Case.layer「{case.layer}」对不上")

        if uncovered:
            checks.append(f"反例没有对全部断言表态，漏了 {'、'.join(uncovered)}")
        if bogus:
            checks.append(f"反例点名了不存在的断言：{'、'.join(bogus)}")
        if set(rows) != set(all_ids):
            checks.append(f"表里的条目变了：{sorted(rows)} 对 {all_ids}")

        for aid in untouched_ids:
            if aid not in rows:
                continue
            b, a = base_rows[aid], rows[aid]
            if a["values"] != b["values"]:
                checks.append(f"{aid} 声称不该动，读数却变了："
                              f"{b['values']} → {a['values']}")
            if a["verdict"] != b["verdict"]:
                checks.append(f"{aid} 声称不该动，结论却变了："
                              f"{b['verdict']} → {a['verdict']}")
            if a["trusted"] != b["trusted"]:
                checks.append(f"{aid} 声称不该动，采信却变了："
                              f"{b['trusted']!r} → {a['trusted']!r}")

        for aid, want in case.expect.items():
            got = rows[aid]["verdict"] if aid in rows else "缺失"
            if got != want:
                checks.append(f"{aid} 该是「{want}」，实得「{got}」"
                              f" —— {rows[aid]['detail'] if aid in rows else ''}")
        for aid, want in case.expect_trusted.items():
            got = rows[aid]["trusted"] if aid in rows else "缺失"
            if got is not want:
                checks.append(f"{aid} 的采信该是 {want!r}，实得 {got!r}"
                              f" —— 不采信原因："
                              f"{rows[aid]['no_trust_because'] if aid in rows else ''}")

        its = {i["id"]: i for i in r["summary"]["interpretations"]}
        for iid, want in case.expect_interpretations.items():
            got = its.get(iid, {}).get("applies")
            if got is not want:
                checks.append(f"解读 `{iid}` 该{'出现' if want else '闭嘴'}，实得 {got!r}")
        # 写错解读 id 的后果是**静默**：`its.get(iid)` 给 None、`want` 是 False
        # 时恰好对上，看起来像「这条解读如我所愿地闭嘴了」。所以先查 id 存在。
        bogus_it = sorted(set(case.expect_interpretations) - known_interps)
        if bogus_it:
            checks.append(f"反例点名了不存在的解读：{'、'.join(bogus_it)}")

        for row in rows.values():
            grids.add((row["verdict"], row["trusted"]))

        results.append({
            "n": n, "layer": case.layer, "what": case.what,
            "ok": not checks, "checks": checks,
            "deviation": r["deviation"], "touched": sorted(r["touched"]),
            "changed": [
                (aid, base_rows[aid]["verdict"], rows[aid]["verdict"])
                for aid in all_ids
                if aid in rows and rows[aid]["verdict"] != base_rows[aid]["verdict"]
            ],
        })
        if checks:
            failures.append((n, case, checks))

    missing_grids = [g for g in GRIDS if g not in grids]
    return {
        "entry_ok": entry_ok,
        "results": results, "failures": failures,
        "grids": sorted(grids, key=lambda g: (str(g[0]), str(g[1]))),
        "missing_grids": missing_grids,
        "base_counts": base_run["summary"]["counts"],
        "ids": all_ids,
    }


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def _grid_name(v, t) -> str:
    return f"{v}·{'采信' if t else '不采信'}" if t is not None else f"{v}"


def render(report: dict, verbose: bool = False) -> str:
    L: list[str] = []
    w = L.append
    n_cases = len(report["results"])
    n_bad = len(report["failures"])
    w("【装置自证】喂它反例，看它会不会被喂倒")
    w("")
    w(f"  基准：{n_cases} 个反例之前的入库产出 —— 通过 "
      f"{report['base_counts'][G.PASS_]} · 否决 {report['base_counts'][G.FAIL_]} · "
      f"不可判定 {report['base_counts'][G.UNDECIDED]}")
    w("")
    w(f"  {'#':>2}  {'改动层':<4}  {'扰动':<46}  {'判定'}")
    w(f"  {'-' * 2}  {'-' * 4}  {'-' * 46}  {'-' * 4}")
    for r in report["results"]:
        what = r["what"]
        if len(what) > 44:
            what = what[:43] + "…"
        w(f"  {r['n']:>2}  {r['layer']:<4}  {what:<46}  "
          + ("✅" if r["ok"] else "❌"))
    w("")

    for r in report["results"]:
        if r["changed"]:
            flips = "、".join(f"{a} {b}→{c}" for a, b, c in r["changed"])
            w(f"      #{r['n']} 结论翻转：{flips}")
        if verbose:
            w(f"      #{r['n']} 改动层 {'、'.join(r['touched']) or '（无）'}"
              f"；重放偏差 {r['deviation']:.2e}")
    w("")

    if report["failures"]:
        w("  失败明细：")
        for n, case, checks in report["failures"]:
            w(f"    #{n} {case.what}")
            for c in checks:
                w(f"        - {c}")
        w("")

    w("  本次触发的输出格（装置声称它有五个，全部都要被触发过）：")
    hit: set = set(report["grids"])
    for grid in GRIDS:
        w(f"    {'✔' if grid in hit else '✗'} {_grid_name(*grid)}")
    w("")

    w(f"  入口一致性：自证用的那条链与产品入口 `build_gold_check` "
      + ("逐字相同 ✅" if report["entry_ok"] else "**不一致 ❌** —— "
         "下面几条反例证明的是另一个东西"))
    w("")
    verdict_ok = not report["failures"] and not report["missing_grids"]
    w(f"  结论：{n_cases - n_bad}/{n_cases} 个反例落在预期格"
      + ("" if verdict_ok else " —— **装置没有通过自证**"))
    # 这两句必须在**反例全过**的前提下才说得出口。原先这里只看 `missing_grids`
    # —— 而反例失败与缺格是两回事：一条反例没落在预期格时并不缺格，于是这句
    # 「五个输出格全部由输入决定」照旧印出来，**在装置已经没通过自证的时候
    # 替它说好话**。观众只读结论那一行，看到的就是一句无条件的背书。
    if report["missing_grids"]:
        w("        **有输出格从未被触发**："
          + "、".join(_grid_name(v, t) for v, t in report["missing_grids"])
          + " —— 那一格是死代码，或者反例不够。")
    elif report["failures"]:
        w("        有反例没落在预期格，所以**这次运行不能用来佐证**"
          "「这张表不是把结论写死的」—— 先说清上面那几条。")
    else:
        w("        五个输出格全部由输入决定：这张表不是把结论写死的。")
    w("")
    w("  这不是分数。它回答的是「这张表会不会不管喂什么都吐同一份结论」，")
    w("  不是「模型有多准」—— 后者需要基准，而这个项目没有、也不打算有。")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ensure_console_encoding()
    ap = argparse.ArgumentParser(description="装置自证：喂它反例")
    ap.add_argument("--rounds-file", default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    rounds_path = None
    if args.rounds_file:
        from pathlib import Path
        rounds_path = Path(args.rounds_file)
        if not rounds_path.is_absolute():
            rounds_path = REPO_ROOT / rounds_path
    scenario_path = None
    if args.scenario:
        from pathlib import Path
        scenario_path = Path(args.scenario)
        if not scenario_path.is_absolute():
            scenario_path = REPO_ROOT / scenario_path

    try:
        report = run(rounds_path=rounds_path, scenario_path=scenario_path)
    except G.BriefError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2
    print(render(report, verbose=args.verbose))
    return 0 if not report["failures"] and not report["missing_grids"] \
        and report["entry_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
