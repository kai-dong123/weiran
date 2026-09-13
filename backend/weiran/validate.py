"""金标校验：用世界状态引擎重放场景，并逐条核对断言。

**这是离线校验，不含任何 LLM 调用、不含随机数。** 因此它可以在没有 API key
的情况下跑、可以进 CI、可以作为「评测确定性」那一半的证据。

它回答两个不同的问题，必须分开看：

  1. **轨迹吻合度** —— 模型跑出来的六维曲线，与金标（来自场景叙事）差多少？
     这不是「正确率」。金标是叙事，模型是假设，两者独立；
     吻合是证据，不吻合是发现。**任何情况下都不许调参去凑。**

  2. **断言是否成立** —— 10 条可机检的断言。这些才是能被打分的东西。

用法：
    python -m weiran.validate
    python -m weiran.validate --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from .world_state import (
    DIMENSIONS,
    DIMENSION_ZH,
    WorldState,
    WorldStateEngine,
    behaviors_from_material,
)

_MATERIAL_PHASE_PREFIX = {"01": "P1", "02": "P2", "03": "P3", "04": "P4", "05": "P5"}


def _phase_dts(phases: list[dict]) -> dict[str, float]:
    """由金标里的 day 字段推出各阶段的天数。

    第一个阶段的 dt 取 1 —— 报告挂网当天就开始发酵。
    """
    dts: dict[str, float] = {}
    prev_day = None
    for i, ph in enumerate(phases):
        day = ph.get("day", 0)
        dts[ph["id"]] = 1.0 if (i == 0 or prev_day is None) else float(day - prev_day)
        prev_day = day
    return dts


def collect_behaviors(scenario_dir: Path, ref: dict) -> dict[str, list[str]]:
    """从种子材料正文收集每个阶段的行为列表，并把金标里的 trigger 也算作一条行为。

    trigger 是这一阶段的**起因**（如「学工部劝删通知下发」），
    它本身就是一个事件，不纳入会低估该阶段的激励。
    """
    seed_dir = scenario_dir / "seed_materials"
    out: dict[str, list[str]] = {ph["id"]: [] for ph in ref.get("phases", [])}

    if seed_dir.is_dir():
        for md in sorted(seed_dir.glob("*.md")):
            pid = _MATERIAL_PHASE_PREFIX.get(md.name[:2])
            if pid and pid in out:
                out[pid].extend(behaviors_from_material(md.read_text(encoding="utf-8")))

    for ph in ref.get("phases", []):
        if ph.get("trigger"):
            out[ph["id"]].append(_classify_trigger(ph["trigger"]))
    return out


def _classify_trigger(text: str) -> str:
    from .world_state import classify

    return classify(text)


def mean_abs_error(actual: dict[str, float], expected: dict[str, float]) -> float:
    return statistics.fmean(
        abs(actual[d] - expected[d]) for d in DIMENSIONS if d in expected
    )


# ---------------------------------------------------------------------------
# 断言
# ---------------------------------------------------------------------------

def check_assertions(
    timeline: list, engine: WorldStateEngine, ref: dict
) -> list[dict]:
    """逐条核对金标里的断言。

    每条返回 {id, claim, passed, detail, note}。
    `note` 用于记录「这条为什么没过 / 为什么现在还不能判」——
    不做 silent skip。
    """
    by_phase = {r.phase_id: r.state_after.values for r in timeline}
    ids = [r.phase_id for r in timeline]
    results: list[dict] = []

    def add(aid: str, claim: str, passed, detail: str) -> None:
        results.append(
            {"id": aid, "claim": claim, "passed": bool(passed), "detail": detail}
        )

    # AS-1 信任单调下降
    seq = [by_phase[p]["trust"] for p in ("P1", "P2", "P3") if p in by_phase]
    ok = len(seq) == 3 and seq[0] > seq[1] > seq[2]
    add("AS-1", "trust 在 P1→P3 单调下降", ok,
        f"trust P1/P2/P3 = {[round(v, 3) for v in seq]}")

    # AS-2 / AS-4 峰值在 P3
    for aid, dim, label in (("AS-2", "attention", "attention"),
                            ("AS-4", "polarization", "polarization")):
        peak = max(ids, key=lambda p: by_phase[p][dim])
        vals = {p: round(by_phase[p][dim], 3) for p in ids}
        add(aid, f"{label} 在 P3 见顶", peak == "P3", f"峰值阶段={peak}，各阶段 {vals}")

    # AS-3 风险越阈值
    r3 = by_phase.get("P3", {}).get("risk", 0.0)
    add("AS-3", "risk 在 P3 越过 0.70", r3 >= engine.RISK_ALERT,
        f"risk(P3)={r3:.3f}，阈值 {engine.RISK_ALERT}")

    # AS-5 不对称性（核心）
    #
    # ⚠️ 这里有一个曾经把断言判错的坑，记下来：
    # 早期版本只判 `关注回落 > 3×信任恢复`。当信任在 P3→P5 继续**下跌**时，
    # 右边是负数，不等式恒成立 —— 断言以错误的方式通过了。
    # 「信任没恢复」被当成了「信任恢复得慢」，两件事完全不同。
    # 现在必须先要求信任确实回升（t5 > t3），再比较幅度。
    a3, a5 = by_phase["P3"]["attention"], by_phase["P5"]["attention"]
    t3, t5 = by_phase["P3"]["trust"], by_phase["P5"]["trust"]
    lhs, rhs = a3 - a5, 3 * (t5 - t3)
    recovered = t5 > t3
    add("AS-5", "关注回落显著大于信任恢复（核心断言）", recovered and lhs > rhs,
        f"关注回落 {lhs:.3f} vs 3×信任恢复 {rhs:.3f}；"
        f"信任是否回升={recovered}（P3={t3:.3f} → P5={t5:.3f}）"
        + ("" if recovered else " ← 未回升，不构成「恢复慢」，而是「没有恢复」"))

    # AS-6 不可逆
    t1 = by_phase["P1"]["trust"]
    add("AS-6", "P5 信任未回到 P1 基线（低于 0.10 以上）", t5 < t1 - 0.10,
        f"trust(P5)={t5:.3f} vs trust(P1)-0.10={t1 - 0.10:.3f}")

    # AS-7 压制事件被检出，且落在 P3
    sup_phases = [
        r.phase_id for r in timeline
        if any(e.kind == "suppression_detected" for e in r.events)
    ]
    add("AS-7", "劝删/压制行为被检出", bool(sup_phases),
        f"检出阶段={sup_phases or '无'}")

    # AS-8 预警是否早于 P3 触发
    early = []
    for r in timeline:
        if r.phase_id in ("P1", "P2") and r.events:
            early.extend(f"{r.phase_id}:{e.kind}" for e in r.events)
    add("AS-8", "风险预警在 P3 之前触发（「未然」的存在理由）", bool(early),
        f"P3 之前的检出事件={early or '无'}")

    # AS-9 需要仿真输出，离线阶段无法判定
    add("AS-9", "辅导员群体角色冲突指标为全群体最高", False,
        "需要多智能体仿真的 actions.jsonl，离线校验阶段无法判定")

    # AS-10 反事实
    cf = ref.get("_counterfactual")
    if cf:
        add("AS-10", "P2 主动公开可提升 P5 信任终值", cf["passed"], cf["detail"])
    else:
        add("AS-10", "P2 主动公开可提升 P5 信任终值", False, "未运行反事实推演")

    return results


# ---------------------------------------------------------------------------
# 反事实推演
# ---------------------------------------------------------------------------

def run_counterfactual(
    engine: WorldStateEngine,
    timeline: list,
    behaviors: dict[str, list[str]],
    dts: dict[str, float],
) -> dict:
    """假设在 P2 主动公开分专业数据，P5 的信任会是多少。

    做两个版本，都报出来 —— 因为「公开之后会怎样」本身就有模型假设成分，
    报一个数会假装它比实际更确定：

      A. 激励版：后续行为不变，只在 P2 叠加一次「公开」的激励。
         这是保守下界 —— 它假设公开了也照样会有人去劝删。

      B. 机制版：P2 公开之后，P3 不再出现压制行为。
         这是乐观上界 —— 它假设「主动说清」真的能消掉「需要压」的动机。

    **分叉点与干预落点必须都在 P2。** 这是修正过的一处错位：原先分叉取
    `timeline[p2_index].state_after`（P2 **结束**时的状态）、而 `later` 从
    `p2_index + 1` 起算，于是 `counterfactual()` 把干预施加到了 **P3** 那一步
    —— 断言文字说「在 P2 公开」，算出来的却是「在删帖争议已经烧起来之后才
    公开」。两个分支都照样「通过」，所以不看数字根本发现不了。
    实测差别：修正前 A 版 trust@P5 = 0.4368（Δ +0.1740），修正后 0.3847
    （Δ +0.1219）—— 错位把收益夸大了 43%。

    正确的口径是「P2 这个阶段自己承担这次干预」，所以 `later` 从 P2 起算、
    分叉状态取 P1 结束时的状态（P2 是首阶段时取基线）。
    """
    p2_index = next(i for i, r in enumerate(timeline) if r.phase_id == "P2")
    fork_state = (timeline[p2_index - 1].state_after
                  if p2_index > 0 else WorldState.baseline(engine.params))

    later = {r.phase_id: behaviors[r.phase_id] for r in timeline[p2_index:]}
    later_dts = {pid: dts[pid] for pid in later}

    branch_a, branch_b = engine.two_branch_futures(
        fork_state, behaviors_after=later, dts=later_dts,
    )

    actual_p5 = timeline[-1].state_after["trust"]
    a_p5 = branch_a["P5"]["trust"]
    b_p5 = branch_b["P5"]["trust"]

    return {
        "actual_trust_p5": round(actual_p5, 4),
        "branch_a_trust_p5": round(a_p5, 4),
        "branch_b_trust_p5": round(b_p5, 4),
        "passed": a_p5 > actual_p5,
        "detail": (
            f"实际 {actual_p5:.3f}；"
            f"A 激励版 {a_p5:.3f}（+{a_p5 - actual_p5:.3f}）；"
            f"B 机制版 {b_p5:.3f}（+{b_p5 - actual_p5:.3f}）"
        ),
        "branch_a": {p: s.as_dict() for p, s in branch_a.items()},
        "branch_b": {p: s.as_dict() for p, s in branch_b.items()},
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(scenario_dir: Path, *, verbose: bool = True) -> dict:
    ref = json.loads((scenario_dir / "reference_data.json").read_text(encoding="utf-8"))
    dts = _phase_dts(ref["phases"])
    behaviors = collect_behaviors(scenario_dir, ref)

    engine = WorldStateEngine()
    phases = [
        {"id": ph["id"], "dt": dts[ph["id"]], "behaviors": behaviors[ph["id"]]}
        for ph in ref["phases"]
    ]
    timeline = engine.run(phases)

    # 轨迹吻合度
    comparison = []
    for r, gold_phase in zip(timeline, ref["phases"]):
        actual = r.state_after.as_dict()
        expected = gold_phase["reference_shape"]
        comparison.append({
            "phase": r.phase_id,
            "day": gold_phase.get("day"),
            "dt": r.dt,
            "behavior_counts": r.behavior_counts,
            "actual": actual,
            "expected": expected,
            "mae": round(mean_abs_error(actual, expected), 4),
        })

    ref["_counterfactual"] = run_counterfactual(engine, timeline, behaviors, dts)
    assertions = check_assertions(timeline, engine, ref)

    overall_mae = statistics.fmean(c["mae"] for c in comparison)

    if verbose:
        _print_report(comparison, assertions, ref, overall_mae)

    return {
        "comparison": comparison,
        "assertions": assertions,
        "counterfactual": ref["_counterfactual"],
        "overall_mae": round(overall_mae, 4),
        "n_passed": sum(1 for a in assertions if a["passed"]),
        "n_total": len(assertions),
    }


def _print_report(comparison, assertions, ref, overall_mae) -> None:
    w = sys.stdout
    w.write("\n" + "=" * 78 + "\n")
    w.write("世界状态引擎 · 离线重放校验（无 LLM，无随机数）\n")
    w.write("=" * 78 + "\n\n")

    w.write("""\
⚠️  先读这一段，否则下面的数字会被误读。

本次重放**不是**预测精度评估，原因有两条，都必须明说：

  1. 输入是**场景描述**，不是 **agent 行为**。
     种子材料是「发生了什么」的叙述文本，而引擎的设计输入是每一轮
     智能体实际发出的动作。把分类过的散文喂进引擎是**范畴错误** ——
     正因为如此，本轮暴露出的是分类器与输入标度的问题，
     而不是模型的预测能力。

  2. 金标的状态值是**作者手写的预期**，不是观测值。
     因此「模型 vs 金标」的 MAE **不是分数，只是诊断量**。
     把人工写下的数字当作评分基准，正是本工作明确拒绝的做法
     （参见对「人工标注评测」的批判）。标准必须一致。

离线重放**能**做的三件事，也是它的全部价值：
  ① 证明引擎确定性、可离线运行、无隐藏的外部依赖；
  ② 检验模型的**结构性质**（信任的不对称、耦合方向、确定性）；
  ③ 暴露输入侧的问题 —— 本轮就查出两个真 bug（关键词黑洞、断言判据错误）。

真正的精度评估要等仿真产出带 agent 角色的 actions.jsonl 之后。
""")
    w.write("\n【一】轨迹对照：模型 vs 作者预期（诊断量，非分数）\n\n")
    header = f"{'阶段':<6}{'天数':>5}  " + "".join(
        f"{DIMENSION_ZH[d]:>6}" for d in DIMENSIONS
    ) + f"{'MAE':>8}\n"
    w.write(header)
    w.write("-" * len(header.rstrip()) + "\n")
    for c in comparison:
        row = f"{c['phase']:<6}{str(c['day']):>5}  "
        row += "".join(f"{c['actual'][d]:>6.2f}" for d in DIMENSIONS)
        row += f"{c['mae']:>8.3f}\n"
        w.write(row)
    w.write("\n")
    for c in comparison:
        exp = "".join(f"{c['expected'][d]:>6.2f}" for d in DIMENSIONS)
        w.write(f"{c['phase']:<6}{'金标':>5}  {exp}\n")
    w.write(f"\n整体 MAE = {overall_mae:.4f}   "
            f"← 诊断量。与作者手写预期的偏离程度，**不是模型准确率**。\n")

    w.write("\n各阶段行为构成：\n")
    for c in comparison:
        items = ", ".join(f"{k}×{v}" for k, v in sorted(c["behavior_counts"].items()))
        w.write(f"  {c['phase']}  {items or '（无）'}\n")

    w.write("\n【二】金标断言\n\n")
    for a in assertions:
        mark = "PASS" if a["passed"] else "FAIL"
        w.write(f"  [{mark}] {a['id']}  {a['claim']}\n")
        w.write(f"         {a['detail']}\n")

    cf = ref["_counterfactual"]
    w.write("\n【三】反事实：若在 P2 主动公开分专业数据\n\n")
    w.write(f"  实际路径   P5 信任 = {cf['actual_trust_p5']:.3f}\n")
    w.write(f"  A 激励版   P5 信任 = {cf['branch_a_trust_p5']:.3f}"
            f"   （后续行为不变，只叠加公开的激励）\n")
    w.write(f"  B 机制版   P5 信任 = {cf['branch_b_trust_p5']:.3f}"
            f"   （公开之后 P3 不再出现压制行为）\n")

    passed = sum(1 for a in assertions if a["passed"])
    w.write(f"\n断言：{passed}/{len(assertions)} 通过\n")
    w.write("=" * 78 + "\n\n")


def main(argv: list[str] | None = None) -> int:
    # 必须在任何 print 之前 —— 否则重定向输出时会先崩在打印上（见函数注释）。
    from .config import ensure_console_encoding
    ensure_console_encoding()

    ap = argparse.ArgumentParser(description="世界状态引擎金标校验")
    ap.add_argument("--scenario", default="benchmark/scenarios/employment_trust_crisis")
    ap.add_argument("--json", default=None, help="把完整结果写到该路径")
    args = ap.parse_args(argv)

    from .config import REPO_ROOT

    scenario = Path(args.scenario)
    if not scenario.is_absolute():
        scenario = REPO_ROOT / scenario

    result = run(scenario)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"结果已写入 {out}")

    # 退出码反映断言通过情况，便于 CI 使用
    return 0 if result["n_passed"] == result["n_total"] else 1


if __name__ == "__main__":
    sys.exit(main())
