"""世界状态引擎测试。

**这里测的是模型的结构性质，不是它与金标的吻合度。**

这个区分很重要。金标里的六维数值是作者手写的预期，把「模型输出 vs 手写数字」
的吻合度当测试，就会把测试变成拟合的帮凶——哪天曲线对不上了，
改的会是测试或参数，而不是去问模型哪里错了。

结构性质不一样：它们是模型**声称**具备的机制，可以被独立地证实或证伪。
例如「信任的恢复比关注慢」——这可以直接构造一个受控输入来检验，
与任何手写数字无关。

    python backend/tests/test_world_state.py
    pytest backend/tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weiran.world_state import (  # noqa: E402
    ASYMMETRY,
    DIMENSIONS,
    EXCITATION,
    REF_WEIGHT,
    VOLUME_REF,
    WorldState,
    WorldStateEngine,
    behaviors_from_material,
    classify,
    sections_from_material,
)


def _engine() -> WorldStateEngine:
    return WorldStateEngine()


def _baseline() -> WorldState:
    return WorldState.baseline()


# -- 确定性 ----------------------------------------------------------------


def test_deterministic():
    """同样的输入必得同样的输出。这是「评测确定性」那一半的地基。"""
    e = _engine()
    phases = [
        {"id": "P1", "dt": 2.0, "behaviors": ["suppression"] * 3 + ["discussion"]},
    ]
    a = e.run(phases)[0].state_after.as_dict()
    b = e.run(phases)[0].state_after.as_dict()
    assert a == b, f"两次运行结果不一致：{a} vs {b}"


def test_values_stay_bounded():
    """无论输入多极端，六个维度都必须留在 [0, 1]。"""
    e = _engine()
    cur = _baseline()
    for behaviors in (
        ["suppression"] * 200,
        ["amplification"] * 200,
        ["disclosure"] * 200,
        ["testimony"] * 200,
    ):
        r = e.step(cur, behaviors, dt=30.0, phase_id="X")
        cur = r.state_after
        for d in DIMENSIONS:
            assert 0.0 <= cur[d] <= 1.0, f"{d} 越界：{cur[d]}"


# -- 核心机制：不对称 ------------------------------------------------------


def test_trust_recovers_slower_than_attention():
    """★ 核心机制。给两个维度同样大小的一次扰动，看谁先回到基线。

    这是模型的全部主张所在。构造上完全受控：
    直接把状态推离基线同样的距离，不给任何新激励，只让它自然回落。

    **一次只扰动一个维度**。早期版本同时扰动关注、恐慌、信任三者，
    结果恐慌的残留比按半衰期算出来的高出一倍——因为信任被压低之后，
    通过 trust→panic 的耦合把恐慌托住了。那个现象本身是真实的、有意义的
    （不信任会维持焦虑），但它会污染对「半衰期」这一性质的测量，
    所以在这里把它隔离掉，另行单独测（见下一条测试）。

    断言用**残留比例**而不是绝对量：绝对量的阈值依赖位移大小，
    换个位移就得重算；比例是维度自身的性质，更稳也更好解释。
    """
    e = _engine()
    params = e.params

    def keep_ratio(dim: str, displacement: float) -> float:
        probe = WorldState.baseline()
        probe.values[dim] = params[dim].baseline + displacement
        after = e.step(probe, [], dt=6.0, phase_id="cool").state_after
        return abs(after[dim] - params[dim].baseline) / abs(displacement)

    att_keep = keep_ratio("attention", +0.30)
    panic_keep = keep_ratio("panic", +0.30)
    trust_keep = keep_ratio("trust", -0.30)

    assert att_keep < 0.10, f"关注 6 天后应基本回落，实际残留 {att_keep:.1%}"
    assert panic_keep < 0.20, f"恐慌 6 天后应大幅回落，实际残留 {panic_keep:.1%}"
    assert trust_keep > 0.60, (
        f"信任 6 天后应几乎没恢复（应残留 >60%），实际残留 {trust_keep:.1%} —— "
        "若此断言失败，说明「信任恢复慢」这一核心主张不再成立"
    )
    assert trust_keep > 5 * att_keep, (
        f"信任残留({trust_keep:.1%})应远大于关注残留({att_keep:.1%})"
    )


def test_low_trust_sustains_panic():
    """耦合的副作用：信任被压低会**托住**恐慌，使它回落得比半衰期暗示的慢。

    这不是 bug，是这个模型想表达的东西之一：焦虑不只来自事件本身，
    也来自「我不再相信官方的说法」。单独测出来，免得日后有人把它当噪声修掉。
    """
    e = _engine()
    params = e.params

    alone = WorldState.baseline()
    alone.values["panic"] = params["panic"].baseline + 0.30
    panic_alone = e.step(alone, [], dt=6.0).state_after["panic"]

    with_distrust = WorldState.baseline()
    with_distrust.values["panic"] = params["panic"].baseline + 0.30
    with_distrust.values["trust"] = params["trust"].baseline - 0.30
    panic_with = e.step(with_distrust, [], dt=6.0).state_after["panic"]

    assert panic_with > panic_alone, (
        f"低信任应使恐慌回落更慢：单独 {panic_alone:.3f} vs 伴随低信任 {panic_with:.3f}"
    )


def test_favourable_excitation_is_discounted():
    """有利方向的激励要打折：道歉一次抵不上造一次伤害。

    直接对比同一个维度在正负两个方向上的响应幅度。
    """
    e = _engine()
    base = e.params["trust"].baseline

    # 纯 disclosure —— 有利于信任
    up = e.step(_baseline(), ["disclosure"] * 6, dt=1.0).state_after["trust"] - base
    # 纯 suppression —— 不利于信任
    down = e.step(_baseline(), ["suppression"] * 6, dt=1.0).state_after["trust"] - base

    assert up > 0, "披露应提升信任"
    assert down < 0, "压制应压低信任"
    assert abs(down) > abs(up) / ASYMMETRY["trust"], (
        f"负向幅度({abs(down):.4f})应显著大于正向({abs(up):.4f})"
    )


def test_suppression_hurts_trust_most():
    """在所有行为类型里，压制对信任的打击最大。

    这是 P3 成为转折点的机制依据：不是「出事」伤信任最深，
    而是「出事之后去捂」伤信任最深。
    """
    e = _engine()
    base = e.params["trust"].baseline
    impacts = {}
    for kind in EXCITATION:
        after = e.step(_baseline(), [kind] * 6, dt=1.0).state_after
        impacts[kind] = after["trust"] - base

    worst = min(impacts, key=lambda k: impacts[k])
    assert worst == "suppression", (
        f"对信任打击最大的应为 suppression，实际是 {worst}；全部：{impacts}"
    )


# -- 耦合方向 --------------------------------------------------------------


def test_coupling_directions():
    """耦合的方向必须符合机制，不能相反。"""
    e = _engine()
    p = e.params

    probe = WorldState.baseline()
    probe.values["polarization"] = p["polarization"].baseline + 0.4
    after = e.step(probe, [], dt=0.0, phase_id="t0").state_after
    # dt=0 时弛豫与激励都不起作用，只剩耦合
    assert after["stability"] < probe["stability"], "极化上升应压低稳定"
    assert after["trust"] < probe["trust"], "极化上升应压低信任"

    probe2 = WorldState.baseline()
    probe2.values["trust"] = p["trust"].baseline + 0.2
    after2 = e.step(probe2, [], dt=0.0, phase_id="t1").state_after
    assert after2["panic"] < probe2["panic"], "信任上升应压低恐慌"
    assert after2["risk"] < probe2["risk"], "信任上升应压低风险"


# -- 参数一致性 ------------------------------------------------------------


def test_ref_weight_matches_excitation_table():
    """REF_WEIGHT 必须等于激励表里单项权重绝对值的上界。

    它是归一化分母。若有人往表里加了一个更大的权重却忘了同步，
    激励会被静默削顶 —— 这类错误不写断言根本发现不了。
    """
    biggest = max(
        abs(w) for targets in EXCITATION.values() for w in targets.values()
    )
    assert abs(biggest - REF_WEIGHT) < 1e-9, (
        f"激励表最大权重为 {biggest}，但 REF_WEIGHT={REF_WEIGHT}，二者必须一致"
    )


def test_half_life_ordering_matches_mechanism():
    """半衰期的相对大小必须符合机制陈述。"""
    e = _engine()
    p = e.params
    assert p["attention"].half_life < p["panic"].half_life < p["trust"].half_life, (
        "关注应比恐慌快，恐慌应比信任快"
    )
    assert p["trust"].half_life < p["polarization"].half_life, "信任恢复应快于极化消退"
    assert VOLUME_REF > 0


# -- 行为分类 --------------------------------------------------------------


def test_classify_priority_suppression_wins():
    """优先级：压制信号最强，不应被误判成普通讨论。

    用例取自真实种子材料。**注意关键词分类的已知局限**：
    它能识别「我是 22 届的」，识别不了「我说的是假的吗」——
    后者语义上是现身说法，但没有命中任何关键词。
    这类漏检是关键词方法的固有代价，靠加词表治不好（加了就会开始误召回）。
    记录在此，不假装没有。
    """
    assert classify("辅导员要求同学删除帖子") == "suppression"
    assert classify("学校公布了分专业数据") == "disclosure"
    assert classify("我是被算进去的那部分") == "testimony"
    assert classify("今天天气不错") == "discussion"


def test_sections_strip_metadata():
    """回归：元数据行（来源 / 表格）不得进入分类。

    早期版本没剥元数据，每份材料开头的「公开性 | 通稿公开」都命中
    `disclosure` 的「公开」，分类器退化成格式识别器。
    """
    text = (
        "# 种子材料 99 · 测试\n\n"
        "| 项 | 值 |\n|---|---|\n"
        "| 公开性 | 通稿公开；论坛为半公开 |\n\n"
        "## 材料 99-A\n"
        "> 来源：某处官网\n"
        "辅导员要求同学删除帖子\n"
    )
    sections = sections_from_material(text)
    assert len(sections) == 1, f"应只切出 1 节，实际 {len(sections)}"
    assert "公开性" not in sections[0], "元数据表行未被剥掉"
    assert "来源" not in sections[0], "来源行未被剥掉"
    assert classify(sections[0]) == "suppression"


def test_disclosure_is_not_a_magnet():
    """回归：`disclosure` 曾因词表含「公开/说明/发布」而吞噬一切。

    一个关键词类如果命中率过高，它就不是在分类，只是在计数。
    在真实种子材料上做整体统计来守住这一点。
    """
    repo_root = Path(__file__).resolve().parents[2]
    seed = repo_root / "benchmark/scenarios/employment_trust_crisis/seed_materials"
    if not seed.is_dir():
        return

    counts: dict[str, int] = {}
    for md in sorted(seed.glob("*.md")):
        for kind in behaviors_from_material(md.read_text(encoding="utf-8")):
            counts[kind] = counts.get(kind, 0) + 1

    total = sum(counts.values())
    assert total > 0, "种子材料未产出任何行为"
    share = counts.get("disclosure", 0) / total
    assert share < 0.30, (
        f"disclosure 占 {share:.1%}，过高 —— 词表很可能又退化成泛用词了。"
        f"当前分布：{counts}"
    )


def test_suppression_detected_in_turning_point_material():
    """03 是转折点材料，其中的压制行为必须被检出。

    这是一条内容层面的断言：模型再不精确，也不该把转折点漏掉。
    """
    repo_root = Path(__file__).resolve().parents[2]
    md = (
        repo_root
        / "benchmark/scenarios/employment_trust_crisis/seed_materials/03_删帖争议.md"
    )
    if not md.is_file():
        return
    kinds = behaviors_from_material(md.read_text(encoding="utf-8"))
    assert kinds.count("suppression") >= 2, (
        f"03 里的压制行为应被多次检出，实际 {kinds}"
    )


# -- 事件检测 --------------------------------------------------------------


def test_detects_risk_crossing():
    e = _engine()
    cur = _baseline()
    found = None
    for i in range(6):
        r = e.step(cur, ["suppression"] * 5 + ["amplification"] * 3,
                   dt=3.0, phase_id=f"P{i}")
        cur = r.state_after
        if any(ev.kind == "risk_alert" for ev in r.events):
            found = i
            break
    assert found is not None, "连续压制+放大六轮仍未触发风险预警，阈值或增益有问题"


def test_detects_suppression_event():
    e = _engine()
    r = e.step(_baseline(), ["suppression"], dt=1.0, phase_id="P3")
    kinds = {ev.kind for ev in r.events}
    assert "suppression_detected" in kinds


def test_no_events_on_quiet_step():
    """平静的一步不该报出任何事件 —— 否则告警会失去意义。"""
    e = _engine()
    r = e.step(_baseline(), [], dt=1.0, phase_id="P0")
    assert r.events == [], f"平静步报出了事件：{[ev.kind for ev in r.events]}"


# -- 反事实 ----------------------------------------------------------------


def test_counterfactual_disclosure_raises_trust():
    """在危机早期主动披露，信任终值应高于不披露。

    这是「未然」的产出形态：不是报告现状，而是比较两条路。
    """
    e = _engine()
    cur = _baseline()
    cur = e.step(cur, ["suppression"] * 3, dt=3.0, phase_id="P2").state_after

    later = {"P3": ["suppression"] * 3, "P4": ["suppression"] * 2}
    dts = {"P3": 3.0, "P4": 6.0}

    no_action = e.counterfactual(cur, intervention={}, behaviors_after=later, dts=dts)
    disclosed = e.counterfactual(
        cur,
        intervention={d: e.excitation["disclosure"].get(d, 0.0) * 3 for d in DIMENSIONS},
        behaviors_after=later,
        dts=dts,
    )
    assert disclosed["P4"]["trust"] > no_action["P4"]["trust"], (
        "主动披露应提升终止信任，否则干预建议功能没有意义"
    )


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
