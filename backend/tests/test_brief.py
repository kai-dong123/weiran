"""决策简报生成的测试。

这一组守的**不是**「简报好不好看」，而是三件不报错的失败：

  1. **窗口无声消失。** 未覆盖的窗口少打印几行，没有任何异常，
     但读者会以为那份简报覆盖了全部 5 个窗口。
  2. **口径声明被悄悄抹掉。** 压缩、不可复现、反事实非预测这三条写在
     渲染函数里，改模板时删掉一行不会有任何提示 —— 而它们删掉之后，
     剩下的数字就变成了「看起来像预测」的样子。
  3. **检查本身恒真。** `check_brief()` 若写的条件永远成立，后来的人会
     以为这里被守住了。所以下面每条检查都有一正一反两个用例。

所以测试**不依赖仓库里那份真实产出，也不依赖 API key**：`build_brief` 的
输入就是两个 dict，合成一份 run 即可。真实数据只用在最后一条冒烟上。

与另几套一样，**刻意不依赖 pytest**：普通 assert + 函数，
`python tests/test_brief.py` 直接跑，`python -m pytest tests/` 也能跑。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weiran import brief as B  # noqa: E402
from weiran.world_state import DIMENSIONS, WorldState, WorldStateEngine  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCENARIO = REPO / "benchmark" / "scenarios" / "employment_trust_crisis"
SIM = REPO / "data" / "simulation"


# ---------------------------------------------------------------------------
# 合成输入：要什么情形就造什么情形
# ---------------------------------------------------------------------------

def _phases(n: int = 5) -> list[dict]:
    days = [0, 2, 5, 8, 14]
    labels = ["引爆", "口径质疑", "删帖争议", "外部介入", "收束"]
    turns = [False, False, True, False, False]
    return [
        {"id": f"P{i + 1}", "day": days[i], "label": labels[i],
         "trigger": f"事件{i + 1}", "is_turning_point": turns[i]}
        for i in range(n)
    ]


def _gold(n: int = 5) -> dict:
    return {
        "phases": _phases(n),
        "decision_windows": [
            {"phase": f"P{i + 1}", "window": f"窗口{i + 1}",
             "actual_choice": f"做法{i + 1}", "is_turning_point": i == 2}
            for i in range(n)
        ],
        "turning_point": {
            "phase": "P3", "day": 5, "event": "劝删通知",
            "why_it_matters": "议题从数据真伪转移到是否压制。",
            "detectable_signal": "极化增速超过关注增速",
        },
        "metric_notes": {"scoring_determinism": "报均值与区间"},
        "core_claim": {"statement": "信任的衰减—恢复不对称。"},
        "dimensions": [{"key": d, "desc": f"{d} 的说明"} for d in DIMENSIONS],
        "gold_status": {"what_is_NOT_solid": "reference_shape 是手写预期"},
    }


def _rounds(n_rounds: int = 3, *, phases_on: bool = True, dpr: float = 7.0,
            behaviors: list[list[str]] | None = None,
            actions: list[list[dict]] | None = None) -> dict:
    """合成一份推演产出 —— **用真引擎跑出来**，不是手编数字。

    手编 `state` 的话，`replay_deviation` 会立刻报「这份产出不是本引擎产生的」，
    那反而说明检查有效 —— 但测试要测的是别的东西，所以这里如实跑一遍。

    `actions` 与 `behaviors` 是两件事，**故意分开传**：`behaviors` 是喂进引擎的
    信号（本例里手写），`actions` 是 OASIS 那一轮真实产生的动作（用来验
    「其中发声」那一列怎么算）。默认给空列表 —— 与老产出一致。
    """
    engine = WorldStateEngine()
    if behaviors is None:
        # **必须循环补齐到 `n_rounds` 条。** 原先写的是 `[...][:n_rounds]`，
        # 而默认列表只有 3 条 —— 于是 `_rounds(n_rounds=5)` **静默地只产出
        # 3 轮**，调用方拿到的轮数比要的少，没有任何提示。
        # 写新用例时踩到过一次（断言 5 行、实得 3 行）。夹具自己犯静默短缺，
        # 是这个项目最不该有的东西。
        _pattern = [
            ["disclosure", "amplification", "discussion", "discussion"],
            ["suppression", "amplification", "discussion", "discussion"],
            ["disclosure", "discussion"],
        ]
        behaviors = [_pattern[i % len(_pattern)] for i in range(n_rounds)]
    phase_ids = (["P1,P2", "P3,P4", "P5"] if n_rounds == 3 else
                 [f"P{i + 1}" for i in range(n_rounds)]) if phases_on else \
        ["" for _ in range(n_rounds)]

    cur = WorldState.baseline(engine.params)
    out = []
    for i, beh in enumerate(behaviors):
        res = engine.step(cur, beh, dt=dpr, phase_id=phase_ids[i])
        cur = res.state_after
        out.append({
            "index": i, "seconds": 2.0, "llm_seconds": 2.0, "calls": 1,
            "prompt_tokens": 1000, "completion_tokens": 100,
            "phase_id": phase_ids[i], "injected": {"a": "x"},
            "injection_sample": "", "behaviors": beh,
            "state": cur.as_dict(),
            "actions": (actions[i] if actions and i < len(actions) else []),
            "errors": 0,
        })
    return {
        "meta": {
            "rounds": n_rounds, "agents": 3, "platform": "twitter",
            "seed_text": "测试种子", "days_per_round": dpr,
            "compressed": dpr != 1.0, "comparable_to_round_day": dpr == 1.0,
            "phases_on": phases_on, "knowledge_on": True, "feedback_on": True,
            "world_state": True, "total_seconds": 6.0,
            "total_actions": sum(len(r["behaviors"]) for r in out),
            # 上下文压力。**与 simulate 现在真的写出去的字段一致** ——
            # 夹具少写字段的话，「字段缺失」那条降级会**永远**出现在每个用例里，
            # 而它本意是只在读旧产出时才出现。
            "context_limit": 16384, "truncations": 0,
            "truncation_unparsed": 0, "truncation_worst_dropped": 0,
        },
        "rounds": out,
        "_provenance": {"rounds_file": "<合成>", "rounds_sha256": "0" * 16},
    }


def _build(rounds_doc=None, gold=None, **kw):
    rd = rounds_doc if rounds_doc is not None else _rounds()
    gd = gold if gold is not None else _gold()
    gd = json.loads(json.dumps(gd))
    gd["_provenance"] = {"scenario_dir": "<合成>", "scenario_file": "<合成>",
                         "scenario_sha256": "0" * 16}
    return B.build_brief(rd, gd, **kw)


# ---------------------------------------------------------------------------
# 1. 确定性
# ---------------------------------------------------------------------------

def test_same_input_same_output():
    """同输入同输出 —— 这是本模块能进 `repro_check.py` 的前提。"""
    rd, gd = _rounds(), _gold()
    b1 = _build(json.loads(json.dumps(rd)), gd)
    b2 = _build(json.loads(json.dumps(rd)), gd)
    m1, m2 = B.render_markdown(b1), B.render_markdown(b2)
    assert m1 == m2, "同一份输入渲染出两份不同的简报 —— 简报层不是确定性的"
    assert (json.dumps(b1, sort_keys=True, ensure_ascii=False)
            == json.dumps(b2, sort_keys=True, ensure_ascii=False)), "结构化简报不一致"


def test_render_is_pure():
    """渲染不改结构化数据 —— 否则第二次渲染会读到被污染的状态。"""
    b = _build()
    before = json.dumps(b, sort_keys=True, ensure_ascii=False)
    B.render_markdown(b)
    B.render_markdown(b)
    assert json.dumps(b, sort_keys=True, ensure_ascii=False) == before, \
        "render_markdown 有副作用"


# ---------------------------------------------------------------------------
# 2. 未覆盖的窗口要逐个出声
# ---------------------------------------------------------------------------

def test_uncovered_windows_are_counted_one_by_one():
    """`phases_on=False` 时 5 个窗口全部未覆盖，且**逐个**标出。

    少打印几行不会有任何异常 —— 这正是最需要测试盯住的那种失败。
    """
    b = _build(_rounds(phases_on=False))
    assert b["coverage"]["n_uncovered"] == 5, \
        f"应 5 个窗口全部未覆盖，实得 {b['coverage']['n_uncovered']}"
    md = B.render_markdown(b)
    assert md.count("本次运行未覆盖") == 5, \
        f"未覆盖标记应出现 5 次，实得 {md.count('本次运行未覆盖')} 次"
    for i in range(5):
        assert f"窗口{i + 1}" in md, f"未覆盖的窗口{i + 1}整行消失了"
    assert B.check_brief(b, md) == [], B.check_brief(b, md)


def test_uncovered_rows_keep_their_window_text():
    """未覆盖的行仍要写出「当时金标是怎么做的」——不能只剩一个「未覆盖」。"""
    b = _build(_rounds(phases_on=False))
    md = B.render_markdown(b)
    for i in range(5):
        assert f"做法{i + 1}" in md, f"未覆盖行把金标的 actual_choice 弄丢了（{i + 1}）"


def test_zero_round_index_is_not_treated_as_uncovered():
    """第 0 轮的 `round_index` 是 `0`，真值判断会把它误判成「未覆盖」。

    这条是**实测踩到的**：第一版渲染器用 `if not r["round_index"]`，
    于是 P1/P2 两行被印成「未覆盖」，而机检当场报了
    「未覆盖窗口 0 个，但正文里只出现 2 次」。
    """
    b = _build()
    r0 = [r for r in b["windows"] if r["round_index"] == 0]
    assert len(r0) >= 1, "合成数据里应当有落在第 0 轮的窗口"
    md = B.render_markdown(b)
    assert "本次运行未覆盖" not in md, \
        "有窗口落在第 0 轮却被判成「未覆盖」——`0` 被当真值用了"


# ---------------------------------------------------------------------------
# 3. 口径声明：正向 + 反向
# ---------------------------------------------------------------------------

def test_caveats_present():
    b = _build()
    md = B.render_markdown(b)
    assert B.MARKERS["nonreproducible"] in md, "缺「不可端到端复现」"
    assert B.MARKERS["compressed"] in md, "缺「压缩模式不可比」"
    assert B.MARKERS["model_internal"] in md, "缺「模型内对照非预测」"
    assert B.MARKERS["assumption"] in md, "缺「干预口径为自设假设」"
    assert B.check_brief(b, md) == [], B.check_brief(b, md)


def test_caveat_check_is_not_vacuous():
    """**把每条声明分别抠掉，检查必须报出来。**

    一条恒真的检查比没有检查更危险 —— 它会让后来的人以为这里被守住了。
    """
    b = _build()
    md = B.render_markdown(b)
    for key, marker in B.MARKERS.items():
        if key == "degraded":
            continue                      # 它由 degradations 非空触发，另测
        broken = md.replace(marker, "")
        assert broken != md, f"标记 {marker} 在正文里根本不存在，无从抠除"
        bad = B.check_brief(b, broken)
        assert bad, f"抠掉 {marker} 之后 check_brief 竟然还通过 —— 这条检查是恒真的"


def test_compressed_declaration_follows_meta():
    """非压缩的产出**不该**出现压缩声明 —— 否则它成了一句口头禅。"""
    b = _build(_rounds(dpr=1.0))
    md = B.render_markdown(b)
    assert not b["meta"]["compressed"]
    assert B.MARKERS["compressed"] not in md, \
        "轮 = 天的产出也印了「压缩模式不可比」"

    b2 = _build(_rounds(dpr=7.0))
    assert B.MARKERS["compressed"] in B.render_markdown(b2), \
        "压缩产出没印压缩声明"


def test_context_declaration_distinguishes_three_states():
    """上下文压力那一句有**三种**状态，一个都不许合并。

    合并的代价是实打实的：把「没记录」并进「没截断」，等于把「不知道」
    伪装成「没问题」；而「没截断」正是本项目要求**主动说出来**的结论
    —— 它和「没在数」在文件里长得一模一样。
    """
    # 1) 记录到 0 次 —— 必须**主动断言**没截断
    b0 = _build()
    md0 = B.render_markdown(b0)
    assert "全程未发生记忆截断" in md0, "0 次的产出没有主动断言「没发生」"
    assert not [d for d in b0["degradations"] if d["kind"] == "truncated"], \
        "0 次截断不该报降级"

    # 2) 记录到 >0 次 —— 必须报降级，且说清丢了多少
    rd = _rounds()
    rd["meta"].update(truncations=3, truncation_worst_dropped=5000)
    b1 = _build(rd)
    md1 = B.render_markdown(b1)
    assert "全程发生 3 次记忆截断" in md1, md1[:400]
    assert "5000" in md1
    assert "全程未发生记忆截断" not in md1, \
        "有截断却仍印「未发生」—— 这是最坏的一种错"
    kinds = [d["kind"] for d in b1["degradations"]]
    assert "truncated" in kinds, f"有截断却没报降级：{kinds}"
    assert B.check_brief(b1, md1) == [], B.check_brief(b1, md1)

    # 3) **压根没有这个字段** —— 必须说「没测」，不许默默按 0 处理
    rd2 = _rounds()
    for k in ("context_limit", "truncations", "truncation_unparsed",
              "truncation_worst_dropped"):
        rd2["meta"].pop(k)
    b2 = _build(rd2)
    md2 = B.render_markdown(b2)
    assert "这一项本次没有记录" in md2, md2[:400]
    assert "全程未发生记忆截断" not in md2, \
        "没有记录却宣称「未发生」—— 把「不知道」当成了「没问题」"
    kinds2 = [d["kind"] for d in b2["degradations"]]
    assert "no_context_record" in kinds2, f"旧产出没报降级：{kinds2}"


def test_context_check_is_not_vacuous():
    """**一个永远印「未发生截断」的实现必须被拦下。**

    光检查标记在场是不够的 —— 那种实现照样能让「标记在场」这条通过，
    而它的全部效果就是把监测变成一句口头禅。
    """
    rd = _rounds()
    rd["meta"].update(truncations=7, truncation_worst_dropped=9000)
    b = _build(rd)
    md = B.render_markdown(b)

    # 把那段换成「没截断」的说法，其余原样 —— 机检必须报出来
    healthy = B._context_line({**rd["meta"], "truncations": 0,
                               "truncation_worst_dropped": 0})
    sick = B._context_line(rd["meta"])
    tampered = md.replace(sick, healthy)
    assert tampered != md, "正文里找不到那段，无从篡改"
    bad = B.check_brief(b, tampered)
    assert bad, "meta 说截断了 7 次，正文说「未发生」，机检竟然通过了"
    assert any("截断" in msg for msg in bad), bad


def test_context_declaration_is_enforced_by_marker_check():
    """标记必须无条件在场 —— 抠掉它，机检要报。"""
    b = _build()
    md = B.render_markdown(b)
    broken = md.replace(B.MARKERS["context"], "")
    assert broken != md, "标记在正文里根本不存在"
    assert B.check_brief(b, broken), "抠掉上下文标记之后机检竟然还通过"


def test_truncation_reports_how_little_survives_not_just_how_much_was_lost():
    """丢弃量必须换算成**保留比例** —— 否则读者判断不出量级。

    「单次最多丢弃 1,234,443 token」读起来只是个大数；
    配上「上限 16384」读者仍要自己做除法，而做不做、做对没有，无从保证。
    换成「能看到的记忆不超过自己的 1.31%」才有量级。
    """
    # 上界公式：after/(after+dropped) ≤ limit/(limit+dropped)
    assert abs(B._visible_fraction(16_384, 1_234_443) - 16_384 / 1_250_827) < 1e-12
    assert 0.0130 < B._visible_fraction(16_384, 1_234_443) < 0.0132

    # 丢得越多，能看到的越少 —— 单调，且恒在 (0, 1]
    fr = [B._visible_fraction(1000, d) for d in (0, 1, 100, 10_000)]
    assert fr[0] is None, "丢弃量为 0 时没有「最坏那一次」，不许编一个比例"
    assert all(0.0 < x <= 1.0 for x in fr[1:])
    assert fr[1] > fr[2] > fr[3]

    # 算不出来就返回空串 —— 不许退化成「100%」那种看着没问题的数
    assert B._visible_sentence(0, 500) == ""
    assert B._visible_sentence(1000, 0) == ""

    # 端到端：这句必须真的进正文，且两个出口（正文 + 降级表）都要有
    rd = _rounds()
    rd["meta"].update(context_limit=16_384, truncations=117,
                      truncation_worst_dropped=1_234_443)
    b = _build(rd)
    md = B.render_markdown(b)
    assert "1.31%" in md, "正文里没有把丢弃量换算成保留比例"
    assert "75 倍" in md, "没有把丢弃量与窗口上限的比例说出来"
    deg = [d for d in b["degradations"] if d["kind"] == "truncated"]
    assert deg and "1.31%" in deg[0]["text"], \
        f"正文有、降级表没有，两处口径不一致：{deg}"
    assert B.check_brief(b, md) == [], B.check_brief(b, md)


def test_visible_fraction_is_not_vacuous():
    """**一个恒返回 100% 的实现必须被拦下。**

    这是本项唯一的失效模式：公式写错一个符号，比例就恒等于 1，
    而正文照样印得出「不超过自己的 100.00%」—— 看着像句人话，
    实际把「丢了九成九」说成了「没丢」。
    """
    assert B._visible_fraction(16_384, 1_234_443) < 0.05, \
        "丢 123 万 token 还报出 ≥5% 的可见比例，公式反了"
    # 上限的代价随丢弃量单调收紧 —— 一个恒定实现过不了这一条
    assert B._visible_fraction(100, 900) < B._visible_fraction(100, 100)


def test_banned_words_rejected():
    """不得把「注入生效」说成「效果变好」。"""
    b = _build()
    md = B.render_markdown(b)
    # 禁用词表里存的是词干（「证明」），正文里可能是「证明了」——
    # 所以断言要看**是否被拦下**，不是比对字面。
    for word, stem in (("改善", "改善"), ("证明了", "证明"),
                       ("建议采取", "建议采取"), ("准确率", "准确率")):
        bad = B.check_brief(b, md + f"\n{word}\n")
        assert any(stem in x for x in bad), f"「{word}」没被拦下：{bad}"


def test_reference_shape_never_appears():
    """`reference_shape` 必须物理上进不来。"""
    gd = _gold()
    gd["phases"][0]["reference_shape"] = {"trust": 0.999, "attention": 0.888}
    b = _build(gold=gd)
    dump = json.dumps(b, ensure_ascii=False)
    assert "reference_shape" not in dump, "reference_shape 混进了结构化简报"
    assert "0.999" not in dump and "0.888" not in dump, \
        "reference_shape 的数值混进了结构化简报"
    assert "reference_shape" not in B.render_markdown(b)


def test_hardcoded_number_is_caught():
    """模板里手写死一个数 → 机检必须报。防的是「数已经不成立却没人发现」。"""
    b = _build()
    md = B.render_markdown(b)
    bad = B.check_brief(b, md + "\n该模型的信任终值为 0.9317。\n")
    assert any("0.9317" in x for x in bad), "手写死的数值没被拦下"


def test_traceable_numbers_pass():
    """反面：真实来源的数值不得被误报（旧版按字符串比，14 个数全误报）。"""
    b = _build()
    md = B.render_markdown(b)
    assert B.check_brief(b, md) == [], B.check_brief(b, md)


# ---------------------------------------------------------------------------
# 4. 分支的行为
# ---------------------------------------------------------------------------

def test_branches_respond_to_scale():
    """干预倍数变了，A 版的终值必须跟着变 —— 否则那两列是装饰。"""
    lo = _build(scale=0.5)
    hi = _build(scale=5.0)
    a_lo = [r["branch_a"] for r in lo["windows"] if r["round_index"] is not None]
    a_hi = [r["branch_a"] for r in hi["windows"] if r["round_index"] is not None]
    assert a_lo and a_hi
    assert a_lo != a_hi, f"K 从 0.5 调到 5.0，A 版终值纹丝不动：{a_lo}"


def test_zero_intervention_degenerates_to_paired_actual():
    """K=0 时 A 版必须**退化成配对实际那一条路** —— 这是「配对」的定义。

    注意 B 版**不跟着退化**，那是**对的**：B 的定义是把劝删这件事从后续
    行为里去掉，是机制改动，与干预强度无关。把 B 也判成「该退化」会把
    「机制版」误读成「强度更大的激励版」。所以这里分开测。
    """
    engine = WorldStateEngine()
    st = WorldState.baseline(engine.params)
    ba = {"R0": ["disclosure"], "R1": ["suppression"]}
    dts = {"R0": 7.0, "R1": 7.0}
    z = engine.counterfactual(st, intervention={}, behaviors_after=ba, dts=dts)

    a, b = engine.two_branch_futures(st, behaviors_after=ba, dts=dts, scale=0.0)
    for pid in ba:
        for d in DIMENSIONS:
            assert abs(a[pid][d] - z[pid][d]) < 1e-12, \
                f"K=0 时 A 版未退化成空跑路径（{pid}/{d}）"
    assert b["R1"]["trust"] > z["R1"]["trust"] + 1e-9, \
        "K=0 时 B 版也退化了 —— 机制版不该随干预强度归零"

    # `drop` 指向一个后续行为里根本不存在的类型时，B 才该退化成空跑
    a2, b2 = engine.two_branch_futures(
        st, behaviors_after=ba, dts=dts, scale=0.0, drop="不存在的行为")
    for pid in ba:
        for d in DIMENSIONS:
            assert abs(b2[pid][d] - z[pid][d]) < 1e-12, \
                f"drop 未命中时 B 版未退化成空跑路径（{pid}/{d}）"


def test_branch_b_removes_suppression():
    """B 版的定义：那件不可逆的事（劝删）不再发生 —— 与 A 版不是同一条路。"""
    engine = WorldStateEngine()
    st = WorldState.baseline(engine.params)
    ba = {"R0": ["discussion"], "R1": ["suppression", "amplification"]}
    dts = {"R0": 7.0, "R1": 7.0}
    a, b = engine.two_branch_futures(st, behaviors_after=ba, dts=dts, scale=2.5)
    assert b["R1"]["trust"] > a["R1"]["trust"], \
        f"去掉劝删之后信任反而没更高：A={a['R1']['trust']:.4f} B={b['R1']['trust']:.4f}"


def test_only_turning_point_gets_branch_b():
    """只有拐点窗口配 B 版。

    给每行都配 B 会让 4/5 行退化成 A==B，反而稀释 B 的信息量 ——
    那是「看起来更全」，不是更全。
    """
    b = _build()
    with_b = [r["phase"] for r in b["windows"] if r["branch_b"] is not None]
    assert with_b == ["P3"], f"配了 B 版的窗口应为拐点 P3，实得 {with_b}"


def test_paired_arm_reproduces_recorded_path():
    """「配对实际」应当复现记录里的路径 —— 这正说明它是个合格的对照。

    空干预的推演**就是**那次记录下来的推演：同样的 behaviors、同样的 dt，
    所以从任何分叉点起跑都会落回同一个终值。实得与产出终值一致
    （差 6e-5 内，来自产出只存 4 位小数）。
    """
    rd = _rounds()
    b = _build(rd)
    tail = rd["rounds"][-1]["state"]["trust"]
    for r in b["windows"]:
        if r["round_index"] is None:
            continue
        assert abs(r["actual_from_fork"] - tail) < 1e-3, \
            f"{r['phase']} 的配对空跑没落回记录终值（{r['actual_from_fork']} vs {tail}）"


def test_paired_arm_is_invariant_so_it_cannot_guard_the_fork():
    """**配对值对分叉点不敏感 —— 所以它不能用来验证分叉规则。**

    这条是一个真实的陷阱，记在这里免得下轮有人（包括我）去写一个
    「配对值 vs 终值」的检查来守分叉点：那个检查**恒真**。空干预的路径
    由 behaviors 唯一决定，分叉点只是这条路径上的一个中间点，从哪儿起跑
    都到同一个终点。

    真正编码分叉规则的是 **Δ**（A − 配对）。见下一条。
    """
    rd = _build(_rounds())
    tail = rd["rounds"][-1]["state"]["trust"]
    pairs = {r["actual_from_fork"] for r in rd["windows"] if r["round_index"] is not None}
    assert len(pairs) <= 2 and all(abs(p - tail) < 1e-3 for p in pairs), \
        f"配对值竟然随窗口变了（{pairs}）——那说明它不是空干预路径"


def test_delta_is_sensitive_to_fork():
    """**Δ 才是编码分叉规则的那个量**，且对错位一格极其敏感。

    实测（真实产出）：分叉写对了 Δ = +0.0457，写成「该轮结束之后」就变成
    +0.0819 —— 同一个窗口、同一份产出，收益被夸大 79%。这与 `validate.py`
    的 AS-10 是**同一类错位**（那边夸大了 43%），而且两个分支都照样「通过」，
    不看数字根本发现不了。所以这里是本模块最需要守住的一条。
    """
    rd = _rounds()
    engine = WorldStateEngine()
    dpr = rd["meta"]["days_per_round"]
    rounds = rd["rounds"]
    n = len(rounds)
    b = _build(rd)
    tail_key = f"R{n - 1}"

    for row in b["windows"]:
        r = row["round_index"]
        if r is None or r + 1 >= n:
            continue                      # 没有「更晚一格」可比的窗口跳过
        inter = {d: engine.excitation["disclosure"].get(d, 0.0) * 2.5
                 for d in DIMENSIONS}

        def delta(fork_round: int, start: int) -> float:
            st = (WorldState.baseline(engine.params) if fork_round == 0
                  else WorldState(dict(rounds[fork_round - 1]["state"])))
            fut = {f"R{k}": rounds[k]["behaviors"] for k in range(start, n)}
            fdt = {f"R{k}": dpr for k in range(start, n)}
            a = engine.counterfactual(st, intervention=inter,
                                      behaviors_after=fut, dts=fdt)[tail_key]["trust"]
            z = engine.counterfactual(st, intervention={},
                                      behaviors_after=fut, dts=fdt)[tail_key]["trust"]
            return a - z

        correct = delta(r, r)
        shifted = delta(r + 1, r + 1)
        assert abs(correct - row["delta_a"]) < 1e-3, \
            f"{row['phase']}：简报里的 Δ 与本测试独立算出的不一致"
        assert abs(correct - shifted) > 0.01, \
            (f"{row['phase']}：分叉错位一格后 Δ 几乎没变（{correct:+.4f} → "
             f"{shifted:+.4f}）—— 那 Δ 就没在编码分叉规则，本模块的核心假设失守")


# ---------------------------------------------------------------------------
# 5. 分叉点的边界
# ---------------------------------------------------------------------------

def test_fork_at_baseline_is_labeled():
    """窗口落在第 0 轮时没有前一轮，以基线为分叉点，且必须写明。"""
    b = _build()
    r0 = [r for r in b["windows"] if r["round_index"] == 0]
    assert r0, "合成数据里应当有落在第 0 轮的窗口"
    for r in r0:
        assert r["fork_day"] == 0, f"{r['phase']} 的分叉日应为 0，实得 {r['fork_day']}"
        assert "基线" in r["note"], f"{r['phase']} 以基线为分叉点却没标注"


def test_last_window_still_gets_a_branch():
    """最后一个阶段的窗口**仍要有分支**。

    若规则写成「分叉在该轮之后」，P5 就没有后续可推，只能报 0 ——
    而报 0 会被读成「在这个窗口做什么都没用」。规则必须是「分叉在该轮之前、
    该轮的行为照旧发生」，这样 5 个窗口一个都不落。
    """
    b = _build()
    last = b["windows"][-1]
    assert last["phase"] == "P5"
    assert last["round_index"] is not None, "最后的窗口没有对应轮次"
    assert last["branch_a"] is not None and last["delta_a"] is not None, \
        "最后的窗口没有分支 —— 规则退化成「分叉在该轮之后」了"


def test_unknown_phase_raises_loudly():
    """金标里出现没登记干预口径的阶段 → **抛**，不静默跳过。

    静默跳过会让那一行无声消失，读者以为简报覆盖了它。
    """
    gd = _gold()
    gd["decision_windows"].append(
        {"phase": "P9", "window": "凭空多出来的窗口",
         "actual_choice": "无", "is_turning_point": False})
    try:
        _build(gold=gd)
    except B.BriefError as exc:
        assert "P9" in str(exc) and "WINDOW_INTERVENTIONS" in str(exc), str(exc)
    else:
        raise AssertionError("未登记的阶段被静默跳过了")


def test_missing_phases_section_is_loud():
    """金标缺 phases 段 → 出声，不是渲染一份空简报。"""
    gd = _gold()
    gd["phases"] = []
    try:
        _build(gold=gd)
    except B.BriefError as exc:
        assert "阶段" in str(exc), str(exc)
    else:
        raise AssertionError("缺 phases 段时没有报错 —— 会交出一份标签全空的简报")


# ---------------------------------------------------------------------------
# 6. 金标里那些「读进来了但没渲染」的字段
# ---------------------------------------------------------------------------

def test_gold_labels_are_rendered():
    """`Phase.label` / `trigger` / `why_it_matters` 必须真的进正文。

    这几个字段是本轮才第一次有消费者的：光「读进对象了」不算数，
    必须出现在给读者看的那一份里。
    """
    b = _build()
    md = B.render_markdown(b)
    for lbl in ("引爆", "口径质疑", "删帖争议", "外部介入", "收束"):
        assert lbl in md, f"阶段标签「{lbl}」没进正文"
    assert "劝删通知" in md, "拐点事件没进正文"
    assert "议题从数据真伪转移到是否压制。" in md, "why_it_matters 没进正文"
    assert "信任的衰减—恢复不对称。" in md, "core_claim 没进正文"
    assert "极化增速超过关注增速" in md, "detectable_signal 没进正文"


def test_turning_point_reports_disagreement_not_agreement():
    """模型侧与叙事侧不一致时必须**并列报出**，不许合并成「拐点被复现」。

    实测：金标说拐点在 P3（第 5 天），而「Δ极化 > Δ关注」这个信号在
    **第 0 轮**就已经成立 —— 信号比金标早了整整两轮。把它写成「一致」
    是这类工作里最容易犯、也最不该犯的错。
    """
    b = _build()
    tp = b["turning_point"]
    assert tp["signal_agrees_with_gold"] is False, \
        "合成数据里信号在 R0 就成立、金标在 P3，不该判成一致"
    md = B.render_markdown(b)
    assert "与金标所指的阶段不一致" in md, "不一致没有被报出来"
    assert "拐点被复现" in md, "应当明说这不是「拐点被复现」"


def test_unimplemented_signal_is_declared_not_faked():
    """角色冲突指标没实现 → 列成「未实现的观测项」并说明原因。"""
    b = _build()
    names = [u["name"] for u in b["turning_point"]["unimplemented_signals"]]
    assert any("角色冲突" in n for n in names), "未实现的观测项没被列出"
    md = B.render_markdown(b)
    assert "未实现的观测项" in md
    assert "角色冲突" in md


def test_replay_deviation_is_reported():
    """重放偏差要报出来 —— 它超过 1e-4 说明产出不是本引擎产生的。"""
    b = _build()
    assert b["replay_max_deviation"] <= 1e-4, b["replay_max_deviation"]

    # 反面：把某一轮的 state 改掉，偏差必须被测出并成为一条降级项
    rd = _rounds()
    rd["rounds"][1]["state"]["trust"] += 0.05
    b2 = _build(rd)
    assert b2["replay_max_deviation"] > 1e-4, "被改过的产出没测出偏差"
    kinds = [d["kind"] for d in b2["degradations"]]
    assert "replay_mismatch" in kinds, f"偏差没成为降级项：{kinds}"


# ---------------------------------------------------------------------------
# 7. 成本的诚实性
# ---------------------------------------------------------------------------

def test_missing_failure_fields_are_not_shown_as_zero():
    """产出没记 `failures`/`slowest` 时，**不能印出「失败 0，最慢 0.0s」**。

    一个编出来的 0 比一个空值危险得多：它会让人以为「这次跑得很干净」，
    而真相是这两个字段根本没记。合成数据里没有这两列，正好覆盖旧产出。
    """
    b = _build()
    assert all(r["failures"] is None for r in b["rounds"]), \
        "合成数据本就不含这两个字段，不该有值"
    assert b["cost"]["failures_recorded"] is False
    summary = b["cost"]["summary"]
    assert "未记录" in summary, f"未记录的字段被印成了数字：{summary}"
    assert "失败 0" not in summary, f"印出了编造的「失败 0」：{summary}"
    md = B.render_markdown(b)
    assert "失败 0" not in md, "正文里出现了编造的「失败 0」"


def test_recorded_failure_fields_are_used():
    """反面：产出里**有**这两个字段时，必须用真实值，而不是继续报「未记录」。"""
    rd = _rounds()
    for i, r in enumerate(rd["rounds"]):
        r["failures"] = i
        r["slowest"] = 10.0 + i
    b = _build(rd)
    assert b["cost"]["failures_recorded"] is True
    assert b["cost"]["summary"].count("失败 3") == 1, b["cost"]["summary"]
    assert "12.0" in b["cost"]["summary"], b["cost"]["summary"]
    assert "未记录" not in b["cost"]["summary"]


# ---------------------------------------------------------------------------
# 8. 装载侧
# ---------------------------------------------------------------------------

def test_load_rounds_rejects_foreign_format():
    """早期版本曾把整个文件写成裸列表 —— 那种文件必须被拦下并说清楚。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "old.json"
        p.write_text(json.dumps([{"index": 0}]), encoding="utf-8")
        try:
            B.load_rounds(p)
        except B.BriefError as exc:
            assert "meta" in str(exc), str(exc)
        else:
            raise AssertionError("裸列表格式没被拦下")


def test_load_rounds_rejects_empty_rounds():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path( td) / "empty.json"
        p.write_text(json.dumps({"meta": {}, "rounds": []}), encoding="utf-8")
        try:
            B.load_rounds(p)
        except B.BriefError as exc:
            assert "空" in str(exc), str(exc)
        else:
            raise AssertionError("空 rounds 没被拦下")


def test_load_gold_strips_reference_shape():
    """装载时就剥掉，而不是靠「记得别引用」。"""
    gd = _gold()
    gd["phases"][0]["reference_shape"] = {"trust": 0.5}
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "reference_data.json").write_text(
            json.dumps(gd, ensure_ascii=False), encoding="utf-8")
        loaded = B.load_gold(d)
    assert all("reference_shape" not in ph for ph in loaded["phases"]), \
        "load_gold 没有剥掉 reference_shape"


# ---------------------------------------------------------------------------
# 9. 冒烟：真实的产出文件读得进
# ---------------------------------------------------------------------------

def test_real_run_smoke():
    """仓库里那份真实付费产出必须读得进、跑得通、五个窗口都算得出分支。

    **只断言结构，不断言数值** —— 数值由 brief.json 自己记录，
    在测试里再抄一份就多了一个会过期的真相来源。

    **同样不能断言「它是压缩的」**：第一版把 `compressed is True` 与
    `days_per_round == 7.0` 写死在这里，那是照着当时那份 3 轮产出抄的。
    仓库换成 15 轮（轮 = 天）的产出之后，测试立刻红了 —— 但红的原因是
    测试绑死了一份会变的产出，不是代码坏了。所以现在改成**从产出自身推导**：
    它声称什么，渲染出来的就必须与之一致。
    """
    path = SIM / "twitter_rounds.json"
    if not path.is_file():
        print("      （跳过：仓库里没有 data/simulation/twitter_rounds.json）")
        return
    rd = B.load_rounds(path)
    gd = B.load_gold(SCENARIO)
    b = B.build_brief(rd, gd, command="test")
    assert len(b["windows"]) == 5, len(b["windows"])
    assert b["coverage"]["n_uncovered"] == 0, \
        f"真实产出里 5 个窗口都该有对应轮次，实得未覆盖 {b['coverage']['n_uncovered']}"
    assert all(r["branch_a"] is not None for r in b["windows"]), "有窗口没算出分支"
    md = B.render_markdown(b)
    assert B.check_brief(b, md) == [], B.check_brief(b, md)

    # 自称与实际必须一致 —— **两个方向都测**，只测一个方向的话，
    # 「非压缩产出却印着压缩声明」这种错就漏掉了（那是真实发生过的缺陷）。
    if b["meta"]["compressed"]:
        assert B.MARKERS["compressed"] in md, "自称压缩，正文却没有压缩标记"
        assert "本轮运行是压缩的" in md, "自称压缩，时点偏差那句却没说是压缩的"
    else:
        assert B.MARKERS["compressed"] not in md, \
            "非压缩产出（轮 = 天）却印着压缩标记 —— 正文在自我否认"
        assert "本轮运行是压缩的" not in md, \
            "非压缩产出却印着「本轮运行是压缩的」—— 这一句曾经是写死的"
    # 「轮 = 天」时 `comparable_to_round_day` 必须为真，两者不能各说各话
    assert b["meta"]["comparable_to_round_day"] == (not b["meta"]["compressed"])


def test_real_run_output_is_current():
    """已经落盘的 `brief.md` 必须与当前代码渲染出来的一致。

    否则改了模板却忘了重跑，仓库里那份样例会静静地过期 ——
    而它正是别人看到的第一份东西。
    """
    md_path = SIM / "brief.md"
    js_path = SIM / "brief.json"
    if not md_path.is_file() or not js_path.is_file():
        print("      （跳过：尚未生成 data/simulation/brief.md）")
        return
    stored = json.loads(js_path.read_text(encoding="utf-8"))
    fresh_md = B.render_markdown(stored)
    assert fresh_md == md_path.read_text(encoding="utf-8"), \
        "data/simulation/brief.md 与 brief.json / 当前模板不一致 —— 需要重跑"

    rd = B.load_rounds(SIM / "twitter_rounds.json")
    gd = B.load_gold(SCENARIO)
    rebuilt = B.build_brief(rd, gd, command=stored["provenance"]["command"])
    assert B.render_markdown(rebuilt) == fresh_md, \
        "落盘的 brief.json 不是当前代码从这份产出重算出来的结果"


# ---------------------------------------------------------------------------
# 10. 证据量：每一轮的状态是拿几条行为算出来的
# ---------------------------------------------------------------------------
#
# 守的失效模式：**状态看着有两个小数位、很精确，背后其实只有两三条行为。**
# 27 agent × 15 轮实测里轮 0 有 21 条进入引擎、轮 14 只剩 3 条，而此前的产出
# 里没有任何字段能区分这两者 —— 落盘之后一模一样。

def _railed_doc(n_rounds: int = 5) -> dict:
    """造一份**有维度贴到夹逼边界**的产出。

    每轮 30 条同向行为、`dt=1`：trust 的半衰期 14 天，弛豫每轮只拉回 4.8%，
    而激励每轮推 0.45 —— 两三轮就顶到 0.0 被 `_clamp` 夹住。
    """
    return _rounds(n_rounds=n_rounds, dpr=1.0,
                   behaviors=[["suppression"] * 30] * n_rounds)


def test_volume_is_recomputed_from_the_engines_own_constant():
    """`volume` 必须是 `tanh(n / VOLUME_REF)` 的复算，**不是本文另定的阈值**。

    若哪天有人把 `VOLUME_REF` 改了而这里没跟着走，这条会红 ——
    那正是我们想知道的：引擎的声势尺度变了，简报里的解释也要重写。
    """
    rd = _rounds(n_rounds=5, dpr=1.0)
    b = _build(rd)
    for row in b["evidence"]["rows"]:
        want = round(math.tanh(row["n_behaviors"] / B.VOLUME_REF), 4)
        assert row["volume"] == want, (
            f"R{row['round']} 的 volume {row['volume']} != "
            f"tanh({row['n_behaviors']}/{B.VOLUME_REF}) = {want}")


def test_talk_actions_exclude_the_silent_ones():
    """「其中发声」那一列必须按 `simulate._SILENT_ACTIONS` 算。

    这条用真动作名单验，**不重抄一份** —— 简报与仿真说两套话，
    比说错一套更糟。
    """
    acts = [
        [{"action": "create_post"}, {"action": "refresh"},
         {"action": "do_nothing"}, {"action": "follow"}],
        [{"action": "quote_post"}, {"action": "like_post"},
         {"action": "mute"}, {"action": "repost"}],
        [{"action": "refresh"}],
        [{"action": "create_post"}, {"action": "create_post"}],
        [{"action": "do_nothing"}, {"action": "sign_up"}],
    ]
    rd = _rounds(n_rounds=5, dpr=1.0, actions=acts)
    rows = _build(rd)["evidence"]["rows"]
    assert [r["n_actions"] for r in rows] == [4, 4, 1, 2, 2]
    # like_post / repost / quote_post 都算发声，refresh/do_nothing/follow/mute/sign_up 不算
    assert [r["n_talk_actions"] for r in rows] == [1, 3, 0, 2, 0]


def _actions_for(n_rounds: int, participants: list[int],
                 speakers: list[int]) -> list[list[dict]]:
    """造每轮动作：`participants[i]` 个 agent 各出一条动作，其中
    `speakers[i]` 个出的是发声动作、其余出 refresh。"""
    out = []
    for i in range(n_rounds):
        row = []
        for u in range(participants[i]):
            row.append({"user_id": u,
                        "action": "create_post" if u < speakers[i] else "refresh"})
        out.append(row)
    return out


def test_presence_and_speech_are_two_counts_that_can_diverge():
    """**在场人数与发声人数必须分开记，而且要能看出它们不同向。**

    这是一次实测逼出来的口径：27 agent × 15 轮里发声条数从 21 掉到 9，
    而**每一轮都有全部 27 个 agent 在动作** —— 只看条数会把它读成
    「人少了 / 集体安静了」。两个症状的模型含义不同，简报里不能合并成一个数。
    """
    acts = _actions_for(5, [3, 3, 3, 3, 3], [3, 2, 2, 1, 1])
    rd = _rounds(n_rounds=5, dpr=1.0, actions=acts)
    ev = _build(rd)["evidence"]
    assert [r["n_participants"] for r in ev["rows"]] == [3, 3, 3, 3, 3]
    assert [r["n_speakers"] for r in ev["rows"]] == [3, 2, 2, 1, 1]
    p = ev["presence"]
    assert p["participants_constant"] is True
    assert p["speakers_fell"] is True
    assert p["speech_dropped_faster"] is True
    assert (p["speakers_first"], p["speakers_last"]) == (3, 1)

    sent = B._evidence_sentence(ev)
    assert "在场人数" in sent and "发声人数" in sent
    assert "不是在场的人" in sent, "两个数不同向时，正文必须把这一点说出来"


def test_presence_is_silent_when_actions_were_not_recorded():
    """**反方向：产出里没有动作记录时，一个字都不许编。**

    `n_participants` 全是 0 有两种来源 —— 老产出没记 `actions`，
    或者真的没人动作。这两种在数据里长得一样，所以这里**不猜**：
    `presence` 为 `None`，正文不提这件事。报一个编出来的「在场 0 人」
    比什么都不说更坏。
    """
    b = _build(_rounds(n_rounds=5, dpr=1.0))     # 夹具默认 actions=[]
    ev = b["evidence"]
    assert all(r["n_participants"] == 0 for r in ev["rows"])
    assert ev["presence"] is None
    assert "在场人数" not in B._evidence_sentence(ev)


def test_falling_volume_alone_does_not_claim_the_wrong_cause():
    """**反方向：在场人数也在掉的时候，不许说「不是在场的人」。**

    上面那段话只在两个数**不同向**时成立。若在场人数与发声人数一起掉，
    那确实是「人退出」—— 此时再念那句解释，就是拿一个模板去覆盖事实。
    """
    # 在场人数与发声人数一起掉（≤ 夹具 meta 里的 3 个 agent，别造出
    # 「4 个 agent 在 3 个 agent 的推演里动作」这种自相矛盾的夹具）。
    acts = _actions_for(5, [3, 3, 2, 2, 1], [3, 3, 2, 2, 1])
    rd = _rounds(n_rounds=5, dpr=1.0, actions=acts)
    ev = _build(rd)["evidence"]
    p = ev["presence"]
    assert p["participants_constant"] is False
    assert p["speakers_fell"] is True          # 发声确实在掉……
    assert p["speech_dropped_faster"] is False  # ……但在场掉得一样快
    assert "1~3" in B._evidence_sentence(ev)
    assert "不是在场的人" not in B._evidence_sentence(ev)


def test_a_railed_round_is_reported_and_named():
    """贴界是 `_clamp` 的产物，不是读数 —— 必须报出来，**而且要点名哪一轮**。

    只印一句「本次有维度贴界」是不够的：读者要能回到表里找到是哪几轮。
    """
    b = _build(_railed_doc())
    ev = b["evidence"]
    assert ev["railed_rounds"], "这份夹具本该把维度推到边界上，却一轮都没贴界"
    md = B.render_markdown(b)
    assert "贴到上下界" in md
    for r in ev["railed_rounds"]:
        assert f"R{r}" in md, f"R{r} 贴界了，正文里却没点名"
    assert any(d["kind"] == "railed" for d in b["degradations"]), \
        "贴界没进降级项 —— 降级项是读者最先看的地方"


def test_a_clean_run_does_not_cry_wolf():
    """**反方向同样要测。** 一份没贴界、也没有零行为轮的产出，
    不许出现「⚠️」与「贴到上下界」。一个永远喊狼来了的实现，
    与一个永远报平安的实现，是同一种失效。"""
    b = _build(_rounds(n_rounds=5, dpr=1.0))
    ev = b["evidence"]
    assert not ev["railed_rounds"] and not ev["no_evidence_rounds"]
    assert not any(d["kind"] in ("railed", "no_evidence_round")
                   for d in b["degradations"])
    sent = B._evidence_sentence(ev)
    assert "⚠️" not in sent
    assert "贴到上下界" not in B.render_markdown(b)


def test_a_zero_behavior_round_under_a_window_is_reported():
    """**零条行为 ⟹ 那一轮没有观测。** 引擎那一步的激励项恒为 0
    （`tanh(0/8)`），状态只由弛豫与耦合外推 —— 落在它上面的窗口数字
    不是「测出来的」。这条判据没有自由参数：0 就是 0。
    """
    rd = _rounds(n_rounds=5, dpr=1.0,
                 behaviors=[["disclosure", "discussion"], [],
                            ["discussion"], ["disclosure"], ["discussion"]])
    b = _build(rd)
    assert b["evidence"]["no_evidence_rounds"] == [1]
    assert any(d["kind"] == "no_evidence_round" for d in b["degradations"])
    md = B.render_markdown(b)
    assert "零条行为" in md and "R1" in md


def test_evidence_check_is_not_vacuous():
    """机检必须**真的能红**。把证据量那一节从正文里删掉再跑一次 —— 若仍然通过，
    这条检查就是恒真的，等于没有。"""
    b = _build(_railed_doc())
    md = B.render_markdown(b)
    assert B.check_brief(b, md) == [], B.check_brief(b, md)

    # 逐块破坏，每一块都必须被逮到。
    for label, broken in (
        ("整节删掉", md.replace(B.MARKERS["evidence"], "「证据量」")),
        ("贴界那句删掉", md.replace("贴到上下界", "（此处应有贴界说明）")),
        ("R 行删掉", md.replace("| R2 |", "| 第2轮 |")),
    ):
        assert B.check_brief(b, broken) != [], f"删掉「{label}」之后机检仍然通过了"


def test_evidence_section_is_unconditional():
    """标记无条件出现 —— 与上下文压力同理：**「证据是够的」也要主动说出来**，
    否则它与「没在数」在文件里长得一模一样。"""
    rd = _rounds()
    b = _build(rd)
    md = B.render_markdown(b)
    assert B.MARKERS["evidence"] in md
    assert B.MARKERS["evidence"] in b["markers"].values()


# ---------------------------------------------------------------------------
# 直接执行时的兜底 runner（不依赖 pytest）

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
