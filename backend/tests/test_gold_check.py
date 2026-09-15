"""金标对照表的测试。

这一组守的**不是**「表里那几条结论对不对」，而是五件不会报错的失败：

  1. **断言静默消失。** 金标里加了第 11 条，表里还是 10 条 —— 没有任何异常，
     而读的人会以为「全都核过了」。
  2. **不可判定被折成通过或失败。** 缺数据的条目折成「通过」是最省事的写法，
     也是把一张能核对的表变成一句表态的最快方式。
  3. **贴界被当成正常读数。** 三条断言踩在 `_clamp` 的边界上，不加标记的话，
     一张表会同时给出「通过」和「否决」两种毫无内容的结论 —— 而它们看起来
     与真结论一模一样。
  4. **判据恒真而没人发现。** AS-5 的右端在信任下降时是负数，判据于是自动
     成立；一条恒真的判据比没有判据更危险，因为它会通过。
  5. **出处被代写。** 表头那句「这不是分数」是金标自己说的，本表只是引用；
     金标没写就必须抛，不能替它说 —— 替它说就是替它背书。

第 4 条有一个**防恒真用例**：把 P5 那一轮改成不贴界（真数据上手工改，
不是另造一份合成输入），AS-5 必须从「通过」变成「否决」。做不到这一点，
就说明「不采信」那套逻辑是装饰 —— 一个永远返回「不采信」的实现也能过。

与另几套一样，**刻意不依赖 pytest**：普通 assert + 函数。
数据一律取**入库的真产出**（27 agent × 15 轮）与**真金标**，不另造合成输入：
合成数据永远比真数据整齐，而这张表全部的意义就是能否吃下真数据。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weiran import gold_check as G  # noqa: E402
from weiran.brief import BriefError  # noqa: E402
from weiran.config import REPO_ROOT  # noqa: E402

ROUNDS_FILE = G.DEFAULT_ROUNDS_FILE
SCENARIO_DIR = REPO_ROOT / G.DEFAULT_SCENARIO

_cache: dict = {}


def _load() -> tuple[dict, dict, dict]:
    """真产出 + 真金标 + 真简报。跑一次缓存起来（`build_brief` 约 0.03s）。"""
    if not _cache:
        rd = G.load_rounds(ROUNDS_FILE)
        # `evaluate` 吃的是**按 id 索引过的**金标 —— `build_gold_check` 在入口
        # 做这一步。测 `evaluate` 时得自己补上，否则测到的是另一个入口。
        gold = G.load_gold(SCENARIO_DIR)
        gold["assertions_by_id"] = {a["id"]: a for a in gold["assertions"]}
        _cache["rd"], _cache["gold"] = rd, gold
        _cache["brief"] = G.build_brief(rd, gold, command=G.DEFAULT_COMMAND)
    return _cache["rd"], _cache["gold"], _cache["brief"]


def _table() -> tuple[dict, dict]:
    """真产出上跑一遍求值。返回 (表, 金标)。"""
    rd, gold, brief = _load()
    return G.evaluate(copy.deepcopy(rd), copy.deepcopy(gold),
                      copy.deepcopy(brief)), gold


def _by_id(doc: dict, aid: str) -> dict:
    return next(r for r in doc["assertions"] if r["id"] == aid)


def _full() -> dict:
    """`build_gold_check` 的完整产物（含 summary / provenance / not_a_score）。"""
    rd, gold, _ = _load()
    return G.build_gold_check(rd, gold, command=G.DEFAULT_COMMAND)


def _unrailed() -> dict:
    """**防恒真用例的数据**：把 P5 那一轮从贴界上摘下来。

    只改**落盘的那条曲线**（`rounds[14].state`），不改行为序列 —— 所以这份
    产出与自己的重放对不上（`replay_max_deviation` 会从 4.9e-05 涨到 0.55），
    这是**手工改曲线的预期代价**，本条测试不碰重放那一层。
    """
    rd, gold, _ = _load()
    rd = copy.deepcopy(rd)
    st = rd["rounds"][14]["state"]
    rd["rounds"][14]["state"] = {
        **st,
        # 信任抬到 P3（0.4448）之上 → AS-5 的右端变正，判据不再恒真；
        # 但仍低于 P1 − 0.10（0.5843）→ AS-6 该通过。
        "trust": 0.55, "polarization": 0.70, "risk": 0.60, "stability": 0.30,
    }
    brief = G.build_brief(rd, gold, command=G.DEFAULT_COMMAND)
    return G.evaluate(rd, copy.deepcopy(gold), brief)


# ---------------------------------------------------------------------------
# 1. 一条都不能少
# ---------------------------------------------------------------------------

def test_every_gold_assertion_appears():
    """金标里几条，表里就几条 —— 且 id 与顺序都对上。

    这条防的是「断言静默消失」：求值函数漏接一条新断言时，最省事的写法是
    跳过它，而跳过的表看起来依然是完整的。所以这里不写死 10，而是拿**金标
    自己的列表**去比 —— 金标加了第 11 条，这条测试立刻红。
    """
    doc, gold = _table()
    assert [r["id"] for r in doc["assertions"]] == [a["id"] for a in gold["assertions"]], \
        "表里的条目与金标的 assertions 对不上"


def test_missing_threshold_in_check_text_raises():
    """金标 `check` 里抠不出阈值时必须**抛**，不许替它补一个默认值。

    补出来的阈值没人会发现它是编的 —— 而它会一路影响判据结果。
    """
    rd, gold, brief = _load()
    gold = copy.deepcopy(gold)
    gold["assertions_by_id"]["AS-3"]["check"] = "risk(P3) >= 高"   # 阈值被删了
    try:
        G.evaluate(copy.deepcopy(rd), gold, copy.deepcopy(brief))
    except BriefError as exc:
        assert "阈值" in str(exc), f"没说是缺阈值：{exc}"
        return
    raise AssertionError("阈值抠不出来却照样算了 —— 应当是抛出，不是回落")


def test_table_refuses_when_the_gold_does_not_say_it_is_not_a_score():
    """金标没写「reference_shape 不是评分基准」时，本表必须拒绝生成。

    那句「这不是分数」是**引用**，不是本表的主张。金标没写而本表照写，
    等于本表替金标背书 —— 这个项目一直在拒绝的正是这个姿态。
    """
    rd, gold, _ = _load()
    gold = copy.deepcopy(gold)
    gold["gold_status"].pop("what_is_NOT_solid", None)
    try:
        G.build_gold_check(rd, gold, command="test")
    except BriefError as exc:
        assert "gold_status" in str(exc), f"没说清缺的是哪一条声明：{exc}"
        return
    raise AssertionError("金标没写这条声明，表却照样生成了 —— 出处成了本表自己")


# ---------------------------------------------------------------------------
# 2. 三种结果：不可判定不许被折成通过或失败
# ---------------------------------------------------------------------------

def test_undecidable_rows_are_not_counted_as_pass_or_fail():
    """AS-7 / AS-9 没有可判的数 → 必须是「不可判定」，且**不谈采信**。"""
    doc, _ = _table()
    for aid in ("AS-7", "AS-9"):
        r = _by_id(doc, aid)
        assert r["verdict"] == G.UNDECIDED, f"{aid} 判成了 {r['verdict']}"
        assert r["trusted"] is None, \
            f"{aid} 的采信应当是「不适用」（None），实得 {r['trusted']!r}"
        assert r["no_trust_because"] == [], f"{aid} 判不出来却在谈采信原因"
        assert r["undecidable_because"], f"{aid} 没说为什么判不了"
        assert r["to_make_decidable"], f"{aid} 没说还缺什么才能判"


def test_undecidable_rows_do_not_enter_the_trusted_denominator():
    """「判得出来的有几条」这个分母里不许混进不可判定的。"""
    doc, _ = _table()
    s = G.summarize(doc["assertions"])
    undecided = [r["id"] for r in doc["assertions"] if r["verdict"] == G.UNDECIDED]
    assert s["judged"] == s["total"] - len(undecided)
    assert not set(s["trusted_ids"]) & set(undecided), \
        "不可判定的条目进了「可采信」名单"
    # 入库那份的两个数：判得出来 8 条，其中 4 条可采信。
    assert (s["judged"], len(s["trusted_ids"])) == (8, 4), \
        f"分母或分子变了：judged={s['judged']} trusted={s['trusted_ids']}"


def test_as7_records_that_the_check_text_itself_is_wrong():
    """AS-7 除了缺数据，**判据原文本身也与它声称的不等价** —— 必须写出来。

    `seq(劝删) in (seq(口径质疑), seq(家长介入))` 说的是「属于其中某一个」，
    而 claim 说的是「排在两者之间」。本表不替金标改写判据，但要说清这件事，
    否则下一个人会把「按字面求值永远不成立」读成「模型没做到」。
    """
    doc, _ = _table()
    r = _by_id(doc, "AS-7")
    assert "不等价" in r["to_make_decidable"], \
        f"没记下判据原文与声称不等价：{r['to_make_decidable']}"


def test_summary_names_the_shared_root_cause():
    """三条否决是**同一个相位缺陷**，表里必须写明，别让人读成三个问题。"""
    doc = _full()
    note = doc["summary"]["one_defect_note"]
    for aid in ("AS-2", "AS-3", "AS-4", "AS-8"):
        assert aid in note, f"{aid} 没被归到那条共同的成因里"
    assert "相位" in note, "没点出是相位缺陷"


# ---------------------------------------------------------------------------
# 3. 贴界：一票降级，且**双向**
# ---------------------------------------------------------------------------

def test_railing_is_detected_from_the_curve_itself():
    """贴界的判定无自由参数：值恰好等于 0.0000 或 1.0000。

    这里不信表里那串轮号，而是**照着曲线自己重算一遍**再去比 ——
    否则这条测试只是在复述实现。
    """
    rd, _, _ = _load()
    expected = sorted({r["index"] for r in rd["rounds"]
                       if any(round(v, 4) in (0.0, 1.0)
                              for v in r["state"].values())})
    doc, _ = _table()
    assert expected == [9, 10, 11, 12, 13, 14], \
        f"入库产出里的贴界轮变了：{expected}（改这条之前先想想为什么）"
    assert doc["railing"]["railed_rounds"] == expected
    assert set(doc["railing"]["railed_dims"]) == {"trust", "polarization",
                                                 "risk", "stability"}
    assert "attention" not in doc["railing"]["railed_dims"], \
        "attention 从未触界，进了贴界维度说明判定算错了"


def test_railing_only_taints_assertions_that_read_the_railed_value():
    """**按读到的操作数判，不按「沾没沾到贴界轮」判。**

    这条运行里有 6 个贴界轮，几乎每条阶段级判据都要读 P5。若按「沾到就染红」，
    整张表会只剩「不采信」，等于什么也没说。所以：AS-1 读的是 P1/P2/P3 的
    信任（都没贴界）→ 必须干净；AS-2 读 attention → 必须干净。
    """
    doc, _ = _table()
    for aid in ("AS-1", "AS-2", "AS-3", "AS-8"):
        r = _by_id(doc, aid)
        assert r["trusted"] is True, \
            f"{aid} 没读贴界值，却被降级：{r['no_trust_because']}"
        assert not r["tainted_by_railing"]
    for aid in ("AS-4", "AS-5", "AS-6", "AS-10"):
        r = _by_id(doc, aid)
        assert r["trusted"] is False, f"{aid} 读过贴界值却没被降级"
        assert r["tainted_by_railing"], f"{aid} 没记下踩在哪个 (维, 轮) 上"


def test_a_pass_that_reads_a_railed_value_is_not_trusted():
    """AS-6 通过，但它读的是贴到 0.0 下界的 trust(P5) —— **通过也不采信**。"""
    doc, _ = _table()
    r = _by_id(doc, "AS-6")
    assert r["verdict"] == G.PASS_
    assert r["trusted"] is False
    assert G.NO_TRUST_RAIL in r["no_trust_because"], r["no_trust_because"]


def test_a_fail_that_reads_a_railed_value_is_also_not_trusted():
    """AS-10 否决，而它比的是两个都贴在下界的数 —— **否决同样不采信**。

    这一条与上一条对读：贴界对判据的破坏是**双向的**。只守一个方向的话，
    一个「只要不通过就没问题」的实现在这里会过。
    """
    doc, _ = _table()
    r = _by_id(doc, "AS-10")
    assert r["verdict"] == G.FAIL_
    assert r["trusted"] is False
    assert G.NO_TRUST_RAIL in r["no_trust_because"], r["no_trust_because"]
    # 两边都贴在 0.0 上、判据方向由夹逼决定这件事，正文要自己说清。
    assert "双向" in r["detail"], f"没点出贴界破坏是双向的：{r['detail']}"


def test_railed_variant_is_reported_for_ordering_assertions():
    """贴界污染次序型判据时，要**同时给出剔掉贴界轮的那个口径**。"""
    doc, _ = _table()
    r = _by_id(doc, "AS-4")
    alt = r["variant"]
    assert alt is not None, "没给出剔掉贴界轮的变体"
    assert alt["argmax_phase"] == "P4", alt
    assert 14 not in alt["rounds_used"], "变体里还留着贴界轮"
    assert alt["value"] > r["values"]["polarization(P3)"], \
        "变体读数没高于 P3，那「剔掉后还是 P4」这句话不成立"


# ---------------------------------------------------------------------------
# 4. 恒真判据：AS-5 的防恒真用例
# ---------------------------------------------------------------------------

def test_as5_is_flagged_vacuous_when_trust_falls():
    """信任下降时 AS-5 的右端为负 → 判据恒真 → **通过也不能采信**。"""
    doc, _ = _table()
    r = _by_id(doc, "AS-5")
    assert r["verdict"] == G.PASS_
    assert r["values"]["right"] <= 0, "右端应当为负（信任是下降的）"
    assert G.NO_TRUST_VACUOUS in r["no_trust_because"], r["no_trust_because"]
    assert r["trusted"] is False


def test_as5_flips_to_fail_once_p5_is_not_railed():
    """**防恒真用例**：把 P5 改成不贴界（信任高于 P3），AS-5 必须变「否决」。

    做不到这一点，就说明「不采信」那套逻辑是装饰 —— 一个永远返回「不采信」
    的实现也能过上面那条测试。这里换的是**真数据**上的一轮，结论必须跟着变。
    """
    doc = _unrailed()
    r = _by_id(doc, "AS-5")
    r0 = _by_id(_table()[0], "AS-5")

    assert r0["verdict"] == G.PASS_ and r["verdict"] == G.FAIL_, \
        f"P5 不再贴界后判据应不成立，实得 {r['verdict']}（{r['detail']}）"
    assert r["values"]["right"] > 0 > r0["values"]["right"], \
        "右端应当由负转正 —— 那正是「恒真」被解除的标志"
    assert G.NO_TRUST_VACUOUS not in r["no_trust_because"], \
        "右端已经为正，不该再判「恒真」"
    assert r0["trusted"] is False and r["trusted"] is True, \
        "判据不再恒真时应当可以采信"
    assert r["values"]["left"] == r0["values"]["left"], \
        "左端（关注回落）不该被这次改动碰到 —— 碰了说明改错了地方"


def test_unrailing_p5_flips_three_more_rows_from_untrusted_to_trusted():
    """把 P5 摘下来还应当**放过另外三条** —— 一票降级不是一票否决。

    AS-4 / AS-6 / AS-10 在原数据里都被 P5 的贴界降级，摘掉之后它们各自的
    结论（否决 / 通过 / 否决）不变，但都变成可采信。这条与上一条合起来说明：
    降级改的是「能不能当证据」，不是「结论是正是反」。
    """
    before = _table()[0]
    doc = _unrailed()
    assert 14 not in doc["railing"]["railed_rounds"]
    for aid in ("AS-4", "AS-6", "AS-10"):
        r0, r = _by_id(before, aid), _by_id(doc, aid)
        assert r0["verdict"] == r["verdict"], \
            f"{aid} 的结论不该因为摘掉贴界而改变（{r0['verdict']} → {r['verdict']}）"
        assert r0["trusted"] is False and r["trusted"] is True, \
            f"{aid} 摘掉贴界后应当可以采信"


def test_mutating_the_recorded_curve_breaks_its_own_replay():
    """那份「不贴界」的数据与自己的重放对不上 —— 这件事要说在明处。

    改的是**落盘曲线**，不是行为序列，所以重放偏差必然从 4.9e-05 涨到
    0.55。若有一天这个数没变，说明改的根本不是被判据读的那份数据。
    """
    rd, gold, _ = _load()
    mut = copy.deepcopy(rd)
    mut["rounds"][14]["state"] = {**mut["rounds"][14]["state"],
                                 "trust": 0.55, "polarization": 0.70,
                                 "risk": 0.60, "stability": 0.30}
    base = G.build_brief(rd, gold, command="test")["replay_max_deviation"]
    hot = G.build_brief(mut, gold, command="test")["replay_max_deviation"]
    assert base < 1e-4, f"原产出本该与自己的重放一致，实得 {base}"
    assert hot > 1e-2, f"改过曲线却没体现在重放偏差上，实得 {hot}"


# ---------------------------------------------------------------------------
# 5. AS-8：定义写清了才判，且不与另一条信号混淆
# ---------------------------------------------------------------------------

def test_as8_uses_the_first_risk_alert_round():
    """`warning_issued_at` = 首个 `risk_alert` 的轮号（本表的定义，写进产物）。"""
    doc, _ = _table()
    r = _by_id(doc, "AS-8")
    assert r["values"]["warning_issued_at"] == 7
    assert r["values"]["risk_alert_rounds"] == [7]
    assert r["verdict"] == G.FAIL_, "轮 7 晚于 P2（轮 2），应当否决"
    assert "本表的定义" in r["definition_used"], \
        "换了定义可以改变结论，定义必须写在产物里"


def test_as8_passes_when_the_alert_comes_early_enough():
    """**反面**：预警若真在 P2 之前触发，这条必须能通过。

    只测「否决」那一面的话，一个永远否决的实现也能过 —— 而这条断言的
    全部意义就在于「未然」能不能提前预警。这里直接把预警事件放进简报的
    第 1 轮（真数据上第 7 轮才有）。
    """
    rd, gold, brief = _load()
    brief = copy.deepcopy(brief)
    brief["rounds"][1]["events"] = [{
        "kind": "risk_alert", "dimension": "risk", "magnitude": 0.7104,
        "message": "风险越过预警线 0.70（当前 0.7104）"}]
    doc = G.evaluate(copy.deepcopy(rd), copy.deepcopy(gold), brief)
    r = _by_id(doc, "AS-8")
    assert r["verdict"] == G.PASS_, f"轮 1 ≤ 轮 2 应当通过，实得 {r['verdict']}"
    assert r["values"]["warning_issued_at"] == 1, "取到的不是最早那一轮"
    assert r["trusted"] is True


def test_as8_says_it_must_not_be_confused_with_the_polarization_signal():
    """产出在 P2 确实有一条极化信号 —— 这条要写明，否则会被混起来读。"""
    doc, _ = _table()
    r = _by_id(doc, "AS-8")
    assert "polarization_surge" in r["detail"], \
        "没提醒「别把风险预警线与极化信号混为一谈」"


# ---------------------------------------------------------------------------
# 6. 对齐口径
# ---------------------------------------------------------------------------

def test_alignment_is_0_based_and_matches_gold_days():
    """阶段 → 轮号必须是 0 基，且与金标 `day` 在轮=天时逐位相等。"""
    doc, gold = _table()
    a = doc["alignment"]
    assert a["index_base"] == 0
    assert a["by_phase"] == {"P1": 0, "P2": 2, "P3": 5, "P4": 8, "P5": 14}
    assert [r["phase"] for r in a["rows"]] == [p["id"] for p in gold["phases"]]
    assert all(r["match"] is True for r in a["rows"]), a["rows"]
    assert "1 基" in a["note_1based"], "没写下「别与 1 基的 R 号混用」"


def test_alignment_declines_to_compare_days_when_round_is_not_a_day():
    """压缩产出里「轮」不是「天」，两者不可直接比 —— 必须返回 null 而不是 False。

    False 会被读成「对不上」，而真相是「不可比」。这个仓库里两种编号混过
    一次，所以这一条要分得清。
    """
    rd, gold, _ = _load()
    rd = copy.deepcopy(rd)
    rd["meta"]["days_per_round"] = 7.0
    a = G.alignment(rd, gold)
    assert all(r["match"] is None for r in a["rows"]), \
        f"轮≠天时不该给出「相等/不相等」的判断：{a['rows']}"


# ---------------------------------------------------------------------------
# 7. 产物：不是分数、有出处、可复现
# ---------------------------------------------------------------------------

def test_the_table_does_not_use_reference_shape_and_says_where_that_comes_from():
    """判据里不许出现 `reference_shape`，且「不是在评分」这句话要有出处。"""
    doc = _full()
    assert "reference_shape" not in json.dumps(doc["assertions"], ensure_ascii=False), \
        "判据里出现了 reference_shape —— 那是作者手写的预期，不是观测值"
    nas = doc["not_a_score"]
    _, gold, _ = _load()
    assert nas["why"] == gold["gold_status"]["what_is_NOT_solid"], \
        "「这不是分数」的理由不是引自金标原文"
    assert nas["source"].endswith("gold_status"), nas["source"]
    assert "reference_shape" in nas["excluded_from_judgement"], \
        "没写明 reference_shape 被排除在判据之外"


def test_carries_provenance_of_both_inputs():
    """表里的数必须能追回产生它的那次运行。"""
    doc = _full()
    p = doc["_provenance"]
    assert p["rounds_sha256"] == G._sha256(ROUNDS_FILE)
    assert p["command"] == G.DEFAULT_COMMAND
    for key in ("rounds_file", "scenario_file", "scenario_sha256"):
        assert p.get(key), f"缺溯源字段 {key}"
    assert doc["replay_max_deviation"] < 1e-4, \
        "这份曲线与引擎重放对不上，整张表都不成立"


def test_render_is_deterministic_and_matches_the_committed_sample():
    """渲染两次逐字一致，且与**入库那份**一致 —— 两者一起才叫「算出来的」。

    只比「两次跑一样」是不够的：把渲染改坏、同时又重跑入库样例，两次仍然
    一致。所以这里比的是仓库里那份给人看的 md。命令用 CLI 的默认值，
    因为 md 正文自己印着生成命令。
    """
    rd, gold, _ = _load()
    a = G.render_markdown(G.build_gold_check(rd, gold, command=G.DEFAULT_COMMAND))
    b = G.render_markdown(G.build_gold_check(rd, gold, command=G.DEFAULT_COMMAND))
    assert a == b, "同一份产出渲染出两样文字"

    sample = ROUNDS_FILE.parent / "gold_check.md"
    if not sample.is_file():
        print("      （跳过：入库样例不在）")
        return
    assert a == sample.read_text(encoding="utf-8"), \
        "入库那份 gold_check.md 与当前代码算出来的不是同一张" \
        "（重跑：python -m weiran.gold_check）"


def test_committed_json_matches_the_code():
    """入库的 JSON 同样要逐字对得上 —— 展示层读的是它，不是 md。"""
    sample = ROUNDS_FILE.parent / "gold_check.json"
    if not sample.is_file():
        print("      （跳过：入库样例不在）")
        return
    on_disk = json.loads(sample.read_text(encoding="utf-8"))
    fresh = _full()
    assert json.dumps(on_disk, ensure_ascii=False, sort_keys=True) == \
        json.dumps(fresh, ensure_ascii=False, sort_keys=True), \
        "入库那份 gold_check.json 与当前代码算出来的不是同一张"


def test_headline_counts_are_locked():
    """三个数印在表的正文里，改一个就得有人来说明 —— 在这里锁住。"""
    doc = _full()
    s = doc["summary"]
    assert s["total"] == 10 and s["judged"] == 8
    assert s["counts"] == {"通过": 3, "否决": 5, "不可判定": 2}, s["counts"]
    assert s["trusted_ids"] == ["AS-1", "AS-2", "AS-3", "AS-8"], s["trusted_ids"]
    assert [n["id"] for n in s["not_trusted"]] == ["AS-4", "AS-5", "AS-6", "AS-10"]


def test_render_prints_every_row_and_every_reason():
    """逐条说明一段都不能少，且不可判定的条目要写明「还缺什么」。"""
    doc = _full()
    md = G.render_markdown(doc)
    for r in doc["assertions"]:
        assert f"### {r['id']} ·" in md, f"{r['id']} 的逐条说明没渲染出来"
    assert md.count("### AS-") == len(doc["assertions"])
    assert "这不是分数" in md
    assert "还缺什么才能判" in md and "为什么不可判定" in md
    assert md.count("为什么不可判定") == 2, "不可判定的应当是 2 条"


# -- 简易 runner（与其余测试文件保持一致）----------------------------------

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
