"""跨 seed 汇总器的测试。**全部离线：不联网、不调 LLM、不跑仿真。**

这一套守的是「**汇总器会不会安静地把一批不可比的臂折成一份读数**」。

它的危险和 `repro_check.py` 那套是同一类，但更难看出来：汇总器**不写任何
被它读的数据**，所以它不可能像 `repro_check.py` 那样把产出覆盖掉。它坏的
方式是另一种 —— **产出一份看起来正常、其实没有依据的读数**：

  - 拿一支臂拷成 N 份 → 跨 seed 离散恒等于 0，而报告看着漂亮得离谱；
  - 12 agent 的臂与 27 agent 的臂混在一起求均值 → 均值没有意义，
    而文件里看不出来（12 agent 的相位读数是已知无效的）；
  - 引擎换过之后旧臂重放不上 → 那些曲线不是当前引擎算出来的，
    拿它们算离散等于在量别的东西。

所以这一套的重点不在「算得对不对」，而在**该拒绝的时候它拒绝没有**。
每一条拒绝都有一个反例测试 —— 一个只会求平均、从不拒绝的实现，
会在这里红一大片。

**人造臂从哪来。** 拿入库那份 27 agent 产出当底本，程序化地改它的
`behaviors`（引擎的直接输入），再**让引擎把状态重算一遍**。不重算的话
重放对不上，臂会被正确地判为不可采信 —— 这一点本身也有一条测试盯着
（`test_hand_edited_state_is_refused`），因为那正是「手改产物」的形状。

    python backend/tests/test_stability_report.py
    pytest backend/tests/
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stability_report as SR  # noqa: E402
from weiran.brief import load_gold, replay  # noqa: E402
from weiran.config import REPO_ROOT, ensure_console_encoding  # noqa: E402
from weiran.gold_check import FAIL_, PASS_  # noqa: E402
from weiran.profiles import DEFAULT_SCENARIO  # noqa: E402
from weiran.world_state import DIMENSIONS, WorldStateEngine  # noqa: E402

SHIPPED = REPO_ROOT / "data" / "simulation" / "twitter_rounds.json"

_BEHAVIOR_LABELS = ("discussion", "testimony", "suppression", "contradiction",
                    "appeal", "ridicule")

#: 哨兵：`meta_extra` 里用 `_DROP` 表示**把这个键整个删掉**，而不是置为
#: `None`。这两件事测的正是「缺席」与「值就是 null」的区别。
_DROP = object()


# ---------------------------------------------------------------------------
# 人造臂
# ---------------------------------------------------------------------------

def _base() -> dict:
    return json.loads(SHIPPED.read_text(encoding="utf-8"))


def _variant(seed: int, *, rate: float = 0.3, base: dict | None = None) -> dict:
    """仿一支「换了个 seed」的臂：动行为序列，再让引擎重算状态。

    **必须重算。** 只改 `behaviors` 而不重算 `state`，就是「记录与它自己的
    输入对不上」，重放偏差会跳到 0.17 以上 —— 那不是一支臂，是一份被手改过
    的产物。`test_hand_edited_state_is_refused` 专门盯这个区别。
    """
    rng = random.Random(seed)
    doc = copy.deepcopy(base if base is not None else _base())
    for r in doc["rounds"]:
        b = list(r["behaviors"])
        for i in range(len(b)):
            if rng.random() < rate:
                b[i] = rng.choice(_BEHAVIOR_LABELS)
        r["behaviors"] = b
    engine = WorldStateEngine()
    dpr = float(doc["meta"]["days_per_round"])
    for res, r in zip(replay(engine, doc["rounds"], dpr), doc["rounds"]):
        r["state"] = {d: round(res.state_after[d], 4) for d in DIMENSIONS}
    return doc


def _write_arm(root: Path, label: str, doc: dict, *, seed: int,
               protocol: bool = True, inputs: str = "same",
               manifest_extra: dict | None = None,
               meta_extra: dict | None = None,
               out_sha: str | None = None) -> Path:
    arm = root / label
    arm.mkdir(parents=True, exist_ok=True)
    if protocol:
        doc["meta"] = {**doc["meta"], "seed": seed, "temperature": 0.7,
                       "random_seeded": True, "events_from": "phases",
                       "chunking_guard": True}
    # 覆盖要**最后**施加，否则会被上面那一行抹掉。
    for key, value in (meta_extra or {}).items():
        if value is _DROP:
            doc["meta"].pop(key, None)
        else:
            doc["meta"][key] = value
    path = arm / "twitter_rounds.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "label": label, "seed": seed, "command": f"simulate --seed {seed}",
        "inputs": {n: inputs for n in SR.INPUT_FILES},
        "out_sha256": out_sha or hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    manifest.update(manifest_extra or {})
    (arm / "arm.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return arm


def _batch(root: Path, *specs) -> list[dict]:
    """`specs` 是 `(label, doc, kwargs)` 三元组。"""
    return [SR.read_arm(_write_arm(root, label, doc, **kw))
            for label, doc, kw in specs]


def _gold() -> dict:
    return load_gold(REPO_ROOT / DEFAULT_SCENARIO)


def _three_arms() -> list[dict]:
    root = Path(tempfile.mkdtemp(prefix="weiran-report-"))
    return _batch(root,
                  ("seed11", _variant(11), {"seed": 11}),
                  ("seed22", _variant(22), {"seed": 22}),
                  ("seed33", _variant(33), {"seed": 33}))


def _build(arms: list[dict], gold: dict | None = None) -> dict:
    return SR.build_stability(arms, gold=gold or _gold(), root="(test)",
                              command="test")


# ---------------------------------------------------------------------------
# 1. 算得对
# ---------------------------------------------------------------------------

def test_a_clean_batch_is_accepted():
    """先立一条「正常批次能过」。**没有它，下面每一条拒绝都可能是
    「什么都拒绝」的实现，而那种实现看着一样绿。**"""
    arms = _three_arms()
    assert SR.check_batch(arms) == []
    doc = _build(arms)
    assert len(doc["arms"]) == 3


def test_curve_mean_and_range_are_computed_across_arms():
    arms = _three_arms()
    doc = _build(arms)
    readings = [SR.read_arm_readings(a, _gold(), scale=SR.DEFAULT_SCALE)
                for a in arms]

    for row in doc["curve"]:
        r = row["round"]
        for d in DIMENSIONS:
            vals = [x["curve"][r][d] for x in readings]
            got = row["dims"][d]
            assert got["min"] == min(vals), (r, d)
            assert got["max"] == max(vals), (r, d)
            assert abs(got["mean"] - sum(vals) / len(vals)) < 1e-12, (r, d)
            assert abs(got["range"] - (max(vals) - min(vals))) < 1e-12, (r, d)


def test_drive_is_the_sum_of_absolute_excitations():
    """驱动量是**已有读数之和**，不是新物理。用一个已知的式子钉住它。"""
    arms = _three_arms()
    doc = _build(arms)
    for a, row in zip(arms, doc["arms"]):
        brief = SR.build_brief(a["rounds_doc"], _gold(), scale=SR.DEFAULT_SCALE,
                               command="test")
        expected = SR._drive_curve(brief)
        readings = SR.read_arm_readings(a, _gold(), scale=SR.DEFAULT_SCALE)
        assert readings["drive_curve"] == expected


def test_drive_peak_of_the_shipped_artifact_is_decidable():
    """入库那份产出的驱动峰是 R9，与次峰差 0.0945 —— 远在舍入上界之上。
    这条钉住「可判定」的门槛没有被设成一个把正常读数也判掉的值。"""
    brief = SR.build_brief(_base(), _gold(), scale=SR.DEFAULT_SCALE,
                           command="test")
    curve = SR._drive_curve(brief)
    peak, gap = SR._peak(curve)
    assert peak == 9
    assert gap > SR._ROUNDING_FLOOR


def test_rounding_floor_scales_with_the_dimension_count():
    """峰-次峰的间隔要跟「六维各按 4 位小数落盘」对上。"""
    assert abs(SR._ROUNDING_FLOOR - len(DIMENSIONS) * 0.5e-4) < 1e-15


def test_signal_separates_cross_arm_spread_from_within_arm_step():
    arms = _three_arms()
    doc = _build(arms)
    readings = [SR.read_arm_readings(a, _gold(), scale=SR.DEFAULT_SCALE)
                for a in arms]
    for d, row in doc["signal"]["per_dim"].items():
        spreads = []
        for r in sorted(readings[0]["curve"]):
            vals = [x["curve"][r][d] for x in readings]
            spreads.append(max(vals) - min(vals))
        assert abs(row["across_arm_spread"] - sum(spreads) / len(spreads)) < 1e-12
        assert row["within_arm_step"] > 0, d
        assert abs(row["ratio"] - row["across_arm_spread"]
                   / row["within_arm_step"]) < 1e-9


def test_signal_ratio_is_none_not_infinity_when_a_dimension_never_moves():
    """维度整段不动时比值的分母是 0 —— **报 `None`，不报 `inf`**。
    `inf` 会被读成一个巨大的读数，而真相是「没有可读的东西」。"""
    arms = _three_arms()
    frozen = []
    for a in arms:
        doc = copy.deepcopy(a)
        for r in doc["rounds_doc"]["rounds"]:
            r["state"] = {d: 0.5 for d in DIMENSIONS}
        frozen.append(doc)
    readings = [SR.read_arm_readings(a, _gold(), scale=SR.DEFAULT_SCALE)
                for a in frozen]
    sig = SR.merge_signal(readings)
    assert sig["per_dim"]["trust"]["ratio"] is None


# ---------------------------------------------------------------------------
# 2. 「翻转」必须真的报出来（这一条是防恒真的关键）
# ---------------------------------------------------------------------------

def test_a_flipped_assertion_is_reported_as_flipped():
    """把一条断言的结论在臂之间做成不一致，汇总**必须**报「翻转」。

    一个只重算第一支臂的实现会在这里红 —— 那正是它存在的理由。
    这里用 `rate=1.0`（行为序列整个换掉）造出真的翻转：实测 AS-4 在
    seed 99 上是「通过」，在 seed 7 上是「否决」。
    """
    root = Path(tempfile.mkdtemp(prefix="weiran-flip-"))
    arms = _batch(root,
                  ("seed99", _variant(99, rate=1.0), {"seed": 99}),
                  ("seed7", _variant(7, rate=1.0), {"seed": 7}))
    doc = _build(arms)

    per = {r["id"]: r for r in doc["gold"]["per_assertion"]}
    assert set(per["AS-4"]["verdicts"]) == {PASS_, FAIL_}, \
        f"夹具本身没造出翻转：{per['AS-4']['verdicts']}"
    assert per["AS-4"]["flipped"] is True
    assert doc["gold"]["agrees"] is False
    assert "AS-4" in doc["gold"]["flipped_ids"]


def test_the_flip_shows_up_in_the_rendered_markdown():
    """翻转要出现在**正文**里。只在 JSON 里报、正文里写「一致」是最坏的一种：
    读的人只看 markdown。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-flip2-"))
    doc = _build(_batch(root,
                        ("seed99", _variant(99, rate=1.0), {"seed": 99}),
                        ("seed7", _variant(7, rate=1.0), {"seed": 7})))
    md = SR.render_markdown(doc)
    assert "翻转" in md
    assert "AS-4" in md


def test_an_agreeing_batch_says_so():
    """反向：真的都一致时，正文不许说翻转。**没有这一条，上面那条可以靠
    「永远印翻转」骗过去。**"""
    doc = _build(_three_arms())
    md = SR.render_markdown(doc)
    assert doc["gold"]["agrees"] is True
    assert "翻转" not in md


# ---------------------------------------------------------------------------
# 3. 拒绝：不是一批，就别想汇总
# ---------------------------------------------------------------------------

def _refused(arms: list[dict]) -> str:
    problems = SR.check_batch(arms)
    assert problems, "本该拒绝，却通过了"
    return "\n".join(problems)


def test_an_arm_copied_n_times_is_refused():
    """**离散恒等于 0 会看着漂亮得离谱。** 这是最容易被误读成好消息的一种。

    夹具刻意做成最像样的那种拷贝：推演内容一模一样，而 `meta.seed` 被改成
    了不同的值 —— 于是**文件**指纹是两样、**内容**指纹是一样。查拷贝如果
    只看文件指纹，这一条就会漏过去（这正是第一版实现踩到的坑）。
    """
    root = Path(tempfile.mkdtemp(prefix="weiran-dup-"))
    arms = _batch(root, ("seed11", _variant(11), {"seed": 11}),
                  ("seed22", _variant(11), {"seed": 22}))
    assert arms[0]["rounds_sha256"] != arms[1]["rounds_sha256"], \
        "夹具没造出「文件不同」的形状"
    assert arms[0]["content_sha256"] == arms[1]["content_sha256"]
    assert "逐字节相同" in _refused(arms)


def test_the_same_seed_twice_is_refused():
    root = Path(tempfile.mkdtemp(prefix="weiran-same-"))
    samedoc = _variant(11)
    arms = _batch(root, ("seed11", copy.deepcopy(samedoc), {"seed": 11}),
                  ("seed11b", copy.deepcopy(samedoc), {"seed": 11}))
    assert "同一个 seed" in _refused(arms)


def test_a_different_scale_across_arms_is_refused():
    """12 agent 的相位读数是已知无效的，混进 27 agent 的一批里，
    均值没有意义而文件里看不出来。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-scale-"))
    arms = _batch(root,
                  ("seed11", _variant(11), {"seed": 11}),
                  ("seed22", _variant(22), {"seed": 22,
                                            "meta_extra": {"agents": 12}}))
    assert "agents" in _refused(arms)


def test_a_different_chunking_guard_across_arms_is_refused():
    """两种切片护栏口径的曲线不可混用（关掉时记忆被切成 1 token 一块、
    膨胀约 20 倍）。**目录名上看不出这件事**，所以只能靠程序拒绝。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-cg-"))
    arms = _batch(root,
                  ("seed11", _variant(11), {"seed": 11}),
                  ("seed22", _variant(22), {"seed": 22,
                                            "meta_extra": {"chunking_guard": False}}))
    assert "chunking_guard" in _refused(arms)


def test_different_inputs_across_arms_are_refused():
    """输入不同源，臂之间的差异就不止是 seed。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-in-"))
    arms = _batch(root, ("seed11", _variant(11), {"seed": 11}),
                  ("seed22", _variant(22), {"seed": 22, "inputs": "other"}))
    assert "不同源" in _refused(arms)


# ---------------------------------------------------------------------------
# 4. 拒绝：「未记录」不是「一致」
# ---------------------------------------------------------------------------

def test_an_arm_whose_meta_lacks_the_protocol_keys_is_refused():
    """老产出的 `meta` 里没有 `chunking_guard` / `temperature` / `events_from`。
    **全批都没记不等于一致** —— 那只是我们不知道。

    这里也是「键缺席」与「键在而值为 null」的分界：一个**记了** `null` 的臂
    是过得了的（确实没传），一个**没这个键**的臂过不了（不知道）。
    """
    root = Path(tempfile.mkdtemp(prefix="weiran-old-"))
    arms = _batch(root,
                  ("seed11", _variant(11), {"seed": 11, "protocol": False}),
                  ("seed22", _variant(22), {"seed": 22, "protocol": False}))
    text = _refused(arms)
    assert "根本没被记录" in text
    assert "chunking_guard" in text

    # 记了 `null` 是另一回事：那是一次口径明确的一致（确实没传）。
    ok_root = Path(tempfile.mkdtemp(prefix="weiran-null-"))
    arms2 = _batch(ok_root,
                   ("seed11", _variant(11), {"seed": 11,
                                             "meta_extra": {"temperature": None}}),
                   ("seed22", _variant(22), {"seed": 22,
                                             "meta_extra": {"temperature": None}}))
    assert SR.check_batch(arms2) == []


def test_the_protocol_table_shows_unrecorded_not_null():
    """口径表本身也要守「缺键 ≠ 0」：缺席印「未记录」，不印 `null`、更不印 `0`。

    **这条走的是 `render_markdown` 而不是 `build_stability`。** 原因要写清楚：
    `build_stability` 会先把「有键没被记录」的批次整个拒掉，所以走它**到不了**
    这个分支。这里直接喂一份人造 doc，守的是渲染器**自己**的保证 ——
    它是个公开函数，喂它 `null` 不许印出 `null`。
    """
    root = Path(tempfile.mkdtemp(prefix="weiran-proto-"))
    arms = _batch(root,
                  ("seed11", _variant(11), {"seed": 11}),
                  ("seed22", _variant(22), {"seed": 22}))
    doc = _build(arms)
    doc["protocol"]["chunking_guard"] = "未记录"
    doc["arms"][0]["seed"] = None

    md = SR.render_markdown(doc)
    assert "未记录" in md
    assert "| None |" not in md
    assert SR.check_stability(doc, md) == []


def test_a_manifest_that_contradicts_the_artifact_is_refused():
    """`arm.json` 是驱动写的、`meta` 是那次运行自己写的。两者对不上，
    说明其中一份被动过 —— **两条都要读，只读一条就会放过去**。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-tamper-"))
    arms = _batch(root, ("seed11", _variant(11), {"seed": 11,
                                                  "manifest_extra": {"temperature": 0.1}}),
                  ("seed22", _variant(22), {"seed": 22,
                                            "manifest_extra": {"temperature": 0.1}}))
    text = _refused(arms)
    assert "arm.json" in text and "0.1" in text


# ---------------------------------------------------------------------------
# 5. 拒绝：读不齐的臂
# ---------------------------------------------------------------------------

def test_a_directory_without_a_manifest_is_not_an_arm():
    """缺 `arm.json` 的老式目录不被当成合法臂 —— 缺这份清单就无法回答
    「它是不是和别的一样跑法」。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-bare-"))
    bare = root / "bare"
    bare.mkdir()
    (bare / "twitter_rounds.json").write_text("{}", encoding="utf-8")
    try:
        SR.read_arm(bare)
    except SR.StabilityError as exc:
        assert "arm.json" in str(exc)
    else:
        raise AssertionError("缺 arm.json 的目录被当成了合法臂")


def test_an_artifact_edited_after_the_run_is_refused():
    """跑完再改产物 → sha256 与 `arm.json` 记的对不上。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-edit-"))
    arm = _write_arm(root, "seed11", _variant(11), seed=11)
    SR.read_arm(arm)  # 先确认它是合法的
    p = arm / "twitter_rounds.json"
    p.write_text(p.read_text(encoding="utf-8").replace("0.3", "0.31", 1),
                 encoding="utf-8")
    try:
        SR.read_arm(arm)
    except SR.StabilityError as exc:
        assert "被改过" in str(exc) or "sha" in str(exc).lower()
    else:
        raise AssertionError("跑完之后被改过的产物没被拦下")


def test_two_artifacts_in_one_arm_directory_are_refused():
    """一支臂的目录里应当恰好一份产出。两份说明有人把两次跑混在了一起。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-two-"))
    arm = _write_arm(root, "seed11", _variant(11), seed=11)
    (arm / "other_rounds.json").write_bytes(
        (arm / "twitter_rounds.json").read_bytes())
    try:
        SR.read_arm(arm)
    except SR.StabilityError as exc:
        assert "恰好一份" in str(exc)
    else:
        raise AssertionError("一个目录里两份产出没被拦下")


# ---------------------------------------------------------------------------
# 6. 拒绝：重放不上的臂
# ---------------------------------------------------------------------------

def test_hand_edited_state_is_refused():
    """**这是「手改产物」的形状。** 只改 `state` 而不重算，重放偏差会跳到
    0.17 以上 —— 那种曲线不是当前引擎按当前参数算出来的。

    与「引擎换过」的形状要分开：这里是**一支**对不上，那里是**全部**
    对不上。两者该做的事不同（去重跑 vs 去查引擎），所以报的话也不同。
    """
    root = Path(tempfile.mkdtemp(prefix="weiran-hand-"))
    d = _variant(11)
    d["rounds"][5]["state"]["trust"] = 0.99
    # **三支臂、坏一支。** 两支里坏一支时会撞上「可采信的臂只剩一支」那条
    # 更早的拒绝 —— 那样这条测试就测不到它本来要测的东西了。
    arms = _batch(root, ("seed11", d, {"seed": 11}),
                  ("seed22", _variant(22), {"seed": 22}),
                  ("seed33", _variant(33), {"seed": 33}))
    doc = _build(arms)
    untrusted = {a["label"] for a in doc["arms"] if not a["trusted"]}
    assert untrusted == {"seed11"}
    assert any("排除" in r for r in doc["refusals"])
    # 被排除的臂不许参与读数。
    assert doc["drive"]["trusted_arms"] == ["seed22", "seed33"]


def test_a_batch_where_every_arm_fails_replay_is_refused_outright():
    """全部对不上 = 「引擎换过」。**这种形状不能给读数** —— 但也必须与
    「一支坏了」分开报，否则读者会去重跑，而该做的是去查引擎。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-allbad-"))
    d1, d2 = _variant(11), _variant(22)
    d1["rounds"][3]["state"]["panic"] = 0.99
    d2["rounds"][7]["state"]["panic"] = 0.99
    arms = _batch(root, ("seed11", d1, {"seed": 11}),
                  ("seed22", d2, {"seed": 22}))
    try:
        _build(arms)
    except SR.StabilityError as exc:
        assert "每一支臂都重放不上" in str(exc)
    else:
        raise AssertionError("全部臂都重放不上时，汇总器还是给出了读数")


def test_a_single_trusted_arm_is_refused():
    """一支臂算不出跨 seed 离散 —— 这时候给一份「均值」是在假装有 N 个样本。

    **注意它与「全部重放不上」是两条不同的拒绝**：这里有一支是好的，
    所以该做的是去重跑坏的那支，而不是去查引擎。
    """
    root = Path(tempfile.mkdtemp(prefix="weiran-one-"))
    bad = _variant(11)
    bad["rounds"][3]["state"]["panic"] = 0.99   # 手改 → 重放不上
    arms = _batch(root, ("seed11", bad, {"seed": 11}),
                  ("seed22", _variant(22), {"seed": 22}))
    assert [a["label"] for a in
            _build_batch_readings(arms) if a["trusted"]] == ["seed22"]
    try:
        _build(arms)
    except SR.StabilityError as exc:
        assert "可采信的臂只剩" in str(exc)
    else:
        raise AssertionError("只剩一支可采信的臂时，汇总器还是给出了读数")


def _build_batch_readings(arms: list[dict]) -> list[dict]:
    return [SR.read_arm_readings(a, _gold(), scale=SR.DEFAULT_SCALE)
            for a in arms]


# ---------------------------------------------------------------------------
# 7. 「这不是分数」
# ---------------------------------------------------------------------------

def test_the_report_says_it_is_not_a_score():
    doc = _build(_three_arms())
    assert "不是评分" in doc["not_a_score"]["statement"]
    assert doc["not_a_score"]["why"], "金标自己那句 what_is_NOT_solid 没被引用"


def test_the_rendered_markdown_leads_with_the_not_a_score_statement():
    doc = _build(_three_arms())
    md = SR.render_markdown(doc)
    assert doc["not_a_score"]["statement"] in md
    assert md.index(doc["not_a_score"]["statement"]) < md.index("## 一")


def test_words_that_read_agreement_as_correctness_are_rejected():
    """**正文里不许出现把一致性读成正确性的说法。** 跨 seed 一致**不等于**
    结论对：一个系统性偏差会在每个 seed 下一模一样地重现。"""
    doc = _build(_three_arms())
    for word in ("更准", "提升了", "准确率"):
        bad = SR.render_markdown(doc) + f"\n结论：这一版{word}。\n"
        problems = SR.check_stability(doc, bad)
        assert any(word in p for p in problems), word


def test_check_stability_accepts_the_real_output():
    """反向：真的产物必须过。没有这一条，上面那条可以靠「永远报问题」骗过去。"""
    doc = _build(_three_arms())
    assert SR.check_stability(doc, SR.render_markdown(doc)) == []


def test_railed_dimensions_are_flagged_but_not_judged():
    """贴过界的维度上跨臂离散是假的（夹逼把它压小了）—— **只标不判**。"""
    doc = _build(_three_arms())
    assert doc["evidence"]["railed_dims"], "夹具里本该有贴界的维度"
    md = SR.render_markdown(doc)
    assert "只标不判" in md


# ---------------------------------------------------------------------------
# 8. 离线
# ---------------------------------------------------------------------------

def test_the_aggregator_imports_nothing_that_can_reach_the_network():
    """汇总器**必须是纯离线的**：它可以被跑在任意一批臂上，而那些人
    不该因此产生任何费用。"""
    import ast
    src = (Path(__file__).resolve().parents[1] / "stability_report.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"requests", "httpx", "urllib", "socket", "openai", "camel",
              "oasis", "sentence_transformers"}
    assert not (imported & banned), imported & banned


def test_the_aggregator_never_invokes_a_subprocess():
    """它读臂、算读数、写字，**不起进程**。"""
    import ast
    src = (Path(__file__).resolve().parents[1] / "stability_report.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert "subprocess" not in [a.name for a in node.names]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"run", "Popen", "system", "call"}


def _run() -> int:
    # 兜底 runner 也要防这一条：**被测代码会往控制台印符号**（汇总器的
    # 正文里有 ⚠️、✓/✗，「未记录」也是中文），Windows 中文控制台是 GBK，
    # 装不下时 print 会抛 UnicodeEncodeError —— 于是「有坏消息要报」的那次
    # 运行反而崩在报消息的路上，看起来像测试坏了。
    ensure_console_encoding()
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
