"""差异化感知层的测试。

这一组测试守的是**注入这件事本身有没有走样**。它比别的测试更要紧一点：
注入是「不报错的失败」高发区 —— 少注入一次、注入成空串、所有 agent
收到同一段文本，这三种情况都不会抛异常，只会让结果悄悄变得没有意义。

所以这里的重点不是「函数返回值对不对」，而是几条**性质**：
  - 不同的角色拿到不同的块（否则「差异化感知」是假的）
  - 同一个角色前后两次拿到一样的块（否则感知成了第二个随机源）
  - 块里没有数值（否则 agent 会复述数字，曲线变成回声）
  - 块里没有 `# RESPONSE FORMAT`（否则将来会被 OASIS 静默切掉）

与另三套一样，**刻意不依赖 pytest**：普通 assert + 函数，
`python tests/test_perception.py` 直接跑，`python -m pytest tests/` 也能跑。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weiran.perception import (  # noqa: E402
    DIRECTION_ZH, FORBIDDEN, INJECT_ATTR, ActorKnowledge, KnowledgeError,
    Phase, active_phase_by_round, build_block, days_per_round,
    inject_round_context, load_knowledge, load_phases, phase_schedule,
    render_knowledge, render_state, tier_of,
)
from weiran.world_state import DIMENSION_ZH, DIMENSIONS, WorldState  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCENARIO = REPO / "benchmark" / "scenarios" / "employment_trust_crisis"
SIM = REPO / "data" / "simulation"


# ---------------------------------------------------------------------------
# 造一个最小的场景，好让「悬空 fact id」这类坏输入能被测到

def _write(dirpath: Path, facts, actors) -> tuple[Path, Path]:
    scenario = dirpath / "scenario"
    out = dirpath / "sim"
    scenario.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (scenario / "reference_data.json").write_text(
        json.dumps({"facts": facts, "phases": [
            {"id": "P1", "label": "引爆", "day": 0, "trigger": "报告挂网"},
            {"id": "P2", "label": "收束", "day": 3, "trigger": "校长说明会"},
        ]}, ensure_ascii=False), encoding="utf-8")
    (out / "actor_knowledge.json").write_text(
        json.dumps({"actors": actors}, ensure_ascii=False), encoding="utf-8")
    return out, scenario


def _actor(uid, name="某人", group="某组", knows=(), hidden=()):
    return {"user_id": uid, "user_name": f"weiran_a{uid + 1:02d}", "name": name,
            "group": group, "knows": list(knows), "hidden": list(hidden)}


def _tmp():
    return Path(tempfile.mkdtemp(prefix="weiran-perception-"))


# ---------------------------------------------------------------------------
# 知情映射

def test_knowledge_joins_ids_back_to_text():
    d = _tmp()
    try:
        out, scen = _write(
            d,
            [{"id": "F1", "text": "落实率 78.6%"},
             {"id": "F7", "text": "实际核实率 32.2%"}],
            {"A01": _actor(0, knows=["F1"], hidden=["F7"])},
        )
        k = load_knowledge(out, scen)
        e = k.by_actor["A01"]
        assert e.knows == ("落实率 78.6%",), "knows 没被还原成正文"
        assert e.hidden == ("实际核实率 32.2%",), "hidden 没被还原成正文"
        assert k.by_user_id == {0: "A01"}, "user_id 反查表不对"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_dangling_fact_id_is_loud_not_silent():
    """**这是本文件最重要的一条。** 悬空 id 若被跳过，注入进 prompt 的
    是一个裸露的 `F99` —— agent 只会当成乱码，而没有任何地方会提示。"""
    d = _tmp()
    try:
        out, scen = _write(
            d,
            [{"id": "F1", "text": "落实率 78.6%"}],
            {"A01": _actor(0, knows=["F1", "F99"])},
        )
        try:
            load_knowledge(out, scen)
        except KnowledgeError as exc:
            assert "F99" in str(exc), "报错里必须点名是哪个 id"
            assert "A01" in str(exc), "报错里必须点名是哪个角色"
        else:
            raise AssertionError("悬空 fact id 被静默跳过了")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_missing_files_say_what_to_do():
    d = _tmp()
    try:
        try:
            load_knowledge(d / "nope", d / "nope")
        except KnowledgeError as exc:
            assert "weiran.profiles" in str(exc) or "scenario" in str(exc), \
                "报错必须说清怎么补，不能只说「文件不存在」"
        else:
            raise AssertionError("缺文件竟然没报错")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_real_scenario_knowledge_loads_and_is_differentiated():
    """拿真场景跑一遍：27 个角色必须有不同的知情范围。"""
    k = load_knowledge(SIM, SCENARIO)
    assert len(k.by_actor) == 27, f"角色数不对：{len(k.by_actor)}"
    sigs = {e.knows for e in k.by_actor.values()}
    assert len(sigs) > 5, (
        f"只有 {len(sigs)} 种不同的知情范围 —— 差异化感知形同虚设")
    # 至少有一个角色确实有「知道但不公开」的事实，否则第二段永远不渲染
    assert any(e.hidden for e in k.by_actor.values()), "没有任何角色有隐藏事实"


# ---------------------------------------------------------------------------
# 态势渲染

def test_tier_boundaries():
    assert tier_of(0.0) == "很低"
    assert tier_of(0.149) == "很低"
    assert tier_of(0.15) == "偏低"
    assert tier_of(0.5) == "中等"
    assert tier_of(0.70) == "偏高"
    assert tier_of(0.85) == "很高"
    assert tier_of(1.0) == "很高"


def test_baseline_reads_like_a_normal_university():
    """常态下六个维度读出来必须都说得通 —— 这是档位分界是否合理的检验。"""
    base = WorldState.baseline().as_dict()
    text = render_state(base)
    for word in ("关注很低", "恐慌很低", "信任偏高", "稳定很高"):
        assert word in text, f"常态渲染里没有「{word}」：{text}"


def test_render_state_has_no_numbers_and_names_every_dimension():
    st = WorldState.baseline().as_dict()
    st["trust"] = 0.31
    text = render_state(st)
    assert not any(ch.isdigit() for ch in text), \
        f"态势文案里出现了数字 —— agent 会把它抄进帖文：{text}"
    for d in DIMENSIONS:
        assert DIMENSION_ZH[d] in text, f"少了维度 {d}"


def test_render_state_is_deterministic():
    st = {"attention": 0.4, "panic": 0.2, "trust": 0.6,
          "polarization": 0.3, "risk": 0.2, "stability": 0.7}
    prev = {k: v - 0.1 for k, v in st.items()}
    assert render_state(st, prev) == render_state(st, prev)
    assert len(render_state(st, prev)) > len(render_state(st)), \
        "给了上一轮却没写方向"


def _segment(text: str, dim: str) -> str:
    """从「关注很低、升温中；恐慌很低；…」里挑出某一维那一段。

    直接对整串做 `in` 断言会**恒真通过** —— 因为档位词夹在维度名与方向词
    中间（「信任中等、下滑中」），`"信任下滑" in text` 永远是 False。
    本文件早期版本就是踩了这个坑：一条断言方向写反的测试，
    在方向真的写反时照样通过。
    """
    for seg in text.split("；"):
        if seg.startswith(DIMENSION_ZH[dim]):
            return seg
    raise AssertionError(f"文案里没有维度 {dim}：{text}")


def test_direction_words_are_per_dimension_not_one_size_fits_all():
    """六维全上升时，六句话应各用各的词，不能是同一个句式。"""
    prev = {d: 0.5 for d in DIMENSIONS}
    st = {d: 0.6 for d in DIMENSIONS}
    text = render_state(st, prev)
    for d, (up, down) in DIRECTION_ZH.items():
        seg = _segment(text, d)
        assert up in seg, f"{d} 上升时那一段应说「{up}」：{seg}"
        assert down not in seg, f"{d} 上升时那一段不该说「{down}」：{seg}"


def test_direction_words_point_the_right_way_for_every_dimension():
    """**逐维钉住 (上升词, 下降词) 的顺序。**

    这一条是被一个真 bug 逼出来的：`trust` 与 `stability` 的方向词曾经写反，
    于是信任在**回升**时注入给 agent 的是「信任在下滑」。它不报错、
    读起来也通顺，只是让 agent 听到与事实相反的话 —— 而六维闭环的全部意义
    就是让 agent 对真实态势做反应。

    上下各测一次：任何一维的顺序反了，这里必失败。
    """
    expected = {
        #        维度            上升词    下降词
        "attention":    ("升温", "降温"),
        "panic":        ("加重", "缓解"),
        "trust":        ("回升", "下滑"),
        "polarization": ("扩大", "收窄"),
        "risk":         ("抬头", "回落"),
        "stability":    ("企稳", "动摇"),
    }
    assert DIRECTION_ZH == expected, "方向词表被改了；若是有意改的，同步改这里"

    for d, (up, down) in expected.items():
        prev = {x: 0.5 for x in DIMENSIONS}
        rising = _segment(render_state(dict(prev, **{d: 0.8}), prev), d)
        falling = _segment(render_state(dict(prev, **{d: 0.2}), prev), d)
        assert up in rising and down not in rising, \
            f"{d} 上升时应说「{up}」而不是「{down}」：{rising}"
        assert down in falling and up not in falling, \
            f"{d} 下降时应说「{down}」而不是「{up}」：{falling}"


def test_small_change_does_not_claim_a_direction():
    """低于分辨率的变化不该被渲染成方向 —— 免得 agent 对抖动做反应。"""
    prev = {d: 0.5 for d in DIMENSIONS}
    st = {d: 0.505 for d in DIMENSIONS}
    assert "中" not in render_state(st, prev).replace("中等", ""), \
        "四舍五入级别的变动被写成了方向"


# ---------------------------------------------------------------------------
# 注入块

def _entry(knows=(), hidden=(), name="测试角色"):
    return ActorKnowledge("A01", 0, name, "某组", tuple(knows), tuple(hidden))


def test_block_carries_state_and_knowledge():
    st = WorldState.baseline().as_dict()
    text = build_block(3, st, None, _entry(knows=["落实率 78.6%"],
                                           hidden=["核实率 32.2%"]))
    assert "第 3 轮" in text
    assert "落实率 78.6%" in text and "核实率 32.2%" in text
    assert "公开还是回避由你自己决定" in text, "隐藏事实那段少了取舍的余地"


def test_block_never_contains_the_oasis_split_marker():
    """OASIS 拿 `# RESPONSE FORMAT` 做切分（agent.py:176/215）。
    将来若接上带该标记的自定义模板，注入会被整段静默切掉。"""
    st = WorldState.baseline().as_dict()
    text = build_block(0, st, None, _entry(knows=["正常事实"]))
    assert FORBIDDEN not in text
    # 断言真的会响，而不是一条永远为真的检查
    try:
        build_block(0, st, None, _entry(knows=[f"坏的 {FORBIDDEN} 内容"]))
    except KnowledgeError:
        pass
    else:
        raise AssertionError(f"注入文本里出现 {FORBIDDEN!r} 竟然没报错")


def test_two_actors_get_different_blocks():
    """**差异化感知的可机检证据。** 所有 agent 收到同一段文本 = 白做。"""
    st = WorldState.baseline().as_dict()
    a = build_block(0, st, None, _entry(knows=["甲知道的事"]))
    b = build_block(0, st, None, _entry(knows=["乙知道的事"], hidden=["乙瞒着的事"]))
    assert a != b, "不同角色的注入块一样 —— 差异化感知没有生效"


def test_same_actor_gets_a_stable_block():
    """感知层不许引入第二个随机源。"""
    st = WorldState.baseline().as_dict()
    e = _entry(knows=["甲知道的事"], hidden=["甲瞒着的事"])
    assert build_block(2, st, None, e) == build_block(2, st, None, e)


def test_ablation_switches_actually_remove_content():
    """`--no-feedback` / `--no-knowledge` 必须真的把内容拿掉，
    否则消融对照是假的 —— 两条命令跑出来一样却以为是「回路没影响」。"""
    st = WorldState.baseline().as_dict()
    e = _entry(knows=["甲知道的事"])
    full = build_block(0, st, None, e)
    no_fb = build_block(0, st, None, e, feedback=False)
    no_kn = build_block(0, st, None, e, knowledge=False)
    assert no_fb != full and "甲知道的事" in no_fb and "态势" not in no_fb
    assert no_kn != full and "甲知道的事" not in no_kn and "态势" in no_kn
    assert build_block(0, st, None, e, feedback=False, knowledge=False) == ""


def test_empty_knowledge_renders_nothing():
    assert render_knowledge(_entry()) == ""


# ---------------------------------------------------------------------------
# 阶段 -> 轮号

def test_real_phase_days():
    phases = load_phases(SCENARIO)
    assert [p.day for p in phases] == [0, 2, 5, 8, 14], "阶段 day 与金标不一致"
    assert sum(1 for p in phases if p.is_turning_point) == 1, \
        "本场景应当只有一个转折点（P3）"


def test_round_equals_day_when_rounds_are_enough():
    phases = load_phases(SCENARIO)
    dpr, compressed = days_per_round(phases, rounds=15)
    assert dpr == 1.0 and not compressed, "轮数够的时候不该压缩"
    schedule, _, _ = phase_schedule(phases, rounds=15)
    assert sorted(schedule) == [0, 2, 5, 8, 14], \
        f"round=day 时阶段应落在各自的 day 上：{sorted(schedule)}"
    assert schedule[5][0].phase_id == "P3"


def test_compression_is_flagged_and_loses_nothing():
    """压缩必须被标出来，且**一个阶段都不能丢** ——
    丢掉一个阶段等于悄悄改掉场景，而曲线看上去仍然正常。"""
    phases = load_phases(SCENARIO)
    dpr, compressed = days_per_round(phases, rounds=3)
    assert compressed, "3 轮覆盖 14 天，必须标为压缩"
    assert dpr > 1.0
    schedule, _, _ = phase_schedule(phases, rounds=3)
    got = [p.phase_id for ps in schedule.values() for p in ps]
    assert got == [p.phase_id for p in phases], \
        f"压缩后阶段丢了或乱了：{got}"
    assert max(schedule) <= 2, "轮号越界了"


def test_last_phase_lands_on_the_last_round():
    """压缩映射的边界：最后一个阶段必须落在最后一轮，不能被舍掉。"""
    phases = load_phases(SCENARIO)
    for rounds in (2, 3, 4, 7, 14, 15):
        schedule, _, _ = phase_schedule(phases, rounds=rounds)
        assert max(schedule) == rounds - 1, \
            f"rounds={rounds} 时最后阶段落在第 {max(schedule)} 轮，越界或提前"


def test_too_few_rounds_is_rejected():
    phases = load_phases(SCENARIO)
    try:
        days_per_round(phases, rounds=1)
    except KnowledgeError:
        pass
    else:
        raise AssertionError("1 轮竟然被接受了")


def test_every_round_gets_an_active_phase_label():
    """与 `phase_schedule` 是两件事：**每一轮**都该有阶段标签，
    哪怕它这一轮不注入任何事件。

    轮=天时 15 轮里只有 5 轮有事件，但 15 轮都处在某个阶段里。
    如果这里跟着 schedule 一起只在 5 轮上给标签，曲线就没法按阶段切片 ——
    而「P3 之后那几轮发生了什么」正是本场景最该回答的问题。"""
    phases = load_phases(SCENARIO)
    schedule, dpr, _ = phase_schedule(phases, rounds=15)
    label = active_phase_by_round(phases, rounds=15, dpr=dpr)

    assert sorted(label) == list(range(15)), "有轮次没拿到标签"
    assert len(schedule) == 5 < len(label), \
        "这一条测的正是「有事件的轮」与「有标签的轮」数量不同"
    # 第 0 轮在第 1 个阶段里（day 0 就是 P1 当天）；第 14 轮在第 5 个阶段里。
    assert label[0] == "P1"
    assert label[14] == "P5"
    # 阶段标签只在阶段当天及之后才切换。
    assert label[4] == label[2] == "P2", "day4 / day2 之间没有阶段边界"
    assert label[5] == "P3", "day5 是 P3 当天，边界日必须算进新阶段"


def test_last_round_always_lands_in_the_last_phase():
    """`dpr = span / (rounds-1)`，所以最后一轮的标称 day 恒等于 `span`。

    但 `(rounds-1) * (span/(rounds-1))` 在浮点下未必正好等于 `span`
    —— 差一点点就会让最后一轮被判成**倒数第二个**阶段，即整段收束期的
    曲线全部挂错标签。这个容差是为此而留的。
    `rounds` 从 2 遍历到 15，覆盖压缩与 round=day 两侧。"""
    phases = load_phases(SCENARIO)
    for rounds in range(2, 16):
        dpr, _ = days_per_round(phases, rounds)
        label = active_phase_by_round(phases, rounds, dpr)
        assert label[rounds - 1] == phases[-1].phase_id, \
            (f"rounds={rounds}（每轮 {dpr:.6f} 天）时最后一轮落进了 "
             f"{label[rounds - 1]!r} 而不是 {phases[-1].phase_id!r}")


def test_event_rounds_and_active_phase_agree_under_round_equals_day():
    """推荐口径（`--rounds 15`）下，两个判据必须**恒等**。

    钉住的是「同一行日志里不会冒出两个互相矛盾的阶段标签」。
    `simulate.py` 在两者冲突时优先用注入的那个，而轮=天时它们不冲突，
    所以那条优先规则不需要额外解释。"""
    phases = load_phases(SCENARIO)
    dpr, compressed = days_per_round(phases, rounds=15)
    assert not compressed and dpr == 1.0
    schedule, _, _ = phase_schedule(phases, rounds=15)
    label = active_phase_by_round(phases, 15, dpr)
    assert len(schedule) == 5
    for r, evs in schedule.items():
        assert [p.phase_id for p in evs] == [label[r]], \
            f"轮 {r} 注入 {[p.phase_id for p in evs]}，标签却是 {label[r]}"


def test_compression_may_shift_the_active_phase_one_step_back():
    """压缩模式下两者**允许**不一致 —— 把这个已知差异钉住，
    免得日后有人「顺手改成一致」，把两个不同的判据混成一个。

    8 轮覆盖 14 天（每轮 2.0 天）：第 2 轮注入 day 5 的 P3，
    但这一轮的起点是 day 4.0，还没走到 P3，所以标签仍是 P2。"""
    phases = load_phases(SCENARIO)
    dpr, compressed = days_per_round(phases, rounds=8)
    assert compressed and dpr == 2.0
    schedule, _, _ = phase_schedule(phases, rounds=8)
    label = active_phase_by_round(phases, 8, dpr)
    fired = [p.phase_id for p in schedule[2]]
    assert fired == ["P3"], f"8 轮时第 2 轮该注入 P3，实际 {fired}"
    assert label[2] == "P2", \
        f"压缩模式下这一轮的起点仍在 P2 地盘，标签应为 P2，实际 {label[2]}"


def test_active_phase_accepts_unsorted_phases():
    """唯一正确的来源是「day 最小的那个最近阶段」，与输入顺序无关。"""
    phases = [Phase(phase_id="P3", day=8, trigger="x"),
              Phase(phase_id="P1", day=0, trigger="x"),
              Phase(phase_id="P2", day=3, trigger="x")]
    label = active_phase_by_round(phases, rounds=9, dpr=1.0)
    assert label[0] == "P1" and label[3] == "P2" and label[8] == "P3", \
        f"乱序输入被判错了：{label}"


# ---------------------------------------------------------------------------
# 挂到 agent 上（需要 OASIS，单独一段）

class _FakeAgent:
    """只带 social_agent_id 的替身，够 inject_round_context 用。"""

    def __init__(self, uid):
        self.social_agent_id = uid


def test_injection_lands_on_each_agent_separately():
    k = load_knowledge(SIM, SCENARIO)
    agents = [_FakeAgent(i) for i in range(3)]
    st = WorldState.baseline().as_dict()
    blocks = inject_round_context(
        agents, round_index=0, state=st, prev=None, knowledge=k)
    assert len(blocks) == 3
    for a in agents:
        assert getattr(a, INJECT_ATTR, ""), "有 agent 拿到空块"
    assert len({getattr(a, INJECT_ATTR) for a in agents}) == 3, \
        "三个 agent 拿到了同样的块 —— 挂错了或共用了"


def test_unknown_agent_id_is_loud():
    """画像与知情映射不同源时必须响，不能默默给个空块。"""
    k = load_knowledge(SIM, SCENARIO)
    try:
        inject_round_context(
            [_FakeAgent(999)], round_index=0,
            state=WorldState.baseline().as_dict(), prev=None, knowledge=k)
    except KnowledgeError as exc:
        assert "999" in str(exc)
    else:
        raise AssertionError("找不到角色的 agent 被静默放过了")


def test_knowledge_can_be_switched_off_entirely():
    agents = [_FakeAgent(0)]
    inject_round_context(
        agents, round_index=0, state=WorldState.baseline().as_dict(),
        prev=None, knowledge=None)
    assert isinstance(getattr(agents[0], INJECT_ATTR), str)


def test_install_injection_is_idempotent_and_replaces_astep():
    """注入点必须是真装上、且只装一次。

    留一个「装没装上都不报错」的口子，等于让「注入没生效」和
    「注入生效了但没差别」长得一模一样 —— 这正是要防的那类失败。
    """
    try:
        from oasis.social_agent.agent import SocialAgent
    except ImportError:  # pragma: no cover - 仿真内核未安装时跳过
        print("      （未安装 camel-oasis，跳过）")
        return

    from weiran.perception import install_injection

    first = install_injection()
    assert first is True, "首次安装应当返回 True"
    assert install_injection() is False, "重复安装应当返回 False"
    assert getattr(SocialAgent, "_weiran_injection_installed", False)
    assert SocialAgent.astep.__name__ == "astep", "包装函数改坏了方法名"


def test_the_wrapper_really_prepends_the_block_to_the_model_message():
    """**这是「注入进了 prompt」这条链上中间那一环，此前没有任何机检。**

    已有的测试只做到「属性挂上去了」，而属性到「模型真的看见」之间还隔着
    一个类级包装函数。它写错了不会报错 —— 只会让注入静默消失，而那在六维
    曲线上与「注入生效了但没差别」完全一样，正是本项目专门要防的那类失败。

    做法：先把 `SocialAgent.astep` 换成一个记录器，再让 `install_injection`
    去包那个记录器，于是记录器收到的就是包装层真正递给下游的东西。
    （用 `__new__` 造实例，避开需要模型的 `__init__`，全程离线。）
    """
    try:
        import asyncio

        from camel.messages import BaseMessage
        from oasis.social_agent.agent import SocialAgent
    except ImportError:  # pragma: no cover - 仿真内核未安装时跳过
        print("      （未安装 camel-oasis，跳过）")
        return

    from weiran.perception import install_injection

    seen: dict = {}

    async def recorder(self, input_message, response_format=None):
        seen["msg"] = input_message
        return "ok"

    saved_astep = SocialAgent.astep
    saved_flag = getattr(SocialAgent, "_weiran_injection_installed", False)
    try:
        SocialAgent.astep = recorder                  # 1. 换成记录器
        SocialAgent._weiran_injection_installed = False
        install_injection()                           # 2. 让包装层包住记录器

        agent = SocialAgent.__new__(SocialAgent)      # 不跑 __init__，不需要模型
        body = BaseMessage.make_user_message(role_name="User", content="原始正文")

        setattr(agent, INJECT_ATTR, "【态势】恐慌偏高")
        asyncio.run(SocialAgent.astep(agent, body))
        got = seen["msg"]
        assert got.content == "【态势】恐慌偏高\n\n原始正文", \
            f"注入没被拼到消息前面：{got.content!r}"
        assert got.role_name == "User", "role_name 被改掉了"

        # 空串 = 本轮不注入，必须**原样透传**。若这里也重建消息，
        # 每一轮的记忆里都会多一个空段落，而没有任何地方会提示。
        seen.clear()
        setattr(agent, INJECT_ATTR, "")
        asyncio.run(SocialAgent.astep(agent, body))
        assert seen["msg"] is body, "空注入竟然也重建了消息"
    finally:
        SocialAgent.astep = saved_astep
        SocialAgent._weiran_injection_installed = saved_flag


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
