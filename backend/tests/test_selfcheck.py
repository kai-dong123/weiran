"""装置自证（`weiran.selfcheck`）的测试。

自证本身跑一遍就知道过没过（`python -m weiran.selfcheck`）。这里守的**不是**
「那 9 个反例过不过」，而是**自证这台 harness 自己会不会骗人** —— 四种不会报错、
只会全绿的失败：

  1. **反例对某条断言不表态。** 扰动顺手把第三条也掀翻了，而反例只声明了它想
     翻的那两条 —— 观众看到 9/9，以为每一条都被盯住了，其实有一条没人管。
  2. **`untouched` 只比结论不比读数。** 「这一条没动」如果用 verdict 判断，
     那么一个把读数改掉却恰好没掀翻结论的扰动，会被记成「没动」。读书的人
     看不出区别，因为两件事都写着「没动」。
  3. **哨兵把检查抹成空转。** `untouched=(*,)` 的唯一意思是「每条都不许动」。
     它一旦在某处被展开成空集合，那条反例就什么都没比，却仍然报 ✅ ——
     恒等反例用的正是这个哨兵，而它全部的职责就是抓扰动器的副作用。
  4. **改动层写错一个字。** 决定简报重不重算的是 `apply` 的返回值，写错不成
     报错：简报不重算，反例读到改动之前的中间产物，静默退化成一个恒等反例，
     然后报「通过」。

第 1、2、3、4 条都各配一个**故意写坏的反例**，断言 harness 会把它拒掉。
少了这些，上面那些检查是装饰 —— 一个什么都不检查的实现也能全绿。
`run(cases=...)` 这个口子就是为它们留的。

与另几套一样，**刻意不依赖 pytest**：普通 assert + 函数 + 兜底 runner。
数据一律取**入库的真产出**与**真金标**，不另造合成输入。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weiran import gold_check as G  # noqa: E402
from weiran import selfcheck as S  # noqa: E402
from weiran.config import REPO_ROOT, ensure_console_encoding  # noqa: E402

_cache: dict = {}


def _report() -> dict:
    """整个自证跑一遍（约 9 个反例，跑一次缓存）。"""
    if "report" not in _cache:
        _cache["report"] = S.run()
    return _cache["report"]


def _refuses(case: S.Case) -> list[str]:
    """注入一个反例，返回 harness 记下的问题清单（空 = 没拦下来）。"""
    rep = S.run(cases=[case])
    for n, _c, checks in rep["failures"]:
        if n == 1:
            return checks
    return []


def _noop(ctx: dict) -> set[str]:
    return {"产出"}


# ---------------------------------------------------------------------------
# 1. 自证本身：9 个反例都落在它说的那一格
# ---------------------------------------------------------------------------

def test_every_negative_example_lands_where_it_says():
    """9/9，且入口一致性成立 —— 后者不成立的话，那些反例证明的是另一个东西。"""
    rep = _report()
    assert rep["entry_ok"] is True, "自证用的链路与产品入口不等价"
    assert not rep["failures"], rep["failures"]
    assert len(rep["results"]) == 9, [r["n"] for r in rep["results"]]
    assert all(r["ok"] for r in rep["results"]), \
        [r for r in rep["results"] if not r["ok"]]


def test_all_five_output_grids_are_triggered():
    """装置声称它有五个输出格，那五个格就都得真被触发过。

    只断言「没缺」是不够的：缺格清单如果是写死的空表，这条看着也过。
    所以比的是**实际触发出来的那五个**与**声称的那五个**相等。
    """
    rep = _report()
    assert rep["missing_grids"] == [], rep["missing_grids"]
    assert set(rep["grids"]) == set(S.GRIDS), rep["grids"]
    assert len(S.GRIDS) == 5


def test_the_base_run_reproduces_the_committed_table():
    """自证用的基准就是**入库那张表**：三个数、采信名单，一个都不许飘。

    这条把 L3 跟第 2.6 层钉在一起。自证如果是在另一份数据上跑的，
    它证明的是另一张表 —— 而读者会以为它证明的是仓库里这张。
    """
    rep = _report()
    assert rep["base_counts"] == {G.PASS_: 3, G.FAIL_: 5, G.UNDECIDED: 2}, \
        rep["base_counts"]
    assert rep["ids"] == [f"AS-{i}" for i in range(1, 11)], rep["ids"]


# ---------------------------------------------------------------------------
# 2. harness 的四条闸：各配一个故意写坏的反例，断言它被拒掉
# ---------------------------------------------------------------------------

def test_a_case_that_stays_silent_about_an_assertion_is_refused():
    """闸 1：对某条断言不表态的反例，必须被拒 —— 且要点名漏了哪几条。"""
    checks = _refuses(S.Case(layer="产出", what="故意写坏：一个字都不声明",
                             apply=_noop))
    assert checks, "对全部断言都不表态的反例竟然通过了"
    joined = " ".join(checks)
    assert "没有对全部断言表态" in joined, checks
    assert "AS-1" in joined and "AS-10" in joined, checks


def test_the_identity_sentinel_compares_every_row():
    """闸 2 + 闸 3：`untouched=(*,)` 必须**逐条比读数**，不只是比结论。

    这里注入的扰动是「只把 trust(P1) 抬一点点」：单调下坠仍成立、AS-6 的
    门槛仍过得去 —— **一条结论都不翻，只有读数变了**。于是：
      · 它证明哨兵确实展开成了全部 10 条（漏展开则一条都不比，见闸 3）；
      · 它证明比的是 `values` 而不只是 `verdict`（只比结论则抓不到）。
    两者缺一，这条反例就会被记成 ✅。
    """
    def nudge(ctx: dict) -> set[str]:
        S._set_phase(ctx, "P1", trust=0.69)
        return {"产出"}

    checks = _refuses(S.Case(layer="产出", what="故意写坏：读数动了，结论没动",
                             apply=nudge, untouched=(S.ALL_ASSERTIONS,)))
    assert checks, "读数动了却声称「每条都没动」，竟然通过了"
    joined = " ".join(checks)
    assert "读数却变了" in joined, joined
    assert "AS-1" in joined, joined


def test_a_mistyped_layer_is_refused():
    """闸 4：改动层名称写错必须被拒。

    写错不成报错，是**简报不重算** —— 反例读到改动之前的中间产物，静默退化成
    恒等反例，还报 ✅。所以这里用「产出」的错别字，断言 harness 认得出。
    """
    checks = _refuses(S.Case(layer="产物", what="故意写坏：改动层错别字",
                             apply=_noop, untouched=(S.ALL_ASSERTIONS,)))
    assert checks, "改动层写了「产物」，竟然通过了"
    joined = " ".join(checks)
    assert "改动层" in joined, joined


def test_a_case_that_names_a_nonexistent_interpretation_is_refused():
    """闸 5：点名一个不存在的解读 id，必须被拒。

    不拒的后果是**静默**的：`its.get(iid)` 给 None，若 `want` 恰好是 False，
    就对上了 —— 看起来像「这条解读如我所愿地闭嘴了」。
    """
    checks = _refuses(S.Case(layer="产出", what="故意写坏：解读 id 不存在",
                             apply=_noop, untouched=(S.ALL_ASSERTIONS,),
                             expect_interpretations={"no_such_reading": False}))
    assert checks, "点名了不存在的解读，竟然通过了"
    assert "不存在的解读" in " ".join(checks), checks


# ---------------------------------------------------------------------------
# 3. 渲染：坏消息要说得出来
# ---------------------------------------------------------------------------

def test_render_shouts_when_the_device_fails_its_own_check():
    """未通过时，渲染必须明说，而不是照旧印一句「全部由输入决定」。"""
    good = S.render(_report())
    assert "装置没有通过自证" not in good
    assert "9/9 个反例落在预期格" in good

    bad = S.render(S.run(cases=[S.Case(layer="产出", what="故意写坏",
                                       apply=_noop)]))
    assert "装置没有通过自证" in bad, bad
    assert "❌" in bad, "没通过的条目要标出来"
    assert "五个输出格全部由输入决定" not in bad, \
        "有反例失败时还宣称「全部由输入决定」，那是替装置说好话"


# -- 简易 runner（与其余测试文件保持一致）----------------------------------

def _run() -> int:
    # 与另几套同理：被测代码会往控制台印 ✅/❌，Windows 中文控制台是 GBK。
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
