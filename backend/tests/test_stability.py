"""稳定性与可复现性的测试。

**全部离线。不联网、不调 LLM。** 唯一碰上游的地方是 `importlib.util.find_spec`
拿到 `oasis` 的**路径**（只解析、不导入），然后读源码的 AST。

这一套守的是一句已经写进文档、却长期**没人能对着它跑测试**的话：

    「六维曲线无法端到端复现……正式口径是多 seed 跑 N 次报均值与方差」

散文的问题不是不准，是**上游改一行它照样成立**。所以这里把它拆成两条
可机检的性质：

  1. **我们这边是确定的。** 引擎与简报那几个模块里不许出现任何随机抽签
     （`test_engine_modules_are_draw_free`）—— 曲线跨次不一样，**不是**
     引擎的错，这条把这个归因钉住。
  2. **上游那边能收窄，而且我们真的收窄到了。** `seed_process(k)` 必须钉住
     **OASIS 抽签用的那一个 RNG 对象** —— 不是「某个 RNG」，是那一个。
     这一条就是本文件的主体：它不能靠「扫文本里有没有 `random.sample`」来断言，
     那样断言的是文本长什么样，不是**名字绑到了哪个对象**。

**为什么第 2 条值得这么多测试。** 一根断了的杆最贵的地方不是它没用，是
**它看起来有用**：如果哪天上游把 `random` 换成一个私有的 `random.Random()`
实例，我们这边什么都不会报错，播种静悄悄地失效，然后有人拿一批真金白银的臂
去算「跨 seed 离散」—— 量出来的是没被钉住的那部分，而结论会写成「播种收窄了
多少」。下面每一条都是冲着这个失效形态去的。

    python backend/tests/test_stability.py
    pytest backend/tests/
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import random
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weiran.config import ensure_console_encoding  # noqa: E402
from weiran.stability import (  # noqa: E402
    SAMPLING_NOTE,
    VARIANCE_SOURCES,
    seed_process,
)

REPO = Path(__file__).resolve().parents[2]


# -- 上游源码：只解析路径，不导入 -------------------------------------------

def _parse(src: str) -> ast.Module:
    """解析上游源码，**吞掉它自己的 DeprecationWarning**。

    上游有些文件里写着 `'\\%'` 这类无效转义序列，`ast.parse` 会为它发一条
    `SyntaxWarning`：那不是我们的问题，但它会以 `<unknown>:58` 的形式混进
    测试输出，看上去像是这一套测试在报错。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.parse(src)


def _upstream_root(pkg: str) -> pathlib.Path | None:
    """拿到包的源码目录。**不导入它** —— 导入 `oasis` 会拉起
    `sentence_transformers` 与 torch，几秒钟起步，而这里只需要读文件。

    找不到就返回 None，由调用方决定是跳过还是失败。
    """
    spec = importlib.util.find_spec(pkg)
    if spec is None or not spec.submodule_search_locations:
        return None
    return pathlib.Path(list(spec.submodule_search_locations)[0])


def _draw_sites(root: pathlib.Path) -> list[tuple[str, int]]:
    """全树里所有 `random.<什么>` 的**属性访问**，返回 (相对路径, 行号)。

    数的是属性访问而不是调用，因为它同时覆盖 `random.sample(...)` 与
    `random.random()`；这两者都是对同一个模块对象的读取。
    """
    found: list[tuple[str, int]] = []
    for p in sorted(root.rglob("*.py")):
        src = p.read_text(encoding="utf-8", errors="replace")
        if "random" not in src:
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        for node in ast.walk(_parse(src)):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "random"):
                found.append((rel, node.lineno))
    return found


def _calls_of(root: pathlib.Path, names: set[str]) -> list[tuple[str, int, str]]:
    """全树里所有 `func` 的**源码文本**落在 `names` 里的调用。"""
    found: list[tuple[str, int, str]] = []
    for p in sorted(root.rglob("*.py")):
        src = p.read_text(encoding="utf-8", errors="replace")
        rel = str(p.relative_to(root)).replace("\\", "/")
        for node in ast.walk(_parse(src)):
            if isinstance(node, ast.Call) and ast.unparse(node.func) in names:
                found.append((rel, node.lineno, ast.unparse(node.func)))
    return found


# -- 0. 我们这边是确定的 ----------------------------------------------------
#
# 这一条是**归因**测试，不是功能测试：它回答「曲线跨次不一样，是谁的错」。
# 本项目在三个地方自认过「一份产出无法端到端复现」，而每次都要重新论证一遍
# 「不是引擎的错」—— 论证的成本就在这里一次性付掉。

#: 全程不许出现随机抽签的模块。**这是一份白名单，不是「扫出来的结果」**：
#: 名单外的模块要么是驱动层（`simulate.py` 自己就要调 `random.seed`），
#: 要么是证据库（`store.py` 用 `uuid4` + 挂钟给每行发 id，与六维曲线无关）。
_DRAW_FREE = ("world_state.py", "perception.py", "brief.py", "gold_check.py",
              "stability.py")


#: 会**取**一个随机数的调用。注意 `random.seed` 不在里面：它是**写**，
#: 而且是本项目唯一允许出现在这批模块里的 random 调用（就在 `stability.py`）。
_DRAWING_CALLS = frozenset({
    "random.random", "random.uniform", "random.sample", "random.choice",
    "random.choices", "random.shuffle", "random.randint", "random.randrange",
    "random.gauss", "random.getrandbits",
})

#: 挂钟。它们不叫「随机」，但同样让同一份输入两次跑出不同的输出。
_CLOCK_CALLS = frozenset({"time.time", "time.monotonic", "time.perf_counter",
                          "datetime.now", "datetime.utcnow"})


def test_engine_modules_are_draw_free():
    """引擎、感知、简报、金标对照、本模块：一次随机抽签都不许有。

    **这条是会变红的** —— 谁哪天在引擎里加一个抖动「让它看起来更真」，
    它会红，而那时跨 seed 的可比性已经没有了。这正是它存在的理由。

    查四类东西：

      - `from random import ...` / `from uuid import ...`：把一个**名字**
        绑进来，是私有 RNG 与 uuid4 的常见入口；
      - 任何取随机数的调用（`_DRAWING_CALLS`），以及 `np.random.*`；
      - `uuid.uuid4()`；
      - 挂钟（`_CLOCK_CALLS`）。

    **`import random` 与 `random.seed(...)` 是允许的，而且只允许在
    `stability.py` 里** —— 那里是唯一一处「写入 RNG」的地方，正是本模块的
    主题。别处出现就说明有人在这批本该确定的模块里动了随机状态。
    """
    problems: list[str] = []
    for name in _DRAW_FREE:
        p = REPO / "backend" / "weiran" / name
        assert p.is_file(), f"{name} 不在了 —— 这份白名单过期了，请更新它"
        tree = _parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in ("random", "uuid"):
                    problems.append(
                        f"{name}:{node.lineno} from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    # `import random` 在 stability.py 里是正当的（它就要播种）；
                    # 在别处出现意味着有人准备在这批模块里抽签。
                    if a.name == "random" and name != "stability.py":
                        problems.append(f"{name}:{node.lineno} import random")
                    if a.name.split(".")[0] == "uuid":
                        problems.append(f"{name}:{node.lineno} import {a.name}")
            elif isinstance(node, ast.Call):
                fn = ast.unparse(node.func)
                if fn.startswith(("np.random.", "numpy.random.")):
                    problems.append(f"{name}:{node.lineno} {fn}()（numpy RNG）")
                elif fn in _DRAWING_CALLS:
                    problems.append(f"{name}:{node.lineno} {fn}()")
                elif fn == "uuid.uuid4" or fn == "uuid4":
                    problems.append(f"{name}:{node.lineno} {fn}()")
                elif fn == "random.seed" and name != "stability.py":
                    problems.append(
                        f"{name}:{node.lineno} random.seed()（播种只该在 "
                        "stability.py 里发生）")
                elif fn in _CLOCK_CALLS:
                    problems.append(f"{name}:{node.lineno} {fn}()（挂钟）")
    assert not problems, (
        "这些「本该确定」的模块里出现了随机抽签或挂钟：\n  "
        + "\n  ".join(problems)
        + "\n\n加了抖动之后，同一份输入两次跑出来的曲线就不一样了 —— "
          "而跨 seed 的离散正是靠「引擎这边不动」才可归因的。")


def test_the_draw_free_allowlist_names_its_exceptions():
    """白名单自己也要被钉住 —— 免得下一个人靠**扩大白名单**来让它变绿。

    两个刻意的例外必须留在名单外，而且理由要写在上面那段注释里：
    `simulate.py`（它就要调 `random.seed`）与 `store.py`（`uuid4` + 挂钟）。
    """
    assert "simulate.py" not in _DRAW_FREE, (
        "simulate.py 进了白名单 —— 它正是调用 random.seed 的那一个")
    assert "store.py" not in _DRAW_FREE, (
        "store.py 进了白名单 —— 它用 uuid4 与挂钟给每一行发 id，"
        "不是无抽签的；把它加进来等于让这条检查少看一个模块")
    assert "stability.py" in _DRAW_FREE, (
        "stability.py 自己必须无抽签：它只有 seed_process 一处 random，"
        "且是**写**不是抽")


# -- 1. seed_process 的行为 -------------------------------------------------

def test_seed_process_none_is_a_true_no_op():
    """`seed=None` 必须**一个随机数都不动**。

    默认路径的行为一个字节都不能变 —— 这是本项目对默认值的一贯要求：
    默认不动，由调用方显式选。这里用 `random.getstate()` 直接比对内部状态，
    而不是「跑两次结果一样」那种弱断言。
    """
    random.seed(12345)
    before = random.getstate()
    out = seed_process(None)
    assert random.getstate() == before, (
        "seed_process(None) 动了全局 RNG —— 默认路径被悄悄改了行为")
    assert out == {"seeded": False}, f"返回值形状变了：{out}"


def test_seed_process_really_pins_the_global_rng():
    """给定种子时必须**真的钉住标准库那个全局 RNG**。

    断言方式是**行为**的而不是结构的：播完种之后取的随机数序列，
    必须与「直接用 `random.seed(k)` 再取」逐位相同。
    一个把 `random.seed` 换成空操作、或者去播一个私有 `Random()` 实例的
    实现，会在这里红 —— 而那种实现在「扫文本找 random.sample」式的检查下
    是看不出来的。
    """
    random.seed(999)                    # 先弄乱，确保下面的相等不是巧合
    seed_process(7)
    a = [random.random() for _ in range(5)]
    random.seed(7)
    b = [random.random() for _ in range(5)]
    assert a == b, (
        "播种后取到的序列与 random.seed(k) 后的序列不同 —— "
        "这根杆没有搭在全局 RNG 上")


# -- 2. 这根杆确实搭在 OASIS 上 ---------------------------------------------

def test_oasis_draws_from_the_standard_library_random_module():
    """上游用的必须是**标准库那个全局模块**，而不是一个私有 RNG 实例。

    只看两件事：`social_platform/` 下那两个文件 `import random`（而不是
    `from random import Random`），且模块里**没有** `random.Random(...)`。
    后者是关键：那个对象是一个独立的流，`random.seed()` 影响不到它，
    而它同样长得像 `random.sample(...)`。扫文本是看不出这个区别的。
    """
    root = _upstream_root("oasis")
    if root is None:
        # 环境里没有 oasis 时跳过（与 test_brief.py 对缺产出的处理一致），
        # 但**先确认它本该在**，否则这个跳过会随着时间变成永久的。
        req = (REPO / "requirements.txt")
        text = req.read_text(encoding="utf-8") if req.is_file() else ""
        assert "camel-oasis" in text, (
            "环境里没有 oasis，而 requirements.txt 里也没有它 —— "
            "这个跳过不该是这个原因")
        print("      （跳过：环境里没有 oasis）")
        return

    for name in ("social_platform/platform.py", "social_platform/recsys.py"):
        src = (root / name).read_text(encoding="utf-8", errors="replace")
        tree = _parse(src)
        imports = {ast.unparse(n) for n in ast.walk(tree)
                   if isinstance(n, (ast.Import, ast.ImportFrom))}
        assert "import random" in imports, (
            f"{name} 不再 import 标准库 random —— 用的是别的对象，"
            f"播种它未必覆盖得到。实际 import：{sorted(imports)}")
        priv = [ast.unparse(n) for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and ast.unparse(n.func).startswith("random.Random")]
        assert not priv, (
            f"{name} 里出现了私有 RNG 实例 {priv} —— "
            "random.seed() 影响不到它，我们的播种在这条路上会静默失效")


def test_no_upstream_module_reseeds_the_global_rng():
    """上游不许自己调 `random.seed()`。

    我们靠「进程里播一次种」覆盖上游抽签，前提是**没有别人再播一次**。
    上游哪天加一句 `random.seed(0)`（很常见的一种「让它可复现」的写法），
    我们的 `--seed` 就变成一个不影响结果的数字 —— 而产出里照样记着它。

    `random.Random(...)` 私有实例不算，那些是各自独立的流；camel 在
    datasets / environments / toolkits 一侧有七处，都不在 oasis 拉起的
    模块里（oasis 只 import camel 的 agents / messages / models / prompts /
    toolkits / types / memories / embeddings）。
    """
    problems = []
    for pkg in ("oasis", "camel"):
        root = _upstream_root(pkg)
        if root is None:
            continue
        for rel, lineno, fn in _calls_of(
                root, {"random.seed", "np.random.seed", "numpy.random.seed"}):
            problems.append(f"{pkg}/{rel}:{lineno} {fn}")
    assert not problems, (
        "上游自己给全局 RNG 播了种 —— 我们的 --seed 从此不再是那根杆：\n  "
        + "\n  ".join(problems))


def test_the_oasis_draw_surface_is_still_exactly_the_one_we_verified():
    """全库的抽签点应当**仍是核过的那六处**，且仍分在两个文件里。

    这条会在上游升级加了一处抽签时变红 —— 这正是它存在的理由：新加的那一处
    在不在我们的路径上、播种覆不覆盖得到，得有人重新看一眼，而不是
    拿着「已核 6 处」这句话继续往下走。

    六处里**只有 platform.py 那一处**在路径上（其余四处分属 RANDOM 推荐器、
    被阈值挡掉的 coarse_filtering、以及 TWITTER 那条我们没走的路；
    逐一核过的依据写在 `weiran/stability.py` 的模块 docstring 里）。
    """
    root = _upstream_root("oasis")
    if root is None:
        print("      （跳过：环境里没有 oasis）")
        return

    sites = _draw_sites(root)
    files = sorted({rel for rel, _ in sites})
    assert files == ["social_platform/platform.py",
                     "social_platform/recsys.py"], (
        f"OASIS 的抽签点跑到了别的文件里：{files} —— "
        "「播一次种够不够」这个结论要重新核")
    assert len(sites) == 6, (
        f"OASIS 里的抽签点从 6 处变成了 {len(sites)} 处：{sites}\n"
        "新增的那一处需要重新判断在不在我们的路径上、播种覆不覆盖得到")


def test_numpy_random_is_not_used_upstream():
    """上游不许走 numpy 的 RNG。

    「播一次种就够」这句话的前提是**所有抽签都走标准库那一个模块**。
    numpy 有自己的全局状态，`random.seed()` 碰不到它。
    """
    for pkg in ("oasis", "camel"):
        root = _upstream_root(pkg)
        if root is None:
            continue
        hits = []
        for p in sorted(root.rglob("*.py")):
            src = p.read_text(encoding="utf-8", errors="replace")
            if "random" not in src:
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            for node in ast.walk(_parse(src)):
                if (isinstance(node, ast.Attribute)
                        and ast.unparse(node.value) in ("np.random",
                                                        "numpy.random")):
                    hits.append(f"{pkg}/{rel}:{node.lineno}")
        assert not hits, (
            "上游开始用 numpy 的 RNG 了，而 random.seed() 覆盖不到它：\n  "
            + "\n  ".join(hits))


# -- 3. 表本身 --------------------------------------------------------------

def test_every_variance_source_can_say_where_it_lives_and_what_it_costs():
    """表里每一行都要能回答四个问题：叫什么、在哪一层、为什么变、能不能收窄。

    这不是形式检查 —— 这张表是「哪一层会变」的**唯一出处**，缺一项就意味着
    某个归因无处可查，而结论会照旧被写出来。
    """
    assert VARIANCE_SOURCES, "方差来源表是空的"
    keys = [v.key for v in VARIANCE_SOURCES]
    assert len(set(keys)) == len(keys), f"key 重复：{keys}"
    for v in VARIANCE_SOURCES:
        assert v.key.isascii(), f"key 应当是稳定标识（ASCII）：{v.key!r}"
        for field in ("layer", "detail", "seedable", "where"):
            assert getattr(v, field).strip(), f"{v.key} 的 {field} 是空的"
        # `where` 是给人去核的，得像个位置。
        assert any(mark in v.where for mark in (".py", "()")), (
            f"{v.key}.where 不像一个代码位置：{v.where!r}")


def test_the_unseedable_layer_is_marked_as_unseedable():
    """`asyncio_interleaving` 必须被标成**不能播种**，而且理由要写出来。

    这一条守的是「不要把播种说成一劳永逸」：并发交错不是随机数，是调度顺序，
    播种种不到它。谁哪天把这一行改成「能收窄」，得先有办法收窄它。
    """
    by_key = {v.key: v for v in VARIANCE_SOURCES}
    assert "asyncio_interleaving" in by_key, "不能播种的那一层从表里消失了"
    v = by_key["asyncio_interleaving"]
    assert "不能播种" in v.seedable, (
        f"asyncio 交错被说成可播种了：{v.seedable!r} —— "
        "它不是一个随机数，random.seed() 影响不到它")


def test_the_sampling_note_does_not_promise_reproducibility():
    """落盘那句话**不许承诺可复现**。

    它会被写进每一份产出的 `meta.sampling_note`，读它的人多半不会再看别的
    文档。所以它必须同时说清三件事：种子只被端点「尽力」遵守、进程 RNG 的
    播种覆盖不到 asyncio 交错、跨次比较要按「多次看离散」读。
    """
    assert "尽力" in SAMPLING_NOTE, "没写清端点只是「尽力」遵守"
    assert "asyncio" in SAMPLING_NOTE, "没写清哪一层播种覆盖不到"
    assert "离散" in SAMPLING_NOTE, "没告诉读者跨次比较该怎么读"

    # 「复现」这个词不是不能出现 —— 出现「不要把单次产出当可复现结论」正是
    # 我们要的。要禁的是**肯定句式**：谁把这句话改写成「本次产出可复现」，
    # 这条会红。所以逐处看它是不是落在否定语境里。
    for m in re.finditer("复现", SAMPLING_NOTE):
        window = SAMPLING_NOTE[max(0, m.start() - 12):m.end() + 2]
        assert any(neg in window for neg in ("不要", "不", "无法", "覆盖不到")), (
            f"「复现」出现在了肯定语境里：…{window}… —— "
            "这句话恰恰不能承诺可复现")


# -- 简易 runner（与其余测试文件保持一致）----------------------------------

def _run() -> int:
    # 与其余十二套一致：兜底 runner 先把控制台编码策略设好。
    # `test_handbook.py` 有 AST 测试盯着这一行，缺了会红。
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
