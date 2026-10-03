"""可复现性检查：分清哪一层是确定的、哪一层不是。

**这个脚本的存在本身就是一次实测的产物。** 第一版它只做一件事：
同一条命令跑两次，比较六维曲线。结果两次**完全不同** —— 于是浮出一个
必须说清楚的结论，而不是一个可以含糊过去的数字：

    「未然」的可复现性是**分层**的，不是一句「我们可复现」能概括的。

    第 3 层  端到端（同一条命令 → 同一条曲线）   ❌ 不可复现，且**原理上做不到**
    第 2.7 层 自证（反例喂进去，结论跟着动吗）   ✅ 确定，`weiran.selfcheck` 本体
    第 2.6 层 金标对照（同一份产出 → 同一张表）  ✅ 确定，`test_gold_check.py` 锁住
    第 2.5 层 简报（同一份产出 → 同一份文字）    ✅ 确定，`test_brief.py` 锁住
    第 2 层  推演（同样的行为序列 → 同样的曲线） ✅ 确定，`test_world_state.py` 锁住
    第 1 层  归类（同样的文本 → 同样的标签）     ✅ 确定，缓存 + `test_stance.py` 锁住

第 3 层做不到的原因不在我们的代码里，而在上游：**agent 每次说的话都不一样**。
缓存只能保证「同一句话给同一个标签」，保证不了「同一场推演说同一句话」。
压 temperature、固定 seed 能收窄抖动，但服务端只「尽力」遵守 seed，
给不了逐位保证 —— 所以正式口径只能是「多 seed 跑 N 次，报均值与方差」，
而不是「跑一次就准」。

**把做不到的那一层如实报出来，比含糊掉它重要得多。**

**这个脚本自己翻过三次车，三次都值得留在文件里。**

第一次：第 3 层没传 `--out`，于是自检把仓库里那份入库产出静默覆盖掉了。

第二次（修第一次的时候引入的）：修法是「把 `--out` 指到临时目录」，
但 **`--out` 的粒度是目录，不是文件** —— `simulate` 从它读画像
（`out_dir/twitter_profiles.csv`）与知情映射（`out_dir/actor_knowledge.json`），
再往 `out_dir/twitter_rounds.json` 写产出。给它一个 `e2e_a.json` 这样的
**文件名**，它会在读画像那一步 `FileNotFoundError` —— 第 3 层整层跑不起来。

第三次（2026-09-15 才发现，一直躺在那里）：第 3 层**读不动子进程的话**。
`encoding="utf-8"` 传给 `subprocess.run`，而子进程在 Windows 中文机器上
被重定向到管道时说的是 cp936/GBK，于是读取线程里解码失败 —— 那个异常
不让 `run()` 抛，只让 `proc.stdout` 变成 `None`，报出来的错是下一行的
`AttributeError: 'NoneType' object has no attribute 'splitlines'`。
**像一个推演失败的报错，实际是一行中文没读进来。** 详见 `child_env()`。

三次的**共同点比差异更要紧**：第一次是「产物被毁但没人看」，第二次与
第三次都是「**测试全绿而工具已经废了**」。第 9 套测试当时测的是
「argv 里有没有 `--out`、它是不是不等于入库产出」—— 两条都真，
而发出去的命令根本跑不通。**只断言「参数在场」的检查，是这个项目最该防的
那种恒真检查。** 所以第二次那一版加的不是断言，是**把运行目录真的造出来**
（`_seed_run_dir`）：能不能跑，取决于磁盘上有没有那几份输入，而不是取决于
字符串长得对不对。第三次同理，加的检查是**真的起一个子进程说一句中文、
按 `_run_sim` 的读法把它读回来**，而不是断言「env 里有没有那个键」。

第 2.5 层是补上去的，理由值得写下来：上面那句「不可复现」说的是**曲线**，
不是**产出物**。一份给人看的简报里没有 LLM、没有随机数、没有时间戳，
所以它**必须**逐字可复现 —— 否则「不可复现」就从一条有边界的性质，
变成了不去做确定性的借口。这一层还顺带比对落盘样例是否与当前代码一致，
防的是「模板改了、样例没重跑」。

第 2.6 层同理由，多一条：**金标对照表的结论必须能被评委自己跑出来**。
那张表里有五条否决、其中四条同源，正是靠这一层才成为可核对的结论而不是一句表态。

第 2.7 层（`weiran.selfcheck`）问的是这些层都问不到的一件事：**上面每一层都是
「同一份产出 → 同一份产物」，那么一个把结论写死的实现，能让它们全部通过。**
把 `evaluate()` 整个换成一张十行的常量表，2.6 层照旧逐字一致、照旧全绿。
**「可复现」与「算出来的」是两个问题**，2.5/2.6 只回答了前一个。
所以这一层喂反例：改一处读数，看结论会不会跟着动、以及**不该动的有没有跟着动**。
它同样进退出码 —— 它是确定的（不调 LLM、无随机数），没有理由不进。

用法：
    cd backend
    python repro_check.py            # 六层都跑
    python repro_check.py --skip-e2e # 只跑确定的五层（快、不花钱，不跑第 3 层）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
REPO = BACKEND.parent
SIM = REPO / "data" / "simulation"

SEED = ("【云溪大学就业指导中心】2025届毕业生就业质量报告今日发布。"
        "报告显示本届毕业生总体去向落实率为78.3%，较上届下降6.1个百分点；"
        "其中灵活就业占比升至21.7%。完整报告与分专业数据见官网附件。")

# 第 2 层用的固定行为序列：刻意不来自 LLM，这样这一层的检查不受上游影响。
REPLAY_BEHAVIORS = [
    ["disclosure"],
    ["discussion", "discussion", "amplification"],
    ["suppression", "testimony", "amplification", "amplification"],
    ["contradiction", "discussion", "disclosure"],
]


#: 仓库里那份**随仓库入库**的推演产出，以及它所在的目录。
#: `check_brief()` 读它，而 `_run_sim()` **绝不能写它** —— 见 `sim_argv()`。
SHIPPED_DIR = SIM
SHIPPED_ROUNDS = SIM / "twitter_rounds.json"

#: 一次推演要能在某个目录里跑起来，必须先有这些输入文件。
#: **`--out` 不是「输出文件」，是「输出目录」** —— 这一条是这个文件里最贵的教训：
#: `simulate` 从 `out_dir` 读画像（`out_dir/twitter_profiles.csv`）与知情映射
#: （`out_dir/actor_knowledge.json`），再往 `out_dir/twitter_rounds.json` 写产出。
#: 所以「把 --out 指到临时目录」的正确做法是**造一个完整的运行目录**，
#: 而不是给一个临时文件名 —— 后者会在读画像那一步就炸（见 `_seed_run_dir`）。
RUN_INPUTS = ("twitter_profiles.csv", "actor_knowledge.json", "stance_cache.json")


def sim_argv(rounds: int, agents: int, out_dir: Path) -> list[str]:
    """构造一条推演命令，产出落在 `out_dir/twitter_rounds.json`。

    **`--out` 必须显式给，而且必须不指向仓库里那个目录。** 这不是洁癖：
    原先这里不传 `--out`，于是 `python -m weiran.simulate` 用了它自己的默认值
    `data/simulation`（**是目录，不是文件**）—— 跑一次自检就把仓库里那份
    27 agent × 15 轮的产出覆盖成 5 agent × 3 轮的临时结果，而且没有任何提示。
    一个验证工具静默销毁它要验证的东西，是这个项目最不该有的东西。

    拆成一个纯函数是为了能被离线测到：真的跑一次第 3 层要调 LLM、要花钱、
    要十几分钟，而这条约束本身是可以用一次字符串比较守住的东西。

    参数名从 `out` 改成 `out_dir` 是故意的：叫 `out` 的时候，调用方
    （包括我自己）会理所当然地传一个 `.json` 文件名进去，而那是跑不起来的。
    """
    cmd = [sys.executable, "-m", "weiran.simulate",
           "--agents", str(agents), "--rounds", str(rounds),
           "--seed-text", SEED, "--out", str(out_dir)]
    # 兜底自检：即便上面的字符串哪天被改错，也不许落到仓库那份产出上。
    # 比两处：既不许是那个文件，也不许是它所在的目录（目录才是 --out 的粒度）。
    resolved = Path(out_dir).resolve()
    for bad, why in ((SHIPPED_ROUNDS, "仓库里那份产出"),
                     (SHIPPED_DIR, "仓库里那份产出所在的目录")):
        if resolved == bad.resolve():
            raise RuntimeError(
                f"自检的推演输出被指到了{why}（{bad}）—— "
                f"跑一次就会把它覆盖掉。改用一个临时目录。"
            )
    return cmd


def _seed_run_dir(src: Path, dst: Path) -> list[str]:
    """把一次推演必需输入复制进 `dst`，返回实际复制了的文件名。

    **不复制就跑步起来。** `simulate` 的 `--out` 是输出目录，而它同时从那个
    目录读画像与知情映射 —— 于是「指到临时目录」这件事不是把输出挪走，
    是**造一个能跑起来的完整目录**。缺 `twitter_profiles.csv` 会直接
    `FileNotFoundError`（这一条实测过），而那个失败发生在任何 LLM 调用之前，
    所以它不花钱、但会让第 3 层整层跑不起来。

    `stance_cache.json` 有就复制、没有就算了：它是**可选加速**，
    缺了会重新归类（花 LLM、也让两次运行的归类结果不再共享）。
    把它带上，两次运行的标签口径才一致。

    Raises:
        RuntimeError: 必需的输入文件在仓库里就找不到。**必须响** ——
            那说明仓库不完整，而第 3 层会以一个看不出原因的推演失败告终。
    """
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in RUN_INPUTS:
        s = src / name
        if not s.is_file():
            if name == "stance_cache.json":
                continue            # 可选加速，不是必需输入
            raise RuntimeError(
                f"仓库里缺少 {s} —— 第 3 层没法在一个临时目录里跑起来。"
                "这份文件随仓库入库，缺了说明仓库不完整。"
            )
        shutil.copy2(s, dst / name)
        copied.append(name)
    return copied


def child_env() -> dict:
    """让子进程按 UTF-8 说话 —— 否则父进程解不出它的话。

    **这是这个文件翻的第三次车，而且是最隐蔽的一次：它把第 3 层藏起来了。**
    上面（`_run_sim`）一直写的是 `capture_output=True, text=True,
    encoding="utf-8"`。三个参数都真、都合理，而子进程在 Windows 中文机器上
    被重定向到管道时，stdout 的编码是**本机 locale（cp936/GBK）**，
    不是 UTF-8 —— `weiran/config.py:ensure_console_encoding()` 只改
    `errors` 策略、**故意不动 encoding**（它要的是中文照原样显示，不是改成
    UTF-8）。于是父进程拿 UTF-8 去解 GBK 字节，实测：

        python repro_check.py
        UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd0 in position 2

    错在父进程读不动，可它炸出来的样子是另一回事：解码发生在
    `subprocess` 的**读取线程**里，那个异常不会让 `run()` 抛出去，只会让
    `proc.stdout` 变成 `None`，然后这里下一行的 `proc.stdout.splitlines()`
    报 `AttributeError: 'NoneType' object has no attribute 'splitlines'`
    —— **指向一个跟真正原因毫无关系的对象**。第 3 层就这样整层停摆，
    而它停摆的方式是「抛异常」，不是「报一个错的数」，所以只看输出像是
    「推演本身跑不起来」，于是会去查推演。

    修法也**不动父进程的 decode**：父进程要读的 `"完成：…"` 是中文，
    拿 `errors="replace"` 兜住会把中文换成问号、那行就永远找不到了
    （那就从「崩掉」退化成「静默少一行」，比崩掉更坏）。正解是让子进程
    真的说 UTF-8：`PYTHONIOENCODING` 是 Python 自己认的开关，
    对推演行为**零影响**（只换输出编码），所以拿它对齐不引入任何混淆变量。

    教训的形状跟前两次**是同一个**：这个文件坏掉的方式，总是「测试全绿、
    而工具已经废了」。前两次是 argv 长得对、跑不通；这一次是参数写得对、
    字节解不开。所以配的检查（`test_repro_check.py::test_child_env_…`）
    不是断言这个字典长什么样，而是**真的起一个子进程说一句中文、父进程
    按 `_run_sim` 的读法把它读回来** —— 恒真检查防不住这一类。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_sim(rounds: int, agents: int, out_dir: Path) -> tuple[dict, str]:
    proc = subprocess.run(
        sim_argv(rounds, agents, out_dir),
        cwd=BACKEND, capture_output=True, text=True, encoding="utf-8",
        env=child_env(),  # TEMP-PULL
    )
    if proc.stdout is None:
        # 解不动子进程的话。这里**必须响**，而且必须响在正确的地方 ——
        # 见 `child_env()`：默认的炸法是下一行把 None 当字符串用。
        raise SystemExit(
            "推演子进程的输出解不成 UTF-8（stdout 为 None）。"
            "原因通常是子进程按本机 locale 说话；"
            f"当前 PYTHONIOENCODING={child_env().get('PYTHONIOENCODING')!r}，"
            "应当由 _run_sim 传下去。"
        )
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"推演失败，退出码 {proc.returncode}")
    # 产出文件名是 simulate 定的，不是调用方定的（`--out` 只给目录）。
    payload = json.loads((Path(out_dir) / "twitter_rounds.json").read_text(encoding="utf-8"))
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("完成：")), "")
    return payload, line


# ---------------------------------------------------------------------------
# 第 1 层：归类 —— 同样的文本，同样的标签
# ---------------------------------------------------------------------------

def check_classifier() -> bool:
    print("\n【第 1 层】归类：同样的文本 → 同样的标签")
    from weiran.stance import StanceClassifier

    texts = ["辅导员要求同学删除帖子，不要在网上讨论",
             "学校公布了分专业数据，并对口径作出说明",
             "我那年也是这么被算进去的"]
    a = StanceClassifier(llm=None, cache_path=None)
    b = StanceClassifier(llm=None, cache_path=None)
    la, lb = a.classify_many(texts), b.classify_many(texts)
    ok = la == lb
    for t, x in zip(texts, la):
        print(f"  {'✅' if ok else '❌'} {x:<14} {t[:28]}")
    print(f"  → 关键词通路确定：{'是' if ok else '否'}")

    # 缓存通路：同一个标签必须能从磁盘原样读回
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        c1 = StanceClassifier(llm=None, cache_path=p)
        c1.cache = {"x": "suppression"}
        c1.save()
        c2 = StanceClassifier(llm=None, cache_path=p)
        same = c2.cache == c1.cache
    print(f"  → 缓存落盘/读回一致：{'是 ✅' if same else '否 ❌'}")
    return ok and same


# ---------------------------------------------------------------------------
# 第 2.5 层：简报 —— 同样的产出，同样的文字
# ---------------------------------------------------------------------------

def check_brief() -> bool | None:
    """把「确定性」从引擎推进到**产出物**。

    前面两层说的是「同样的输入给同样的输出」，这一层说的是
    「同一份产出给同一份简报」—— 一个给人看的东西也应当是算出来的，
    而不是每次带点不一样。

    这一层的确定性是**真的**（简报里没有 LLM、没有随机数、没有时间戳），
    所以它进退出码，与第 3 层不同规格。

    Returns:
        True / False / **None**。None 表示「没测到」（缺产出文件）。
        这里刻意返回三态而不是把「没测到」折成 True：一份没跑过的检查
        在汇总表里显示 `✅ 确定`，比没有这条检查更坏 —— 它会让整张表
        的可信度一起贬值。第一版就是折成 True 的。
    """
    print("\n【第 2.5 层】简报：同一份产出 → 同一份文字")
    import json

    from weiran import brief as B
    from weiran.config import REPO_ROOT

    run_path = REPO_ROOT / "data" / "simulation" / "twitter_rounds.json"
    if not run_path.is_file():
        print(f"  ⏭  缺少 {run_path.name} —— **这不是「通过」，是没测到**。"
              f"该文件随仓库入库，缺了说明仓库不完整或产出被删。")
        return None

    rd = B.load_rounds(run_path)
    gold = B.load_gold(REPO_ROOT / B.DEFAULT_SCENARIO)
    a = B.build_brief(rd, gold, command="repro_check")
    b = B.build_brief(rd, gold, command="repro_check")
    md_a, md_b = B.render_markdown(a), B.render_markdown(b)
    j_a = json.dumps(a, ensure_ascii=False, sort_keys=True)
    j_b = json.dumps(b, ensure_ascii=False, sort_keys=True)

    ok = md_a == md_b and j_a == j_b
    print(f"  {'✅' if md_a == md_b else '❌'} markdown 逐字一致（{len(md_a)} 字符）")
    print(f"  {'✅' if j_a == j_b else '❌'} JSON 逐字一致")

    # 落盘的那份样例必须与当前代码算出来的一致 —— 否则模板改了而样例没重跑，
    # 仓库里给人看的第一份东西就是过期的。
    sample = REPO_ROOT / "data" / "simulation" / "brief.md"
    if sample.is_file():
        fresh = md_a == sample.read_text(encoding="utf-8")
        ok = ok and fresh
        print(f"  {'✅' if fresh else '❌'} 落盘样例 data/simulation/brief.md 与当前代码一致"
              + ("" if fresh else "（模板改了没重跑：python -m weiran.brief）"))
    else:
        print("  ⏭  尚无落盘样例，跳过一致性比对")

    print(f"  → 简报确定且样例最新：{'是 ✅' if ok else '否 ❌'}")
    return ok


def check_gold_check() -> bool | None:
    """第 2.6 层：金标对照表 —— 同一份产出 → 同一张表。

    这一层与第 2.5 层同规格，但**多守一件事**：那张表的结论要能被评委
    自己跑出来。所以这里比对的不只是「两次跑一样」，还包括**落盘的那份
    与当前代码一致** —— 否决算不算数，取决于这张表是不是算出来的。

    为什么不能只靠单元测试：单测守的是「函数按契约工作」，守不住
    「仓库里那份给人看的结论，与当前代码算出来的不是同一张」。
    """
    print("\n【第 2.6 层】金标对照表：同一份产出 → 同一张表")
    import json

    from weiran import gold_check as G
    from weiran.config import REPO_ROOT

    run_path = REPO_ROOT / "data" / "simulation" / "twitter_rounds.json"
    if not run_path.is_file():
        print(f"  ⏭  缺少 {run_path.name} —— **这不是「通过」，是没测到**。")
        return None

    rd = G.load_rounds(run_path)
    gold = G.load_gold(REPO_ROOT / G.DEFAULT_SCENARIO)
    # 这一层要**整份**逐字比对落盘样例，而对照表的正文里印着它自己的生成
    # 命令（第五节「来源」）。所以重算时必须用 CLI 的默认命令，不能用
    # "repro_check" —— 否则两边只差那一行 provenance，比对会被顶成
    # 「结论与代码不一致」，而那个结论其实是同一个。命令从模块里取，
    # 不在这里重抄：CLI 改了默认命令而样例没重跑，这一层就该红。
    a = G.build_gold_check(rd, gold, command=G.DEFAULT_COMMAND)
    b = G.build_gold_check(rd, gold, command=G.DEFAULT_COMMAND)
    md_a, md_b = G.render_markdown(a), G.render_markdown(b)
    j_a = json.dumps(a, ensure_ascii=False, sort_keys=True)
    j_b = json.dumps(b, ensure_ascii=False, sort_keys=True)

    ok = md_a == md_b and j_a == j_b
    print(f"  {'✅' if md_a == md_b else '❌'} markdown 逐字一致（{len(md_a)} 字符）")
    print(f"  {'✅' if j_a == j_b else '❌'} JSON 逐字一致")

    sample = REPO_ROOT / "data" / "simulation" / "gold_check.md"
    if sample.is_file():
        # 比整份文件（含 provenance）。上面用的是 CLI 的默认命令，与产出
        # 它的那条命令一致，所以能整份比；整份比严格，那就整份比。
        fresh = md_a == sample.read_text(encoding="utf-8")
        ok = ok and fresh
        print(f"  {'✅' if fresh else '❌'} 落盘样例 data/simulation/gold_check.md 与当前代码一致"
              + ("" if fresh else "（代码改了没重跑：python -m weiran.gold_check）"))
    else:
        print("  ⏭  尚无落盘样例，跳过一致性比对")
        ok = False

    s = a["summary"]
    print(f"  → 对照表确定且样例最新：{'是 ✅' if ok else '否 ❌'}"
          f"（通过 {s['counts']['通过']} · 否决 {s['counts']['否决']} · "
          f"不可判定 {s['counts']['不可判定']}；判得出来的 {s['judged']} 条里"
          f"只有 {len(s['trusted_ids'])} 条可采信）")
    return ok


# ---------------------------------------------------------------------------
# 第 2.7 层：自证 —— 反例喂进去，结论跟着动吗
# ---------------------------------------------------------------------------

def check_selfcheck() -> bool | None:
    """第 2.7 层：装置自证 —— 那张表是不是**算出来的**。

    这一层跟 2.6 的区别不是精度，是**问法**：2.6 问「同一份产出是不是给同一张
    表」，这一层问「那张表有没有可能本来就是写死的」。**一个把结论写死的实现
    能让 2.5 与 2.6 全部逐字通过** —— 它们比的是「两次跑一样」，而常量表当然
    每次都一样。把 `evaluate()` 换成十行 `if/else`，2.6 及以前那几层一个都不会红。

    所以这一层不比两次运行，它**喂反例**：一次改一处读数，看结论会不会跟着动；
    同时看**声明不该动的那几条有没有跟着动**。它进退出码：不调 LLM、无随机数，
    是确定的 —— 没有理由不进。

    这里只跑判定，不重抄自证的逻辑：连「五个格都要被触发过」这句话都是
    `weiran.selfcheck` 自己算出来的，它自己说不全，这里就红。
    """
    print("\n【第 2.7 层】自证：反例喂进去，结论跟着动吗")
    from weiran import selfcheck as S
    from weiran.config import REPO_ROOT

    run_path = REPO_ROOT / "data" / "simulation" / "twitter_rounds.json"
    if not run_path.is_file():
        print(f"  ⏭  缺少 {run_path.name} —— **这不是「通过」，是没测到**。")
        return None

    rep = S.run()
    n, n_bad = len(rep["results"]), len(rep["failures"])
    for r in rep["results"]:
        if not r["ok"]:
            print(f"  ❌ #{r['n']} {r['what']}")
            for c in r["checks"]:
                print(f"       - {c}")
    ok = not rep["failures"] and not rep["missing_grids"]

    print(f"  {'✅' if not rep['failures'] else '❌'} "
          f"{n - n_bad}/{n} 个反例落在预期格")
    hit = set(rep["grids"])
    print(f"  {'✅' if not rep['missing_grids'] else '❌'} "
          f"五个输出格被触发 {len(hit)}/5"
          + ("" if not rep["missing_grids"] else "，缺 "
             + "、".join(S._grid_name(v, t) for v, t in rep["missing_grids"])))
    print(f"  {'✅' if rep['entry_ok'] else '❌'} "
          f"自证用的链路与产品入口 `build_gold_check` 等价")
    print(f"  → 这张表的结论由输入决定：{'是 ✅' if ok else '否 ❌'}"
          "（这一层不比两次运行 —— 常量表每次运行也一样。）")
    return ok


# ---------------------------------------------------------------------------
# 第 3 层：端到端 —— 同一条命令，同一条曲线
# ---------------------------------------------------------------------------

def check_engine() -> bool:
    print("\n【第 2 层】引擎：同样的行为序列 → 同样的曲线")
    from weiran.world_state import WorldState, WorldStateEngine

    def replay():
        eng = WorldStateEngine()
        st = WorldState.baseline(eng.params)
        out = []
        for i, beh in enumerate(REPLAY_BEHAVIORS):
            r = eng.step(st, beh, dt=1.0, phase_id=f"R{i}")
            st = r.state_after
            out.append(st.as_dict())
        return out

    a, b = replay(), replay()
    ok = a == b
    for i, (x, y) in enumerate(zip(a, b)):
        print(f"  {'✅' if x == y else '❌'} 轮{i}  " + "  ".join(
            f"{k[:4]}={v:.4f}" for k, v in x.items()))
    print(f"  → 引擎确定：{'是 ✅' if ok else '否 ❌'}")
    return ok


# ---------------------------------------------------------------------------
# 第 3 层：端到端 —— 同一条命令，同一条曲线
# ---------------------------------------------------------------------------

def check_end_to_end(rounds: int, agents: int, workdir: Path) -> bool:
    print(f"\n【第 3 层】端到端：同一条命令跑两次（{agents} agent × {rounds} 轮）")
    # 两次运行各占一个**完整的运行目录**（画像 + 知情映射 + 归类缓存），
    # 都不碰仓库那份产出（见 `sim_argv` / `_seed_run_dir`）。
    a_dir, b_dir = workdir / "e2e_a", workdir / "e2e_b"
    for d in (a_dir, b_dir):
        copied = _seed_run_dir(SHIPPED_DIR, d)
        print(f"  {d.name}/ 就绪：" + "、".join(copied))
    a_meta, line_a = _run_sim(rounds, agents, a_dir)
    print("  第一次 " + line_a)
    b_meta, line_b = _run_sim(rounds, agents, b_dir)
    print("  第二次 " + line_b)

    # 压缩过的曲线与 round=day 不可比（激励项按步累加）。这一层比的是
    # 「同一条命令的两个不同输出」，所以压缩不影响结论，但必须在输出里
    # 说明，免得有人把这行数字搬去和 round=day 的结果并列。
    meta = a_meta.get("meta", {})
    if meta.get("compressed"):
        print(f"  ⚠️ 本次为压缩模式（每轮 {meta['days_per_round']:.2f} 天）——"
              "结论只适用于「两次运行彼此比较」，不可与 round=day 的结果比")
    a, b = a_meta["rounds"], b_meta["rounds"]

    beh_same = [r["behaviors"] for r in a] == [r["behaviors"] for r in b]
    ok = [r["state"] for r in a] == [r["state"] for r in b]
    for x, y in zip(a, b):
        eq = x["state"] == y["state"]
        print(f"  {'✅' if eq else '❌'} 轮{x['index']}  " + "  ".join(
            f"{k[:4]}={v:.4f}" for k, v in x["state"].items()))
        if not eq:
            print("       第二次 " + "  ".join(
                f"{k[:4]}={v:.4f}" for k, v in y["state"].items()))

    print(f"  → 行为序列一致：{'是' if beh_same else '否'}")
    print(f"  → 六维曲线一致：{'是 ✅' if ok else '否 ❌'}")
    if not ok:
        print("     这是**预期结果**，不是回归。原因在上游：agent 每次说的话不同，"
              "\n     于是喂给引擎的行为序列不同。缓存只能保证「同一句话给同一标签」。")
        print(f"     第一次行为标签: {[r['behaviors'] for r in a]}")
        print(f"     第二次行为标签: {[r['behaviors'] for r in b]}")
        print("     正式口径应为「多 seed 跑 N 次，报均值与方差」，见 进度.md 第 5 步。")
    return ok


def main() -> int:
    from weiran.config import ensure_console_encoding
    ensure_console_encoding()

    ap = argparse.ArgumentParser(description="可复现性分层检查")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--skip-e2e", action="store_true",
                    help="跳过第 3 层（不调 LLM，秒回）")
    args = ap.parse_args()

    results = {"归类": check_classifier(), "引擎": check_engine(),
               "简报": check_brief(), "对照表": check_gold_check(),
               "自证": check_selfcheck()}
    if not args.skip_e2e:
        # **第 3 层的产物写进临时目录。** 它跑的是完整的 LLM 推演，
        # 若沿用默认输出路径，跑一次自检就会覆盖 `data/simulation/` 里
        # 那份随仓库入库的产出（曾经真的发生过）。
        with tempfile.TemporaryDirectory(prefix="weiran-repro-") as tmp:
            results["端到端"] = check_end_to_end(args.rounds, args.agents,
                                                 Path(tmp))

    print("\n" + "=" * 62)
    for k, v in results.items():
        print(f"  {k:<8} "
              + ("⏭ 未测到" if v is None else "✅ 确定" if v else "❌ 不确定"))
    print("=" * 62)
    # **第 3 层不进退出码** —— 它不可复现是已知且已解释的性质，不是失败。
    # 把它算进去会让这个脚本天天红，然后就没人看了。前两层与简报/对照表层进。
    # 「未测到」(None) **算失败**：产出文件随仓库入库，缺了就是仓库不完整，
    # 而一个「跑不动所以全绿」的自检脚本正好是它要防的那种东西。
    gated = [results[k] for k in ("归类", "引擎", "简报", "对照表", "自证")]
    if any(v is None for v in gated):
        print("  ⚠️ 有检查未测到（见上）—— 未测到按失败计，不按通过计")
    return 0 if all(v is True for v in gated) else 1


if __name__ == "__main__":
    sys.exit(main())
