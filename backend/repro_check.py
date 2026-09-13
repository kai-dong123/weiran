"""可复现性检查：分清哪一层是确定的、哪一层不是。

**这个脚本的存在本身就是一次实测的产物。** 第一版它只做一件事：
同一条命令跑两次，比较六维曲线。结果两次**完全不同** —— 于是浮出一个
必须说清楚的结论，而不是一个可以含糊过去的数字：

    「未然」的可复现性是**分层**的，不是一句「我们可复现」能概括的。

    第 3 层  端到端（同一条命令 → 同一条曲线）   ❌ 不可复现，且**原理上做不到**
    第 2.5 层 简报（同一份产出 → 同一份文字）    ✅ 确定，`test_brief.py` 锁住
    第 2 层  推演（同样的行为序列 → 同样的曲线） ✅ 确定，`test_world_state.py` 锁住
    第 1 层  归类（同样的文本 → 同样的标签）     ✅ 确定，缓存 + `test_stance.py` 锁住

第 3 层做不到的原因不在我们的代码里，而在上游：**agent 每次说的话都不一样**。
缓存只能保证「同一句话给同一个标签」，保证不了「同一场推演说同一句话」。
压 temperature、固定 seed 能收窄抖动，但服务端只「尽力」遵守 seed，
给不了逐位保证 —— 所以正式口径只能是「多 seed 跑 N 次，报均值与方差」，
而不是「跑一次就准」。

**把做不到的那一层如实报出来，比含糊掉它重要得多。**

第 2.5 层是补上去的，理由值得写下来：上面那句「不可复现」说的是**曲线**，
不是**产出物**。一份给人看的简报里没有 LLM、没有随机数、没有时间戳，
所以它**必须**逐字可复现 —— 否则「不可复现」就从一条有边界的性质，
变成了不去做确定性的借口。这一层还顺带比对落盘样例是否与当前代码一致，
防的是「模板改了、样例没重跑」。

用法：
    cd backend
    python repro_check.py            # 四层都跑
    python repro_check.py --skip-e2e # 只跑确定的三层（快、不花钱）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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


def _run_sim(rounds: int, agents: int) -> tuple[dict, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "weiran.simulate",
         "--agents", str(agents), "--rounds", str(rounds), "--seed-text", SEED],
        cwd=BACKEND, capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"推演失败，退出码 {proc.returncode}")
    payload = json.loads((SIM / "twitter_rounds.json").read_text(encoding="utf-8"))
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

def check_brief() -> bool:
    """把「确定性」从引擎推进到**产出物**。

    前面两层说的是「同样的输入给同样的输出」，这一层说的是
    「同一份产出给同一份简报」—— 一个给人看的东西也应当是算出来的，
    而不是每次带点不一样。

    这一层的确定性是**真的**（简报里没有 LLM、没有随机数、没有时间戳），
    所以它进退出码，与第 3 层不同规格。
    """
    print("\n【第 2.5 层】简报：同一份产出 → 同一份文字")
    import json

    from weiran import brief as B
    from weiran.config import REPO_ROOT

    run_path = REPO_ROOT / "data" / "simulation" / "twitter_rounds.json"
    if not run_path.is_file():
        print(f"  ⏭  缺少 {run_path.name}，本层跳过 —— "
              f"**这不是「通过」**，是没测到")
        return True

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

def check_end_to_end(rounds: int, agents: int) -> bool:
    print(f"\n【第 3 层】端到端：同一条命令跑两次（{agents} agent × {rounds} 轮）")
    a_meta, line_a = _run_sim(rounds, agents)
    print("  第一次 " + line_a)
    b_meta, line_b = _run_sim(rounds, agents)
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
               "简报": check_brief()}
    if not args.skip_e2e:
        results["端到端"] = check_end_to_end(args.rounds, args.agents)

    print("\n" + "=" * 62)
    for k, v in results.items():
        print(f"  {k:<8} {'✅ 确定' if v else '❌ 不确定'}")
    print("=" * 62)
    # **第 3 层不进退出码** —— 它不可复现是已知且已解释的性质，不是失败。
    # 把它算进去会让这个脚本天天红，然后就没人看了。前两层与简报层进。
    return 0 if all(results[k] for k in ("归类", "引擎", "简报")) else 1


if __name__ == "__main__":
    sys.exit(main())
