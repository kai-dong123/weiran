"""可复现性检查：分清哪一层是确定的、哪一层不是。

**这个脚本的存在本身就是一次实测的产物。** 第一版它只做一件事：
同一条命令跑两次，比较六维曲线。结果两次**完全不同** —— 于是浮出一个
必须说清楚的结论，而不是一个可以含糊过去的数字：

    「未然」的可复现性是**分层**的，不是一句「我们可复现」能概括的。

    第 3 层  端到端（同一条命令 → 同一条曲线）   ❌ 不可复现，且**原理上做不到**
    第 2 层  推演（同样的行为序列 → 同样的曲线） ✅ 确定，`test_world_state.py` 锁住
    第 1 层  归类（同样的文本 → 同样的标签）     ✅ 确定，缓存 + `test_stance.py` 锁住

第 3 层做不到的原因不在我们的代码里，而在上游：**agent 每次说的话都不一样**。
缓存只能保证「同一句话给同一个标签」，保证不了「同一场推演说同一句话」。
压 temperature、固定 seed 能收窄抖动，但服务端只「尽力」遵守 seed，
给不了逐位保证 —— 所以正式口径只能是「多 seed 跑 N 次，报均值与方差」，
而不是「跑一次就准」。

**把做不到的那一层如实报出来，比含糊掉它重要得多。**

用法：
    cd backend
    python repro_check.py            # 三层都跑
    python repro_check.py --skip-e2e # 只跑确定的两层（快、不花钱）
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


def _run_sim(rounds: int, agents: int) -> tuple[list[dict], str]:
    proc = subprocess.run(
        [sys.executable, "-m", "weiran.simulate",
         "--agents", str(agents), "--rounds", str(rounds), "--seed-text", SEED],
        cwd=BACKEND, capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"推演失败，退出码 {proc.returncode}")
    data = json.loads((SIM / "twitter_rounds.json").read_text(encoding="utf-8"))
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("归类")), "")
    return data, line


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
# 第 2 层：引擎 —— 同样的行为序列，同样的曲线
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
    a, line_a = _run_sim(rounds, agents)
    print("  第一次 " + line_a)
    b, line_b = _run_sim(rounds, agents)
    print("  第二次 " + line_b)

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

    results = {"归类": check_classifier(), "引擎": check_engine()}
    if not args.skip_e2e:
        results["端到端"] = check_end_to_end(args.rounds, args.agents)

    print("\n" + "=" * 62)
    for k, v in results.items():
        print(f"  {k:<8} {'✅ 确定' if v else '❌ 不确定'}")
    print("=" * 62)
    # **只有前两层算「必须通过」** —— 第 3 层不可复现是已知且已解释的性质，
    # 不是失败。把它算进退出码，会让这个脚本天天红，然后就没人看了。
    return 0 if results["归类"] and results["引擎"] else 1


if __name__ == "__main__":
    sys.exit(main())
