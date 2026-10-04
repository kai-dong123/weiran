"""按 seed 起一批臂。**默认什么钱都不花。**

    python backend/run_seeds.py --seeds 11,22,33 --agents 27 --rounds 15 \
        --temperature 0.7 --out data/runs/seeds --preflight
    python backend/run_seeds.py --seeds 11,22,33 --agents 27 --rounds 15 \
        --temperature 0.7 --out data/runs/seeds --yes

**这个脚本的全部价值在于它的拒绝。** 起一批臂这件事本身只是一行循环；难的是
「跑完之后这批臂**是**一批可比的臂」这件事不靠人记得。三种坏法都能安静地
产出看起来很正常的目录树，而每一种都只有在汇总时才会暴露 —— 那时钱已经花了：

  1. **`stance_cache.json` 被上一支臂改过。** 它是跨运行共享且会被改写的，
     谁先跑谁占缓存。第二支臂于是有一半的归类直接命中缓存，两次运行差的不是
     seed。开发期探针 `run_ab2.py`（不随仓库交付）已经写过这条纪律，本脚本
     把它从外部草稿搬进仓库，并变成**逐臂核对 sha256**。
     （这份缓存是**可选**的：它不入库，底本没有它也照样跑 —— 见
     `OPTIONAL_INPUTS`。所以这里防的是「有它在时被共享」，不是「必须有它」。）
  2. **臂之间口径不同还照样跑。** 12 agent 的臂与 27 agent 的臂混在一批里，
     而 12 agent 的相位读数是已知无效的（体量项跳动 75%）。
  3. **中途崩了还接着跑。** `simulate` **没有断点续跑**（每次先删库，JSON 只在
     最后写一次），所以一支坏臂不会自己好；闷头跑完只会得到一批第二支开始
     就污染了的产出。

**顺序跑、不并行，是刻意的**：墙钟是一个被记录的量，并行会把它变成一个
没人能解释的数。所以本脚本不提供 `--jobs` / `--parallel`。

**它不报金额。** 仓库里没有写死单价（`smoke.py` 只在调用方给
`--price-in/--price-out` 时才报钱），所以这里只报 token 与墙钟，
**金额一律不猜** —— 报一个编出来的数字比不报更坏。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from weiran.config import (  # noqa: E402
    REPO_ROOT, child_env, ensure_console_encoding, rel_path, sha256_text)
from weiran.profiles import DEFAULT_SCENARIO  # noqa: E402

BACKEND = REPO_ROOT / "backend"

#: `simulate` 的默认产出目录。**臂一律不许写进这里** —— 那里面是入库那份
#: 产出，`repro_check.py` 第 2.5 / 2.6 层对它逐字锁定。
SHIPPED_OUT = REPO_ROOT / "data" / "simulation"

DEFAULT_MASTER = SHIPPED_OUT
DEFAULT_ROOT = REPO_ROOT / "data" / "runs" / "seeds"
DEFAULT_ROUNDS_FILE = SHIPPED_OUT / "twitter_rounds.json"

#: 每支臂要用**自己那份**的输入。`simulate` 是从 `out_dir` 里读它们的
#: （`simulate.py:1092` / `:1121` / `:1276`），所以「逐臂各拷一份」既不是
#: 保险措施、也不是洁癖 —— 那是唯一能把输入钉住的办法。
#:
#: **这三个的地位不一样，缺了会怎样也不一样。**
REQUIRED_INPUTS = ("twitter_profiles.csv", "actor_knowledge.json")

#: `stance_cache.json` 是**可选加速**：它不入库（`.gitignore:42`），删掉只是让
#: 归类重跑一遍（花 LLM、也让两次运行的标签不再共享），**不会让推演跑不起来**。
#: `repro_check._seed_run_dir()` 对同一件事就是这么办的（「有就复制、没有就算了」）。
#: 所以底本没有它时本脚本**不拒绝** —— 只把它记成 `None`（「本次底本没有这份」），
#: 因为把一份**不入库**的文件当必需件，等于让每个新 clone 都跑不起批次。
OPTIONAL_INPUTS = ("stance_cache.json",)

#: 用来遍历的那一份。**必需在前、可选在后**，顺序只影响打印。
INPUT_FILES = REQUIRED_INPUTS + OPTIONAL_INPUTS

#: 相位读数只在 27 agent 上有效（12 agent 实测体量项跳动 75%，十支臂里
#: 7 支驱动峰假落在 P3）。所以小规模要显式开口子。
PHASE_READING_MIN_AGENTS = 27


class BatchRefused(Exception):
    """这一批不该跑。**拒绝时要给出理由与下一步**，不是一句「不合法」。"""


# ---------------------------------------------------------------------------
# 规划（纯函数：不碰磁盘、不起进程，所以能直接测）
# ---------------------------------------------------------------------------

@dataclass
class ArmPlan:
    label: str
    seed: int
    out_dir: Path
    argv: list[str] = field(default_factory=list)

    def command(self) -> str:
        return "python -m weiran.simulate " + " ".join(self.argv)


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            seeds.append(int(part))
        except ValueError:
            raise BatchRefused(f"--seeds 里 {part!r} 不是整数") from None
    if not seeds:
        raise BatchRefused("--seeds 是空的")
    dupes = {s for s in seeds if seeds.count(s) > 1}
    if dupes:
        # 同一个 seed 跑两次**是** preflight 的用法，但在一批里它只会得到
        # 两支一模一样的臂，把跨 seed 离散拉低。要查一致性请用 --preflight。
        raise BatchRefused(
            f"--seeds 里有重复的种子：{sorted(dupes)}。"
            "同一个 seed 跑两次不会给出两支可比的臂（那只会把跨 seed 离散"
            "往零拉），要验同一 seed 的一致性请用 --preflight")
    return seeds


def check_target(out_root: Path) -> None:
    """臂不许写进入库产出目录。"""
    try:
        out_root.resolve().relative_to(SHIPPED_OUT.resolve())
        inside = True
    except ValueError:
        inside = False
    if inside:
        raise BatchRefused(
            f"--out 指向 {SHIPPED_OUT} 里面（{out_root}）—— 那是入库那份产出的家，"
            "`repro_check.py` 第 2.5 / 2.6 层对它逐字锁定。"
            f"请换一个目录，例如 {DEFAULT_ROOT}")


def check_scale(agents: int, *, allow_small: bool) -> None:
    if agents < PHASE_READING_MIN_AGENTS and not allow_small:
        raise BatchRefused(
            f"--agents {agents} 小于 {PHASE_READING_MIN_AGENTS}：相位读数只在 "
            f"{PHASE_READING_MIN_AGENTS} agent 上有效（12 agent 实测体量项跳动 "
            "75%，十支臂里 7 支驱动峰假落在 P3）。**小规模跑出来的臂不能用来"
            "判相位**，拿它们汇总会得到一条看着有、其实是假的驱动峰。"
            "确实只是想验流程的话，加 --allow-small-scale，"
            "并注意那样的臂只能看「跑没跑通」，读不了相位。")


def check_consistent(plans: list[ArmPlan], options: dict) -> None:
    """臂与臂之间口径必须全等。**这不是形式检查** —— 一批里混进一支
    `--chunking-guard` 不同的臂，它的记忆会被 camel 切成 1 token 一块、
    膨胀约 20 倍，两条口径的曲线不可混用，而目录名上看不出来。"""
    keys = ("agents", "rounds", "chunking_guard", "events_from", "platform",
            "scenario", "temperature", "no_world_state", "no_phases",
            "no_knowledge", "no_feedback")
    for key in keys:
        seen: dict[str, list[str]] = {}
        for p in plans:
            seen.setdefault(repr(options[p.label][key]), []).append(p.label)
        if len(seen) > 1:
            detail = "；".join(f"{v}（{','.join(ls)}）" for v, ls in seen.items())
            raise BatchRefused(f"臂之间 {key!r} 不一致：{detail} —— 那不是一批")


def plan_batch(seeds: list[int], out_root: Path, simulate_args: list[str],
               options: dict, *, master: Path, allow_small: bool,
               agents: int, force: bool) -> list[ArmPlan]:
    """规划一批臂。**只读磁盘，不起进程、不发请求。**"""
    check_target(out_root)
    check_scale(agents, allow_small=allow_small)

    plans = []
    for seed in seeds:
        label = f"seed{seed}"
        out_dir = out_root / label
        plans.append(ArmPlan(
            label=label, seed=seed, out_dir=out_dir,
            argv=simulate_args + ["--seed", str(seed), "--out", rel_path(out_dir)]))

    check_consistent(plans, options)

    # 已存在的臂：seed/规模对得上才算续跑，对不上就拒绝而不是覆盖。
    # **覆盖是不可逆的** —— 那支臂花过钱，且没有断点续跑能把它找回来。
    for p in plans:
        manifest_path = p.out_dir / "arm.json"
        if not manifest_path.is_file():
            if p.out_dir.exists() and any(p.out_dir.iterdir()) and not force:
                raise BatchRefused(
                    f"{p.out_dir} 已存在且非空，但没有 arm.json —— "
                    "它可能是别的什么东西，本脚本不覆盖。"
                    "确认可以丢弃就加 --force。")
            continue
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatch = [
            k for k, v in (("seed", p.seed), ("agents", agents))
            if k in old and old[k] != v
        ]
        if mismatch and not force:
            detail = "、".join(f"{k}：已有 {old[k]!r} vs 本次 {v!r}"
                              for k, v in (("seed", p.seed), ("agents", agents))
                              if k in old and old[k] != v)
            raise BatchRefused(
                f"{p.out_dir} 里已有一支臂，口径与本次不符（{detail}）—— "
                "拒绝覆盖。要么换 --out 根目录，要么换 seed 标签，"
                "要么确认这支臂可以丢弃后加 --force。")
    return plans


# ---------------------------------------------------------------------------
# 输入同源
# ---------------------------------------------------------------------------

def hash_master(master: Path) -> dict:
    """底本输入的指纹。**必需的那两个缺一个就拒绝**；可选那个缺了记成 `None`。

    缺件不能靠「跳过它」继续 —— 那是对**必需**输入说的：少了画像或知情映射，
    推演根本起不来，「臂之间输入同源」也就没有依据。

    可选那个（`stance_cache.json`）反过来：它**本来就不入库**，所以「底本没有」
    是**正常状态**，不是残缺。它按 `None` 记进指纹表，于是 ①「这次没有」与
    「那次有」在 `check_against_previous_batch` 里**比得出来**（跨批次不可比时
    照样会响），② 它不会被静默当成「三个都在」。
    """
    out = {}
    for name in INPUT_FILES:
        p = master / name
        if not p.is_file():
            if name in OPTIONAL_INPUTS:
                out[name] = None
                continue
            raise BatchRefused(
                f"底本缺 {p} —— 一次批次要的正是这两个必需输入各拷一份。"
                "缺了它，「臂之间输入同源」这句话就没有依据。"
                f"（{OPTIONAL_INPUTS[0]} 是另一回事：它不入库，没有也照样跑。）")
        out[name] = sha256_text(p)
    return out


def absent_optional(master_hashes: dict) -> list[str]:
    """指纹表里那些「底本没有」的可选输入。**要打印给人看，不是内部状态。**"""
    return [n for n in OPTIONAL_INPUTS if master_hashes.get(n) is None]


def check_against_previous_batch(out_root: Path, master_hashes: dict) -> str:
    """跨批次比较的前提是输入同源。**换过底本就得知道。**"""
    batch_path = out_root / "batch.json"
    if not batch_path.is_file():
        return ""
    old = json.loads(batch_path.read_text(encoding="utf-8"))
    changed = [k for k, v in master_hashes.items()
               if k in old.get("master", {}) and old["master"][k] != v]
    if changed:
        return ("**底本已经换过**：" + "、".join(changed) + "。"
                "新臂与旧臂**不可比** —— 它们的差异不止是 seed。"
                "请换一个 --out 根目录，不要把两批混在一个目录里。")
    return ""


# ---------------------------------------------------------------------------
# 成本估算（不是测量）
# ---------------------------------------------------------------------------

def measure_baseline(rounds_file: Path) -> dict | None:
    """从入库那份产出读出真实的单次读数。**只读，免费。**

    这是「估算」唯一的诚实来源：仓库里没有写死单价，但**实测过的墙钟与
    token 是真实存在的**，把它们线性外推至少标明了一个量级。
    """
    if not rounds_file.is_file():
        return None
    try:
        doc = json.loads(rounds_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    meta, rounds = doc.get("meta", {}), doc.get("rounds", [])
    if not rounds or not meta.get("agents"):
        return None
    return {
        "agents": meta["agents"], "rounds": len(rounds),
        "seconds": meta.get("total_seconds"),
        "actions": meta.get("total_actions"),
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rounds),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in rounds),
        "whence": str(rounds_file),
    }


def format_estimate(baseline: dict | None, *, n_arms: int, agents: int,
                    rounds: int) -> list[str]:
    L: list[str] = []
    if baseline is None:
        L.append("  估算：**没有可用的底本产出**，无法外推。"
                 "（本脚本不猜墙钟与 token。）")
        return L
    unit = baseline["agents"] * baseline["rounds"]
    scale = (agents * rounds) / unit if unit else 0.0
    L.append(f"  推算依据（实测，来自 {baseline['whence']}）：")
    L.append(f"    一次 {baseline['agents']} agent × {baseline['rounds']} 轮："
             f"{baseline['seconds']}s 墙钟、"
             f"入 {baseline['prompt_tokens']:,} / 出 "
             f"{baseline['completion_tokens']:,} token、"
             f"{baseline['actions']} 个动作")
    L.append(f"  **按「agent×轮数」线性外推**到 {agents} agent × {rounds} 轮：")
    per = baseline["seconds"] * scale if baseline["seconds"] else None
    tot = per * n_arms if per else None
    L.append(f"    每支臂约 {per:,.0f}s（{per / 60:.1f} 分钟）"
             if per else "    每支臂：墙钟未记录，无法外推")
    if tot:
        L.append(f"    {n_arms} 支臂合计约 {tot:,.0f}s（{tot / 3600:.2f} 小时），"
                 f"**顺序跑**")
    L.append(f"    入 token 约 {baseline['prompt_tokens'] * scale * n_arms:,.0f}、"
             f"出 token 约 "
             f"{baseline['completion_tokens'] * scale * n_arms:,.0f}")
    L.append("  **这是外推，不是测量。** 真实值只能由真跑量出来 —— "
             "上游服务端的排队与限流都不在这条直线里。")
    L.append("  **不含金额**：仓库里没有写死单价，本脚本不猜。")
    return L


# ---------------------------------------------------------------------------
# 跑
# ---------------------------------------------------------------------------

def run_arm(plan: ArmPlan, master: Path, master_hashes: dict, *,
            timeout: float | None = None) -> dict:
    """跑一支臂。**输入先落地并逐份核对 sha256，再起进程。**

    顺序不能反：先把输入拷进 `out_dir` 并核对，是为了让「这支臂用的是
    哪份输入」在**它开跑之前**就已经定死。跑完再拷的话，中途改过的底本会
    被当成它当时的输入。

    可选那个（`stance_cache.json`）在底本里没有时**不拷、也不核对** —— 但它
    在 `out_dir` 里的**旧副本要被删掉**：那一份是上一轮的遗留，不是这份底本的
    输入，留着就正好是这份脚本要防的那种「谁先跑谁占缓存」。
    """
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    for name in INPUT_FILES:
        target = plan.out_dir / name
        if master_hashes.get(name) is None:
            target.unlink(missing_ok=True)
            continue
        shutil.copy2(master / name, target)
        got = sha256_text(target)
        if got != master_hashes[name]:
            raise BatchRefused(
                f"拷进 {target} 之后 sha256 与底本不符"
                f"（{got[:12]}… vs {master_hashes[name][:12]}…）—— 停。")

    # `env=child_env()` 不是可选项：这里按 UTF-8 读子进程，就必须先让子进程按
    # UTF-8 说。少了它，子进程在被重定向到管道时按本机 locale（Windows 中文 =
    # cp936）输出，而 `errors="replace"` 会让这个错误**不声不响**地发生 ——
    # 实测三支臂的 2 KB 日志尾巴各被换成 630 个 `U+FFFD`，中文全废。
    # 「跑得动」把「这份记录已经废了」盖住了，所以配了真子进程来回读的检查
    # （`tests/test_run_seeds.py::test_the_log_tail_comes_back_in_chinese`）。
    proc = subprocess.run(
        [sys.executable, "-m", "weiran.simulate", *plan.argv],
        cwd=BACKEND, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=child_env(), timeout=timeout)

    doc = None
    rounds_path = plan.out_dir / "twitter_rounds.json"
    if rounds_path.is_file():
        doc = json.loads(rounds_path.read_text(encoding="utf-8"))

    manifest = {
        "label": plan.label,
        "seed": plan.seed,
        "command": plan.command(),
        "returncode": proc.returncode,
        "inputs": dict(master_hashes),
        # 「底本没有哪一份可选输入」也记下来：`inputs` 里那个 `None` 单看像是
        # 漏写，而它是**本次的实情**。汇总器按值比较，不受这个键影响。
        "inputs_absent": absent_optional(master_hashes),
        "out_sha256": sha256_text(rounds_path) if rounds_path.is_file() else None,
        "ok": proc.returncode == 0 and doc is not None,
        "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
    }
    # 口径键从**产出自己的 meta** 抄一份进 arm.json。汇总器只认 meta
    # （理由见 stability_report.recorded），这里抄进来是为了让人能一眼看出
    # 「驱动打算跑的」与「产出自己说的」是否一致 —— 不一致就是有问题。
    if doc is not None:
        for key in ("agents", "rounds", "platform", "days_per_round",
                    "compressed", "events_from", "phases_on", "knowledge_on",
                    "feedback_on", "world_state", "chunking_guard",
                    "temperature"):
            if key in doc.get("meta", {}):
                manifest[key] = doc["meta"][key]
    (plan.out_dir / "arm.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def run_preflight(master: Path, master_hashes: dict, *, seed: int,
                  tmp_root: Path) -> dict:
    """同一个 seed、`--agents 1 --rounds 2`，跑两次，逐位比对。

    **它验的是一个当前只是假设的前提**：播种 + 压缩采样能把抖动收窄。
    这件事便宜到几乎免费，而它挡住的是一笔贵得多的开销 ——
    **若同一 seed 两次都跑不一致，先别跑那批 27 agent 的臂**：
    该如实报告「seed 达不到可复现」，而不是花钱买一批更贵的不可复现产出。

    **结论推不到 27 agent。** 这里的规模小到几乎不触发并发交错（1 个 agent
    没有 agent 间的写入次序问题），而 asyncio 交错恰恰是我们播种**覆盖不到**
    的那一层。所以两次一致**不足以**推断 27 agent 上也一致。
    """
    results = []
    for i in (1, 2):
        out_dir = tmp_root / f"preflight{i}"
        argv = ["--agents", "1", "--rounds", "2", "--seed", str(seed),
                "--out", str(out_dir)]
        plan = ArmPlan(label=f"preflight{i}", seed=seed, out_dir=out_dir,
                       argv=argv)
        manifest = run_arm(plan, master, master_hashes)
        doc = None
        rp = out_dir / "twitter_rounds.json"
        if rp.is_file():
            doc = json.loads(rp.read_text(encoding="utf-8"))
        results.append({"manifest": manifest, "doc": doc})

    a, b = results
    if a["doc"] is None or b["doc"] is None:
        return {"ok": False, "reason": "有一次没跑出产出",
                "returncodes": [a["manifest"]["returncode"],
                                b["manifest"]["returncode"]]}

    states_a = [r.get("state") for r in a["doc"]["rounds"]]
    states_b = [r.get("state") for r in b["doc"]["rounds"]]
    beh_a = [r.get("behaviors") for r in a["doc"]["rounds"]]
    beh_b = [r.get("behaviors") for r in b["doc"]["rounds"]]
    return {
        "ok": states_a == states_b and beh_a == beh_b,
        "state_identical": states_a == states_b,
        "behaviors_identical": beh_a == beh_b,
        "seed": seed,
        "scale": {"agents": 1, "rounds": 2},
        "caveat": (
            "**只在这个规模上成立。** 1 个 agent 几乎没有 agent 间的写入次序"
            "问题，而 asyncio 的交错正是播种覆盖不到的那一层 —— "
            "所以这里一致**不足以**推断 27 agent 上也一致。"
            "真批次跑完仍需按 stability_report 的离散读数看。"),
        "if_inconsistent": (
            "同一 seed 两次就不一致 ⇒ **先别跑那批 27 agent 的臂**。"
            "该如实报告「seed 达不到可复现」，而不是花钱买一批更贵的"
            "不可复现产出。"),
    }


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ensure_console_encoding()
    ap = argparse.ArgumentParser(
        description="按 seed 起一批臂（默认不花钱；顺序跑，不并行）")
    ap.add_argument("--seeds", required=True, help="逗号分隔，如 11,22,33")
    ap.add_argument("--agents", type=int, default=PHASE_READING_MIN_AGENTS,
                    help=f"agent 数，默认 {PHASE_READING_MIN_AGENTS}")
    ap.add_argument("--rounds", type=int, default=15, help="轮数，默认 15")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--out", default=str(DEFAULT_ROOT),
                    help=f"臂根目录（默认 {DEFAULT_ROOT}）")
    ap.add_argument("--master", default=str(DEFAULT_MASTER),
                    help=f"底本目录，三个输入的来源（默认 {DEFAULT_MASTER}）")
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO)
    ap.add_argument("--chunking-guard", action="store_true")
    ap.add_argument("--events-from", choices=("phases", "event_order"),
                    default="phases")
    ap.add_argument("--allow-small-scale", action="store_true",
                    help=f"允许少于 {PHASE_READING_MIN_AGENTS} agent"
                         "（那样的臂读不了相位）")
    ap.add_argument("--preflight", action="store_true",
                    help="只跑同 seed 两次的小规模一致性检查，不起真批次")
    ap.add_argument("--preflight-seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="允许覆盖已有的臂")
    ap.add_argument("--yes", action="store_true",
                    help="真的开始跑。**不给这个参数就只是打印计划**")
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    master = Path(args.master)
    if not master.is_absolute():
        master = REPO_ROOT / master

    try:
        seeds = parse_seeds(args.seeds)
    except BatchRefused as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2

    simulate_args = ["--agents", str(args.agents), "--rounds", str(args.rounds),
                     "--scenario", args.scenario,
                     "--events-from", args.events_from]
    if args.temperature is not None:
        simulate_args += ["--temperature", str(args.temperature)]
    if args.chunking_guard:
        simulate_args.append("--chunking-guard")

    options = {f"seed{s}": {
        "agents": args.agents, "rounds": args.rounds,
        "chunking_guard": args.chunking_guard, "events_from": args.events_from,
        "platform": "twitter", "scenario": args.scenario,
        "temperature": args.temperature, "no_world_state": False,
        "no_phases": False, "no_knowledge": False, "no_feedback": False,
    } for s in seeds}

    # --- 先做全部只读的检查，一个请求都还没发 ---
    try:
        plans = [] if args.preflight else plan_batch(
            seeds, out_root, simulate_args, options, master=master,
            allow_small=args.allow_small_scale, agents=args.agents,
            force=args.force)
        master_hashes = hash_master(master)
    except BatchRefused as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2

    print(f"底本：{master}")
    for name in INPUT_FILES:
        got = master_hashes[name]
        print(f"  {name}  {got[:12]}…" if got
              else f"  {name}  **底本没有**（可选加速，见下）")
    if absent := absent_optional(master_hashes):
        print(f"  ⚠️ 底本没有 {'、'.join(absent)} —— **不拦**（它不入库，新 clone "
              "本来就没有），但本批每支臂会各自归类：多花 LLM 调用，"
              "而且与带缓存的那批**不可比**（`batch.json` 把「没有」也记成一个值，"
              "换回来时会报「底本已经换过」）。要带上就先把那份文件放进底本目录。")

    if args.preflight:
        print()
        print(f"preflight：同一个 seed（{args.preflight_seed}）"
              f"跑两次 --agents 1 --rounds 2，比对 state 与 behaviors")
        import tempfile
        with tempfile.TemporaryDirectory(prefix="weiran-preflight-") as tmp:
            try:
                result = run_preflight(master, master_hashes,
                                       seed=args.preflight_seed,
                                       tmp_root=Path(tmp))
            except (BatchRefused, subprocess.TimeoutExpired) as exc:
                print(f"✗ preflight 没跑完：{exc}", file=sys.stderr)
                return 2
        print(json.dumps({k: v for k, v in result.items()
                          if k != "caveat"}, ensure_ascii=False, indent=2))
        print()
        if result["ok"]:
            print("✓ 这个规模上两次一致。")
            print("  " + result["caveat"])
            return 0
        print("✗ 两次不一致 —— " + result["if_inconsistent"], file=sys.stderr)
        return 1

    # --- 计划与估算。不给 --yes 就到此为止 ---
    baseline = measure_baseline(DEFAULT_ROUNDS_FILE)
    print()
    print(f"计划：{len(plans)} 支臂，顺序跑，不并行")
    for p in plans:
        print(f"  {p.label} → {p.out_dir}")
    print(f"  参数：{args.agents} agent × {args.rounds} 轮"
          + (f"，temperature={args.temperature}" if args.temperature is not None
             else "，temperature 未传")
          + (f"，seed={seeds}" if len(seeds) <= 5 else ""))
    for line in format_estimate(baseline, n_arms=len(plans),
                               agents=args.agents, rounds=args.rounds):
        print(line)

    if not args.yes:
        print()
        print("**没有 --yes：不会发任何请求，也没有花任何钱。**")
        print("确认上面这份计划与估算是你要的之后，再加 --yes 重跑。")
        return 0

    previous_warning = check_against_previous_batch(out_root, master_hashes)
    if previous_warning:
        print(f"✗ {previous_warning}", file=sys.stderr)
        return 2

    out_root.mkdir(parents=True, exist_ok=True)
    print()
    print(f"开始跑 {len(plans)} 支臂 —— 中途失败会停下并报第几支坏了，"
          "不接着跑（没有断点续跑）。")
    for i, plan in enumerate(plans, 1):
        print(f"[{i}/{len(plans)}] {plan.label} …", flush=True)
        try:
            manifest = run_arm(plan, master, master_hashes)
        except BatchRefused as exc:
            print(f"✗ {plan.label} 的输入没落地：{exc}", file=sys.stderr)
            print(f"停在 [{i}/{len(plans)}]，后面 {len(plans) - i} 支没跑。",
                  file=sys.stderr)
            return 1
        if not manifest["ok"]:
            print(f"✗ {plan.label} 退出码 {manifest['returncode']}",
                  file=sys.stderr)
            if manifest["stderr_tail"]:
                print(manifest["stderr_tail"], file=sys.stderr)
            print(f"停在 [{i}/{len(plans)}]，后面 {len(plans) - i} 支没跑。"
                  "（臂没有断点续跑，修好原因后按 --force 重跑这一支。）",
                  file=sys.stderr)
            return 1
        print(f"      ok  {manifest['out_sha256'][:12]}…")

    (out_root / "batch.json").write_text(json.dumps({
        "master": master_hashes,
        "master_dir": rel_path(master),
        "arms": [p.label for p in plans],
        "options": options[plans[0].label],
        "command": "python backend/run_seeds.py "
                   + " ".join(argv if argv is not None else sys.argv[1:]),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"✓ {len(plans)} 支臂都跑完了。汇总：")
    print(f"    python backend/stability_report.py --root {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
