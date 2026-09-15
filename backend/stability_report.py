"""跨 seed 稳定性汇总：把 N 支臂的产出折成**一份读数**。

**离线、无网络、无 LLM。** 输入只有各臂目录里的 `twitter_rounds.json` 与
`arm.json`；输出 `stability.json` + `stability.md`。

**它为什么必须在仓库里、而且必须自己会拒绝。**

本项目自认过「正式口径是多 seed 跑 N 次报均值与方差」，但这句话此前只有
前半句能兑现：没有工具能汇总，也没有工具能**拒绝**。而「汇总一批臂」这件事
有三种坏法，每一种都会安静地产出一份看起来正常的读数：

  1. **把一支臂拷成 N 份。** 于是跨 seed 离散恒等于 0，读数看着漂亮得离谱。
  2. **臂之间口径不同还照样平均。** 27 agent 的臂与 12 agent 的臂混在一起，
     均值没有意义；而文件里看不出来（12 agent 的相位读数是已知无效的）。
  3. **把「不可采信」的臂折进结论。** 引擎换过之后旧臂重放不上，
     `replay_max_deviation` 会爆掉 —— 那些臂的曲线不是本引擎按当前参数
     算出来的，拿它们算离散等于在量别的东西。

所以下面每一条读数都配一条拒绝：**能算的才算，不能算的写清楚为什么不算**，
而不是给个数字让读者自己当心。拒绝的判据全是没有自由参数的那种（相等 / 不相等、
两个不同的值），**不设「波动小于 N 就算稳」**——理由是 `brief.py` 已经为
「不设自由参数」写过一次：`_weiran_decay` 那套阈值一旦写死，它就会变成一个
可以被调到达标的旋钮。

**它报的不是分数。** 跨 seed 一致**不等于**结论对：一个系统性偏差（引擎读
构成占比而不是体量、相位缺陷）在每一个 seed 下都会一模一样地重现。一致性
只能界定**抽样误差**那部分。这句话必须由金标自己说，本模块只引用 ——
与 `gold_check.py` 同款做法。

    python backend/stability_report.py --root data/runs/seeds
    python backend/stability_report.py --root data/runs/seeds --stdout
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from weiran.brief import (  # noqa: E402
    DEFAULT_SCALE,
    BriefError,
    build_brief,
    load_gold,
    load_rounds,
)
from weiran.config import REPO_ROOT, ensure_console_encoding  # noqa: E402
from weiran.gold_check import UNDECIDED, build_gold_check  # noqa: E402
from weiran.profiles import DEFAULT_SCENARIO  # noqa: E402
from weiran.world_state import DIMENSIONS  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "data" / "runs" / "seeds"

#: 一支臂的曲线「确实是本引擎按当前参数算出来的」的判据。**没有自由参数** ——
#: 它取自 `brief.py` 自己的重放自检，本模块只是引用它，不另立一个阈值。
REPLAY_TRUSTED_BELOW = 1e-4

#: 口径键：同一批臂里这些**必须全等**，否则那不是一批。
#:
#: 取值与比较都走 `_recorded()`：键**缺席**与键**在而值为 `None`** 是两件
#: 不同的事（前者是「不知道」，后者是「本次确实没传」）。所以「都没记」
#: **不**等于「一致」—— 那只是我们不知道它们一不一致。
PROTOCOL_KEYS = (
    "agents", "rounds", "platform", "days_per_round", "compressed",
    "events_from", "phases_on", "knowledge_on", "feedback_on", "world_state",
    "chunking_guard", "temperature",
)

#: 臂目录里必须与底本逐字节一致的那三个输入。**缺一个都不是一批** ——
#: `stance_cache.json` 尤其：它是跨运行共享且会被改写的，谁先跑谁占缓存。
INPUT_FILES = ("twitter_profiles.csv", "actor_knowledge.json",
               "stance_cache.json")

#: 渲染出来的文字里**不许出现**的说法。它们是「把一致性读成正确性」的入口，
#: 而这份产物恰恰不能支持那种读法。
_FORBIDDEN_IN_MD = ("更准", "更准确", "提升了", "稳健地证明", "证明了模型",
                    "准确率", "得分")


class StabilityError(Exception):
    """这一批臂不构成一次可汇总的稳定性批次。"""


# ---------------------------------------------------------------------------
# 读臂
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(doc: dict) -> str:
    """**只对 `rounds` 取指纹，不含 `meta`。**

    这是「这支臂的推演内容」的指纹，而 `out_sha256` 是「这个文件」的指纹。
    两个都要，因为它们回答不同的问句：

      - 一支臂被拷成 N 份、只把 `meta.seed` 改成不同的值 → **文件**指纹
        不同（`meta.seed` 在里面），但**内容**指纹相同。查拷贝只能靠内容
        指纹；拿全文件指纹去查会漏掉，因为每一支臂都会记自己的 seed。
      - 跑完再手改产物 → **文件**指纹与 `arm.json` 记的对不上。查篡改
        靠文件指纹。
    """
    payload = json.dumps(doc.get("rounds", []), ensure_ascii=False,
                         sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_arm(arm_dir: Path) -> dict:
    """读一支臂。缺件一律当场抛 —— **不做「尽量读」**。

    一支读不齐的臂如果被宽容地放进来，它的缺项会以「未记录」的样子混进
    均值，而读者看不出少了一支。
    """
    manifest_path = arm_dir / "arm.json"
    if not manifest_path.is_file():
        raise StabilityError(
            f"{arm_dir} 里没有 arm.json —— 这不像是按 seed 起出来的臂。"
            "（缺这份清单就无法回答「它是不是和别的一样跑法」，"
            "所以它不能被当成一支合法的臂，哪怕它的产出看着很完整。）")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("label", "seed", "inputs", "command"):
        if key not in manifest:
            raise StabilityError(f"{arm_dir}/arm.json 少了 {key!r}")

    rounds_files = sorted(arm_dir.glob("*_rounds.json"))
    if len(rounds_files) != 1:
        raise StabilityError(
            f"{arm_dir} 下有 {len(rounds_files)} 份 *_rounds.json，"
            "应当恰好一份")
    rounds_path = rounds_files[0]

    recorded = _sha256(rounds_path)
    if manifest.get("out_sha256") and manifest["out_sha256"] != recorded:
        raise StabilityError(
            f"{arm_dir} 的产出在开跑之后被改过：arm.json 记的是 "
            f"{manifest['out_sha256'][:12]}…，文件现在是 {recorded[:12]}…")

    doc = load_rounds(rounds_path)
    return {
        "label": manifest["label"],
        "dir": str(arm_dir),
        "manifest": manifest,
        "rounds_path": rounds_path,
        "rounds_sha256": recorded,
        "content_sha256": content_sha256(doc),
        "rounds_doc": doc,
    }


_ABSENT = object()


def recorded(arm: dict, key: str):
    """这个口径键在这支臂上取值多少。**只认产出自己的 `meta`。**

    **为什么不在 `meta` 缺席时回落到 `arm.json`。** `arm.json` 是驱动写的
    —— 它写的是「我打算怎么跑」，而且一个写错的驱动完全可能把 `None` 灌进
    自己根本没读到的键里。那样一来「不知道」就被洗成了「记了、值是空」，
    而这是两种必须分开的状态。`meta` 是那次运行自己结束时写的，驱动想洗也
    洗不到它，所以口径一律以它为准。

    **`None` 是一个值，`缺席` 不是。** 全批都没传 `--temperature` 时
    `temperature` 键在、值为 `None`：那是一次口径明确的一致（确实没传）；
    而键整个不存在时我们**不知道**当时传了没有 —— 后者不能当一致读。
    """
    meta = arm["rounds_doc"]["meta"]
    return (meta[key], True) if key in meta else (_ABSENT, False)


def check_batch(arms: list[dict]) -> list[str]:
    """这一批能不能汇总。返回问题清单，空 = 能。"""
    problems: list[str] = []

    # 口径必须一致，且必须以产出自己的 `meta` 为准（理由见 `recorded`）。
    for key in PROTOCOL_KEYS:
        seen: dict[str, list[str]] = {}
        unrecorded: list[str] = []
        for a in arms:
            value, present = recorded(a, key)
            if not present:
                unrecorded.append(a["label"])
                continue
            seen.setdefault(repr(value), []).append(a["label"])
        if unrecorded:
            problems.append(
                f"口径 {key!r} 在 {'、'.join(unrecorded)} 的产出里**根本没被记录** —— "
                "「都没记」不是「一致」，只是我们无从确认这批臂跑法相同。"
                "（这正是 `OASIS_*` / `FLASK_*` 那类「设了不生效、记了看不出」"
                "的失效形状，所以这里出声而不是放过。）")
        if len(seen) > 1:
            detail = "；".join(f"{v} → {','.join(ls)}" for v, ls in seen.items())
            problems.append(f"口径 {key!r} 跨臂不一致：{detail}")

    # `arm.json` 与产出自己的 `meta` 不许互相矛盾。**这一条防的是手改
    # arm.json**：两边都读、且要求相等，改一处就会露。
    for a in arms:
        meta = a["rounds_doc"]["meta"]
        for key in PROTOCOL_KEYS:
            if key in meta and key in a["manifest"] and \
                    meta[key] != a["manifest"][key]:
                problems.append(
                    f"{a['label']}：arm.json 说 {key}={a['manifest'][key]!r}，"
                    f"产出 meta 说 {meta[key]!r} —— 两者必须一致"
                    "（产出是那次运行自己写的，arm.json 是驱动写的；"
                    "对不上说明其中一份被动过）")

    # 输入必须同源。不同源的臂之间，差异不止是 seed。
    for name in INPUT_FILES:
        seen = {}
        for a in arms:
            value = a["manifest"]["inputs"].get(name)
            seen.setdefault(value, []).append(a["label"])
        if len(seen) > 1:
            detail = "；".join(f"{str(v)[:12]}…（{','.join(ls)}）"
                              for v, ls in seen.items())
            problems.append(f"输入 {name} 跨臂不同源：{detail}")

    # **必须真的是多 seed。** 这一条是整份报告的立足点：
    # 拿一支臂拷贝 N 份，或拿 N 支臂却只记了一个种子，都不能叫稳定性批次。
    seeds = {a["manifest"]["seed"] for a in arms}
    # **查拷贝用内容指纹，不用文件指纹** —— 每一支臂都会把 seed 记进自己的
    # `meta`，所以「拷贝 + 改 seed」之后文件指纹是不同的，而推演内容一模一样。
    contents = {a["content_sha256"] for a in arms}
    if any(s is None for s in seeds):
        problems.append(
            "有臂没记 seed —— 无法证明这是多 seed 批次。"
            "（「没记」与「记了同一个」是两回事，但都不能当多 seed 用）")
    elif len(seeds) < 2:
        problems.append(f"所有臂记的是同一个 seed（{seeds}）—— 这不是多 seed")
    if len(contents) < 2:
        problems.append(
            "所有臂的推演内容逐字节相同 —— 这是一支臂被拷成了 N 份"
            "（只把 meta 里的 seed 改成了不同的值）。"
            "它的「跨 seed 离散」恒等于 0，没有任何信息")

    return problems


# ---------------------------------------------------------------------------
# 逐臂读数
# ---------------------------------------------------------------------------

def _drive_curve(brief: dict) -> list[float]:
    """逐轮的驱动量 = 该轮六维激励项绝对值之和。

    **这不是新物理**，只是把 `brief` 已经算好的 `excitation` 加起来 ——
    它与 `_weiran_decay/report_drive.py` 量的是同一个东西，那一套的读法
    （「驱动峰在 P3 之后」）也是在这个量上定案的。本模块不另起一套重放。
    """
    return [sum(abs(v) for v in r["excitation"].values())
            for r in brief["rounds"]]


def _peak(curve: list[float]) -> tuple[int, float]:
    """驱动峰所在轮，以及它与次高峰的间隔。

    **间隔要一起报。** `excitation` 在 `brief` 里是按 4 位小数落盘的，
    六维求和之后峰与次峰可能落在同一个数上 —— 那种轮次的「峰」是舍入的
    产物，不是读数。间隔小于舍入上界的臂**不可判定**，不assign 峰。
    """
    order = sorted(range(len(curve)), key=lambda i: curve[i], reverse=True)
    top = curve[order[0]]
    second = curve[order[1]] if len(order) > 1 else 0.0
    return order[0], top - second


#: 六维各按 4 位小数落盘，求和的舍入上界就是这一串。
_ROUNDING_FLOOR = len(DIMENSIONS) * 0.5e-4


def read_arm_readings(arm: dict, gold: dict, *, scale: float) -> dict:
    """一支臂的全部读数。**曲线、驱动峰、金标三态三样都要。**"""
    doc = arm["rounds_doc"]
    brief = build_brief(doc, gold, scale=scale, command=arm["manifest"]["command"])
    gold_check = build_gold_check(doc, gold, scale=scale,
                                  command=arm["manifest"]["command"])
    curve = _drive_curve(brief)
    peak, gap = _peak(curve)
    deviation = brief["replay_max_deviation"]
    return {
        "label": arm["label"],
        "seed": arm["manifest"]["seed"],
        "rounds_sha256": arm["rounds_sha256"],
        "replay_max_deviation": deviation,
        "trusted": deviation <= REPLAY_TRUSTED_BELOW,
        "drive_curve": curve,
        "drive_peak_round": peak,
        "drive_peak_gap": gap,
        # 间隔在舍入上界以下时，峰是舍入挑出来的，不是量出来的。
        "peak_undecidable": gap < _ROUNDING_FLOOR,
        "curve": {r["index"]: dict(r["state"]) for r in doc["rounds"]},
        "railed_rounds": brief["evidence"]["railed_rounds"],
        "railed_dims": brief["evidence"]["railed_dims"],
        "gold_verdicts": {a["id"]: a["verdict"] for a in gold_check["assertions"]},
    }


# ---------------------------------------------------------------------------
# 归并
# ---------------------------------------------------------------------------

def _spread(values: list[float]) -> dict:
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values),
            "range": max(values) - min(values)}


def merge_curves(readings: list[dict]) -> list[dict]:
    """逐轮逐维的跨臂均值与范围。"""
    rounds = sorted(readings[0]["curve"])
    out = []
    for r in rounds:
        per_dim = {}
        for d in DIMENSIONS:
            vals = [x["curve"][r][d] for x in readings]
            per_dim[d] = _spread(vals)
        out.append({"round": r, "dims": per_dim})
    return out


def merge_drive(readings: list[dict]) -> dict:
    """驱动峰是否跨臂一致。**不可判定的臂不参与判定，但点名列出。**"""
    undecidable = [x["label"] for x in readings if x["peak_undecidable"]]
    deciding = [x for x in readings if not x["peak_undecidable"]]
    peaks = {x["drive_peak_round"]: [] for x in deciding}
    for x in deciding:
        peaks[x["drive_peak_round"]].append(x["label"])
    return {
        "peaks": {str(k): v for k, v in sorted(peaks.items())},
        "agrees": len(peaks) == 1,
        "undecidable_arms": undecidable,
        "trusted_arms": [x["label"] for x in readings if x["trusted"]],
        "untrusted_arms": [{"label": x["label"],
                            "replay_max_deviation": x["replay_max_deviation"]}
                           for x in readings if not x["trusted"]],
    }


def merge_signal(readings: list[dict]) -> dict:
    """逐维比较两个量：**跨臂离散** 与 **臂内相邻轮的变动**。

    这是「这条曲线有没有信号」在稳定性口径下的问法：如果换个 seed 造成的
    差，和曲线自己一轮走出来的差是同一个量级，那么读出来的「上升 / 下降」
    就分不清是模型在动还是抽样在动。**跨臂离散明显小于臂内步长**才谈得上
    有信号。

    两个量都是**读数**，本函数只把它们的比印出来，**不设阈值、不下判定**
    —— 「离散小于步长就算有信号」这种话一旦写死就是一个可以被调到达标的
    旋钮，而它本来就没有客观分界。比值只供人看。
    """
    dims = list(DIMENSIONS)
    per_dim = {}
    for d in dims:
        # 跨臂散：每一轮先取该轮跨臂的极差，再在轮上平均。
        spreads = []
        for r in sorted(readings[0]["curve"]):
            vals = [x["curve"][r][d] for x in readings]
            spreads.append(max(vals) - min(vals))
        across = sum(spreads) / len(spreads)

        # 臂内步长：每条臂先取相邻轮的 |Δ| 均值，再在臂上平均。
        steps = []
        for x in readings:
            rs = sorted(x["curve"])
            deltas = [abs(x["curve"][rs[i + 1]][d] - x["curve"][rs[i]][d])
                      for i in range(len(rs) - 1)]
            if deltas:
                steps.append(sum(deltas) / len(deltas))
        within = sum(steps) / len(steps) if steps else 0.0

        per_dim[d] = {
            "across_arm_spread": across,
            "within_arm_step": within,
            # 步长为 0 时比值无定义（维度整段不动）——**不是无穷**，报 None。
            "ratio": (across / within) if within > 0 else None,
        }
    # 贴过界的维度上，两个量都被夹逼压小了，比值不可读。
    railed = sorted({d for x in readings for d in x["railed_dims"]})
    return {
        "per_dim": per_dim,
        "railed_dims": railed,
        "note": (
            "比值 = 跨臂离散 ÷ 臂内相邻轮步长。**比值小**才谈得上有信号"
            "（换 seed 的影响小于曲线自己的走动）。**本表不设阈值** —— "
            "没有客观分界，这里只把两个量并排印出来。"
        ),
        "railed_note": (
            "贴过界的维度上两个量都被 `_clamp` 压小了，比值不可读，"
            "**只标不判**。" if railed else ""),
    }


def merge_gold(readings: list[dict]) -> dict:
    """金标三态逐臂重算后归并成「稳 / 翻转」。

    这是本块最想要的读数：那四条同源否决到底是结构性缺陷，还是单次抽样的
    运气。**只有「每个臂上都是同一个结论」与「至少两个臂上不同」两种结果**
    —— 不设「多数一致就算稳」，那又是一个自由参数。
    """
    ids = sorted(readings[0]["gold_verdicts"])
    per = []
    for aid in ids:
        verdicts = [x["gold_verdicts"][aid] for x in readings]
        per.append({
            "id": aid,
            "verdicts": verdicts,
            "flipped": len(set(verdicts)) > 1,
            "undecided": sum(1 for v in verdicts if v == UNDECIDED),
        })
    flipped = [r["id"] for r in per if r["flipped"]]
    return {"per_assertion": per, "flipped_ids": flipped,
            "agrees": not flipped}


def build_stability(arms: list[dict], *, gold: dict, scale: float = DEFAULT_SCALE,
                    root: str = "", command: str = "") -> dict:
    """汇总一批臂。**不成立就抛，不产出半份读数。**"""
    problems = check_batch(arms)
    if problems:
        raise StabilityError(
            "这一批臂不构成一次可汇总的稳定性批次：\n  - " + "\n  - ".join(problems))

    readings = [read_arm_readings(a, gold, scale=scale) for a in arms]
    trusted = [x for x in readings if x["trusted"]]

    # 引擎换过 / 产物被改过，都会让重放对不上。**两种形状要区别对待** ——
    # 报成同一句话会把「去重跑」和「去查引擎」两件事混成一件。
    refusals = []
    if not trusted:
        raise StabilityError(
            "**每一支臂都重放不上**（replay_max_deviation 全部超过 "
            f"{REPLAY_TRUSTED_BELOW:g}）——这是「引擎换过」的形状，"
            "不是「产物可疑」。例如 `VOLUME_REF` 或每轮注入体量改过之后，"
            "旧臂的曲线就不是当前引擎算出来的，拿它们算离散没有意义。\n"
            "  " + "\n  ".join(
                f"{x['label']}: {x['replay_max_deviation']:.3g}" for x in readings))
    if len(trusted) < len(readings):
        refusals.append(
            f"{len(readings) - len(trusted)} 支臂重放不上、**已排除在读数之外**："
            + "、".join(x["label"] for x in readings if not x["trusted"])
            + "（单支对不上 = 该臂的产物可能被手改过；全部对不上 = 引擎换过）")
    if len(trusted) < 2:
        raise StabilityError(
            f"可采信的臂只剩 {len(trusted)} 支 —— 一支臂算不出跨 seed 离散")

    drive = merge_drive(trusted)
    gold_merged = merge_gold(trusted)

    return {
        "not_a_score": {
            "statement": (
                "本表报的是**同一批设定下 N 次抽样的离散**，不是评分。"
                "跨 seed 一致**不等于**结论对：一个系统性偏差会在每一个 seed 下"
                "一模一样地重现，一致性只能界定**抽样误差**那一部分。"
            ),
            "why": gold.get("gold_status", {}).get("what_is_NOT_solid", ""),
            "source": "benchmark/scenarios/*/reference_data.json → gold_status",
            "what_agreement_bounds": (
                "只界定「换一个 seed 会不会换个结论」。"
                "**它不界定**「这个结论是不是对的」—— 那要靠金标断言本身，"
                "而金标自己已经写明了它哪里不硬。"
            ),
        },
        # 口径表：**缺席显示「未记录」，不显示 `null`、更不显示 `0`**。
        # `null` 是一个信息（本次确实没传），「未记录」是另一个（不知道）；
        # 把它们印成同一个东西，读者就再也分不出这批臂的口径是不是明确的。
        "protocol": {
            k: (arms[0]["rounds_doc"]["meta"][k]
                if k in arms[0]["rounds_doc"]["meta"] else "未记录")
            for k in PROTOCOL_KEYS
        },
        "arms": [
            {k: x[k] for k in ("label", "seed", "rounds_sha256", "trusted",
                               "replay_max_deviation", "drive_peak_round",
                               "drive_peak_gap", "peak_undecidable",
                               "railed_dims")}
            for x in readings
        ],
        "curve": merge_curves(readings),
        "signal": merge_signal(readings),
        "drive": drive,
        "gold": gold_merged,
        "refusals": refusals,
        "evidence": {
            "railed_dims": sorted({d for x in readings for d in x["railed_dims"]}),
            "railed_note": (
                "贴过界的维度上「跨臂离散」是假的 —— 夹逼把它压小了。"
                "这些维度**只标不判**。"),
        },
        "_provenance": {
            "root": root,
            "command": command,
            "arms": [a["label"] for a in arms],
            "arm_dirs": [a["dir"] for a in arms],
        },
    }


# ---------------------------------------------------------------------------
# 文字
# ---------------------------------------------------------------------------

def _unrecorded(value) -> str:
    """缺失一律显示「未记录」，**绝不显示 0**。"""
    return "未记录" if value is None else str(value)


def render_markdown(doc: dict) -> str:
    L: list[str] = []
    w = L.append

    w("# 跨 seed 稳定性读数")
    w("")
    w(f"> {doc['not_a_score']['statement']}")
    w("")
    w(f"> 出处：{doc['not_a_score']['source']}")
    w("")

    if doc["refusals"]:
        w("## ⚠️ 本批次被拒绝的部分")
        w("")
        for r in doc["refusals"]:
            w(f"- {r}")
        w("")

    w("## 一、这批臂")
    w("")
    w("| 臂 | seed | 重放偏差 | 可采信 | 驱动峰 | 峰-次峰 |")
    w("|---|---|---|---|---|---|")
    for a in doc["arms"]:
        peak = "不可判定" if a["peak_undecidable"] else f"R{a['drive_peak_round']}"
        w(f"| {a['label']} | {_unrecorded(a['seed'])} | "
          f"{a['replay_max_deviation']:.2e} | "
          f"{'是' if a['trusted'] else '否，已排除'} | {peak} | "
          f"{a['drive_peak_gap']:.2e} |")
    w("")
    w(f"口径（跨臂全等才算一批）：`{json.dumps(doc['protocol'], ensure_ascii=False)}`")
    w("")

    w("## 二、驱动峰")
    w("")
    d = doc["drive"]
    if d["agrees"]:
        pk = next(iter(d["peaks"]))
        w(f"所有可判定的臂，驱动峰都在 **R{pk}**。")
    else:
        w("**驱动峰在臂之间不一致。**")
        w("")
        for pk, labels in d["peaks"].items():
            w(f"- R{pk}：{'、'.join(labels)}")
    if d["undecidable_arms"]:
        w("")
        w(f"不可判定（峰与次峰的间隔小于六维求和后的舍入上界）："
          f"{'、'.join(d['undecidable_arms'])}")
    w("")
    w("驱动量 = 该轮六维激励项绝对值之和，取自各臂简报的 `excitation`。"
      "**它不是一个「越大越好」的量**，只是一个相位读数。")
    w("")

    w("## 三、金标三态")
    w("")
    g = doc["gold"]
    if g["agrees"]:
        w("每条断言在可采信的臂上结论一致。")
    else:
        w(f"**有 {len(g['flipped_ids'])} 条断言在臂之间翻转**："
          + "、".join(g["flipped_ids"]))
        w("")
        w("翻转意味着「它在这一次运行里发生了没有」这件事本身要换个 seed 才能"
          "定论 —— 那几条断言不能按单次产出的读数读。")
    w("")
    w("| 断言 | 各臂结论 | 其中不可判定 |")
    w("|---|---|---|")
    for r in g["per_assertion"]:
        w(f"| {r['id']} | {'、'.join(r['verdicts'])} | {r['undecided']} |")
    w("")

    w("## 四、有信号吗：跨臂离散 ÷ 臂内步长")
    w("")
    s = doc["signal"]
    w(s["note"])
    w("")
    w("| 维度 | 跨臂离散 | 臂内相邻轮步长 | 比值 |")
    w("|---|---|---|---|")
    for d, row in s["per_dim"].items():
        ratio = "—（该维整段不动）" if row["ratio"] is None \
            else f"{row['ratio']:.2f}"
        w(f"| {d} | {row['across_arm_spread']:.4f} | "
          f"{row['within_arm_step']:.4f} | {ratio} |")
    if s["railed_note"]:
        w("")
        w(f"**{'、'.join(s['railed_dims'])}** 贴过界 —— {s['railed_note']}")
    w("")

    w("## 五、逐轮逐维（跨臂均值 / 范围）")
    w("")
    rail = doc["evidence"]
    w(f"贴过界的维度：{'、'.join(rail['railed_dims']) or '（无）'} —— "
      f"{rail['railed_note']}")
    w("")
    dims = list(DIMENSIONS)
    w("| 轮 | " + " | ".join(dims) + " |")
    w("|---|" + "---|" * len(dims))
    for row in doc["curve"]:
        cells = [f"{row['dims'][d]['mean']:.3f}"
                 f"<br><sub>{row['dims'][d]['min']:.3f}~"
                 f"{row['dims'][d]['max']:.3f}</sub>" for d in dims]
        w(f"| R{row['round']} | " + " | ".join(cells) + " |")
    w("")
    w("每一格是 `均值`，下面一行小字是 `最小~最大`。**本表不设「波动小于 N 就"
      "算稳」** —— 那种阈值一旦写死就会变成一个可以被调到达标的旋钮。")
    w("")

    w("## 六、来源")
    w("")
    w(f"- 臂目录：`{doc['_provenance']['root']}`")
    for label, adir in zip(doc["_provenance"]["arms"],
                           doc["_provenance"]["arm_dirs"]):
        w(f"  - {label} → `{adir}`")
    w(f"- 生成命令：`{doc['_provenance']['command']}`")
    w("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# 机检
# ---------------------------------------------------------------------------

def check_stability(doc: dict, md: str) -> list[str]:
    """产物自己的口径检查。违规就返回问题（调用方负责出声）。

    两条：① **这是读数不是评分** —— 那一句声明要在正文里；
    ② 正文里不许出现把一致性读成正确性的说法。
    """
    problems = []
    if doc["not_a_score"]["statement"] not in md:
        problems.append("正文里没有「这不是评分」那句声明")
    for word in _FORBIDDEN_IN_MD:
        if word in md:
            problems.append(
                f"正文里出现了 {word!r} —— 这份产物支持不了那种读法："
                "跨 seed 一致只界定抽样误差，不界定结论对不对")
    # 「缺键 ≠ 0」：没有任何一条臂的记录时，正文里不许出现数字 0 顶替它。
    if any(a["seed"] is None for a in doc["arms"]) and "未记录" not in md:
        problems.append("有臂没记 seed，正文里却没有出现「未记录」")
    return problems


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ensure_console_encoding()
    ap = argparse.ArgumentParser(
        description="跨 seed 稳定性读数（不是分数）")
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help=f"臂目录的父目录（默认 {DEFAULT_ROOT}）")
    ap.add_argument("--scenario", default=None, help="场景目录（默认同 simulate）")
    ap.add_argument("--intervention-scale", type=float, default=DEFAULT_SCALE,
                    help=f"离线分支的干预倍数，默认 {DEFAULT_SCALE}")
    ap.add_argument("--out", default=None, help="markdown 输出路径")
    ap.add_argument("--json", default=None, help="结构化输出路径")
    ap.add_argument("--stdout", action="store_true", help="把全文也打到终端")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    scenario_dir = Path(args.scenario) if args.scenario else Path(DEFAULT_SCENARIO)
    if not scenario_dir.is_absolute():
        scenario_dir = REPO_ROOT / scenario_dir

    if not root.is_dir():
        print(f"✗ 没有 {root} —— 先用 run_seeds.py 起一批臂", file=sys.stderr)
        return 2

    arm_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not arm_dirs:
        print(f"✗ {root} 下没有任何臂目录", file=sys.stderr)
        return 2

    command = "python backend/stability_report.py --root " + str(args.root)
    try:
        arms = [read_arm(d) for d in arm_dirs]
        gold = load_gold(scenario_dir)
        doc = build_stability(arms, gold=gold,
                              scale=args.intervention_scale,
                              root=str(args.root), command=command)
    except (StabilityError, BriefError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2

    md = render_markdown(doc)
    problems = check_stability(doc, md)
    if problems:
        print("✗ 口径机检未通过 —— 读数**不落盘**：\n  - "
              + "\n  - ".join(problems), file=sys.stderr)
        return 2

    out_md = Path(args.out) if args.out else root / "stability.md"
    out_js = Path(args.json) if args.json else root / "stability.json"
    out_js.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    out_md.write_text(md, encoding="utf-8")

    print(f"已写出 {out_md}")
    print(f"  {len(doc['arms'])} 支臂，可采信 {len(doc['drive']['trusted_arms'])}")
    print(f"  驱动峰："
          + ("一致" if doc["drive"]["agrees"] else "**不一致**"))
    print(f"  金标："
          + ("一致" if doc["gold"]["agrees"]
             else f"翻转 {len(doc['gold']['flipped_ids'])} 条"))
    if args.stdout:
        print()
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
