"""最小连通性冒烟：端点 / 模型 / JSON 模式 / 中文输出 / 推理开关 / 延迟与用量。

**这个脚本回答五个问题，每个都必须是「实测」而不是「照文档推断」：**

  1. 端点通不通、模型 id 对不对？
  2. JSON 模式能用吗？—— 服务端要求提示词里出现 "json"，缺了直接 400。
     第 2 项探测**故意不写 "json"**，用来验证客户端会自动补上。
     不支持 JSON 模式的话，整条抽取管线（实体/关系/画像）就建立在
     一个不存在的特性上。
  3. 中文输出正常吗？（本项目全部语料是中文，这一步不能想当然）
  4. **推理开关值多少钱？** 实测本项目所用模型默认开启推理，思维链占输出
     token 的九成以上；关掉后输出降到约 1/13。这个比例随模型而变，
     所以每次冒烟都重测一遍 —— 换模型时不重测，成本估算会无声地错一个数量级。
  5. 一次调用多少 token、多少秒？—— 这个数字决定演示规模能做多大。

用法：
    cd backend
    python -m weiran.smoke
    python -m weiran.smoke --price-in 0.5 --price-out 2.0   # 每百万 token 的单价
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass

from .config import ConfigError, load_config
from .llm import LLMClient, LLMError, Ledger


@dataclass
class Probe:
    """一次探测的结果。失败不抛出，如实记录 —— 冒烟脚本的价值在于报出坏消息。"""

    name: str
    ok: bool
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    detail: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _usage_of(ledger: Ledger, tag: str) -> tuple[int, int]:
    u = ledger.by_tag.get(tag)
    return (u.prompt_tokens, u.completion_tokens) if u else (0, 0)


def _timed(fn) -> tuple[float, object]:
    t0 = time.monotonic()
    out = fn()
    return time.monotonic() - t0, out


def run_probes(client: LLMClient, ledger: Ledger) -> list[Probe]:
    probes: list[Probe] = []

    # --- 1. 最朴素的一问一答，确认链路通 --------------------------------
    p = Probe("基础对话（中文）", ok=False)
    try:
        secs, text = _timed(
            lambda: client.chat(
                [{"role": "user", "content": "用一句话说明什么是舆情。"}],
                tag="smoke.basic",
                max_tokens=128,
            )
        )
        p.seconds, p.ok = secs, bool(text.strip())
        p.prompt_tokens, p.completion_tokens = _usage_of(ledger, "smoke.basic")
        p.detail = text.strip().replace("\n", " ")[:120]
    except (LLMError, ConfigError) as exc:
        p.detail = str(exc)[:300]
    probes.append(p)

    # --- 2. JSON 模式，且**故意不写 "json"** -----------------------------
    # 这是对修复的回归验证。服务端硬性要求提示词含 "json"，否则 400；
    # `chat_json` 现在会自动补上。所以这一条的**提示词里刻意不出现 json**——
    # 它必须通过。若哪天它开始报 400，说明自动补词那层被改坏了。
    p = Probe("JSON 模式（提示词不含 json，应由客户端自动补）", ok=False)
    try:
        secs, data = _timed(
            lambda: client.chat_json(
                [
                    {
                        "role": "user",
                        "content": "把「学校公布了就业数据，同学们不太相信」拆成 情绪 与 对象 两个字段。",
                    }
                ],
                tag="smoke.json_auto",
                max_tokens=512,
            )
        )
        p.seconds, p.ok = secs, True
        p.prompt_tokens, p.completion_tokens = _usage_of(ledger, "smoke.json_auto")
        p.detail = json.dumps(data, ensure_ascii=False)[:160]
    except (LLMError, ConfigError) as exc:
        p.detail = str(exc)[:300]
    probes.append(p)

    # --- 3. 推理开关的代价 ------------------------------------------------
    # 本项目最大的一个成本事实：关掉推理，输出 token 降到约 1/13。
    # 每跑一次冒烟都重新测一遍，是因为这个比例取决于模型，
    # 换模型时如果不重测，成本估算会在无声中错一个数量级。
    prompt = [
        {"role": "user", "content": "用 json 输出：把「学校公布了就业数据，同学们不太相信」"
                                    "拆成 情绪 与 对象 两个字段。"}
    ]
    tok = {}
    for flag, tag in ((False, "smoke.think_off"), (True, "smoke.think_on")):
        label = "推理关闭（本项目默认）" if not flag else "推理开启"
        p = Probe(f"推理开关对比 · {label}", ok=False)
        try:
            secs, _ = _timed(
                lambda f=flag, t=tag: client.chat_json(
                    prompt, tag=t, max_tokens=4096, thinking=f
                )
            )
            pin, pout = _usage_of(ledger, tag)
            p.seconds, p.ok = secs, True
            p.prompt_tokens, p.completion_tokens = pin, pout
            reasoning = ledger.by_tag[tag].reasoning_tokens
            tok[flag] = pout
            p.detail = f"输出 {pout} tok，其中推理 {reasoning} tok"
        except (LLMError, ConfigError) as exc:
            p.detail = str(exc)[:300]
        probes.append(p)

    if tok.get(False) and tok.get(True):
        ratio = tok[True] / tok[False]
        probes.append(
            Probe(
                "推理开关对比 · 结论",
                ok=True,
                detail=f"开启推理的输出 token 是关闭时的 {ratio:.1f} 倍"
                       f"（{tok[True]} vs {tok[False]}）",
            )
        )

    return probes


def _report(probes: list[Probe], ledger: Ledger, price_in: float, price_out: float) -> None:
    w = sys.stdout
    w.write("\n" + "=" * 78 + "\n")
    w.write("「未然」LLM 端点冒烟\n")
    w.write("=" * 78 + "\n\n")

    for p in probes:
        mark = "OK  " if p.ok else "FAIL"
        w.write(f"  [{mark}] {p.name}\n")
        if p.ok:
            w.write(
                f"         {p.seconds:.2f}s  "
                f"{p.total_tokens} tok（入 {p.prompt_tokens} / 出 {p.completion_tokens}）\n"
            )
        w.write(f"         {p.detail}\n\n")

    w.write("-" * 78 + "\n")
    w.write(ledger.summary() + "\n")

    if price_in or price_out:
        cost = (
            ledger.total.prompt_tokens / 1e6 * price_in
            + ledger.total.completion_tokens / 1e6 * price_out
        )
        w.write(f"\n本次总花费 ≈ {cost:.6f} 元"
                f"（单价 入 {price_in}/M、出 {price_out}/M）\n")
    else:
        w.write("\n未提供单价，故不报金额 —— **不猜价格**。\n"
                "  要算钱请把服务商价目表上的单价传进来：\n"
                "    --price-in <每百万输入token价格> --price-out <每百万输出token价格>\n")

    ok = sum(1 for p in probes if p.ok)
    w.write(f"\n{ok}/{len(probes)} 项通过\n")
    w.write("=" * 78 + "\n\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LLM 端点最小连通性冒烟")
    ap.add_argument("--price-in", type=float, default=0.0,
                    help="每百万输入 token 的单价（元）；不填则不报金额")
    ap.add_argument("--price-out", type=float, default=0.0,
                    help="每百万输出 token 的单价（元）")
    args = ap.parse_args(argv)

    try:
        config = load_config(require_llm=True)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    # 密钥只在必要时现形：报长度与前缀，够判断「是不是空的/拿错了」，
    # 又不至于把凭据打进终端记录。
    key = config.llm.api_key
    print(f"端点  {config.llm.base_url}")
    print(f"模型  {config.llm.model}")
    print(f"密钥  长度 {len(key)}，前缀 {key[:3]!r}……（其余不显示）")

    ledger = Ledger()
    client = LLMClient(config.llm, ledger)
    probes = run_probes(client, ledger)
    _report(probes, ledger, args.price_in, args.price_out)

    return 0 if all(p.ok for p in probes) else 1


if __name__ == "__main__":
    sys.exit(main())
