"""OpenAI 兼容的对话补全客户端。

只做一件事：把 messages 发出去、把 content 拿回来，**顺便把用量记下来**。

为什么只用 requests 而不引入 openai SDK：需求就是一个 HTTP POST。
依赖越少，"评委一条命令跑起来"越不容易失败。代价是要自己写重试——
但那本来也就二十行。

为什么强调记账：本项目的成本是可控性的一部分。仿真要跑几十轮 × 几十个 agent，
不记账就不知道一次演示要花多少钱，也就没法决定演示规模。Ledger 是为此存在的。

--------------------------------------------------------------------------
**关于推理模型：本文件的三条设计都由实测逼出来，不是照文档写的。**

实测对象：`deepseek-flash` / `deepseek-v4-pro`（2026-09-13，见 进度.md）。

  1. **这两个模型默认是推理模型**，思维链走 `reasoning_content` 字段，
     与答案字段 `content` 分开。实测推理 token 占输出的 **90%–96%**。

  2. **`max_tokens` 是推理与答案共享的预算。** 给 200 时，推理吃掉全部 200，
     `content` 返回**空串**，而 HTTP 状态码是 **200**、`finish_reason` 是 `length`。
     也就是说：**预算不足的失败看起来像成功。** 因此本客户端必须自己检查
     `finish_reason`，不能只看状态码。

  3. **推理可以关掉，且关掉后输出 token 降到约 1/13。**
     实测有效的参数只有两个：`thinking={"type":"disabled"}` 与
     `reasoning_effort="none"`。同样看似合理的 `enable_thinking` /
     `thinking_budget` / `chat_template_kwargs` **全部无效**（推理照跑）。
     本项目默认关闭推理：仿真里几百次调用问的是「这个学生接下来会说什么」，
     不需要长思维链，而代价是 13 倍。**需要推理质量的调用点显式传 thinking=True。**
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field

import requests

from .config import LLMConfig

# 只在这些状态码上重试。4xx（除 429）是请求本身有问题，重试没有意义。
_RETRYABLE = {408, 409, 429, 500, 502, 503, 504}

# 关闭推理。见模块开头第 3 条：只有这个参数实测有效。
_THINKING_OFF = {"type": "disabled"}
_THINKING_ON = {"type": "enabled"}

# json_object 模式要求提示词里出现 "json" 字样。这一条由服务端强制：
#   HTTP 400 Prompt must contain the word 'json' in some form to use
#   'response_format' of type 'json_object'
# 与其指望每个调用点都记得，不如由 chat_json 自己兜住。
_JSON_HINT = "json"
_JSON_HINT_SUFFIX = "\n\n请严格以 json 格式输出，不要包含任何解释性文字。"


class LLMError(RuntimeError):
    """调用失败。消息里必须包含：哪个模型、哪一步、服务端原话。"""


@dataclass
class Usage:
    """单次调用的用量。字段名对齐 OpenAI 的 usage 结构。

    `reasoning_tokens` 单独记：它**计费**，且在本项目的模型上占输出的九成以上。
    不单列的话，账本会告诉你「花了 1 万 token」却说不出其中 9 千是思维链，
    成本优化就无从下手。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    # 前缀缓存的命中 / 未命中。**必须单列 —— 它决定的是单价，不是数量。**
    # 服务端对命中前缀的那段输入给很大折扣（本项目所用端点即如此），于是
    # 「同样 1 万 token 的输入」按命中计费与按未命中计费不是一个价。
    # 只记 prompt_tokens 的话，账本只能回答「花了多少 token」，回答不了
    # 「按什么价花的」—— 而后者正是两条调用路径账单不同的地方。
    #
    # 两端点都不报时留 0。这时**不能读成「一次都没命中」**：见 cache_hit_rate。
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    #: 端点到底报没报这两个字段。**必须与数值分开存** —— 否则「没报」只能
    #: 用「数值为 0」来表达，而那正好也是「报了，一次都没命中」的表达。
    #: 与 `brief.py` 里 `cost.failures_recorded` 是同一条纪律。
    cache_recorded: bool = False

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def answer_tokens(self) -> int:
        """输出里真正是答案的那部分。"""
        return max(0, self.completion_tokens - self.reasoning_tokens)

    @property
    def cache_hit_rate(self) -> float | None:
        """前缀缓存命中率。**端点没报这个字段时返回 None，不是 0.0。**

        把「没报」显示成 0% 会把一句「我们不知道」读成一句「一次都没命中」，
        而这两件事的下一步动作完全相反（前者要去查端点，后者要去查提示词前缀）。
        """
        if not self.cache_recorded:
            return None
        seen = self.cache_hit_tokens + self.cache_miss_tokens
        if seen == 0:
            return None
        return self.cache_hit_tokens / seen

    def merge(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cache_hit_tokens += other.cache_hit_tokens
        self.cache_miss_tokens += other.cache_miss_tokens
        self.cache_recorded = self.cache_recorded or other.cache_recorded


@dataclass
class Ledger:
    """累计用量，按用途（tag）分组。

    `tag` 用来回答「钱花在哪了」——抽取？画像？仿真？简报？
    没有它，成本优化就是瞎猜。
    """

    by_tag: dict[str, Usage] = field(default_factory=dict)
    calls: int = 0
    failures: int = 0
    wall_seconds: float = 0.0

    def record(self, tag: str, usage: Usage, elapsed: float) -> None:
        self.by_tag.setdefault(tag, Usage()).merge(usage)
        self.calls += 1
        self.wall_seconds += elapsed

    def record_failure(self) -> None:
        self.failures += 1

    @property
    def total(self) -> Usage:
        agg = Usage()
        for u in self.by_tag.values():
            agg.merge(u)
        return agg

    def as_dict(self) -> dict:
        """落盘用的快照。**给「另一条调用路径」记账用。**

        本项目有两条路径：camel 驱动的 agent，和本模块的直调。直调走
        `requests`，不经过 camel 的模型对象，所以产出里逐轮的
        `calls` / `prompt_tokens` **一个数都不含它** —— 那笔钱在服务商
        账单上看得见，在自己的产物里看不见。这个快照就是把它补上。
        """
        t = self.total
        return {
            "calls": self.calls,
            "failures": self.failures,
            "wall_seconds": round(self.wall_seconds, 2),
            "prompt_tokens": t.prompt_tokens,
            "completion_tokens": t.completion_tokens,
            "reasoning_tokens": t.reasoning_tokens,
            "prompt_cache_hit_tokens": t.cache_hit_tokens,
            "prompt_cache_miss_tokens": t.cache_miss_tokens,
            "cache_recorded": t.cache_recorded,
            "by_tag": {
                tag: {
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "reasoning_tokens": u.reasoning_tokens,
                }
                for tag, u in sorted(self.by_tag.items())
            },
            "summary": self.summary(),
        }

    def summary(self) -> str:
        t = self.total
        rate = t.cache_hit_rate
        lines = [
            f"调用 {self.calls} 次（失败 {self.failures} 次），"
            f"耗时 {self.wall_seconds:.1f}s，"
            f"token 合计 {t.total}"
            f"（输入 {t.prompt_tokens} / 输出 {t.completion_tokens}"
            f"，其中推理 {t.reasoning_tokens}）",
            # 输入侧的价格几乎全由这一行决定。少了它，「两条路径花了同样多的
            # token」与「花了同样多的钱」就分不开 —— 而这两件事的账单能差几倍。
            # 未报时**不印任何数字**：印「命中率 0%」等于替端点回答了一个它
            # 没回答的问题。
            ("  前缀缓存：端点未报此字段，命中率无从判断"
             if not t.cache_recorded else
             ("  前缀缓存：已报字段但样本为空，命中率无从判断"
              if rate is None else
              f"  前缀缓存：命中率 {rate:.1%}"
              f"（命中 {t.cache_hit_tokens} / 未命中 {t.cache_miss_tokens} tok）")),
        ]
        for tag, u in sorted(self.by_tag.items(), key=lambda kv: -kv[1].total):
            r = u.cache_hit_rate
            lines.append(
                f"  {tag:22s} {u.total:8d} tok  "
                f"输入 {u.prompt_tokens} / 输出 {u.completion_tokens}"
                f"（推理 {u.reasoning_tokens}）"
                + ("" if r is None else f"  缓存命中 {r:.0%}")
            )
        return "\n".join(lines)


def _uget(obj, key: str):
    """从 dict 或对象上取一个字段。没有就 None（**不是 0**）。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def parse_cache(raw_usage) -> tuple[int, int, bool]:
    """从 usage 里取前缀缓存的（命中, 未命中, 端点是否报过）。

    见过两套命名，按优先级依次试：

      1. DeepSeek：`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`（都给）
      2. OpenAI：`prompt_tokens_details.cached_tokens`（只给命中）
      3. 都没有 → 第三个返回值为 False，含义是**端点没报**

    第 3 种必须与「报了，但都是 0」区分开，所以标志位单列而不靠数值推断。
    第 2 种只给命中数时，未命中按 `prompt_tokens - 命中` 推 —— 这是恒等式，
    不是猜测。

    **两条调用路径共用这一个解析器**：直调走 `requests` 拿到的是 dict，
    camel 那条拿到的是 SDK 的对象，形状不同、字段名相同。写两份的话，
    哪天补第三套命名就要记得改两处，而漏掉那处是无声的。
    """
    hit = _uget(raw_usage, "prompt_cache_hit_tokens")
    miss = _uget(raw_usage, "prompt_cache_miss_tokens")
    if hit is None and miss is None:
        details = _uget(raw_usage, "prompt_tokens_details")
        cached = _uget(details, "cached_tokens")
        if cached is None:
            return 0, 0, False
        n_hit = int(cached)
        # 只给命中数时，未命中是恒等式推出来的，不是猜的
        n_miss = max(0, int(_uget(raw_usage, "prompt_tokens") or 0) - n_hit)
        return n_hit, n_miss, True
    return int(hit or 0), int(miss or 0), True


def _strip_code_fence(text: str) -> str:
    """去掉 ```json ... ``` 包裹。

    即使要求了 json_object，部分服务商仍会裹一层 fence。
    这不是我们的 bug，但必须在这里兜住，否则整个管线会在深夜崩掉。
    """
    t = text.strip()
    if not t.startswith("```"):
        return t
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _ensure_json_hint(messages: list[dict]) -> list[dict]:
    """保证 messages 里出现 "json" 字样。

    服务端硬性要求，缺了直接 400。做法是复制一份再改 —— **不原地修改调用方
    传入的 list**，否则调用方复用同一个 messages 变量时会莫名其妙多出一段话。
    """
    if any(_JSON_HINT in str(m.get("content", "")).lower() for m in messages):
        return messages

    out = copy.deepcopy(messages)
    for m in reversed(out):
        if m.get("role") == "user":
            m["content"] = str(m.get("content", "")) + _JSON_HINT_SUFFIX
            return out
    # 没有 user 消息（罕见）。附到最后一条上，总比直接失败好。
    if out:
        out[-1]["content"] = str(out[-1].get("content", "")) + _JSON_HINT_SUFFIX
    return out


class LLMClient:
    """对话补全客户端。

    Args:
        config: 端点配置。
        ledger: 共用的用量账本；不传则自建一个（可通过 .ledger 取回）。
    """

    def __init__(self, config: LLMConfig, ledger: Ledger | None = None) -> None:
        self.config = config
        self.ledger = ledger if ledger is not None else Ledger()
        self._session = requests.Session()
        # 最近一次的思维链。调试用：答案不对时，第一件事是看它「想了什么」。
        self.last_reasoning: str = ""

    # -- 内部 --------------------------------------------------------------

    def _post(self, payload: dict, *, tag: str, max_retries: int) -> dict:
        last: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._session.post(
                    self.config.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.config.timeout,
                )
            except requests.RequestException as exc:
                last = exc
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 16))
                    continue
                self.ledger.record_failure()
                raise LLMError(
                    f"[{tag}] 网络错误，已重试 {max_retries} 次：{exc}"
                ) from exc

            if resp.status_code in _RETRYABLE and attempt < max_retries:
                time.sleep(min(2 ** attempt, 16))
                continue

            if resp.status_code != 200:
                self.ledger.record_failure()
                # 服务端原话要带出来 —— 否则「400 Bad Request」什么也说明不了
                body = resp.text[:500]
                raise LLMError(
                    f"[{tag}] HTTP {resp.status_code}（模型 {self.config.model}）：{body}"
                )

            try:
                return resp.json()
            except ValueError as exc:
                self.ledger.record_failure()
                raise LLMError(f"[{tag}] 响应不是合法 JSON：{resp.text[:300]}") from exc

        self.ledger.record_failure()
        raise LLMError(f"[{tag}] 重试耗尽：{last}")

    # -- 公开接口 ----------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        tag: str = "chat",
        json_mode: bool = False,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        thinking: bool = False,
        max_retries: int = 3,
    ) -> str:
        """发一轮对话，返回 assistant 的答案文本。

        Args:
            thinking: 是否启用推理。**默认关闭**，理由是实测它使输出 token
                增至约 13 倍，而仿真里的多数调用（「这个学生接下来会说什么」）
                并不需要长思维链。抽取、归因、简报这类要质量的调用点显式传 True。
            max_tokens: **推理与答案共享这个预算**。给得太小会出现
                「HTTP 200 + 空内容」这种伪装成成功的失败。经验值：
                关闭推理时 512 足够；开启推理时至少 2048。

        Raises:
            LLMError: 网络失败、非 200、或**输出被预算截断**。
        """
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # 部分 OpenAI 兼容端点不认识这个参数并会 400。实测本机所用端点支持；
        # 换端点若报错，在 .env 里设 LLM_SEND_THINKING_PARAM=0 即可关掉。
        if self.config.send_thinking_param:
            payload["thinking"] = _THINKING_ON if thinking else _THINKING_OFF

        started = time.monotonic()
        data = self._post(payload, tag=tag, max_retries=max_retries)
        elapsed = time.monotonic() - started

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        finish = choice.get("finish_reason")
        content = msg.get("content") or ""
        self.last_reasoning = msg.get("reasoning_content") or ""

        raw_usage = data.get("usage") or {}
        details = raw_usage.get("completion_tokens_details") or {}
        n_hit, n_miss, has_cache = parse_cache(raw_usage)
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
            completion_tokens=int(raw_usage.get("completion_tokens", 0)),
            reasoning_tokens=int(details.get("reasoning_tokens", 0)),
            cache_hit_tokens=n_hit,
            cache_miss_tokens=n_miss,
            cache_recorded=has_cache,
        )
        self.ledger.record(tag, usage, elapsed)

        # 预算被截断。必须在返回前拦下 —— 此时 content 可能是空串，
        # 也可能是一段截断的 JSON，两种都会在下游变成难查的错。
        if finish == "length":
            self.ledger.record_failure()
            raise LLMError(
                f"[{tag}] 输出被 max_tokens 截断（finish_reason=length）。"
                f"本次输出 {usage.completion_tokens} tok，其中推理 "
                f"{usage.reasoning_tokens} tok。"
                + (
                    "推理吃掉了几乎全部预算，答案还没开始写 —— "
                    "请提高 max_tokens，或确认该调用是否真的需要 thinking=True。"
                    if usage.reasoning_tokens >= usage.answer_tokens
                    else "答案写到一半被截断，请提高 max_tokens。"
                )
                + f"（当前 max_tokens={max_tokens}）"
            )

        if not content:
            self.ledger.record_failure()
            # 这里曾经把原因写成「可能被内容策略拦截」。那是错的，而且会
            # 把排查引向错误方向。真实原因几乎总是推理耗尽预算。
            raise LLMError(
                f"[{tag}] 返回了空内容。finish_reason={finish!r}，"
                f"推理 {usage.reasoning_tokens} tok / 输出 {usage.completion_tokens} tok。"
                f"若推理占了绝大多数，是 max_tokens 太小而非内容被拦截。"
                + (f" 思维链前 200 字：{self.last_reasoning[:200]}" if self.last_reasoning else "")
            )
        return content

    def chat_json(
        self,
        messages: list[dict],
        *,
        tag: str = "chat",
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
        thinking: bool = False,
        expect: type = dict,
    ):
        """发一轮对话并要求 JSON 输出。

        temperature 默认压到 0.2：抽取类任务要的是稳定，不是创意。

        会自动保证 messages 里含 "json" 字样（见 `_ensure_json_hint`）——
        服务端强制要求，缺了直接 400。

        Raises:
            LLMError: 返回内容无法解析为期望的类型。
        """
        text = self.chat(
            _ensure_json_hint(messages),
            tag=tag,
            json_mode=True,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
        )
        cleaned = _strip_code_fence(text)
        try:
            parsed = json.loads(cleaned)
        except ValueError as exc:
            raise LLMError(
                f"[{tag}] 要求 JSON 但解析失败：{exc}\n原文前 500 字：{cleaned[:500]}"
            ) from exc
        if not isinstance(parsed, expect):
            raise LLMError(
                f"[{tag}] 期望 {expect.__name__}，实际是 {type(parsed).__name__}"
            )
        return parsed
