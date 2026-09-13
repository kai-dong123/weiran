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

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def answer_tokens(self) -> int:
        """输出里真正是答案的那部分。"""
        return max(0, self.completion_tokens - self.reasoning_tokens)

    def merge(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens


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

    def summary(self) -> str:
        t = self.total
        lines = [
            f"调用 {self.calls} 次（失败 {self.failures} 次），"
            f"耗时 {self.wall_seconds:.1f}s，"
            f"token 合计 {t.total}"
            f"（输入 {t.prompt_tokens} / 输出 {t.completion_tokens}"
            f"，其中推理 {t.reasoning_tokens}）"
        ]
        for tag, u in sorted(self.by_tag.items(), key=lambda kv: -kv[1].total):
            lines.append(
                f"  {tag:22s} {u.total:8d} tok  "
                f"输入 {u.prompt_tokens} / 输出 {u.completion_tokens}"
                f"（推理 {u.reasoning_tokens}）"
            )
        return "\n".join(lines)


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
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
            completion_tokens=int(raw_usage.get("completion_tokens", 0)),
            reasoning_tokens=int(details.get("reasoning_tokens", 0)),
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
