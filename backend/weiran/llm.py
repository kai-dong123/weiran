"""OpenAI 兼容的对话补全客户端。

只做一件事：把 messages 发出去、把 content 拿回来，**顺便把用量记下来**。

为什么只用 requests 而不引入 openai SDK：需求就是一个 HTTP POST。
依赖越少，"评委一条命令跑起来"越不容易失败。代价是要自己写重试——
但那本来也就二十行。

为什么强调记账：本项目的成本是可控性的一部分。仿真要跑几十轮 × 几十个 agent，
不记账就不知道一次演示要花多少钱，也就没法决定演示规模。Ledger 是为此存在的。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import requests

from .config import LLMConfig

# 只在这些状态码上重试。4xx（除 429）是请求本身有问题，重试没有意义。
_RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """调用失败。消息里必须包含：哪个模型、哪一步、服务端原话。"""


@dataclass
class Usage:
    """单次调用的用量。字段名对齐 OpenAI 的 usage 结构。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def merge(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens


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
        lines = [
            f"调用 {self.calls} 次（失败 {self.failures} 次），"
            f"耗时 {self.wall_seconds:.1f}s，"
            f"token 合计 {self.total.total}"
            f"（输入 {self.total.prompt_tokens} / 输出 {self.total.completion_tokens}）"
        ]
        for tag, u in sorted(self.by_tag.items(), key=lambda kv: -kv[1].total):
            lines.append(f"  {tag:20s} {u.total:8d} tok  ({self.calls and ''}"
                         f"输入 {u.prompt_tokens} / 输出 {u.completion_tokens}）")
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
        max_retries: int = 3,
    ) -> str:
        """发一轮对话，返回 assistant 的文本。"""
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        started = time.monotonic()
        data = self._post(payload, tag=tag, max_retries=max_retries)
        elapsed = time.monotonic() - started

        raw_usage = data.get("usage") or {}
        self.ledger.record(
            tag,
            Usage(
                prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
                completion_tokens=int(raw_usage.get("completion_tokens", 0)),
            ),
            elapsed,
        )

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"[{tag}] 响应结构异常，缺少 choices[0].message.content："
                f"{json.dumps(data, ensure_ascii=False)[:300]}"
            ) from exc

        if not content:
            raise LLMError(f"[{tag}] 返回了空内容（可能被内容策略拦截）")
        return content

    def chat_json(
        self,
        messages: list[dict],
        *,
        tag: str = "chat",
        temperature: float = 0.2,
        max_tokens: int | None = None,
        expect: type = dict,
    ):
        """发一轮对话并要求 JSON 输出。

        temperature 默认压到 0.2：抽取类任务要的是稳定，不是创意。

        Raises:
            LLMError: 返回内容无法解析为期望的类型。
        """
        text = self.chat(
            messages,
            tag=tag,
            json_mode=True,
            temperature=temperature,
            max_tokens=max_tokens,
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
