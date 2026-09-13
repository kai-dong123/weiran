"""LLM 客户端测试。

**全部离线。不需要 API key，不联网，不发一次真实调用。**
做法是把 `_post` 换成一个假的：它记下送出去的 payload，返回预先编排的响应。
这样测的是**客户端自己的逻辑**——参数怎么组装、响应怎么解读、错误怎么报——
而不是服务商今天心情如何。

这里守的是三条**实测出来的**事实（见 `weiran/llm.py` 模块注释与 进度.md）：

  1. 服务端要求 JSON 模式的提示词里必须出现 "json"，缺了直接 400。
     客户端必须自己补，不能指望每个调用点都记得。
  2. `max_tokens` 由推理与答案共享。预算不足时返回**空内容 + HTTP 200**，
     这不是成功，必须被拦住。
  3. 推理默认开着，占输出 token 的九成以上；关掉它靠 `thinking` 参数。

    python backend/tests/test_llm.py
    pytest backend/tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weiran.config import LLMConfig  # noqa: E402
from weiran.llm import (  # noqa: E402
    Ledger,
    LLMClient,
    LLMError,
    Usage,
    _ensure_json_hint,
    _strip_code_fence,
)


# -- 假的传输层 ------------------------------------------------------------

def _response(
    content: str = "ok",
    *,
    finish: str = "stop",
    reasoning: str = "",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    reasoning_tokens: int = 0,
) -> dict:
    return {
        "choices": [
            {
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                },
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


class _FakeClient(LLMClient):
    """把 _post 换成假的，并记录送出的 payload。"""

    def __init__(self, reply: dict, *, ledger: Ledger | None = None,
                 send_thinking: bool = True) -> None:
        super().__init__(
            LLMConfig(
                api_key="test-key",
                base_url="https://example.invalid/v1",
                model="test-model",
                send_thinking_param=send_thinking,
            ),
            ledger,
        )
        self.reply = reply
        self.sent: list[dict] = []

    def _post(self, payload, *, tag, max_retries):  # type: ignore[override]
        self.sent.append(payload)
        return self.reply


# -- 1. json 提示词：服务端的硬性要求 --------------------------------------


def test_json_hint_added_when_missing():
    """提示词不含 json 时必须自动补上 —— 否则服务端直接 400。"""
    out = _ensure_json_hint([{"role": "user", "content": "把这句话拆开"}])
    assert "json" in out[-1]["content"].lower(), f"未补上 json 提示：{out}"


def test_json_hint_left_alone_when_present():
    """已经含 json 的就不要再动它 —— 无谓地改提示词会改变模型行为。"""
    msgs = [{"role": "user", "content": "请以 json 输出结果"}]
    out = _ensure_json_hint(msgs)
    assert out[-1]["content"] == "请以 json 输出结果"


def test_json_hint_does_not_mutate_caller_input():
    """**回归**：不得原地修改调用方传来的 messages。

    早期实现直接在原 list 上追加。调用方若复用同一个 messages 变量
    （很常见——先 chat 一次看效果，再 chat_json），
    第二遍就会莫名其妙多出一段提示词，且完全看不出是谁加的。
    """
    msgs = [{"role": "user", "content": "把这句话拆开"}]
    before = [dict(m) for m in msgs]
    _ensure_json_hint(msgs)
    assert msgs == before, f"调用方的 messages 被改动了：{msgs}"


def test_json_hint_goes_to_last_user_message():
    msgs = [
        {"role": "system", "content": "你是分析助手"},
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "答"},
        {"role": "user", "content": "第二问"},
    ]
    out = _ensure_json_hint(msgs)
    assert "json" not in out[1]["content"].lower(), "补到了错误的消息上"
    assert "json" in out[3]["content"].lower()


def test_chat_json_succeeds_without_json_in_prompt():
    """端到端：提示词里没有 json，chat_json 仍应拿到结果。"""
    c = _FakeClient(_response('{"情绪": "怀疑"}'))
    got = c.chat_json([{"role": "user", "content": "把这句话拆成两个字段"}], tag="t")
    assert got == {"情绪": "怀疑"}
    assert "json" in c.sent[0]["messages"][-1]["content"].lower(), "没补 json 就发出去了"


# -- 2. finish_reason=length：伪装成成功的失败 -----------------------------


def test_length_finish_raises_instead_of_returning_truncated_text():
    """**回归**：预算被推理吃光时必须报错，不能把空串当答案返回。

    这是本项目实测踩到的坑：max_tokens=200 时推理吃掉全部 200，
    content 返回空串，而 HTTP 状态码是 200。**只看状态码会以为成功了。**
    """
    c = _FakeClient(
        _response(content="", finish="length", completion_tokens=200, reasoning_tokens=200)
    )
    try:
        c.chat([{"role": "user", "content": "问题"}], max_tokens=200)
    except LLMError as exc:
        msg = str(exc)
        assert "max_tokens" in msg, f"错误信息未指出 max_tokens：{msg}"
        assert "截断" in msg, f"错误信息未说明是截断：{msg}"
        return
    raise AssertionError("finish_reason=length 却正常返回了，应当抛 LLMError")


def test_length_finish_error_names_reasoning_as_the_cause():
    """推理占大头时，错误信息必须点明是推理吃掉了预算。"""
    c = _FakeClient(
        _response(content="", finish="length", completion_tokens=200, reasoning_tokens=190)
    )
    try:
        c.chat([{"role": "user", "content": "问题"}], max_tokens=200)
    except LLMError as exc:
        assert "推理" in str(exc), f"未点明推理是原因：{exc}"
        return
    raise AssertionError("应当抛 LLMError")


# -- 3. 空内容：错误信息不许撒谎 -------------------------------------------


def test_empty_content_does_not_blame_content_policy():
    """**回归**：空内容的真实原因几乎总是推理耗尽，不是内容策略。

    早期版本的错误文案是「返回了空内容（可能被内容策略拦截）」。
    那是错的，而且会把排查引向完全错误的方向——去看提示词有没有违规，
    而真正的问题是 max_tokens 太小。
    """
    c = _FakeClient(
        _response(
            content="",
            finish="stop",
            reasoning="让我想想……",
            completion_tokens=50,
            reasoning_tokens=40,
        )
    )
    try:
        c.chat([{"role": "user", "content": "问题"}])
    except LLMError as exc:
        msg = str(exc)
        assert "可能被内容策略拦截" not in msg, f"错误文案仍在误导：{msg}"
        assert "推理" in msg, f"未提到推理：{msg}"
        assert "让我想想" in msg, f"未带出思维链片段，难以定位：{msg}"
        return
    raise AssertionError("空内容应当抛 LLMError")


def test_reasoning_is_kept_for_debugging():
    c = _FakeClient(_response(content="答案", reasoning="我的推理过程"))
    c.chat([{"role": "user", "content": "问题"}])
    assert c.last_reasoning == "我的推理过程", "思维链未被保留，答案不对时无从查起"


# -- 4. 推理开关 -----------------------------------------------------------


def test_thinking_disabled_by_default():
    """默认关闭推理。理由：实测它使输出 token 增至约 12 倍。

    这条断言是**成本纪律的看门人**。哪天有人把默认值改成 True，
    仿真成本会在无声中涨一个数量级，而所有测试仍然通过——
    除非有这一条拦着。
    """
    c = _FakeClient(_response())
    c.chat([{"role": "user", "content": "问题"}])
    assert c.sent[0].get("thinking") == {"type": "disabled"}, (
        f"默认未关闭推理：{c.sent[0].get('thinking')}"
    )


def test_thinking_can_be_enabled_per_call():
    c = _FakeClient(_response())
    c.chat([{"role": "user", "content": "问题"}], thinking=True)
    assert c.sent[0].get("thinking") == {"type": "enabled"}


def test_thinking_param_omitted_when_config_says_so():
    """不支持该参数的端点会 400，配置里要能整个关掉。"""
    c = _FakeClient(_response(), send_thinking=False)
    c.chat([{"role": "user", "content": "问题"}])
    assert "thinking" not in c.sent[0], "配置已要求不发 thinking，却仍发了"


def test_chat_json_forwards_thinking():
    c = _FakeClient(_response('{"a": 1}'))
    c.chat_json([{"role": "user", "content": "拆字段"}], thinking=True)
    assert c.sent[0].get("thinking") == {"type": "enabled"}


# -- 5. 用量记账 -----------------------------------------------------------


def test_usage_separates_reasoning_from_answer():
    u = Usage(prompt_tokens=100, completion_tokens=200, reasoning_tokens=180)
    assert u.answer_tokens == 20, f"答案 token 算错：{u.answer_tokens}"
    assert u.total == 300


def test_ledger_records_reasoning_tokens():
    """账本必须能回答「这 1 万 token 里有多少是思维链」。

    答不出来的话，成本优化就无从下手——因为真正的大头看不见。
    """
    ledger = Ledger()
    c = _FakeClient(
        _response(content="答案", completion_tokens=200, reasoning_tokens=180),
        ledger=ledger,
    )
    c.chat([{"role": "user", "content": "问题"}], tag="extract")
    assert ledger.by_tag["extract"].reasoning_tokens == 180
    assert ledger.total.reasoning_tokens == 180
    assert "推理" in ledger.summary(), f"摘要里没体现推理用量：{ledger.summary()}"


# -- 6. 解析兜底 -----------------------------------------------------------


def test_strip_code_fence_handles_json_block():
    assert _strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fence_leaves_plain_json_alone():
    assert _strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_chat_json_parses_fenced_response():
    """即使要求了 json_object，仍有服务商会裹一层 fence。"""
    c = _FakeClient(_response('```json\n{"a": 1}\n```'))
    assert c.chat_json([{"role": "user", "content": "拆字段"}]) == {"a": 1}


def test_chat_json_rejects_wrong_type():
    c = _FakeClient(_response("[1, 2, 3]"))
    try:
        c.chat_json([{"role": "user", "content": "拆字段"}])
    except LLMError as exc:
        assert "list" in str(exc), f"未说明实际类型：{exc}"
        return
    raise AssertionError("期望 dict 却拿到 list，应当抛 LLMError")


# -- 简易 runner（与其余测试文件保持一致）----------------------------------

def _run() -> int:
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
