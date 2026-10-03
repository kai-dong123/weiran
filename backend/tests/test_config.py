"""运行配置的测试。重点是日志编码那件事。

**全部离线。不联网、不调 LLM、不 import oasis / camel。**

守的是一个**实测撞到过**的静默失效，而且是两层：

  oasis 自建了一个 `logging.FileHandler` 却**没传 `encoding`**
  （`oasis/social_agent/agent.py:44`；同库的 `oasis/environment/env.py:40`
  反而传了 `encoding="utf-8"` —— 所以这是上游疏漏，不是设计）。
  Windows 下 `FileHandler` 默认取 locale 编码 = GBK，于是：

    - 第一层：日志里冒出一段 `--- Logging error ---` 堆栈。演示时看着像崩了。
    - 第二层（**更要命**）：`logging` 吞掉异常，**那一行记录直接没了**。

  触发条件只是「agent 在帖文里发了一个 emoji」。实测 27 agent × 15 轮，
  第 0 轮就撞上，不是偶发。

这里守四件事：

  1. 缺陷是真的 —— 不修的话那一行**确实丢**（这条是下面几条的地基：
     没有它，「修好了」无从验证）。
  2. 修过之后那一行**在**。
  3. 只动 errors 策略、**不动 encoding**（中文不能因为修这个而变成乱码）。
  4. 流不是 `TextIOWrapper` 时不炸（`StringIO`、socket、已关闭的流）。

    python backend/tests/test_config.py
    pytest backend/tests/
"""

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weiran.config import (  # noqa: E402
    LOOPBACK_HOSTS,
    ConfigError,
    ViewerConfig,
    ensure_console_encoding,
    ensure_log_handler_encoding,
    load_config,
)

# agent 帖文里实测出现过的那个字符（🤍 白心）
EMOJI = "\U0001f90d"


def _gbk_file_handler(tmp: str, name: str) -> logging.FileHandler:
    """复刻 oasis 的写法：**不传 encoding**。"""
    return logging.FileHandler(os.path.join(tmp, name))


def _read(path) -> str:
    """按字节读，**用 GBK 解码**。

    不能按 UTF-8 解：这个修法刻意**不动 encoding**，所以中文写进去是 GBK 字节，
    按 UTF-8 解会得到一堆问号，于是「那一行在不在」这个断言会因为**测试自己
    解错了码**而失败 —— 第一版就是这么错的，看起来像实现坏了。
    """
    return Path(path).read_bytes().decode("gbk", errors="replace")


class _SilenceLoggingErrors:
    """临时把 `logging.raiseExceptions` 关掉。

    不修的情况下那条 emoji 日志会在 stderr 上打一段堆栈，把测试输出淹掉。
    关掉它**不影响被测行为** —— 无论打不打堆栈，那一行都是丢的，
    而这正是我们要断言的事。
    """

    def __enter__(self):
        self._saved = logging.raiseExceptions
        logging.raiseExceptions = False
        return self

    def __exit__(self, *exc):
        logging.raiseExceptions = self._saved
        return False


# -- 地基：不修的话那一行确实丢 --------------------------------------------

def test_unfixed_handler_really_loses_the_line():
    """**没有这一条，后面几条全是空转。**

    断言的是「那一行**不在**文件里」——本项目最看重的那类断言。
    若哪天 Python 或 logging 变了、默认不再丢行，这条会红，
    那正是我们想知道的（说明可以撤掉这个修法了）。
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "oasis-like.log")
        handler = _gbk_file_handler(tmp, "oasis-like.log")
        lg = logging.getLogger("test.oasis.unfixed")
        lg.propagate = False
        lg.addHandler(handler)
        try:
            if getattr(handler.stream, "encoding", "").lower() not in ("gbk", "cp936"):
                print(f"      (跳过：本机 locale 编码是 {handler.stream.encoding}，"
                      "这条缺陷只在 GBK 下成立)")
                return
            with _SilenceLoggingErrors():
                lg.warning(f"带 emoji {EMOJI} 的一句")
            assert "带 emoji" not in _read(path), (
                "预期这一行会因 GBK 编码失败而丢失，它却在文件里 —— "
                "说明这个缺陷在本机不复现，本文件的其余测试也失去了前提")
        finally:
            lg.removeHandler(handler)
            handler.close()


# -- 修过之后那一行必须在 --------------------------------------------------

def test_fix_makes_the_line_survive():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "fixed.log")
        handler = _gbk_file_handler(tmp, "fixed.log")
        lg = logging.getLogger("test.oasis.fixed")
        lg.propagate = False
        lg.addHandler(handler)
        try:
            fixed = ensure_log_handler_encoding()
            assert fixed >= 1, "没有修正任何 handler —— 修法没生效"

            with _SilenceLoggingErrors():
                lg.warning(f"带 emoji {EMOJI} 的一句")
            content = _read(path)
            assert "带 emoji" in content, (
                f"修正之后那一行**仍然**丢了：{content!r}")
            assert EMOJI not in content, "emoji 应当退化成替换字符而不是原样写入"
        finally:
            lg.removeHandler(handler)
            handler.close()


def test_fix_keeps_encoding_and_only_changes_errors():
    """**只动 errors、不动 encoding** —— 这是 `ensure_console_encoding()` 定下的策略，
    这里必须一致：中文按原编码照常写，不能因为修 emoji 就让中文变乱码。"""
    with tempfile.TemporaryDirectory() as tmp:
        handler = _gbk_file_handler(tmp, "policy.log")
        lg = logging.getLogger("test.oasis.policy")
        lg.propagate = False
        lg.addHandler(handler)
        try:
            before = handler.stream.encoding
            ensure_log_handler_encoding()
            assert handler.stream.encoding == before, (
                f"encoding 被改掉了：{before} → {handler.stream.encoding}")
            assert handler.stream.errors == "replace"
        finally:
            lg.removeHandler(handler)
            handler.close()


def test_fix_writes_chinese_correctly_not_as_mojibake():
    """中文必须还能读出来。只断言「没崩」是不够的 —— 一个把 encoding
    改成 ascii 的实现也能不崩，但会把中文全变成问号。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cn.log")
        handler = _gbk_file_handler(tmp, "cn.log")
        lg = logging.getLogger("test.oasis.cn")
        lg.propagate = False
        lg.addHandler(handler)
        try:
            ensure_log_handler_encoding()
            with _SilenceLoggingErrors():
                lg.warning("校园舆情态势：信任偏高、下滑中")
            raw = Path(path).read_bytes()
            assert "校园舆情态势".encode("gbk") in raw, (
                f"中文没有按 GBK 写入（可能被改成了别的编码）：{raw!r}")
        finally:
            lg.removeHandler(handler)
            handler.close()


# -- 幂等 / 不炸 ------------------------------------------------------------

def test_second_call_reports_nothing_to_fix():
    """已经是 replace 的流不该被重复计数 —— 否则那个返回值没法当信号用。"""
    with tempfile.TemporaryDirectory() as tmp:
        handler = _gbk_file_handler(tmp, "idem.log")
        lg = logging.getLogger("test.oasis.idem")
        lg.propagate = False
        lg.addHandler(handler)
        try:
            first = ensure_log_handler_encoding()
            second = ensure_log_handler_encoding()
            assert first >= 1 and second == 0, (
                f"第二次调用仍报告修正了 {second} 个 —— 返回值不可当信号")
        finally:
            lg.removeHandler(handler)
            handler.close()


def test_handler_without_text_stream_does_not_crash():
    """`StringIO` 没有 `reconfigure`。**不许因为改不了就抛** ——
    目标是「能改的改掉」，不是「保证全都改掉」。"""
    lg = logging.getLogger("test.oasis.stringio")
    lg.propagate = False
    handler = logging.StreamHandler(io.StringIO())
    lg.addHandler(handler)
    try:
        ensure_log_handler_encoding()   # 不抛即通过
    finally:
        lg.removeHandler(handler)
        handler.close()


def test_closed_handler_does_not_crash():
    """已关闭的流 reconfigure 会抛 ValueError。同理不许炸。"""
    with tempfile.TemporaryDirectory() as tmp:
        handler = _gbk_file_handler(tmp, "closed.log")
        lg = logging.getLogger("test.oasis.closed")
        lg.propagate = False
        lg.addHandler(handler)
        try:
            handler.close()
            ensure_log_handler_encoding()   # 不抛即通过
        finally:
            lg.removeHandler(handler)


def test_console_encoding_is_safe_to_call():
    """`ensure_console_encoding()` 在流不支持 reconfigure 时（如 pytest 的
    捕获对象）也必须安静通过，而不是把整个程序带崩。"""
    ensure_console_encoding()   # 不抛即通过
    ensure_console_encoding()   # 且可重复调用


# -- 展示层监听地址：两个键**必须真的有消费方** ----------------------------
#
# 这两条守的还是同一类静默失效，换个位置再守一遍：`OASIS_*` 曾经是「写在
# .env 里、没有任何消费方、设了不生效也不报错」，`FLASK_HOST` / `FLASK_PORT`
# 有一段时间**一模一样**。第 2 条就是那条缺陷的回归测试 —— 谁把接线删了，
# 它会红，而不是安静地回到「设了不生效」。
#
# 第 1 条守的是接线接错方向的后果：展示层没有鉴权，绑到 0.0.0.0 就把
# 「本地查看工具」变成对局域网开放的服务。

class _env:
    """临时设几个环境变量，退出时恢复原状。

    `_load_dotenv` **不覆盖已存在的环境变量**，所以这里设的值一定赢过仓库里
    那份 `.env` —— 测试不依赖开发机上的 `.env` 长什么样。
    """

    def __init__(self, **kv):
        self._kv = kv

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in self._kv}
        for k, v in self._kv.items():
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def test_flask_env_keys_are_actually_consumed():
    """**这一条就是「死旋钮」的回归测试。**

    守的是：`.env` 里写下 `FLASK_PORT`，`load_config` 读出来，展示层拿到手。
    少任何一环，这个键就退回到「设了不生效也不报错」—— 那正是本项目专门
    猎杀、且已经在 `OASIS_*` 上踩过一次的失效。
    """
    with _env(FLASK_HOST="localhost", FLASK_PORT="8123"):
        cfg = load_config(require_llm=False)
    assert cfg.viewer.port == 8123, (
        f"FLASK_PORT 没有被消费：读出来是 {cfg.viewer.port} —— "
        "这个键又变回死的了")
    assert cfg.viewer.host == "localhost", (
        f"FLASK_HOST 没有被消费：读出来是 {cfg.viewer.host!r}")


def test_flask_port_rejects_non_integer():
    """坏的端口要立刻停下并说清楚，而不是回落到默认值假装没事。"""
    with _env(FLASK_PORT="八千"):
        try:
            load_config(require_llm=False)
        except ConfigError as exc:
            assert "FLASK_PORT" in str(exc), f"报错里没说清是哪个键：{exc}"
            return
    raise AssertionError("FLASK_PORT 是「八千」却通过了 —— 这意味着它被静默忽略了")


def test_non_loopback_host_is_refused_loudly():
    """`0.0.0.0` 必须**抛**，不许静默忽略、也不许静默改回默认值。

    静默忽略是这个缺陷的另一半：用户以为自己绑上去了，实际没有 ——
    或者反过来，以为自己关掉了，实际开着。两种都不可接受。
    """
    for bad in ("0.0.0.0", "192.168.1.7", "::"):
        try:
            ViewerConfig(host=bad)
        except ConfigError as exc:
            assert "回环" in str(exc), f"报错没说清原因：{exc}"
        else:
            raise AssertionError(f"{bad!r} 被接受了 —— 展示层会暴露到本机之外")


def test_loopback_check_is_an_exact_list_not_a_prefix():
    """**白名单是精确匹配，不是 `startswith("127.")`。**

    后者看着更宽容，其实是开了个洞：`127.0.0.1.evil.com` 也以 `127.` 开头，
    而这个域名解析到哪就不由我们决定了。这里把它连同 `localhost` 的变体
    一起钉住。
    """
    for good in LOOPBACK_HOSTS:
        ViewerConfig(host=good)                  # 不抛即通过
    ViewerConfig(host="  LOCALHOST  ")           # 大小写与空白先规整，再比对

    # `127.0.0.2` 确实落在回环段，但不在白名单里 —— 刻意的取舍：
    # 精确匹配比「看起来更全的规则」更难写错。`127.0.0.1.evil.com` 才是
    # 前缀写法真正的洞：它也以 `127.` 开头，而解析到哪不由我们决定。
    for bad in ("127.0.0.1.evil.com", "localhost.evil.com", "127.0.0.2"):
        try:
            ViewerConfig(host=bad)
        except ConfigError:
            continue
        raise AssertionError(f"{bad!r} 不应被接受")


# -- 缺件报错：**一次报全** ------------------------------------------------
#
# `load_config` 的文档承诺是「消息会一次性列出全部缺失项，避免『修一个报一个』
# 的来回」。`LLM_MODEL_NAME` / `LLM_API_KEY` / `EMBEDDING_*` 确实都汇进了同一个
# `missing` 列表，唯独 `LLM_BASE_URL` 原先在聚合检查**之后**单独抛 —— 于是它总
# 比另外几项晚一轮露面：按报错把模型名和密钥补齐、再跑一次，才轮得到它。
# 下面这条就是把「晚一轮」钉死的回归测试。

def test_all_missing_items_are_reported_in_one_go():
    """缺的配置必须在**同一次**报错里列全，包括基址。"""
    with _env(
        LLM_MODEL_NAME="",
        LLM_API_KEY="",
        LLM_BASE_URL="",
        EMBEDDING_MODEL="",
        EMBEDDING_API_KEY="",
    ):
        try:
            load_config(require_llm=True, require_embedding=True)
        except ConfigError as exc:
            msg = str(exc)
        else:
            raise AssertionError("五项全空却通过了 —— 这等于根本没查")

    for key in (
        "LLM_MODEL_NAME",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "EMBEDDING_MODEL",
        "EMBEDDING_API_KEY",
    ):
        assert key in msg, (
            f"{key} 没出现在这一次报错里 —— 它会被推到下一轮才报，"
            f"也就是「修一个报一个」：\n{msg}")


# -- 简易 runner（与其余测试文件保持一致）----------------------------------

def _run() -> int:
    # 与其余十一套一致：兜底 runner 先设好控制台编码策略。这一套本身就在测
    # 那个函数，但**被测它与调用它是两件事** —— 少了这一行，
    # `test_handbook.py` 里那条「每一套的 runner 都要调它」会红。
    ensure_console_encoding()
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
