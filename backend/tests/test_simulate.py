"""推演驱动的环境准备测试。

**全部离线。不联网、不调 LLM、不 import oasis。**

守的是一条**实测出来的**浪费（见 `weiran/simulate.py` 里 `TW_HIN_REPO`
那段注释与 进度.md）：

  OASIS 的 Twitter 推荐系统要从 `huggingface.co` 取 `Twitter/twhin-bert-base`，
  而 `huggingface_hub` **哪怕模型已在本地缓存**也会先发一个 HEAD 校验请求。
  该请求失败要重试 5 次（退避 1+2+4+8+8s，每次先等 10s 连接超时）——
  实测每轮白等约 127 秒（第 1 轮墙钟 134.1s，LLM 只占 7.2s）。

这里守三件事，每一件都是「静默失效」的形态：

  1. 缓存**完整**时确实切了离线（否则优化没生效，而没人会发现）。
  2. 缓存**不完整**时**不切**（只下到一半就强制离线，`from_pretrained`
     会直接抛错，比联网重试更难查）。
  3. 无论切没切，**都返回一句说明**（「静默生效」会让人把网络问题当成
     环境问题，「静默没生效」会让人以为优化生效了其实没有）。

    python backend/tests/test_simulate.py
    pytest backend/tests/
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weiran.simulate import (  # noqa: E402
    TW_HIN_REPO,
    CallStats,
    RoundLog,
    SimulationResult,
    _cache_delta,
    _round_cache_fields,
    build_run_meta,
    enable_hf_offline_if_cached,
    hf_cache_dir,
)

# 本函数会读写的环境变量。**逐个存还原**，不能整体 `os.environ = {}`
# ——那会把 PATH 一起洗掉，后续 import 直接崩。
_ENV_KEYS = ("HF_HUB_CACHE", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


@contextmanager
def _env(**values):
    """临时设置这几个变量，退出时还原成原样（含「原本不存在」）。"""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    try:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _make_snapshot(root: Path, *, with_weights: bool = True, with_tokenizer: bool = True) -> Path:
    """在 `root/models--Twitter--twhin-bert-base/snapshots/<rev>/` 造一份缓存。

    只造两个判据文件 —— 判据就是这两个（`tokenizer.json` + `model.safetensors`），
    多造会让测试与实现同时漂移到一个不存在的契约上。
    """
    repo = root / f"models--{TW_HIN_REPO.replace('/', '--')}"
    snap = repo / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    if with_tokenizer:
        (snap / "tokenizer.json").write_text("{}", encoding="utf-8")
    if with_weights:
        (snap / "model.safetensors").write_bytes(b"\x00")
    return root


def _hf_home_with(tmp: str, **kw) -> str:
    """按 `HF_HOME` 的语义造缓存 —— **缓存根是 `HF_HOME/hub`，不是 `HF_HOME`**。

    第一版把快照直接造在 `HF_HOME` 下，于是「缓存不完整就不切离线」那两条
    测的是「压根没找到缓存」这条**另一条分支** —— 测试是红的说明实现错了，
    测试是绿的却说明它没测到该测的东西。这正是本项目最防的那种假通过。
    """
    _make_snapshot(Path(tmp) / "hub", **kw)
    return tmp


# -- 缓存目录解析 ----------------------------------------------------------

def test_cache_dir_prefers_HF_HUB_CACHE():
    with _env(HF_HUB_CACHE="/tmp/a", HF_HOME="/tmp/b"):
        assert hf_cache_dir() == Path("/tmp/a")


def test_cache_dir_falls_back_to_HF_HOME_then_hub():
    with _env(HF_HOME="/tmp/b"):
        assert hf_cache_dir() == Path("/tmp/b") / "hub"


def test_cache_dir_defaults_to_user_cache_when_unset():
    with _env():
        assert hf_cache_dir() == Path.home() / ".cache" / "huggingface" / "hub"


# -- 完整缓存：必须切离线 --------------------------------------------------

def test_complete_cache_switches_offline():
    with tempfile.TemporaryDirectory() as tmp:
        with _env(HF_HOME=_hf_home_with(tmp)):
            msg = enable_hf_offline_if_cached()
            assert os.environ.get("HF_HUB_OFFLINE") == "1", msg
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1", msg
            assert "已切离线" in msg, msg


def test_complete_cache_via_HF_HUB_CACHE_also_works():
    """两个优先级都要能命中 —— 只测一个的话，另一个分支写错了也测不出来。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_snapshot(Path(tmp))
        with _env(HF_HUB_CACHE=str(root)):
            msg = enable_hf_offline_if_cached()
            assert os.environ.get("HF_HUB_OFFLINE") == "1", msg


# -- 不完整 / 不存在的缓存：必须**不**切 -----------------------------------

def test_incomplete_cache_missing_weights_does_not_switch():
    with tempfile.TemporaryDirectory() as tmp:
        with _env(HF_HOME=_hf_home_with(tmp, with_weights=False)):
            msg = enable_hf_offline_if_cached()
            assert os.environ.get("HF_HUB_OFFLINE") is None, (
                f"缓存缺权重却切了离线 —— from_pretrained 会直接抛错：{msg}")
            assert "保持联网" in msg, msg


def test_incomplete_cache_missing_tokenizer_does_not_switch():
    with tempfile.TemporaryDirectory() as tmp:
        with _env(HF_HOME=_hf_home_with(tmp, with_tokenizer=False)):
            msg = enable_hf_offline_if_cached()
            assert os.environ.get("HF_HUB_OFFLINE") is None, msg
            assert "保持联网" in msg, msg


def test_no_cache_at_all_does_not_switch():
    with tempfile.TemporaryDirectory() as tmp, _env(HF_HOME=tmp):
        msg = enable_hf_offline_if_cached()
        assert os.environ.get("HF_HUB_OFFLINE") is None, msg
        assert "保持联网" in msg, msg
        # 命令里没写清楚「找过哪儿」的话，用户没法自己排查
        assert "找过" in msg, msg


# -- 已经离线：补齐另一半并如实报 ------------------------------------------

def test_already_offline_still_sets_transformers_flag():
    with tempfile.TemporaryDirectory() as tmp:
        with _env(HF_HOME=_hf_home_with(tmp), HF_HUB_OFFLINE="1"):
            msg = enable_hf_offline_if_cached()
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1", (
                f"只设了 HF_HUB_OFFLINE 时没有补齐 TRANSFORMERS_OFFLINE：{msg}")
            assert "调用方已设离线" in msg, msg


# -- 幂等 / 不影响无关变量 -------------------------------------------------

def test_calling_twice_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp, _env(HF_HOME=_hf_home_with(tmp)):
        first = enable_hf_offline_if_cached()
        second = enable_hf_offline_if_cached()
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert "已切离线" in first
        # 第二次说的是「调用方已设离线」——这正是要的：它如实区分了
        # 「这次调用切的」与「之前就切好的」，而不是每次都报同一句话
        assert "调用方已设离线" in second


def test_does_not_touch_unrelated_env():
    with tempfile.TemporaryDirectory() as tmp, _env(HF_HOME=_hf_home_with(tmp)):
        os.environ["WEIRAN_SENTINEL"] = "keep-me"
        try:
            enable_hf_offline_if_cached()
            assert os.environ.get("WEIRAN_SENTINEL") == "keep-me"
        finally:
            os.environ.pop("WEIRAN_SENTINEL", None)


# -- 上下文截断监测 --------------------------------------------------------
#
# 守的是另一种「静默失效」，比上面那个 HF 的更难发现：camel 在 agent 记忆
# 超限时会**丢弃旧记录**，只在日志里留一条 warning。agent 静默失忆之后，
# 六维曲线照画、简报照出、测试照绿 —— 全场没有任何东西会红。
#
# 所以这里守三件事：
#
#   1. 真的截断了，**必须收到**（用真的 camel 截一次，不是自己 emit 一条假日志）。
#   2. 没截断时**不许报**（否则「有截断」这个信号没有意义）。
#   3. 格式读不懂时**不许静默当没发生**（否则监测退化成永远报平安）。

import logging  # noqa: E402

from weiran.simulate import (  # noqa: E402
    TRUNCATION_MARKER,
    TruncationWatch,
    install_truncation_watch,
    remove_truncation_watch,
)
from weiran.config import ensure_console_encoding  # noqa: E402


@contextmanager
def _watching():
    """装一个干净的监听，退出时摘掉。**不做成全局 fixture** ——
    泄漏的 handler 会让别的测试莫名其妙多收到事件。"""
    watch = install_truncation_watch()
    if watch.events:          # 上一个用例留下的，清掉
        watch.events.clear()
    try:
        yield watch
    finally:
        remove_truncation_watch(watch)


def _emit(msg: str, name: str = "camel.camel.memories.context_creators.score_based"):
    """模拟 camel 打一条警告。**只用来测「没截断/读不懂」这两条** ——
    测「真的截断了」用的是真 camel（见下一条）。"""
    logging.getLogger(name).warning(msg)


def test_watch_parses_a_real_truncation_message():
    with _watching() as w:
        _emit(f"{TRUNCATION_MARKER} before=9000, after=1200, limit=16384")
        assert w.count == 1, w.events
        assert w.events[0].before == 9000
        assert w.events[0].after == 1200
        assert w.events[0].limit == 16384
        assert w.events[0].dropped == 7800
        assert w.events[0].parsed


def test_watch_is_silent_when_nothing_is_truncated():
    """**第二条最容易被写漏。** 只测「能收到」的话，一个永远返回 True
    的实现也能过 —— 而那种实现让「有截断」这个信号变得毫无意义。"""
    with _watching() as w:
        _emit("Some other camel warning entirely")
        _emit("Context building finished normally")
        assert w.count == 0, f"没截断却报了：{w.events}"


def test_watch_counts_unparsed_instead_of_silently_ignoring():
    """camel 改了格式 → 解析失败。**不许当没发生** —— 那会让监测从
    「报了就是出事了」退化成「永远报平安」，恰是本项目最防的方向。"""
    with _watching() as w:
        _emit(f"{TRUNCATION_MARKER} before=?; after=?; limit=?")   # 读不懂
        assert w.count == 1, "解析失败却当成没发生 —— 监测已不可信"
        assert w.unparsed == 1
        assert not w.events[0].parsed
        assert w.limit is None


def test_watch_is_idempotent():
    """重复安装不叠加 handler —— 叠加会让同一次截断被数成 2 次、3 次。"""
    first = install_truncation_watch()
    try:
        second = install_truncation_watch()
        assert first is second
        first.events.clear()
        _emit(f"{TRUNCATION_MARKER} before=100, after=10, limit=50")
        assert first.count == 1, f"叠加了 handler，数成了 {first.count} 次"
    finally:
        remove_truncation_watch(first)


def test_watch_catches_both_possible_logger_names():
    """真实名字是 `camel.camel.memories…`（上游 `get_logger` 无条件加前缀，
    而模块 `__name__` 已含 `camel.`）。上游哪天修好这个，名字会变成
    `camel.memories…`。**两个都必须收到** —— 否则上游一改，监测就悄悄失效。"""
    with _watching() as w:
        _emit(f"{TRUNCATION_MARKER} before=100, after=10, limit=50",
              name="camel.camel.memories.context_creators.score_based")
        _emit(f"{TRUNCATION_MARKER} before=200, after=20, limit=50",
              name="camel.memories.context_creators.score_based")
        assert w.count == 2, f"有一种命名收不到：{w.events}"


def test_watch_reports_worst_by_dropped_tokens_not_by_count():
    """`worst` 要按「丢了多少」取，不是按「第几次」。截断是丢多少出事。"""
    with _watching() as w:
        _emit(f"{TRUNCATION_MARKER} before=1000, after=900, limit=16384")
        _emit(f"{TRUNCATION_MARKER} before=9000, after=100, limit=16384")
        assert w.worst is not None and w.worst.dropped == 8900


def test_the_watch_really_catches_a_real_camel_truncation():
    """**这条是整套监测的地基。** 前面几条只证明「我们的 handler 能解析
    我们造的字符串」—— 那测的是自己的解析器，不是 camel 真的会打这条日志。

    所以这里**让 camel 真的截断一次**：造一个 token 上限很小的
    `ScoreBasedContextCreator`，喂进远超上限的记录，断言我们收到了。

    变异检验（已做过）：把 `TRUNCATION_MARKER` 改掉，本条必红 ——
    说明它确实在测那条日志，不是在自我循环。

    不 import oasis（只有 camel）。camel 与 oasis 是两回事：本项目必须
    在切 HF 离线之前不碰的是 oasis/huggingface_hub，而 watch 本身
    **连 camel 都不 import**，只装 handler。
    """
    try:
        from camel.memories.context_creators.score_based import (
            ScoreBasedContextCreator,
        )
        from camel.memories.records import ContextRecord, MemoryRecord
        from camel.messages import BaseMessage
        from camel.types import RoleType, UnifiedModelType
        from camel.utils.token_counting import OpenAITokenCounter
    except ImportError:  # pragma: no cover - 没装 camel 时跳过而非误报
        print("      (跳过：未安装 camel-ai)")
        return

    def rec(text: str, score: float) -> "ContextRecord":
        msg = BaseMessage(role_name="user", role_type=RoleType.USER,
                          meta_dict={}, content=text)
        return ContextRecord(
            memory_record=MemoryRecord(message=msg, role_at_backend="user"),
            score=score,
        )

    creator = ScoreBasedContextCreator(
        OpenAITokenCounter(UnifiedModelType("gpt-4o-mini")), token_limit=300)

    with _watching() as w:
        records = [rec("系统提示", 1.0)]
        records += [rec(f"msg{i} " + "y" * 200, 0.1) for i in range(40)]
        creator.create_context(records)
        assert w.count >= 1, (
            "camel 真的截断了，我们却没收到 —— 上下限监控是空转的"
            "（可能 camel 换了 logger 名或改了警告格式）")
        assert w.unparsed == 0, f"收到了但解析不了，格式变了：{w.events}"
        assert w.worst is not None and w.worst.dropped > 0
        assert w.limit == 300, f"上限读错了：{w.limit}"


def test_the_watch_is_silent_on_a_real_camel_run_within_limit():
    """反向：**上限够大时 camel 不截断，我们也不许报。**
    只测「截断能收到」的话，一个永远报「截断了」的实现也能过。"""
    try:
        from camel.memories.context_creators.score_based import (
            ScoreBasedContextCreator,
        )
        from camel.memories.records import ContextRecord, MemoryRecord
        from camel.messages import BaseMessage
        from camel.types import RoleType, UnifiedModelType
        from camel.utils.token_counting import OpenAITokenCounter
    except ImportError:  # pragma: no cover
        print("      (跳过：未安装 camel-ai)")
        return

    msg = BaseMessage(role_name="user", role_type=RoleType.USER,
                      meta_dict={}, content="很短的一句话")
    record = ContextRecord(
        memory_record=MemoryRecord(message=msg, role_at_backend="user"),
        score=1.0,
    )
    creator = ScoreBasedContextCreator(
        OpenAITokenCounter(UnifiedModelType("gpt-4o-mini")), token_limit=100_000)

    with _watching() as w:
        creator.create_context([record])
        assert w.count == 0, f"远未超限却报了截断：{w.events}"


# ---------------------------------------------------------------------------
# 切片护栏。**这一组守的是一个第三方库的退化行为**，所以要三件东西齐全：
# 造得出那个退化（复现缺陷）、拦得住（我们的修法）、不误伤（自己超上限的还得切）。
# 只测后两件的话，一个「永远不切」的实现也能全绿。
# ---------------------------------------------------------------------------

def _camel_pieces():
    """camel 的那几个零件。**没装就返回 None**，与上面几条 camel 测试一致。"""
    try:
        from camel.agents.chat_agent import ChatAgent
        from camel.memories.agent_memories import ChatHistoryMemory
        from camel.memories.context_creators.score_based import (
            ScoreBasedContextCreator,
        )
        from camel.types import RoleType, UnifiedModelType
        from camel.utils.token_counting import OpenAITokenCounter
        from camel.memories.records import MemoryRecord
        from camel.messages import BaseMessage
    except ImportError:  # pragma: no cover - 没装 camel 时跳过而非误报
        return None
    return (ChatAgent, ChatHistoryMemory, ScoreBasedContextCreator,
            OpenAITokenCounter, UnifiedModelType, RoleType, BaseMessage,
            MemoryRecord)


def _bare_agent(token_limit: int):
    """造一个**只带记忆、不带模型**的 ChatAgent。不发 LLM、不联网。

    用 `__new__` 绕过 `__init__`：真造一个要模型后端，而这里要测的是
    `update_memory` 的写入决策，与模型无关。`update_memory` 用到的只有
    `memory`、`agent_id` 两个属性。
    """
    parts = _camel_pieces()
    if parts is None:
        return None
    (ChatAgent, ChatHistoryMemory, ScoreBasedContextCreator,
     OpenAITokenCounter, UnifiedModelType, _RoleType, _BaseMessage,
     _MemoryRecord) = parts

    creator = ScoreBasedContextCreator(
        OpenAITokenCounter(UnifiedModelType("gpt-4o-mini")),
        token_limit=token_limit,
    )
    agent = ChatAgent.__new__(ChatAgent)
    agent.agent_id = "probe"
    # `memory` 是个 setter，赋值时会调 `init_messages()`，而它要读
    # `_system_message`（`__new__` 绕过了 `__init__`，这个属性还不存在）。
    agent._system_message = None
    agent.memory = ChatHistoryMemory(context_creator=creator)
    return agent, creator


def _fill_until_truncating(agent, creator, filler_tokens: int = 400):
    """把记忆灌到**截断已经启动**为止 —— 也就是「预算归零」那个状态。

    这正是实测里发生退化的前提条件（3 agent 跑到 R9 起）。
    """
    from camel.messages import BaseMessage
    from camel.types import OpenAIBackendRole

    msg = BaseMessage.make_user_message(
        role_name="User", content="填 " + "x" * (filler_tokens * 3))
    per = creator.token_counter.count_tokens_from_messages(
        [msg.to_openai_message(OpenAIBackendRole.USER)])
    n = 0
    while True:
        agent.update_memory(msg, OpenAIBackendRole.USER)
        n += 1
        records = agent.memory.retrieve()
        raw = sum(creator.token_counter.count_tokens_from_messages(
            [r.memory_record.to_openai_message()]) for r in records)
        if raw > creator.token_limit and n >= 3:
            return n, per, raw


def test_camel_shreds_a_fitting_message_once_the_budget_is_gone():
    """**先把缺陷复现出来，再谈修法。** 不改任何东西。

    实测形态：记忆一装满、截断一启动，`remaining_budget` 就归零，
    camel 的 `update_memory` 于是把**一条远远塞得下的消息**切成
    「一块一个 token」（`base_chunk_size = max(1, 0)//10 = 0`
    → `chunk_body_limit = max(1, 0-前缀) = 1`），每块还各带一个
    `[chunk i/N of a long message]` 前缀。

    真实规模下这条让记忆里的记录数从 83 涨到 11798、总量 26 万 token。
    这条测试是那个现象的**缩小版**：上限 300、灌满、再写一条 60 token 的消息。
    """
    made = _bare_agent(token_limit=300)
    if made is None:
        print("      (跳过：未安装 camel-ai)")
        return
    agent, creator = made
    from camel.messages import BaseMessage
    from camel.types import OpenAIBackendRole

    _fill_until_truncating(agent, creator)

    small = BaseMessage.make_user_message(
        role_name="User", content="这是一条很短的消息。" * 5)
    own = creator.token_counter.count_tokens_from_messages(
        [small.to_openai_message(OpenAIBackendRole.USER)])
    assert own < creator.token_limit, "构造失败：这条消息本该远小于上限"

    before = len(agent.memory.retrieve())
    agent.update_memory(small, OpenAIBackendRole.USER)
    added = len(agent.memory.retrieve()) - before

    assert added > 1, (
        "camel 没有切 —— 那这条测试守的现象在本次 camel 版本上不存在，"
        "护栏也就没有存在理由，先查 camel 版本再决定要不要留它")
    tails = [r.memory_record.message.content for r in agent.memory.retrieve()[-3:]]
    assert any("[chunk" in (c or "") for c in tails), (
        f"加了 {added} 条，但没看到切片前缀：{tails}")


def test_guard_writes_a_fitting_message_whole_instead_of_shredding_it():
    """修法生效：同样那条 60 token 的消息，**只增加 1 条记录、内容是完整的**。"""
    made = _bare_agent(token_limit=300)
    if made is None:
        print("      (跳过：未安装 camel-ai)")
        return
    agent, creator = made
    from camel.messages import BaseMessage
    from camel.types import OpenAIBackendRole

    from weiran.simulate import install_chunking_guard, remove_chunking_guard

    install_chunking_guard()
    try:
        _fill_until_truncating(agent, creator)
        guard = install_chunking_guard()
        written_before, sliced_before = guard.written_whole, guard.still_sliced

        small = BaseMessage.make_user_message(
            role_name="User", content="这是一条很短的消息。" * 5)
        before = len(agent.memory.retrieve())
        agent.update_memory(small, OpenAIBackendRole.USER)
        added = len(agent.memory.retrieve()) - before
        last = agent.memory.retrieve()[-1].memory_record.message.content

        assert added == 1, f"护栏没拦住，仍然写进了 {added} 条"
        assert "[chunk" not in last, f"内容仍被切碎：{last[:80]}"
        assert guard.written_whole == written_before + 1
        assert guard.still_sliced == sliced_before
    finally:
        remove_chunking_guard()


def test_guard_still_slices_a_message_that_is_itself_over_the_limit():
    """**反向，这条是护栏的刹车。** 一条**自己就超上限**的消息必须照切 ——
    不切的话它谁也装不进去（截断要保最新，而它比整个窗口还大），
    camel 会直接抛 RuntimeError。

    只测「不切」的话，一个「永远不切」的实现也能全绿。
    """
    made = _bare_agent(token_limit=300)
    if made is None:
        print("      (跳过：未安装 camel-ai)")
        return
    agent, creator = made
    from camel.messages import BaseMessage
    from camel.types import OpenAIBackendRole

    from weiran.simulate import install_chunking_guard, remove_chunking_guard

    install_chunking_guard()
    try:
        guard = install_chunking_guard()
        written_before, sliced_before = guard.written_whole, guard.still_sliced

        huge = BaseMessage.make_user_message(
            role_name="User", content="超 " + "z" * 3000)
        own = creator.token_counter.count_tokens_from_messages(
            [huge.to_openai_message(OpenAIBackendRole.USER)])
        assert own > creator.token_limit, "构造失败：这条消息本该自己就超上限"

        before = len(agent.memory.retrieve())
        agent.update_memory(huge, OpenAIBackendRole.USER)
        added = len(agent.memory.retrieve()) - before

        assert added > 1, f"自己超上限的消息没被切，仍写成 {added} 条"
        assert guard.still_sliced == sliced_before + 1
        assert guard.written_whole == written_before, "不该由护栏写它"
    finally:
        remove_chunking_guard()


def test_guard_is_idempotent_and_removable():
    """装两次是同一个、摘掉之后**类上必须还原成原方法**。

    不还原的话，后面所有测试（乃至同一进程里的下一次运行）都在一个
    被改过的 camel 上跑，而没人知道。
    """
    parts = _camel_pieces()
    if parts is None:
        print("      (跳过：未安装 camel-ai)")
        return
    ChatAgent = parts[0]
    from weiran.simulate import install_chunking_guard, remove_chunking_guard

    original = ChatAgent.update_memory
    try:
        g1 = install_chunking_guard()
        g2 = install_chunking_guard()
        assert g1 is g2, "装两次拿到了两个护栏"
        assert ChatAgent.update_memory is not original, "装了却没换上去"
        assert g1.original is original
    finally:
        remove_chunking_guard()
    assert ChatAgent.update_memory is original, "摘掉之后没还原"
    remove_chunking_guard()  # 再摘一次不许炸


def test_guard_keeps_its_own_timestamps_out_of_the_tool_call_window():
    """护栏自己写记录时，时间戳必须**跨出 camel 给回执留的那扇门**。

    这扇门有多窄：`_record_tool_calling` 用 `base + 1e-6` 排「请求 → 回执」，
    而这条写入路径的时钟分辨率约 1ms，所以那是一微秒宽的一条缝。紧跟着的
    那条记录（`_record_final_output` 写的终稿）只要自己的读数落进缝里，
    排序键 `(timestamp, -score)` 就排出「请求 → 终稿 → 回执」，端点回 400
    `insufficient tool messages following tool_calls message`，而 OASIS 只记
    一行 error 继续跑 —— **该 agent 这一轮的动作整条消失**。

    实测（12 agent × 3 轮，见开发期探针 `epsilon_window.py`，不随仓库交付）：护栏开时
    27 个配对里 6 个落进缝里、正好 6 次 400，逐请求 1:1、零反例；护栏关时
    0/30，终稿与请求的最小间隔 9.956e-04（≈整整一拍，够不到缝）。护栏不是
    缺陷的来源 —— 它是那扇门唯一的把手：写入够快才够得到，而它就是要让
    写入变快。

    **这条测试不是恒真的**：它把时钟**钉死在缝里**（终稿那次读数取
    `base + 5e-7`），也就是把现实中约 22% 概率的那一拍变成必然 —— 旧实现
    下它必然失败（同一构造单独跑过，给的就是「请求 → 终稿 → 回执」）；
    修完终稿被推到至少 `base + 1e-3`，缝够不着。

    判据用**现象**（组装出来的上下文里请求与回执必须相邻），不用时间戳本身：
    时间戳是手段，端点拒的是那个形状。
    """
    import time as _time

    made = _bare_agent(token_limit=4000)
    if made is None:
        print("      (跳过：未安装 camel-ai)")
        return
    agent, creator = made

    from camel.messages import BaseMessage, FunctionCallingMessage
    from camel.types import OpenAIBackendRole, RoleType

    from weiran.simulate import install_chunking_guard, remove_chunking_guard

    base = 1_789_470_229.0
    clock = {"t": base}
    real_time_ns = _time.time_ns
    _time.time_ns = lambda: int(clock["t"] * 1e9)   # 假钟：把那一拍钉死
    install_chunking_guard()
    guard = install_chunking_guard()
    pushed_before = guard.timestamp_pushed
    try:
        agent.update_memory(
            BaseMessage.make_user_message(role_name="User", content="第 0 轮态势"),
            OpenAIBackendRole.USER)
        tid = "call_00_WINDOW"
        agent.update_memory(
            FunctionCallingMessage(
                role_name="assistant", role_type=RoleType.ASSISTANT,
                meta_dict=None, content="", func_name="like_post",
                args={"post_id": 3}, tool_call_id=tid),
            OpenAIBackendRole.ASSISTANT, timestamp=base)
        agent.update_memory(
            FunctionCallingMessage(
                role_name="assistant", role_type=RoleType.ASSISTANT,
                meta_dict=None, content="", func_name="like_post",
                result="{'success': True}", tool_call_id=tid),
            OpenAIBackendRole.FUNCTION, timestamp=base + 1e-6)
        # 最坏情形：终稿这次读数落在 (base, base + 1e-6) 里面
        clock["t"] = base + 5e-7
        agent.update_memory(
            BaseMessage(role_name="assistant", role_type=RoleType.ASSISTANT,
                        meta_dict=None, content=""),
            OpenAIBackendRole.ASSISTANT)
        # 计数器要在**摘护栏之前**读：`remove_chunking_guard` 会把它清零，
        # 在 finally 之后读就永远是 0（第一版就是这么写的，于是这条断言恒假）。
        pushed_after = guard.timestamp_pushed
    finally:
        _time.time_ns = real_time_ns
        remove_chunking_guard()

    messages, _ = creator.create_context(agent.memory.retrieve())
    shape = []
    for m in messages:
        if m.get("tool_calls"):
            shape.append("assistant(tool_calls)")
        elif m.get("role") == "tool":
            shape.append("tool")
        elif m.get("role") == "assistant" and not (m.get("content") or "").strip():
            shape.append("assistant(空)")
        else:
            shape.append(m.get("role"))

    at = shape.index("assistant(tool_calls)")
    assert shape[at + 1] == "tool", (
        f"请求与回执之间被插了东西：{shape} —— 端点会回 400，"
        "这一轮该 agent 的动作整条丢失")
    assert pushed_after > pushed_before, (
        "时间戳一次都没被推进 —— 这条臂是空的，它证明不了任何事")


def test_guard_is_off_by_default():
    """**默认关闭必须机检。** 它改的是第三方库的写入行为、会让曲线换一条口径；
    「默认是开的」这件事如果只写在注释里，某次重构把它翻过来谁也不会发现。
    """
    import inspect

    from weiran.simulate import _run

    default = inspect.signature(_run).parameters["chunking_guard"].default
    assert default is False, f"切片护栏默认值变成了 {default!r}"


# -- 前缀缓存的逐轮归属 ----------------------------------------------------
#
# 这一段守的是**「未记录」在产物里的形态**。本项目有两条调用路径，其中直调
# 那条（归类器）走 requests、不经过 camel 的模型对象，所以逐轮的
# calls / prompt_tokens 里一个数都不含它 —— 那笔钱只在服务商账单上。
# 补账的时候最容易犯的错，是把「没记」写成 0：0 是「报了，一次都没命中」的
# 形态，两者对账单的指向相反，且一旦落盘就再也分不开了。


def test_cache_delta_is_none_when_no_call_reported():
    """本轮没有调用报过这个字段 → None，**不是 (0, 0)**。"""
    got = _cache_delta(hit=0, miss=0, calls=0,
                       base_hit=0, base_miss=0, base_calls=0)
    assert got is None, f"没报却给出了差分：{got}"


def test_cache_delta_is_none_when_calls_happened_but_endpoint_stayed_silent():
    """本轮**调用了**、但端点没报 → 仍是 None。

    这一支是最容易写错的：命中数一点没变，看上去像「本轮全未命中」。
    但「端点没报」与「报了且全未命中」不是一件事，所以判据取的是
    「报过的调用次数有没有前进」，不是命中数有没有变。
    """
    got = _cache_delta(hit=900, miss=100, calls=1,
                       base_hit=900, base_miss=100, base_calls=1)
    assert got is None, f"端点没报却被记成了差分：{got}"


def test_cache_delta_zero_zero_is_real_when_endpoint_did_report():
    """本轮报了、且两端都是 0 → (0, 0)，**而且必须是 (0, 0)**。

    与上一条对读：两个都「什么都没变」，一个是不知道，一个是确知全未命中。
    只测其中一条的话，把判据写成常数也能过。
    """
    got = _cache_delta(hit=500, miss=300, calls=2,
                       base_hit=500, base_miss=300, base_calls=1)
    assert got == (0, 0), f"报了全未命中却没落数：{got}"


def test_cache_delta_subtracts_the_previous_round():
    """差分要对上：本轮增量 = 当前累计 − 轮初快照。"""
    got = _cache_delta(hit=1800, miss=600, calls=3,
                       base_hit=1400, base_miss=500, base_calls=2)
    assert got == (400, 100), f"差分算错：{got}"


def test_round_cache_fields_serialize_unrecorded_as_none():
    """未记录的一轮落盘必须是 None（JSON 里是 null），**不是 0**。"""
    d = _round_cache_fields(RoundLog(index=7))
    assert d == {"prompt_cache_hit_tokens": None,
                 "prompt_cache_miss_tokens": None}, f"未记录落成了 {d}"


def test_round_cache_fields_keep_a_real_zero():
    """记到的 0 要原样落 0 —— 它与「未记录」在 JSON 里必须是两个值。"""
    d = _round_cache_fields(RoundLog(index=7, prompt_cache_hit_tokens=0,
                                     prompt_cache_miss_tokens=1200))
    assert d["prompt_cache_hit_tokens"] == 0
    assert d["prompt_cache_miss_tokens"] == 1200
    assert d["prompt_cache_hit_tokens"] is not None, "报了的 0 被写成了 null"


def test_callstats_cache_line_is_absent_when_unrecorded():
    """整体统计那一行：端点没报就**不印任何数字**。"""
    s = CallStats(calls=10, prompt_tokens=5000)
    assert s.cache_line() is None, f"未报却印了缓存行：{s.cache_line()}"
    s.prompt_cache_miss_tokens = 5000      # 报了、全未命中
    assert s.cache_line() is None, "没标记 recorded 就不该印"
    s.cache_recorded = True
    line = s.cache_line()
    assert line is not None and "0.0%" in line, f"应为 0.0%：{line}"


def test_callstats_cache_line_reports_a_real_ratio():
    s = CallStats(calls=10, prompt_tokens=10000, cache_recorded=True,
                  prompt_cache_hit_tokens=7500, prompt_cache_miss_tokens=2500)
    line = s.cache_line()
    assert line is not None and "75.0%" in line, f"比例算错：{line}"
    assert "7500" in line and "2500" in line, f"未给原始数：{line}"


# -- 采样口径：产出必须记得住自己是怎么跑出来的 ----------------------------
#
# 这一组守的是一条**审计缺口**，不是一条功能：`--seed` / `--temperature` 长期
# 只被送进端点、一个字都没落盘，于是「多 seed 跑 N 次报均值与方差」这句写在
# 文档与产物里的话，建在一份**不可审计**的记录上 —— N 支臂跑进一个目录之后，
# 文件里没有任何东西能区分谁是谁。
#
# 三个状态**必须分得开**，下面三条一正两反地钉住它们：
#
#   键缺失          → 老产出   → 「未记录」（不是 0，也不是「未传种子」）
#   键在、值 null   → 没传     → 「未传种子」
#   键在、有值      → 传了     → 那个值
#
# 中间那一支最容易塌：`meta.get("seed")` 对「老产出」与「没传」返回的都是
# None，消费方若只写 `.get()` 就再也分不出来了 —— 而这两件事完全不同，
# 一件是**我们选择不传**，另一件是**不知道**。

def _meta_for(**result_kw) -> dict:
    """拼一份 meta。args 只放 `build_run_meta` 真会读的那几个字段。"""
    args = argparse.Namespace(platform="twitter", seed_text="",
                              no_world_state=False)
    return build_run_meta(SimulationResult(**result_kw), args, agents=27)


def test_meta_records_the_sampling_caliber():
    """传了种子就必须真的落在产出里 —— 包括**进程 RNG 播没播**这个独立事实。

    `random_seeded` 与 `seed` 不是一回事：前者说的是本进程的全局 RNG
    有没有被播种（覆盖 OASIS 的抽签），后者说的是请求里带了什么。
    两者都由同一次 `seed_process` 调用产生，所以不会互相矛盾。
    """
    m = _meta_for(seed=7, temperature=0.6, random_seeded=True)
    assert m["seed"] == 7, f"种子没落盘：{m.get('seed')!r}"
    assert m["temperature"] == 0.6, f"温度没落盘：{m.get('temperature')!r}"
    assert m["random_seeded"] is True, "进程 RNG 播过种却没记"
    assert m["sampling_note"], "没带上限制条件 —— 产出要自己说清它保证不了什么"


def test_meta_distinguishes_not_passed_from_not_recorded():
    """**这一条是整组的核心。**

    「没传 `--seed`」记成 `null` 并且**键在**；老产出是**键不在**。
    消费方靠 `"seed" in meta` 分这两支。这条用例把它钉死：
    新产出即使没传种子，也必须带着这一组键。
    """
    m = _meta_for()          # 全新跑的一次、没传 seed
    for key in ("seed", "temperature", "random_seeded", "sampling_note"):
        assert key in m, (
            f"新产出少了 {key!r} 这个键 —— 它会把「本次没传」"
            "伪装成「老产出、未记录」")
    assert m["seed"] is None, "没传种子时该记 null（= 未传），不是 0"
    assert m["temperature"] is None
    assert m["random_seeded"] is False, "没播种却记成播过"


def test_meta_never_claims_a_seed_that_was_not_used():
    """反向：没传种子时，产出里不许出现任何看起来像种子的数。

    防的是「补个默认值 0 让它别是 None」这种改法 —— 那会把「不知道」
    写成一个具体的数，而 0 是一个**看起来合法**的种子。
    """
    m = _meta_for()
    assert m["seed"] != 0, "没传种子却记成了 0 —— 0 是一个合法的种子，会被当真"
    assert m["temperature"] != 0, (
        "没传温度却记成了 0 —— 而 0 与「不传、用端点默认值」是两回事")


# -- 简易 runner（与其余测试文件保持一致）----------------------------------

def _run() -> int:
    # 兜底 runner 也要防这一条：**被测代码会往控制台印符号**（repro_check 印
    # ✅/❌、viewer 的失败路径印 ❌）。Windows 中文控制台是 GBK，装不下这些
    # 字符时 print 会抛 UnicodeEncodeError —— 于是「有坏消息要报」的那次运行
    # 反而崩在报消息的路上，看起来像测试坏了。这正是 config.py 里那个
    # `ensure_console_encoding()` 存在的理由，这里用上它。
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
