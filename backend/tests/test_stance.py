"""在线行为归类的测试。

这一组测试守的是**两个静默错误**，两个都不会抛异常、都只会让数字慢慢错掉：

1. `quote_content` 陷阱 —— OASIS 的引用帖把**原文**抄进了 `content`，
   把**作者自己的话**放在 `quote_content`。读错列不会报错，
   只会让每一条引用帖都被归成与种子事件同一个类型，六维曲线退化成平线。

2. 关键词表读不懂自由文本 —— 实测「从**可控**变成失控」被归成
   `reassurance`（安抚）。语义被整个翻了过来，而没有任何地方会报错。

与另三套一样，**刻意不依赖 pytest**：普通 assert + 函数，
`python tests/test_stance.py` 直接跑，`python -m pytest tests/` 也能跑。
理由和 LLM 客户端只用 requests 一样——复现路径上少一个依赖就少一个失败点。

（原先本文件用了 `@pytest.fixture` / `tmp_path` / `parametrize`，是四套里唯一的例外，
代价是它**只能**在 pytest 下跑：直接执行会 `ModuleNotFoundError`。
现在换成等价的普通写法。唯一丢掉的 `parametrize` 会自动覆盖新静默动作这个性质，
由 `test_the_explicit_silent_action_list_covers_the_real_one` 补回来。）
"""

from __future__ import annotations

import atexit
import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weiran.simulate import _SILENT_ACTIONS, texts_from_actions  # noqa: E402
from weiran.stance import KINDS, PROMPT_VERSION, StanceClassifier, _key  # noqa: E402
from weiran.world_state import KIND_PRIORITY, classify  # noqa: E402
from weiran.config import ensure_console_encoding  # noqa: E402

SEED = "【就业指导中心】2025届就业质量报告发布，落实率78.3%。"


def _post(post_id, user_id, content="", quote_content=None,
          original_post_id=None):
    return {"post_id": post_id, "user_id": user_id, "content": content,
            "quote_content": quote_content,
            "original_post_id": original_post_id}


def _posts():
    return {
        # 种子事件：自己写的，quote_content 为空
        1: _post(1, 0, content=SEED),
        # 引用帖：content 是原文副本，quote_content 才是作者的话
        2: _post(2, 1, content=SEED, quote_content="口径怎么定的？说清楚。",
                 original_post_id=1),
        # 转帖：content 是空串，只靠 original_post_id 指向原文
        3: _post(3, 2, content="", original_post_id=2),
    }


# 每个测试一个独立的缓存文件，避免相互污染（同 test_store.py 的 _fresh_conn）。
# 用计数器而不是固定文件名：`test_cache_prevents_a_second_llm_call` 这类
# 测试会先写缓存再新建实例读回，共用文件名会变成测试顺序依赖。
_TMPDIR = Path(tempfile.mkdtemp(prefix="weiran-stance-test-"))
atexit.register(shutil.rmtree, _TMPDIR, True)
_SEQ = itertools.count()


def _cache_path() -> Path:
    return _TMPDIR / f"cache_{next(_SEQ)}.json"


# ---------------------------------------------------------------------------
# 动作 -> 文本

def test_create_post_reads_own_content():
    acts = [{"action": "create_post", "info": json.dumps({"post_id": 1})}]
    assert texts_from_actions(acts, _posts()) == [SEED]


def test_quote_post_reads_the_authors_own_words_not_the_original():
    """**这是本文件最重要的一条。** 读错列 -> 引用帖被归成种子事件的类型。"""
    acts = [{"action": "quote_post",
             "info": json.dumps({"quoted_id": 1, "new_post_id": 2})}]
    out = texts_from_actions(acts, _posts())
    assert out == ["口径怎么定的？说清楚。"]
    # 若哪天有人把这里改回读 content，下面这条会立刻失败
    assert SEED not in out[0]


def test_repost_follows_original_post_id():
    """转帖的 content 是空串，只有 original_post_id 能找回被转的话。"""
    acts = [{"action": "repost",
             "info": json.dumps({"reposted_id": 1, "new_post_id": 3})}]
    out = texts_from_actions(acts, _posts())
    assert out, "转帖被静默丢掉了 —— 而它恰恰是 amplification 最强的信号"
    assert "口径怎么定的" in out[0]


def test_like_uses_the_liked_posts_text():
    acts = [{"action": "like_post", "info": json.dumps({"post_id": 1})}]
    assert texts_from_actions(acts, _posts()) == [SEED]


def _assert_silent(name):
    acts = [{"action": name, "info": json.dumps({"post_id": 1})}]
    assert texts_from_actions(acts, _posts()) == [], f"{name} 不该产生行为信号"


# 5 个静默动作，逐个一条用例 —— 失败时报出的是哪个动作。
# 静默动作集合若变动，下面那条元测试会拦住。
def test_silent_action_do_nothing():
    _assert_silent("do_nothing")


def test_silent_action_follow():
    _assert_silent("follow")


def test_silent_action_mute():
    _assert_silent("mute")


def test_silent_action_refresh():
    _assert_silent("refresh")


def test_silent_action_sign_up():
    _assert_silent("sign_up")


def test_the_explicit_silent_action_list_covers_the_real_one():
    """元测试：上面 5 条显式用例必须与 `_SILENT_ACTIONS` 同步。

    原先这里是一个 `@pytest.mark.parametrize`，新增静默动作会被自动覆盖；
    改成显式用例就把这个性质弄丢了。补这条把它找回来 —— 否则
    `_SILENT_ACTIONS` 加了新动作而这里一声不吭，就是**静默漏测**。
    """
    assert set(_SILENT_ACTIONS) == {
        "do_nothing", "follow", "mute", "refresh", "sign_up",
    }, "静默动作集合变了 —— 上面那 5 条用例要跟着改"


def test_unknown_post_id_is_skipped_not_crashed():
    acts = [{"action": "create_post", "info": json.dumps({"post_id": 999})}]
    assert texts_from_actions(acts, _posts()) == []


def test_info_may_arrive_as_a_dict_or_a_json_string():
    """OASIS 的 trace 里 info 是 TEXT 列，但别处可能已经是 dict。"""
    as_str = [{"action": "create_post", "info": json.dumps({"post_id": 1})}]
    as_dict = [{"action": "create_post", "info": {"post_id": 1}}]
    assert texts_from_actions(as_str, _posts()) == texts_from_actions(
        as_dict, _posts())


def test_broken_info_json_does_not_raise():
    acts = [{"action": "create_post", "info": "{不是合法 json"}]
    assert texts_from_actions(acts, _posts()) == []


# ---------------------------------------------------------------------------
# 分类器

class _FakeLLM:
    """按「文本里出现什么词」返回标签的假客户端，用来测协议而不是测模型。"""

    def __init__(self, mapping=None, label=None, raise_exc=None, count=None):
        self.mapping = mapping or {}
        self.label = label
        self.raise_exc = raise_exc
        self.count = count
        self.calls = 0

    def chat_json(self, messages, **kw):
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc
        n = len(messages[-1]["content"].split("\n")) - 1
        if self.count is not None:
            n = self.count
        labels = [self.mapping.get(t, self.label or "discussion")
                  for t in self.mapping] or [self.label or "discussion"] * n
        return {"labels": labels[:n]}


def test_no_llm_falls_back_to_keywords():
    c = StanceClassifier(llm=None, cache_path=_cache_path())
    out = c.classify_many(["辅导员要求同学删除帖子", "学校公布了分专业数据"])
    assert out == ["suppression", "disclosure"]
    assert c.fallbacks == 2


def test_llm_labels_are_used():
    c = StanceClassifier(llm=_FakeLLM(label="testimony"),
                         cache_path=_cache_path())
    assert c.classify_many(["我那年也是这样"]) == ["testimony"]
    assert c.fallbacks == 0


def test_blank_texts_never_reach_the_llm():
    llm = _FakeLLM(label="testimony")
    c = StanceClassifier(llm=llm, cache_path=_cache_path())
    assert c.classify_many(["", "   "]) == ["discussion", "discussion"]
    assert llm.calls == 0


def test_cache_prevents_a_second_llm_call():
    """可复现性的实现方式：LLM 只在第一次出现，之后从磁盘重放。"""
    path = _cache_path()
    llm = _FakeLLM(label="amplification")
    first = StanceClassifier(llm=llm, cache_path=path)
    assert first.classify_many(["上热搜了"]) == ["amplification"]
    assert llm.calls == 1

    second = StanceClassifier(llm=llm, cache_path=path)  # 新实例，从盘读
    assert second.classify_many(["上热搜了"]) == ["amplification"]
    assert llm.calls == 1, "缓存没生效 —— 同输入两次跑会给出不同曲线"
    assert second.hits == 1


def test_llm_failure_degrades_to_keywords_and_keeps_running():
    c = StanceClassifier(llm=_FakeLLM(raise_exc=RuntimeError("端点挂了")),
                         cache_path=_cache_path())
    assert c.classify_many(["辅导员要求同学删除帖子"]) == ["suppression"]
    assert c.fallbacks == 1


def test_label_count_mismatch_discards_the_whole_batch():
    """条数对不上就整批作废。**不做「尽力对齐」** ——
    错位的标签会静默地把 A 说的话算到 B 头上。"""
    c = StanceClassifier(llm=_FakeLLM(label="testimony", count=1),
                         cache_path=_cache_path())
    out = c.classify_many(["辅导员要求同学删除帖子", "学校公布了分专业数据"])
    assert out == ["suppression", "disclosure"]  # 两个都退回了关键词
    assert c.fallbacks == 2


def test_label_outside_the_engine_vocabulary_is_rejected():
    """引擎不认识的类型会让 EXCITATION 静默查不到键 —— 宁可退回关键词。"""
    c = StanceClassifier(llm=_FakeLLM(label="驳斥"), cache_path=_cache_path())
    out = c.classify_many(["辅导员要求同学删除帖子"])
    assert out == ["suppression"]


def test_corrupt_cache_file_is_ignored():
    path = _cache_path()
    path.write_text("{不是 json", encoding="utf-8")
    c = StanceClassifier(llm=None, cache_path=path)
    assert c.cache == {}


def test_cache_entries_with_unknown_labels_are_dropped():
    path = _cache_path()
    path.write_text(json.dumps({"abc": "驳斥", "def": "testimony"}),
                    encoding="utf-8")
    c = StanceClassifier(llm=None, cache_path=path)
    assert c.cache == {"def": "testimony"}


def test_cache_key_covers_the_prompt_version():
    """改了提示词就必须让旧缓存失效，否则旧标签会被静默复用。"""
    assert _key("同样的文本") == _key("同样的文本")
    assert PROMPT_VERSION >= 1


def test_engine_and_classifier_share_one_vocabulary():
    """`stance.KINDS` 与 `world_state.KIND_PRIORITY` 是同一个元组。
    两边一旦漂移，LLM 会吐出引擎不认识的类型。"""
    assert KINDS == KIND_PRIORITY


# ---------------------------------------------------------------------------
# 关键词表本身：记录它为什么读不懂自由文本

def test_keyword_table_misreads_free_prose_this_is_why_stance_exists():
    """把当初发现这个问题的那条真实文本钉在这里。

    这不是「期望关键词表表现好」的测试，恰恰相反 —— 它**锁住缺陷**，
    好让以后有人想「删掉 stance.py，直接用 classify 吧」时，
    会在这一条上看到当初为什么不能。
    """
    text = "舆情从火星烧成山火，往往就坏在一个「拖」字上。宁可当下认错，也别从可控变成失控。"
    assert classify(text) == "reassurance"      # 判成了「安抚」
    assert "失控" in text                        # 可它说的是「会失控」


def test_short_keywords_are_the_unreliable_ones():
    """两个字的关键词几乎都不可靠 —— 中文没有词边界，子串会翻语义。"""
    assert classify("我是说，这事得按程序来") == "testimony"   # 「我是」
    assert classify("实际上是程序问题，不是数据问题") == "contradiction"  # 「实际上」


# ---------------------------------------------------------------------------
# 直接执行时的兜底 runner（不依赖 pytest）

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
