"""在线行为归类的测试。

这一组测试守的是**两个静默错误**，两个都不会抛异常、都只会让数字慢慢错掉：

1. `quote_content` 陷阱 —— OASIS 的引用帖把**原文**抄进了 `content`，
   把**作者自己的话**放在 `quote_content`。读错列不会报错，
   只会让每一条引用帖都被归成与种子事件同一个类型，六维曲线退化成平线。

2. 关键词表读不懂自由文本 —— 实测「从**可控**变成失控」被归成
   `reassurance`（安抚）。语义被整个翻了过来，而没有任何地方会报错。
"""

from __future__ import annotations

import json

import pytest

from weiran.simulate import texts_from_actions, _SILENT_ACTIONS
from weiran.stance import KINDS, PROMPT_VERSION, StanceClassifier, _key
from weiran.world_state import KIND_PRIORITY, classify


# ---------------------------------------------------------------------------
# 动作 -> 文本
# ---------------------------------------------------------------------------

SEED = "【就业指导中心】2025届就业质量报告发布，落实率78.3%。"


def _post(post_id, user_id, content="", quote_content=None,
          original_post_id=None):
    return {"post_id": post_id, "user_id": user_id, "content": content,
            "quote_content": quote_content,
            "original_post_id": original_post_id}


@pytest.fixture
def posts():
    return {
        # 种子事件：自己写的，quote_content 为空
        1: _post(1, 0, content=SEED),
        # 引用帖：content 是原文副本，quote_content 才是作者的话
        2: _post(2, 1, content=SEED, quote_content="口径怎么定的？说清楚。",
                 original_post_id=1),
        # 转帖：content 是空串，只靠 original_post_id 指向原文
        3: _post(3, 2, content="", original_post_id=2),
    }


def test_create_post_reads_own_content(posts):
    acts = [{"action": "create_post", "info": json.dumps({"post_id": 1})}]
    assert texts_from_actions(acts, posts) == [SEED]


def test_quote_post_reads_the_authors_own_words_not_the_original(posts):
    """**这是本文件最重要的一条。** 读错列 -> 引用帖被归成种子事件的类型。"""
    acts = [{"action": "quote_post",
             "info": json.dumps({"quoted_id": 1, "new_post_id": 2})}]
    out = texts_from_actions(acts, posts)
    assert out == ["口径怎么定的？说清楚。"]
    # 若哪天有人把这里改回读 content，下面这条会立刻失败
    assert SEED not in out[0]


def test_repost_follows_original_post_id(posts):
    """转帖的 content 是空串，只有 original_post_id 能找回被转的话。"""
    acts = [{"action": "repost",
             "info": json.dumps({"reposted_id": 1, "new_post_id": 3})}]
    out = texts_from_actions(acts, posts)
    assert out, "转帖被静默丢掉了 —— 而它恰恰是 amplification 最强的信号"
    assert "口径怎么定的" in out[0]


def test_like_uses_the_liked_posts_text(posts):
    acts = [{"action": "like_post", "info": json.dumps({"post_id": 1})}]
    assert texts_from_actions(acts, posts) == [SEED]


@pytest.mark.parametrize("name", sorted(_SILENT_ACTIONS))
def test_silent_actions_produce_no_signal(name, posts):
    acts = [{"action": name, "info": json.dumps({"post_id": 1})}]
    assert texts_from_actions(acts, posts) == []


def test_unknown_post_id_is_skipped_not_crashed(posts):
    acts = [{"action": "create_post", "info": json.dumps({"post_id": 999})}]
    assert texts_from_actions(acts, posts) == []


def test_info_may_arrive_as_a_dict_or_a_json_string(posts):
    """OASIS 的 trace 里 info 是 TEXT 列，但别处可能已经是 dict。"""
    as_str = [{"action": "create_post", "info": json.dumps({"post_id": 1})}]
    as_dict = [{"action": "create_post", "info": {"post_id": 1}}]
    assert texts_from_actions(as_str, posts) == texts_from_actions(as_dict, posts)


def test_broken_info_json_does_not_raise(posts):
    acts = [{"action": "create_post", "info": "{不是合法 json"}]
    assert texts_from_actions(acts, posts) == []


# ---------------------------------------------------------------------------
# 分类器
# ---------------------------------------------------------------------------

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


def test_no_llm_falls_back_to_keywords(tmp_path):
    c = StanceClassifier(llm=None, cache_path=tmp_path / "c.json")
    out = c.classify_many(["辅导员要求同学删除帖子", "学校公布了分专业数据"])
    assert out == ["suppression", "disclosure"]
    assert c.fallbacks == 2


def test_llm_labels_are_used(tmp_path):
    c = StanceClassifier(
        llm=_FakeLLM(label="testimony"), cache_path=tmp_path / "c.json")
    assert c.classify_many(["我那年也是这样"]) == ["testimony"]
    assert c.fallbacks == 0


def test_blank_texts_never_reach_the_llm(tmp_path):
    llm = _FakeLLM(label="testimony")
    c = StanceClassifier(llm=llm, cache_path=tmp_path / "c.json")
    assert c.classify_many(["", "   "]) == ["discussion", "discussion"]
    assert llm.calls == 0


def test_cache_prevents_a_second_llm_call(tmp_path):
    """可复现性的实现方式：LLM 只在第一次出现，之后从磁盘重放。"""
    path = tmp_path / "c.json"
    llm = _FakeLLM(label="amplification")
    first = StanceClassifier(llm=llm, cache_path=path)
    assert first.classify_many(["上热搜了"]) == ["amplification"]
    assert llm.calls == 1

    second = StanceClassifier(llm=llm, cache_path=path)  # 新实例，从盘读
    assert second.classify_many(["上热搜了"]) == ["amplification"]
    assert llm.calls == 1, "缓存没生效 —— 同输入两次跑会给出不同曲线"
    assert second.hits == 1


def test_llm_failure_degrades_to_keywords_and_keeps_running(tmp_path):
    c = StanceClassifier(llm=_FakeLLM(raise_exc=RuntimeError("端点挂了")),
                         cache_path=tmp_path / "c.json")
    assert c.classify_many(["辅导员要求同学删除帖子"]) == ["suppression"]
    assert c.fallbacks == 1


def test_label_count_mismatch_discards_the_whole_batch(tmp_path):
    """条数对不上就整批作废。**不做「尽力对齐」** ——
    错位的标签会静默地把 A 说的话算到 B 头上。"""
    c = StanceClassifier(llm=_FakeLLM(label="testimony", count=1),
                         cache_path=tmp_path / "c.json")
    out = c.classify_many(["辅导员要求同学删除帖子", "学校公布了分专业数据"])
    assert out == ["suppression", "disclosure"]  # 两个都退回了关键词
    assert c.fallbacks == 2


def test_label_outside_the_engine_vocabulary_is_rejected(tmp_path):
    """引擎不认识的类型会让 EXCITATION 静默查不到键 —— 宁可退回关键词。"""
    c = StanceClassifier(llm=_FakeLLM(label="驳斥"), cache_path=tmp_path / "c.json")
    out = c.classify_many(["辅导员要求同学删除帖子"])
    assert out == ["suppression"]


def test_corrupt_cache_file_is_ignored(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{不是 json", encoding="utf-8")
    c = StanceClassifier(llm=None, cache_path=path)
    assert c.cache == {}


def test_cache_entries_with_unknown_labels_are_dropped(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"abc": "驳斥", "def": "testimony"}),
                    encoding="utf-8")
    c = StanceClassifier(llm=None, cache_path=path)
    assert c.cache == {"def": "testimony"}


def test_cache_key_covers_the_prompt_version(tmp_path):
    """改了提示词就必须让旧缓存失效，否则旧标签会被静默复用。"""
    assert _key("同样的文本") == _key("同样的文本")
    assert PROMPT_VERSION >= 1


def test_engine_and_classifier_share_one_vocabulary():
    """`stance.KINDS` 与 `world_state.KIND_PRIORITY` 是同一个元组。
    两边一旦漂移，LLM 会吐出引擎不认识的类型。"""
    assert KINDS == KIND_PRIORITY


# ---------------------------------------------------------------------------
# 关键词表本身：记录它为什么读不懂自由文本
# ---------------------------------------------------------------------------

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
