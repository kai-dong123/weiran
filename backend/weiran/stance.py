"""在线行为归类：给世界状态引擎喂一个能读得懂自由文本的分类器。

**为什么需要这个模块 —— 一个实测出来的问题。**

`world_state.classify()` 是确定性关键词匹配，它服务于**离线重放**：
喂进去的是 5 份手写的种子材料，叙述性强、用词可预期（「删除帖子」「公布了分专业数据」），
关键词表在那条路上是对的，测试也锁住了它。

但**在线推演的输入不是种子材料，是 agent 自己写的自由文本**。实测一次
5 agent × 4 轮的推演，12 条生成帖文里有 **10 条落进默认类 `discussion`**，
只有 2 条被归到别的类 —— 而那 2 条**都是错的**：

    文本：「舆情从火星烧成山火，往往就坏在一个"拖"字上……
          从**可控**变成失控」
    归类：reassurance（安抚）
    实际：这是一句**警告**，说的是「会失控」，被归成了「安抚」。

关键词表里的 `可控` 命中了「从可控变成失控」里的子串。
**中文没有词边界，子串匹配会把语义整个翻过来** —— 而翻过来的方向
正好落在 ASYMMETRY 折扣覆盖的那三个维度上，于是核心指标被静默推向反面。

同类风险词还有一批：`我是`（会命中「我是说」）、`实际上`（会命中
「实际上是程序问题」，那不是矛盾）、`并没有`（会命中「并没有说清楚」）。
两个字的词几乎都不可靠。

**结论：不继续往关键词表里加词。** 那条路永远不会收敛 ——
agent 每换一种说法，就要补一条规则，而漏掉的那条是无声的。
正确的分工是：

    离线重放（手写材料）  → world_state.classify()   快、确定、已被测试锁住
    在线推演（自由文本）  → 本模块                    读得懂话，结果落盘缓存

**可复现性怎么保证。** `world_state` 坚持「分类不含 LLM」的理由是
**数字必须可复现**，这个理由是对的，不能丢。所以这里不是把 LLM 塞进
演化循环，而是**把 LLM 的判定结果烘焙进缓存文件**：

    缓存键 = sha256(文本 + PROMPT_VERSION)
    命中   = 直接返回，不调 LLM
    未命中 = 一次调用归类**整轮**的文本，写回缓存

于是「同样输入 → 同样六维曲线」仍然成立（缓存是产物的一部分，
随仓库提交），而缓存没覆盖到的部分由关键词表兜底。
这与 `profiles.py` 的 `profiles_cache.json` 是同一个套路 ——
**LLM 只在生成期出现一次，之后一切都从磁盘重放。**

**失败不抛异常。** 归类失败就退回 `classify()`。宁可让某一轮的分类糙一点，
也不能让整场推演因为一次分类调用挂掉。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .world_state import KIND_PRIORITY, classify

#: 改了 _SYSTEM 就必须 +1 —— 它是缓存键的一部分，否则旧标签会被静默复用。
PROMPT_VERSION = 1

DEFAULT_CACHE = "data/simulation/stance_cache.json"

#: 合法的行为类型。**单一来源**：从 world_state 取，不在这里重复写一遍，
#: 否则两边一旦漂移，LLM 会吐出引擎不认识的类型，而 EXCITATION 查不到键
#: 会静默按「零激励」处理 —— 又一处无声的错误。
KINDS: tuple[str, ...] = KIND_PRIORITY

_KIND_GUIDE = """\
- suppression  压制：要求、劝导或施压让人删帖/不要说/不要转发/注意言行；封禁、撤回
- disclosure   披露：权威方主动公布、承认、道歉、更正，或出面说明
- contradiction 矛盾：指出说法与事实对不上、前后不一致、数据失实
- amplification 放大：转载、转发、扩散、上热搜、媒体介入，重点在**扩大传播面**
- testimony    亲历：当事人或校友以第一人称讲述自己的经历与遭遇
- reassurance  安抚：意在平息情绪，说「不用担心」「总体可控」「请放心」「暂不回应」
- discussion   讨论：其余一切 —— 就事论事的提问、分析、表态、附和、质疑

判断要点：
1. **看意图，不要只看词。** 「从可控变成失控」是在**警告**，不是安抚；
   「我是说」不是亲历；「实际上是程序问题」不是矛盾。
2. 一条文本只给**一个**类型，取**最主要**的那个。
3. 拿不准就给 discussion —— 它是中性默认，代价最小。"""

_SYSTEM = f"""你在为一个校园舆情推演系统做行为归类。

把给定文本归到下面**恰好一个**类型：

{_KIND_GUIDE}

只输出 json，格式为 {{"labels": ["类型1", "类型2", ...]}}，
labels 的长度与顺序必须与输入文本一一对应，不要输出任何解释。"""


def _key(text: str) -> str:
    return hashlib.sha256(
        f"{PROMPT_VERSION}\x00{text}".encode("utf-8")
    ).hexdigest()[:32]


class StanceClassifier:
    """带磁盘缓存的在线行为分类器。

    Args:
        llm: `LLMClient`，或任何有 `chat_json(messages, ...)` 的对象。
            传 None 则完全退化为关键词分类（离线跑测试时用）。
        cache_path: 缓存文件。为 None 则不落盘（只在内存里缓存）。
    """

    def __init__(self, llm=None, cache_path: Path | str | None = DEFAULT_CACHE):
        self.llm = llm
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = {}
        self.total = 0
        self.hits = 0
        self.misses = 0
        self.fallbacks = 0
        self.calls = 0
        self._load()

    # -- 缓存 --------------------------------------------------------------

    def _load(self) -> None:
        if self.cache_path is None or not self.cache_path.is_file():
            return
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # 缓存坏了就当没有，不能让一场推演起不来
        if isinstance(raw, dict):
            self.cache = {k: v for k, v in raw.items() if v in KINDS}

    def save(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )

    # -- 分类 --------------------------------------------------------------

    def classify_many(self, texts: list[str]) -> list[str]:
        """归类一批文本。**整批只发一次 LLM 调用。**

        批量而不是逐条，理由不只是省钱：同一条推文里几条帖子的类型往往互相
        依赖（「这位老师提的两点，从程序上讲…」是回应，不是独立表态），
        一次看到全部，判定比逐条更稳。
        """
        out: list[str | None] = [None] * len(texts)
        pending: list[tuple[int, str]] = []
        self.total += len(texts)

        for i, t in enumerate(texts):
            t = (t or "").strip()
            if not t:
                out[i] = "discussion"
                continue
            hit = self.cache.get(_key(t))
            if hit:
                out[i] = hit
                self.hits += 1
            else:
                pending.append((i, t))

        if pending:
            labels = self._ask([t for _, t in pending])
            added = False
            for (i, t), lab in zip(pending, labels):
                out[i] = lab
                if lab is not None:
                    self.cache[_key(t)] = lab
                    added = True
            # **拿到新标签就立刻落盘，不等调用方记得调 save()。**
            # 这是被测试逼出来的：第一版把落盘交给调用方，于是「同输入同曲线」
            # 这个保证就挂在一个「别忘了调 save」上 —— 忘了不会报错，
            # 只会让下次复跑悄悄换一组标签。可复现性不该依赖调用纪律。
            if added:
                self.save()

        # 兜底：LLM 没给、或给了个引擎不认识的类型，都退回关键词表。
        for i, t in enumerate(texts):
            if out[i] is None:
                self.fallbacks += 1
                out[i] = classify(t or "")
        return [str(x) for x in out]

    def _ask(self, texts: list[str]) -> list[str | None]:
        """问一次 LLM。返回与 texts 等长的列表，失败处为 None。"""
        if self.llm is None or not texts:
            return [None] * len(texts)
        self.misses += len(texts)

        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(texts))
        try:
            self.calls += 1
            data = self.llm.chat_json(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",
                     "content": f"共 {len(texts)} 条，请逐条归类：\n{numbered}"},
                ],
                tag="stance",
                max_tokens=1024,
                expect=dict,
            )
        except Exception:  # noqa: BLE001 —— 归类失败绝不能让推演挂掉
            return [None] * len(texts)

        labels = data.get("labels") if isinstance(data, dict) else None
        if not isinstance(labels, list) or len(labels) != len(texts):
            # 条数对不上就整批作废。**不做「尽力对齐」** ——
            # 错位的标签比没有标签更坏：它会静默地把 A 的话算到 B 头上。
            return [None] * len(texts)

        return [str(x) if str(x) in KINDS else None for x in labels]

    # -- 观测 --------------------------------------------------------------

    def summary(self) -> str:
        return (f"{self.total} 条"
                f"（缓存命中 {self.hits}，LLM 归类 {self.misses} 条 / "
                f"{self.calls} 次调用，退回关键词 {self.fallbacks}）"
                f" · 缓存 {len(self.cache)} 条")
