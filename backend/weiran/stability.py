"""稳定性：把「哪一层会变、哪一层不会」写成**唯一的、可机检的出处**。

**为什么要有这个模块。** 本项目在三个地方自认过同一件事：一份推演产出的六维
曲线**无法端到端复现**，正式口径只能是「多 seed 跑 N 次、报均值与方差」

    simulate.py:655-661    「正式的可复现性方案仍然是多 seed 跑 3 次报均值与方差」
    brief.py:128           把「这只是一个样本」写进产物的 disclaimers
    docs/使用手册.md:475    「而这件事**还没做**」

但「哪一层会变」这句话此前只是散文：散在注释、文档、和人的记忆里。散文的
问题不是不准，是**没人能对着它跑测试** —— 上游改一行、或者在引擎里顺手加个
抖动，散文照样成立，而结论已经不成立了。所以这里把那张表搬进代码，作为
**唯一出处**；`docs/使用手册.md` 里那张表要逐行与它对上（由
`tests/test_handbook.py` 盯着），谁改了这边不改那边会红。

**这张表要读对的两件事。**

1. **「不能播种」不等于「没有办法」。** `asyncio` 交错那一条是真的不能播种，
   而它的后果是实的：交错顺序决定 DB 写入次序 → `post_id` 分配 → 后一个
   agent 点赞/引用的是哪条帖。修它的办法不是播种，是**改架构**（把并发降到 1），
   那是另一件事，本表只如实标注。
2. **这条也钉住了四个「看着像、其实不是」的抽签点。** 上游 `oasis` 全库一共
   只有六处走 `random` 的抽签，分在两个文件里。按**函数边界**逐处核过之后，
   在我们的路径上的只有一处：

       platform.py:277    Platform.refresh（258-327）          **在路径上**
       recsys.py:163      rec_sys_random（136-167）            RANDOM 推荐器；我们走 TWHIN
       recsys.py:413      coarse_filtering（403-418）          `len<=scale` 整段返回，scale=4000
       recsys.py:646,647  swap_random_posts（633-654）         只被 TWITTER 那条路调用
       recsys.py:749      rec_sys_personalized_with_trace（682-800）  同上

   冷门的那四条各有各的「不在」的理由，但共同点是：读上游代码时它们全都长得
   像是每轮都在抽。写下来是为了不让下一个人照着上游文件去「修」一个不在路径
   上的东西，也是为了让「播一次种就够」这句话有个可核的出处。

   **那唯一一处是每轮每人都走的。** `refresh` 不在我们的可用动作表里
   （`simulate.TWITTER_ACTIONS_NAMES` 没有它），所以它不是 LLM 选的 ——
   它由上游 `social_agent/agent_environment.py:59` 在**构造每个 agent 每轮的
   观察**时自动发一次（`get_posts_env` → `action.refresh()`）。入库那份产出里
   有 324 条 `refresh` 动作，占 774 个动作的最大一项；**实际抽签次数 ≤ 324** ——
   `platform.py:276` 那层护栏要求该用户的推荐表里至少有 2 条，不足 2 条时
   这一段不抽、但仍然会记一条 trace。「324 条 trace = 324 次抽签」是没核过的
   下半句，不许当结论写。

**上报的离散度不许叫「分数」。** 本模块只描述方差来源；真正的跨 seed 离散由
`stability_report.py` 报，且报的是「同一批设定下 N 次抽样的离散」，不是评分 ——
见金标自己的 `gold_status.what_is_NOT_solid`。
"""

from __future__ import annotations

import random
from typing import NamedTuple


class VarianceSource(NamedTuple):
    """一处会跨次变化的地方。

    Attributes:
        key: 稳定标识（ASCII）。**手册里那张表必须逐字出现这些 key** ——
            这是两侧唯一的对接点，所以 key 一旦改过就要同步改文档。
        layer: 中文名，给人读的。
        detail: 现状与依据。写清「它为什么会变」，不是「它叫什么」。
        seedable: 能不能收窄，以及靠什么。
        where: 代码位置，便于去核。
    """

    key: str
    layer: str
    detail: str
    seedable: str
    where: str


#: 一次推演里全部会跨次变化的地方。**只此一处定义。**
#:
#: 不在表里的东西就是确定的 —— 而且这一半同样有测试兜着
#: （`tests/test_stability.py` 用 AST 扫引擎与简报，断言它们不出现随机抽签）。
VARIANCE_SOURCES: tuple[VarianceSource, ...] = (
    VarianceSource(
        key="llm_agent_sampling",
        layer="camel 侧 agent 的 LLM 采样",
        detail="agent 每轮「接下来做什么 / 说什么」都过一次采样，同一份输入"
               "两次跑出来不一样。这是六维曲线不可复现的**主因**：缓存只保证"
               "「同一句话给同一标签」，保证不了「同一场推演说出同一句话」。",
        seedable="能收窄，但收不干净：`--seed` / `--temperature` 已接通，"
                 "而 `seed` 只被服务端「尽力」遵守，OpenAI 兼容端点不保证逐位一致。",
        where="simulate.py build_model（cfg[\"temperature\"] / cfg[\"seed\"]）",
    ),
    VarianceSource(
        key="llm_direct_sampling",
        layer="直调路径（归类器）的 LLM 采样",
        detail="归类器不走 camel，自己发 HTTP。它把 agent 的自由文本映射成引擎"
               "认得的行为类别，因此是**引擎的直接输入**。",
        seedable="能（默认关）：`LLMClient.chat` 收了 `seed` 只是不传；归类器"
                 "配置了种子才透传。缓存键里不含种子，所以同一句话仍给同一标签，"
                 "缓存语义不变。",
        where="llm.py LLMClient.chat / stance.py classify_many",
    ),
    VarianceSource(
        key="oasis_feed_sampling",
        layer="OASIS 的 refresh（决定 agent 看到什么 feed）",
        detail="上游在**构造每个 agent 每轮的观察**时自动发一次 `refresh`"
               "（`social_agent/agent_environment.py:59`，不是 LLM 选的 —— "
               "我们的可用动作表里根本没有 `refresh`），该处理器再从他的推荐表里"
               "`random.sample` 抽若干条当 feed；twitter 平台把"
               "`refresh_rec_post_count` 设成 2，于是只要推荐表里有 ≥2 条就抽。"
               "**这是 `--seed` 此前完全够不到的一层**：agent 看到哪几条帖，"
               "决定它接下来做什么。",
        seedable="能，且是决定性的：上游用的就是标准库那个全局 random 模块，"
                 "而 oasis 与 camel **全库从不调用 `random.seed()`**（七处"
                 "`random.Random(...)` 私有实例全在 camel 的 datasets / "
                 "environments / toolkits 一侧，不在 oasis 拉起的模块里）—— "
                 "所以在本进程里播一次种就能覆盖它。",
        where="oasis/social_platform/platform.py:277（阈值来自"
              " oasis/environment/env.py:82 的 refresh_rec_post_count=2）",
    ),
    VarianceSource(
        key="asyncio_interleaving",
        layer="asyncio 并发交错顺序",
        detail="每轮为所有 agent 起一个 task 后 `asyncio.gather`，并发上限 10。"
               "谁先写完不由我们决定，而写入次序决定 `post_id` 分配、进而决定"
               "后一个 agent 点赞/引用的是哪一条帖。",
        seedable="**不能播种。** 它不是随机数，是一个调度顺序。要收窄只能把并发"
                 "降到 1，那是动架构的另一件事，不在这里做。",
        where="oasis/environment/env.py 的 env.step（Semaphore 由 simulate.py 传入）",
    ),
    VarianceSource(
        key="camel_retry_jitter",
        layer="camel 重试抖动",
        detail="camel 在限流重试时 `random.uniform(0, delay)` 睡一段时间。",
        seedable="顺带被全局播种覆盖；它只影响墙钟，不影响内容。"
                 "**但墙钟是本项目记录并比较的量**，所以「墙钟差异」在跨次比较里"
                 "要算上它。",
        where="camel/agents/chat_agent.py（RateLimitError 重试分支）",
    ),
)


def seed_process(seed: int | None) -> dict:
    """给本进程的全局 RNG 播种，返回一段可落盘的说明。

    **买到什么**：`VARIANCE_SOURCES` 里那一层 `oasis_feed_sampling` 从此可复现
    —— agent 每轮看到哪几条帖不再随次变化。顺带把 `camel_retry_jitter` 也钉住
    （只影响墙钟）。

    **买不到什么**（写在这里，因为一整批臂都要围着这个限制读）：

    - LLM 那一层收不干净。`--seed` 只是「请求里带了个字段」，端点尽力而已；
      而且 `asyncio` 的交错顺序根本不是随机数，播种种不到它。
    - 所以「同一个 seed 两次跑出同样的曲线」**不是本函数能保证的事**，
      也不该在测试里断言。它是**测量**出来的，见 `run_seeds.py --preflight`。

    `seed=None` 时**什么都不做**并返回 `{"seeded": False}` —— 默认路径的行为
    必须一个字节都不变（这是本项目对「默认值动不动」的一贯要求：默认不动，
    由调用方显式选）。
    """
    if seed is None:
        return {"seeded": False}
    random.seed(seed)
    return {"seeded": True, "seed": seed}


#: 落进产出 `meta.sampling_note` 的那句话。**产出要自己带上限制条件**，
#: 不能指望读它的人先看过本文档。
SAMPLING_NOTE = (
    "seed 只在请求里携带、由服务端「尽力」遵守；进程级 RNG 已播种，"
    "但 asyncio 的并发交错顺序不是随机数，播种覆盖不到。"
    "跨次比较请按「多 seed 多次、看离散」读，不要把单次产出当可复现结论。"
)
