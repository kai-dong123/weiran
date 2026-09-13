"""角色画像生成：金标 actor → OASIS 可用的 profile。

**这个模块的核心是一条架构决定，不是一段提示词。**

`persona` 只装**稳定的身份**：他是谁、站在什么位置、什么性情、倾向怎么做。
**不装事实。** 事实由世界状态引擎在**每一轮**按该 actor 的 `knows` / `hidden`
注入。（该映射另写一份 `actor_knowledge.json`，供感知层读取。）

为什么要这样切：本项目的卖点是「差异化感知」——同一个事件，家长只知道
落实率与灵活就业占比，郑维却知道实际核实率只有 32.2% 且对外不披露。
如果把这些事实写死在 persona 里，感知就是**静态**的，六维状态无从分化，
整个创新点就退化成「给每个 agent 写一段不同的背景故事」。
知识必须是**逐轮注入**的，才谈得上「感知」。

---

两种生成模式：

  - **模板模式（`--offline`）**：不调 LLM，由 role/stance 拼出 persona。
    用于无密钥环境下打通管线、也可进 CI。质量够用但不生动。
  - **LLM 模式**：每个 actor 一次调用生成 persona。**默认关闭推理**
    （`thinking=False`）—— 27 次调用，开推理要多付约 12 倍的输出 token，
    而写一段人物小传并不需要长思维链。

生成结果**落盘缓存**。重复生成不会重复付费，除非显式 `--refresh`。
缓存键包含输入哈希，改了提示词或改了 actor 会自动失效。

用法：
    cd backend
    python -m weiran.profiles --offline              # 无需密钥
    python -m weiran.profiles                        # 调 LLM，写入 sim/ 目录
    python -m weiran.profiles --refresh              # 忽略缓存重新生成
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import REPO_ROOT, ConfigError, load_config
from .llm import LLMClient, LLMError

DEFAULT_SCENARIO = "benchmark/scenarios/employment_trust_crisis"

# 缓存与输出的默认位置。放在 data/ 下 —— 它是产物，不是源码，不入库。
DEFAULT_OUT = "data/simulation"

# 提示词版本。**改了 _PERSONA_SYSTEM 或 _llm_persona 的提问方式就必须 +1。**
# 它是缓存键的一部分：不带上它，改了提示词之后旧结果会被静默复用，
# 看起来「重新生成过了」，其实一个字都没变。这类问题不会报错，
# 只会在某天让人对着两份本该不同的结果发懵。
PROMPT_VERSION = 2

# ---------------------------------------------------------------------------
# 社交足迹：按群体给定，不按人。**确定性**由 actor_id 的哈希保证，
# 同一 actor 每次生成得到同样的数字 —— 否则两次仿真不可比。
#
# 数值本身是作者设定的（这是虚构场景），量级参照真实社交平台的分布：
# 官方号关注者多、关注少；学生相反；媒体/自媒体触达最广。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ⚠️ 实测：**下面这三个数字 OASIS 根本不读。**
#
# 在 oasis 包内全文检索 `follower_count` / `friend_count` / `statuses_count`
# 的结果是零引用 —— 图构建只取 `user_char` / `username` / `description`
# （Twitter）与 `persona` / `username` / `bio` / `mbti` / `gender` / `age` /
# `country`（Reddit）。上游写这两个字段是为了自己的前端展示。
#
# 所以：它们**不是仿真输入**，对本项目的六维状态没有任何影响。
# 保留它们只为画像元数据完整（以及我们自己的前端可以展示），
# 技术报告里不会把它们说成仿真的一部分。
# ---------------------------------------------------------------------------

# group -> (followers, friends, statuses) 各自的 (下限, 上限)
FOOTPRINT: dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = {
    "校领导":   ((2000, 9000),   (5, 40),     (40, 160)),
    "校方":     ((3000, 15000),  (10, 80),    (80, 400)),
    "学工":     ((1500, 6000),   (20, 90),    (60, 300)),
    "辅导员":   ((300, 1500),    (80, 400),   (150, 900)),
    "应届生":   ((30, 500),      (100, 800),  (80, 1200)),
    "往届生":   ((40, 600),      (120, 900),  (200, 2500)),
    "在校低年级": ((20, 400),    (80, 600),   (30, 700)),
    "家长":     ((10, 250),      (30, 300),   (20, 400)),
    "媒体":     ((5000, 30000),  (100, 600),  (800, 4000)),
    "自媒体":   ((8000, 60000),  (200, 1200), (1500, 6000)),
    "外部":     ((100, 900),     (60, 400),   (20, 300)),
}

# 兜底：分组表里没有的一律按这个给，避免 KeyError 让整条管线停摆。
_FOOTPRINT_FALLBACK = ((50, 500), (50, 500), (50, 500))

# 事实清单的机器可读映射（供感知层逐轮注入）。
KNOWLEDGE_FILENAME = "actor_knowledge.json"


@dataclass
class ActorProfile:
    """一个 agent 的完整画像。

    字段分三类，边界要清楚：
      - **来自金标**（actor_id / name / group / role / stance / knows / hidden）：
        是叙事事实，不由本模块发明。
      - **本模块生成**（persona / bio）：LLM 或模板产出，可重新生成。
      - **本模块设定**（user_id / user_name / 社交计数）：虚构场景的作者设定，
        确定性可复现。
    """

    actor_id: str
    name: str
    group: str
    role: str
    stance: str
    persona: str
    bio: str
    user_id: int
    user_name: str
    profession: str
    age: int
    # Reddit 那一路 OASIS 直接索引这两个字段，缺了会 KeyError。
    # 它们是场景作者给虚构角色设定的属性，不是对真人的推断。
    gender: str = "female"
    mbti: str = "ISFJ"
    topics: list[str] = field(default_factory=list)
    follower_count: int = 0
    friend_count: int = 0
    statuses_count: int = 0
    created_at: str = "2024-09-01"
    knows: list[str] = field(default_factory=list)
    hidden: list[str] = field(default_factory=list)
    generated_by: str = "template"  # template | llm

    def twitter_row(self) -> dict:
        """OASIS Twitter 需要的列。

        **列名是实测出来的，不是照文档写的。**

        OASIS 的 `generate_twitter_agent_graph` 只读三列：`user_char`
        （→ agent 的 user_profile，即系统提示里的人设）、`username`、
        `description`。上游 `test_profile_format.py` 声称的必需列
        （`user_name` / `bio` / `friend_count` / ...）**一个都没被读到**——
        那份校验测的是一套它自己的生成器从不产出的列，是失效的测试。

        上游真实写出的表头（`app/services/oasis_profile_generator.py:1139`）：
            ['user_id', 'name', 'username', 'user_char', 'description']
        本方法照此对齐，语义也照上游注释：
            user_char   —— 内部用，喂给 LLM 系统提示，决定 agent 怎么想怎么做
            description —— 外部用，别人能看到的公开简介
        """
        # 换行会破坏 CSV 的一行一记录，也必须清掉（上游同样处理）。
        persona = " ".join(self.persona.split())
        bio = " ".join(self.bio.split())
        return {
            "user_id": self.user_id,
            "name": self.name,
            "username": self.user_name,
            "user_char": f"{bio} {persona}".strip(),
            "description": bio,
        }

    def reddit_row(self) -> dict:
        """OASIS Reddit 需要的结构。

        **`gender` 与 `mbti` 是硬要求，不是可选字段。**
        OASIS 的 `generate_reddit_agent_graph` 直接索引它们、没有 `.get`，
        缺了当场 KeyError。早期版本出于「金标里没有依据就不编」的考虑
        省掉了这两个字段 —— 想法是好的，但会让 Reddit 那一路根本跑不起来。

        它们的性质要说清楚：这**不是**对真实人物的推断，而是场景作者
        为虚构角色设定的属性，与名字本身同一性质。LLM 生成时要求它与
        自己写的人设自洽（人称、性情对得上）；模板模式则按 actor_id
        确定性分配，不假装有依据。
        """
        return {
            "realname": self.name,
            "username": self.user_name,
            "bio": self.bio,
            "persona": self.persona,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": "中国",
            "profession": self.profession,
            "interested_topics": self.topics,
        }


# ---------------------------------------------------------------------------
# 确定性数值
# ---------------------------------------------------------------------------

def _seed(*parts: str) -> int:
    """由字符串稳定地导出一个整数。用 sha256 而非内置 hash——
    后者在 Python 进程间带随机盐，会让两次运行的数字不一样。"""
    raw = "\x1f".join(parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:12], 16)


def _ranged(lo: int, hi: int, *seed_parts: str) -> int:
    return lo + _seed(*seed_parts) % (hi - lo + 1)


def footprint(group: str, actor_id: str) -> tuple[int, int, int]:
    """按群体给出该 actor 的关注者 / 关注 / 发帖数。确定性。"""
    spec = FOOTPRINT.get(group, _FOOTPRINT_FALLBACK)
    return tuple(  # type: ignore[return-value]
        _ranged(lo, hi, actor_id, kind)
        for kind, (lo, hi) in zip(("followers", "friends", "statuses"), spec)
    )


def handles(actor_id: str) -> str:
    """OASIS 的 `user_name`（@ 后面的那个）。

    用纯 ASCII 且可追溯到金标 id —— 中文用户名在部分平台适配层会被
    转义或截断，而 `weiran_a11` 一眼就能对回 `A11 周雨桐`，
    对调试和写报告都方便。中文名走 `name`（显示名）。
    """
    return f"weiran_{actor_id.lower()}"


# ---------------------------------------------------------------------------
# 年龄与专业：由 role 文本推得，规则写在明面上
# ---------------------------------------------------------------------------

_AGE_RULES: tuple[tuple[tuple[str, ...], int], ...] = (
    (("大一",), 18),
    (("大二",), 19),
    (("大三",), 20),
    (("大四",), 21),
    (("2025 届", "2025届"), 22),
    (("2023 届", "2023届"), 24),
    (("2022 届", "2022届"), 25),
    (("2021 届", "2021届"), 26),
    (("校长",), 56),
    (("副校长",), 52),
    (("部长", "主任", "负责人", "主笔", "记者", "HR"), 40),
    (("辅导员",), 32),
)

# 按群体的兜底年龄。**家长必须靠这一条** —— 他们的 role 是
# 「周雨桐之父」这类亲属关系描述，里面没有任何可匹配的年龄线索，
# 落到默认值会得到一个 30 岁的、有一个 22 岁孩子的家长。
_AGE_BY_GROUP: dict[str, int] = {
    "家长": 47,
    "校领导": 54,
    "校方": 42,
    "学工": 44,
    "辅导员": 32,
    "应届生": 22,
    "往届生": 25,
    "在校低年级": 19,
    "媒体": 35,
    "自媒体": 33,
    "外部": 38,
}

_AGE_DEFAULT = 30


def guess_age(role: str, actor_id: str, group: str = "") -> int:
    """推定年龄。先看 role 里的硬线索，再看 group，最后才用默认值。

    这**不是**在推断真实的人 —— 角色是虚构的，年龄是场景作者设定的属性。
    规则写在明面上，是为了让它可复现、可被质疑，而不是藏在提示词里
    让 LLM 每次给一个不同的数。
    """
    base: int | None = None
    for keys, age in _AGE_RULES:
        if any(k in role for k in keys):
            base = age
            break
    if base is None:
        base = _AGE_BY_GROUP.get(group, _AGE_DEFAULT)
    return base + _seed(actor_id, "age") % 3 - 1


# 群体的职业标签。**不能拿 role 当职业** —— role 常是身份描述而非职业，
# 例如应届生的 role 是「经管学院 2025 届」，填进 profession 字段就成了
# 「职业：经管学院 2025 届」，一眼就露怯。
_PROFESSION_BY_GROUP: dict[str, str] = {
    "应届生": "学生",
    "往届生": "职场人士",
    "在校低年级": "学生",
    "家长": "家长",
    "媒体": "记者",
    "自媒体": "自媒体从业者",
    "外部": "企业人力资源",
}

# 这些群体里 role 本身就是职业名，直接用 role 更准确。
_ROLE_IS_PROFESSION = {"校领导", "校方", "学工", "辅导员"}


def profession_of(group: str, role: str) -> str:
    if group in _ROLE_IS_PROFESSION:
        return role
    return _PROFESSION_BY_GROUP.get(group, role)


_TOPICS_BY_GROUP: dict[str, list[str]] = {
    "校领导": ["教育治理", "招生", "舆情应对"],
    "校方": ["就业政策", "统计口径", "公共关系"],
    "学工": ["学生事务", "校园秩序"],
    "辅导员": ["学生工作", "心理支持", "就业指导"],
    "应届生": ["求职", "就业形势", "考研"],
    "往届生": ["职业发展", "就业数据"],
    "在校低年级": ["学业规划", "考研", "校园生活"],
    "家长": ["升学", "就业前景", "教育投资"],
    "媒体": ["教育新闻", "公共政策"],
    "自媒体": ["教育观察", "热点评论"],
    "外部": ["招聘", "人才市场"],
}


# ---------------------------------------------------------------------------
# persona 生成
# ---------------------------------------------------------------------------

_PERSONA_SYSTEM = """\
你在为一个**校园舆情推演仿真**撰写其中一个智能体的人物设定。

这是一个**完全虚构**的场景：虚构的大学、虚构的人物、虚构的事件。
你要写的是一段人物小传，供多智能体仿真使用。

要求：
1. 用第三人称写，一段话，120–200 字，中文。
2. 写**这个人的身份、处境、性情、说话方式、以及他\她在这件事上的立场倾向**。
3. **不要写具体的数据、事实、事件细节**。那些会在仿真运行时按各人
   知情范围单独注入 —— 写进来会破坏「不同角色看到不同信息」这一设计。
4. 不要写「他知道……」「他不知道……」。只写他是个什么样的人。
5. 不要出现任何真实学校、真实人物、真实机构的名字。

另外要给出这个角色的 `gender`（male 或 female）与 `mbti`（四字母）。
这两项是**你为这个虚构角色设定的属性**，请与你写的人设自洽 ——
人称、性情、行为方式要能对得上。

只输出 json，字段为：
  persona（字符串）、bio（不超过 40 字的签名档）、
  gender（"male" 或 "female"）、mbti（四字母）。\
"""


def _default_gender(actor_id: str) -> str:
    """模板模式下确定性分配性别。

    **不做任何基于名字或角色的推断** —— 中文名里看似「像女性」的字，
    用来推性别既不可靠也不该做。这里就是按 id 稳定地二选一，
    它唯一的性质是「可复现」。
    """
    return "female" if _seed(actor_id, "gender") % 2 == 0 else "male"


# 模板模式的 MBTI 池。同样不做推断，按 id 稳定取一个。
_MBTI_POOL = ("ISFJ", "ESFJ", "ISTJ", "ESTJ", "INFJ", "ENFJ", "INFP", "ENFP")


def _default_mbti(actor_id: str) -> str:
    return _MBTI_POOL[_seed(actor_id, "mbti") % len(_MBTI_POOL)]


def _template_persona(actor: dict) -> tuple[str, str]:
    """无 LLM 时的 persona：由 role / stance / group 拼装。

    够用但不生动。它的价值在于**让整条管线在没有密钥的情况下也能跑通**——
    评委、CI、以及我自己调试时都不该被一个 API key 挡住。
    """
    name, group, role, stance = (
        actor["name"], actor["group"], actor["role"], actor["stance"],
    )
    # 措辞要**与群体无关**。早期版本写的是「这既是职责使然」——
    # 用在应届生身上明显不对：一个学生哪来的职责。
    #
    # 也**不要试图从 stance 里抽「第一反应」**：曾经取 `stance.split("，")[0]`，
    # 于是校长的 stance「最终决策者，倾向一次性说清」被拼成了
    # 「第一反应是最终决策者」—— 头衔不是动作。宁可原样引用 stance。
    persona = (
        f"{name}是{role}，属于{group}这一群体。"
        f"{name}在这件事上的立场是：{stance}。"
        f"{name}日常接触到的事件信息、以及感受到的压力，"
        f"大多来自{group}这个位置，"
        f"因此说话时会不自觉地带着这个位置带来的立场与顾虑。"
    )
    bio = f"{role}｜{stance[:16]}"
    return persona, bio


def _llm_persona(client: LLMClient, actor: dict) -> tuple[str, str, str, str]:
    """调 LLM 写一段人物小传。**默认关闭推理** —— 写小传不需要长思维链。

    Returns:
        (persona, bio, gender, mbti)。后两项由 LLM 与人设自洽地设定；
        取不到就退回确定性默认值，不让一个缺字段毁掉整批生成。
    """
    prompt = (
        f"人物：{actor['name']}\n"
        f"群体：{actor['group']}\n"
        f"角色：{actor['role']}\n"
        f"立场倾向：{actor['stance']}\n\n"
        f"请以 json 输出 persona、bio、gender、mbti。"
    )
    data = client.chat_json(
        [
            {"role": "system", "content": _PERSONA_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        tag="profiles.persona",
        temperature=0.8,   # 这里要的是生动，不是稳定
        max_tokens=800,
        thinking=False,
    )
    persona = str(data.get("persona", "")).strip()
    bio = str(data.get("bio", "")).strip()
    if not persona:
        raise LLMError(f"{actor['id']} 的 persona 为空")

    gender = str(data.get("gender", "")).strip().lower()
    if gender not in ("male", "female"):
        gender = _default_gender(actor["id"])
    mbti = str(data.get("mbti", "")).strip().upper()
    if len(mbti) != 4 or not mbti.isalpha():
        mbti = _default_mbti(actor["id"])
    return persona, bio[:40], gender, mbti


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------

def build_profiles(
    actors: list[dict],
    *,
    client: LLMClient | None = None,
    cache: dict | None = None,
    refresh: bool = False,
) -> tuple[list[ActorProfile], dict]:
    """由金标 actor 列表生成画像。

    Returns:
        (画像列表, 更新后的缓存)

    缓存键是 actor 内容的哈希 —— 改了金标里的角色设定会自动失效，
    没改就直接复用，不重复付费。
    """
    cache = dict(cache or {})
    out: list[ActorProfile] = []

    for idx, actor in enumerate(actors):
        aid = actor["id"]
        # 缓存键要含**生成模式**：早期版本只哈希 actor 内容，
        # 于是先跑 --offline 再跑 LLM 模式时会命中模板缓存，
        # 静默地不调用 LLM —— 看起来「跑过了」，其实用的是兜底文案。
        mode = "llm" if client is not None else "template"
        key = hashlib.sha256(
            json.dumps(
                {"actor": actor, "mode": mode, "prompt": PROMPT_VERSION},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]

        cached = cache.get(aid)
        if not refresh and cached and cached.get("key") == key:
            persona, bio = cached["persona"], cached["bio"]
            gender = cached.get("gender") or _default_gender(aid)
            mbti = cached.get("mbti") or _default_mbti(aid)
            source = cached.get("source", "cache")
        elif client is not None:
            persona, bio, gender, mbti = _llm_persona(client, actor)
            source = "llm"
        else:
            persona, bio = _template_persona(actor)
            gender, mbti = _default_gender(aid), _default_mbti(aid)
            source = "template"

        cache[aid] = {
            "key": key, "persona": persona, "bio": bio,
            "gender": gender, "mbti": mbti, "source": source,
        }

        followers, friends, statuses = footprint(actor["group"], aid)
        out.append(
            ActorProfile(
                actor_id=aid,
                name=actor["name"],
                group=actor["group"],
                role=actor["role"],
                stance=actor["stance"],
                persona=persona,
                bio=bio,
                user_id=idx,
                user_name=handles(aid),
                profession=profession_of(actor["group"], actor["role"]),
                age=guess_age(actor["role"], aid, actor["group"]),
                gender=gender,
                mbti=mbti,
                topics=list(_TOPICS_BY_GROUP.get(actor["group"], [])),
                follower_count=followers,
                friend_count=friends,
                statuses_count=statuses,
                knows=list(actor.get("knows", [])),
                hidden=list(actor.get("hidden", [])),
                generated_by=source,
            )
        )
    return out, cache


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------

TWITTER_FILENAME = "twitter_profiles.csv"
REDDIT_FILENAME = "reddit_profiles.json"


def write_twitter_csv(profiles: list[ActorProfile], path: Path) -> None:
    """写 CSV。**表头顺序固定**并在末尾补一个换行 —— 上游按 DictReader 读，
    顺序不影响解析，但固定的表头让 diff 可读。"""
    if not profiles:
        raise ValueError("没有任何画像可写")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [p.twitter_row() for p in profiles]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_reddit_json(profiles: list[ActorProfile], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([p.reddit_row() for p in profiles], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_knowledge_map(profiles: list[ActorProfile], path: Path) -> None:
    """写「谁知道什么」的机器可读映射。

    **这是本模块输出里最重要的一个文件。** 仿真每轮据此决定往哪个 agent
    的上下文里塞哪些事实 —— 差异化感知靠的就是它。

    `hidden` 与 `knows` 分开存，不是冗余：前者是「知道但不披露」，
    它驱动的行为与「不知道」完全不同（会去辩解、会回避、会压话题）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": "由 weiran.profiles 生成。供感知层逐轮注入，勿手工编辑。",
        "actors": {
            p.actor_id: {
                "user_id": p.user_id,
                "user_name": p.user_name,
                "name": p.name,
                "group": p.group,
                "knows": p.knows,
                "hidden": p.hidden,
            }
            for p in profiles
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _scenario_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def generate(
    scenario_dir: Path,
    out_dir: Path,
    *,
    offline: bool = False,
    refresh: bool = False,
) -> list[ActorProfile]:
    ref = json.loads((scenario_dir / "reference_data.json").read_text(encoding="utf-8"))
    actors = ref["actors"]

    client: LLMClient | None = None
    if not offline:
        try:
            config = load_config(require_llm=True)
            client = LLMClient(config.llm)
        except ConfigError as exc:
            print(f"配置不可用，退回模板模式：\n{exc}\n", file=sys.stderr)

    cache_path = out_dir / "profiles_cache.json"
    cache = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}

    profiles, cache = build_profiles(
        actors, client=client, cache=cache, refresh=refresh
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_twitter_csv(profiles, out_dir / TWITTER_FILENAME)
    write_reddit_json(profiles, out_dir / REDDIT_FILENAME)
    write_knowledge_map(profiles, out_dir / KNOWLEDGE_FILENAME)
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if client is not None:
        print(client.ledger.summary())
    return profiles


def main(argv: list[str] | None = None) -> int:
    from .config import ensure_console_encoding
    ensure_console_encoding()

    ap = argparse.ArgumentParser(description="角色画像生成")
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--offline", action="store_true",
                    help="不调 LLM，用模板生成（无需密钥）")
    ap.add_argument("--refresh", action="store_true",
                    help="忽略缓存重新生成")
    args = ap.parse_args(argv)

    scenario = _scenario_path(args.scenario)
    if not (scenario / "reference_data.json").is_file():
        print(f"找不到 {scenario / 'reference_data.json'}", file=sys.stderr)
        return 2

    out = _scenario_path(args.out)
    profiles = generate(scenario, out, offline=args.offline, refresh=args.refresh)

    by_source: dict[str, int] = {}
    for p in profiles:
        by_source[p.generated_by] = by_source.get(p.generated_by, 0) + 1

    print(f"\n生成 {len(profiles)} 个画像 → {out}")
    print(f"  来源分布：{by_source}")
    print(f"  {TWITTER_FILENAME} / {REDDIT_FILENAME} / {KNOWLEDGE_FILENAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
