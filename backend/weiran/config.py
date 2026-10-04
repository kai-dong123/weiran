"""「未然」运行配置。

配置来源优先级：真实环境变量 > 仓库根目录的 .env > 默认值。

设计取向：**缺失的配置必须在第一时间、以能读懂的方式报错。**
这类项目最常见的失败不是算法错，而是跑到一半发现某个 key 是空的。
所以这里不做「宽容降级」——该有的没有，就立刻停下来，并说清楚缺什么、怎么补。
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# backend/weiran/config.py -> backend/weiran -> backend -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]


def rel_path(path) -> str:
    """相对仓库根的路径（正斜杠）；不在仓库内则原样返回。

    凡是要**写进产物或清单**的路径都用它：绝对路径把「本机目录」焊进了
    入库的东西，换台机器、或仓库换个位置，那份记录就和产生它的那次运行
    对不上 —— 而它们的全部意义就是「我看到的和它说的是同一件事」。

    **只此一处定义。** 原先 `brief.py` 自己有一份，而 `simulate` 的落盘
    提示、`run_seeds` 的臂清单、`stability_report` 的来源段另有三处要写
    路径；各写一份必然漂移，且漂移了没有任何地方会报错。
    """
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


#: 文本里的路径**长什么样**：盘符 + 分隔符，或两种 POSIX 家目录。
_PATH_TOKEN = re.compile(r"[A-Za-z]:[\\/][^\s\"'`|<>]*|/(?:mnt|home)/[^\s\"'`|<>]*")

#: 路径后面跟着的成对/句读符号，切掉再判：正文里的路径常写在「」里或以「，」结尾。
_TRAILING = "，。；：、）」』】》”’,;:)]}>\"'`."


def absolute_repo_paths(text: str) -> list[str]:
    """文本里**以绝对形式写出的、本仓库内**的路径。**只做否证用。**

    与 `rel_path` 是一件事的两半，所以放在一起：那个负责**写的时候**写成仓库
    相对，这个负责**写完之后**能自己查出来。少了这一半，`rel_path` 就只在
    **调用它的地方**生效 —— 漏掉一处不会有任何报错。这不是假设：
    `stability_report` 的臂目录就是漏掉的那一处，一份已经写好的汇总里带着
    `D:\\…\\weiran\\data\\runs\\seeds\\seed11`，而它自己的三条口径检查一条都没
    提这件事。**能查出来的东西不要靠记得。**

    **边界写在名字里：只管本仓库内。** 仓库外的路径（测试用的临时目录、
    别人机器上的家目录）不归这条规矩管 —— `rel_path` 对它们本来就原样返回，
    它们也没有等价的「仓库相对」写法。要抓的是「**我们的**东西被写成了绝对
    形式」，那是换台机器就对不上的那种记录。
    """
    hits = []
    for raw in _PATH_TOKEN.findall(text):
        token = raw.rstrip(_TRAILING)
        if not token:
            continue
        try:
            if Path(token).resolve().is_relative_to(REPO_ROOT):
                hits.append(token)
        except (OSError, ValueError):
            continue
    return hits


# 场景库的默认位置。**只此一处定义。**
# 原先 `Config.db_path` 与 `scenario.py` 各算一次同一个路径，两者会悄悄漂移
# —— 改了一处、另一处仍指向老地方，而没有任何地方会报错。
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "weiran.db"


class ConfigError(RuntimeError):
    """配置缺失或非法。消息里必须包含「缺什么」与「怎么补」。"""


def ensure_console_encoding() -> None:
    """让 stdout / stderr 在重定向到 GBK 控制台时不因符号而崩掉。

    实测（Windows 中文控制台，默认 GBK）：

        python -m weiran.validate > out.txt
        UnicodeEncodeError: 'gbk' codec can't encode character '\\u26a0'

    脚本**整个失败**，连断言统计都走不到 —— 而 README 把这条命令写成验证步骤，
    只要重定向输出（或走 CI、或 `| tee`）就会撞上。这类项目最常见的失败
    不是算法错，而是「跑到一半停下来」。

    修法只动 errors 策略、**不动 encoding**：中文仍按控制台原本的编码正确显示，
    装不下的符号退化成 '?'，而不是让整条命令崩掉。
    （交互式控制台下 Python 走 Windows 控制台 API，本就不受影响；这里管重定向与管道。）
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # 不是 TextIOWrapper（测试里的 StringIO、已关闭的流等）


def child_env() -> dict:
    """起子进程时套上这个 env —— **让子进程按 UTF-8 说话**。

    与上面 `ensure_console_encoding()` 是同一件事的两半，所以放在一起：那一半
    管「本进程输出时别因为编码装不下符号而崩」，这一半管「父进程按什么编码去读
    子进程的话」。**两半必须对齐**，否则字节在管道里对不上。

    对不上会怎样，这个项目踩过两次，两种表现都留在这里：

    1. **崩掉**（`repro_check.py` 第三次翻车）：父进程写的是
       `subprocess.run(..., text=True, encoding="utf-8")`，而子进程被重定向到
       管道时按本机 locale（Windows 中文 = cp936/GBK）输出。解码发生在
       `subprocess` 的**读取线程**里，那个异常不让 `run()` 抛，只让
       `proc.stdout` 变成 `None`，最后报成 `'NoneType' object has no attribute
       'splitlines'` —— 指向一个跟真正原因毫无关系的对象。
    2. **不崩，但记录被毁**（`run_seeds.py` 的臂清单，2026-10-04 入库前发现）：
       那一处多写了 `errors="replace"`，于是异常没了、中文被逐个换成 `U+FFFD`
       —— 实测三支臂的 2 KB 日志尾巴里各有 **630 处**替换符，中文不可还原。
       「跑得动」把「这份记录已经废了」盖住了。

    修法**不动父进程的解码**：父进程要读的中文，用 `errors="replace"` 兜住就
    永远读不回来了（从「崩掉」退化成「静默少一行」，比崩掉更坏）。正解是让子
    进程真的说 UTF-8 —— `PYTHONIOENCODING` 是 Python 自己认的开关，对推演行为
    **零影响**（只换输出编码），拿它对齐不引入任何混淆变量。

    **只此一处定义。** 原先只有 `repro_check.py` 自己有一份（它第三次翻车的
    修法），而 `run_seeds.py` 起子进程处另写了一遍 `encoding="utf-8"` 却没有这
    个 env —— 同一个契约两份实现，跟上修的正好是不入库的那一份。与 `rel_path`
    是同一个病：**同一件事抄两份，必然有一份没跟着修。**
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def ensure_log_handler_encoding() -> int:
    """把**已经建好的**日志 handler 的流也改成 `errors="replace"`。返回改了几个。

    **为什么 `ensure_console_encoding()` 不够。** 它只覆盖 `sys.stdout`/`sys.stderr`。
    而 oasis 自建了一个 `logging.FileHandler` 却**没传 `encoding`**
    （`oasis/social_agent/agent.py:44`）—— 同一个库里 `oasis/environment/env.py:40`
    就老老实实传了 `encoding="utf-8"`，所以这是上游的疏漏，不是设计。Windows 下
    `FileHandler` 默认取 `locale` 编码 = GBK，于是 agent 只要在帖文里发一个 emoji：

        --- Logging error ---
        UnicodeEncodeError: 'gbk' codec can't encode character '\\U0001f90d'

    `logging` 会吞掉这个异常并**把那一行丢掉**。后果有两层，第二层更要命：
    日志里冒出一段吓人的堆栈（录演示视频时看着像崩了），**而真正的那条记录没了**。
    实测 27 agent × 15 轮第一次跑，第 0 轮就撞上，且不是偶发 —— agent 发 emoji 很常见。

    修法沿用 `ensure_console_encoding()` 已定的策略：**只动 errors 策略、不动 encoding**
    —— 中文仍按原编码正确写入，装不下的符号退化成 U+FFFD，而不是让整条日志消失。

    **必须在 oasis / camel 被 import 之后调用** —— handler 是它们 import 时建的。
    只处理当时已有的 handler，所以调用点要放在 import 之后而不是 `main()` 开头。
    对已经 `errors="replace"` 的流是空操作，可重复调用。
    """
    import logging

    fixed = 0
    loggers = [logging.getLogger()] + [
        logging.getLogger(name) for name in list(logging.root.manager.loggerDict)
    ]
    for lg in loggers:
        for h in list(lg.handlers):
            stream = getattr(h, "stream", None)
            if stream is None or getattr(stream, "errors", None) == "replace":
                continue
            try:
                stream.reconfigure(errors="replace")
                fixed += 1
            except (AttributeError, ValueError, OSError):
                # 不是 TextIOWrapper（StringIO、socket 流、已关闭的流等）。
                # **不报错**：这里的目标是「能改的改掉」，不是「保证全都改掉」。
                continue
    return fixed


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析。

    支持：`KEY=VALUE`、`#` 注释、值两侧的成对引号、空行。
    不支持：多行值、变量插值、export 前缀 —— 本项目用不到，不引入复杂度。

    刻意不引入 python-dotenv：需求只有十几行，少一个依赖就少一个安装失败点。

    已存在的环境变量**不被覆盖** —— 这样临时用 `LLM_MODEL_NAME=x python ...`
    调试时，命令行传入的值优先于文件。
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI 兼容的对话补全端点。"""

    api_key: str
    base_url: str
    model: str
    timeout: float = 120.0
    # 是否发送 `thinking` 参数以控制推理开关。实测本项目所用端点（DeepSeek）
    # 支持该参数，且**只有这个参数真的能关掉推理**（见 llm.py 模块注释）。
    # 但并非所有 OpenAI 兼容端点都认识它，不认识会直接 400 ——
    # 换端点时若报 400，在 .env 里设 LLM_SEND_THINKING_PARAM=0。
    send_thinking_param: bool = True

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def describe(self) -> str:
        """用于日志与《开源及第三方资源使用清单》——**不含密钥**。"""
        return f"{self.model} @ {self.base_url}"


@dataclass(frozen=True)
class EmbeddingConfig:
    """嵌入模型端点。可与对话模型共用同一个 key 与 base_url。"""

    api_key: str
    base_url: str
    model: str
    timeout: float = 60.0

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/embeddings"


@dataclass(frozen=True)
class SimulationConfig:
    """仿真规模。这两个值是控制成本的旋钮，改动前先看 进度.md 的成本记录。

    **它们曾经是死的**：字段建好了、`.env.example` 也写了，但没有任何消费方 ——
    设了不生效、不报错、不降级。这是本项目专门猎杀的那类「静默失效」，
    所以现在由 `simulate.py` 的 `--rounds` / `--agents` 默认值消费（显式传参优先）。
    """

    max_rounds: int = 10
    max_agents: int = 30


#: 展示层允许绑定的地址。**只此一处定义**：`config.load_config` 与
#: `viewer.main` 的 `--host` 都要过这一关，两个入口共用一个集合，
#: 免得「`.env` 里挡住了、命令行又绕过去」。
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


@dataclass(frozen=True)
class ViewerConfig:
    """展示层的监听地址。**只准绑回环。**

    `FLASK_HOST` / `FLASK_PORT` 两个键此前与 `OASIS_*` 一样是**死的**：写在
    `.env` 里、没有任何消费方 —— 设了不生效、不报错、不降级。现在由展示层
    消费，但两半的处理不一样，理由是这一层的性质：

    - **端口**照读（`FLASK_PORT`）—— 换端口无害。
    - **地址只接受回环**（`127.0.0.1` / `localhost` / `::1`）。容器或共享
      机器上，一个 `0.0.0.0` 会把「本地查看工具」变成对局域网开放的只读服务；
      这一层没有任何鉴权，也没有理由被外部访问。所以非回环值**直接抛**，
      而不是**静默忽略** —— 静默忽略正是这个项目专门猎杀的那类失效：
      配置写了、看着生效了、实际没生效。

    校验放在 `__post_init__` 而不是 `load_config` 里，是为了让 `--host`
    这个命令行入口也过同一道闸：两份代码各查一次，迟早会有一份忘了查。
    """

    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        if self.host.strip().lower() not in LOOPBACK_HOSTS:
            raise ConfigError(
                f"监听地址只能是回环（{' / '.join(LOOPBACK_HOSTS)}），"
                f"当前值 {self.host!r}。\n"
                "  来源是 .env 的 FLASK_HOST 或命令行的 --host。\n"
                "  展示层没有鉴权，绑到外部地址会把「本地查看工具」变成对局域网"
                "开放的只读服务。\n"
                "  要给别人看，请让对方在本机跑同一条命令，而不是把这一层暴露出去。"
            )

    @property
    def is_loopback(self) -> bool:
        return self.host.strip().lower() in LOOPBACK_HOSTS


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    embedding: EmbeddingConfig
    simulation: SimulationConfig
    data_dir: Path
    viewer: ViewerConfig = ViewerConfig()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "weiran.db"


def load_config(*, require_llm: bool = True, require_embedding: bool = False) -> Config:
    """读取配置。

    Args:
        require_llm: 需要对话模型。凡是会调 LLM 的入口都该传 True。
        require_embedding: 需要嵌入模型。仅向量检索相关入口需要。

    Raises:
        ConfigError: 缺少必需项。消息会**一次性列出全部**缺失项，
            避免「修一个报一个」的来回。
    """
    _load_dotenv(REPO_ROOT / ".env")

    missing: list[str] = []

    llm_model = os.environ.get("LLM_MODEL_NAME", "").strip()
    if require_llm and not llm_model:
        missing.append(
            "LLM_MODEL_NAME 未设置 —— 在 .env 中填入对话模型 id"
            "（形如 'Qwen/Qwen2.5-72B-Instruct'，取决于所用服务商）"
        )
    llm_key = os.environ.get("LLM_API_KEY", "").strip()
    if require_llm and not llm_key:
        missing.append("LLM_API_KEY 未设置 —— 在 .env 中填入服务商密钥")

    emb_model = os.environ.get("EMBEDDING_MODEL", "").strip()
    emb_key = os.environ.get("EMBEDDING_API_KEY", "").strip() or llm_key
    if require_embedding and not emb_model:
        missing.append("EMBEDDING_MODEL 未设置 —— 在 .env 中填入嵌入模型 id")
    if require_embedding and not emb_key:
        missing.append(
            "EMBEDDING_API_KEY 未设置，且 LLM_API_KEY 也为空 —— 二者至少填一个"
        )

    # `LLM_BASE_URL` 和上面三项是同一类：没填就起不来。它原先在聚合检查**之后**
    # 单独抛，于是它会晚一轮才露面 —— 先把模型名和密钥补齐、再跑一次，才轮到它。
    # 那正是本函数承诺不做的「修一个报一个」，所以把它并进同一个 `missing`。
    llm_base = os.environ.get("LLM_BASE_URL", "").strip()
    if require_llm and not llm_base:
        missing.append(
            "LLM_BASE_URL 未设置 —— OpenAI 兼容端点的根地址，"
            "形如 'https://api.deepseek.com/v1'（注意结尾的 /v1）"
        )

    if missing:
        raise ConfigError(
            "配置不完整，缺少以下 "
            + str(len(missing))
            + " 项：\n"
            + "\n".join(f"  {i}. {m}" for i, m in enumerate(missing, 1))
            + f"\n\n配置文件位置：{REPO_ROOT / '.env'}"
            + "\n（可从 .env.example 复制：cp .env.example .env）"
        )

    def _int(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} 必须是整数，当前值 {raw!r}") from exc

    def _bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name, "").strip().lower()
        if not raw:
            return default
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"{name} 必须是布尔值（1/0/true/false），当前值 {raw!r}")

    emb_base = os.environ.get("EMBEDDING_BASE_URL", "").strip() or llm_base

    flask_host = os.environ.get("FLASK_HOST", "").strip() or ViewerConfig.host

    return Config(
        llm=LLMConfig(
            api_key=llm_key,
            base_url=llm_base,
            model=llm_model,
            send_thinking_param=_bool("LLM_SEND_THINKING_PARAM", True),
        ),
        embedding=EmbeddingConfig(api_key=emb_key, base_url=emb_base, model=emb_model),
        simulation=SimulationConfig(
            max_rounds=_int("OASIS_DEFAULT_MAX_ROUNDS", 10),
            max_agents=_int("OASIS_MAX_AGENTS", 30),
        ),
        data_dir=DEFAULT_DATA_DIR,
        viewer=ViewerConfig(host=flask_host, port=_int("FLASK_PORT", ViewerConfig.port)),
    )
