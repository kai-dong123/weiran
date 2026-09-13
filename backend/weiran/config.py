"""「未然」运行配置。

配置来源优先级：真实环境变量 > 仓库根目录的 .env > 默认值。

设计取向：**缺失的配置必须在第一时间、以能读懂的方式报错。**
这类项目最常见的失败不是算法错，而是跑到一半发现某个 key 是空的。
所以这里不做「宽容降级」——该有的没有，就立刻停下来，并说清楚缺什么、怎么补。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# backend/weiran/config.py -> backend/weiran -> backend -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """配置缺失或非法。消息里必须包含「缺什么」与「怎么补」。"""


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
    """仿真规模。这三个值是控制成本的旋钮，改动前先看 进度.md 的成本记录。"""

    max_rounds: int = 10
    max_agents: int = 30


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    embedding: EmbeddingConfig
    simulation: SimulationConfig
    host: str
    port: int
    data_dir: Path

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

    llm_base = os.environ.get("LLM_BASE_URL", "").strip()
    if require_llm and not llm_base:
        raise ConfigError(
            "LLM_BASE_URL 未设置 —— OpenAI 兼容端点的根地址，"
            "形如 'https://api.siliconflow.cn/v1'（注意结尾的 /v1）"
        )

    emb_base = os.environ.get("EMBEDDING_BASE_URL", "").strip() or llm_base

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
        host=os.environ.get("FLASK_HOST", "127.0.0.1").strip() or "127.0.0.1",
        port=_int("FLASK_PORT", 5001),
        data_dir=REPO_ROOT / "data",
    )
