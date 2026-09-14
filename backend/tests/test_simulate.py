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

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weiran.simulate import (  # noqa: E402
    TW_HIN_REPO,
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
