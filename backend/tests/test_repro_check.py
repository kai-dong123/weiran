"""可复现性自检脚本本身的测试。

**全部离线。不联网、不调 LLM、不跑仿真。**

这一套守的是**一个自检脚本把仓库搞坏**这件事 —— 不是假想，是实测撞到的：

  `repro_check.py` 的第 3 层要跑两次完整的 LLM 推演来比曲线，而它构造命令时
  **没有传 `--out`**。`weiran.simulate` 的 `--out` 默认值是
  `data/simulation/twitter_rounds.json` —— **正是那份随仓库入库的产出**。
  于是 `python repro_check.py` 跑一次，就把一次 27 agent × 15 轮（961 秒、
  756 条动作）的结果，**静默覆盖成 5 agent × 3 轮的临时结果**。

  没有任何提示。发现它是因为 `test_real_run_output_is_current` 红了，
  而那条测试守的是「落盘的简报与落盘的产出必须对得上」—— 它成了唯一的哨兵。

这个失效的形状值得记住：**验证工具销毁了它要验证的东西。**
本项目一直在防「不报错的失败」，而这是其中最难看的一种 ——
跑自检的人以为自己在让仓库更可信。

守三件事：

  1. 构造出来的命令**必须**显式带 `--out`，且不指向那份入库产出。
  2. 把入库产出当输出传进去时，**当场抛**，不靠调用方自觉。
  3. 第 3 层真的跑起来时，两次推演都落在临时目录里（用替身跑通这条路径，
     不花一分钱、不花一秒等待）。
  4. **子进程说的话，父进程读得动** —— 拿一个真会说话的真子进程测，
     不拿替身测（替身不出声，永远读得动）。见 `child_env()` 那段。

第 4 条是 2026-09-15 补的。它和 1–3 条守的是同一个东西，但坏在另一层：
前三条是「发出去的命令对不对」，这条是「发出去的字节读不读得回来」。

    python backend/tests/test_repro_check.py
    pytest backend/tests/
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import repro_check as R  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 构造出来的命令
# ---------------------------------------------------------------------------

def test_shipped_rounds_is_the_artifact_under_data_simulation():
    """先钉住常量本身。若哪天它被改到别处，下面几条守的就成了另一个文件。"""
    assert R.SHIPPED_ROUNDS == R.SIM / "twitter_rounds.json"
    assert R.SHIPPED_ROUNDS.parent.name == "simulation"


def test_sim_argv_always_passes_out():
    """**这条是地基。** 缺陷的成因就是「没传 `--out`」—— 没传就走默认值，
    而默认值正好指向那份入库产出。"""
    out = Path(tempfile.mkdtemp(prefix="weiran-test-")) / "e2e_a"
    argv = R.sim_argv(3, 5, out)
    assert "--out" in argv, f"命令里没有 --out：{argv}"
    assert argv[argv.index("--out") + 1] == str(out)
    assert "--agents" in argv and "--rounds" in argv


def test_sim_argv_passes_a_directory_not_a_json_file():
    """**第二次翻车的指纹就在这一条里。**

    第一版修法是「把 `--out` 指到临时目录」，实际传的是 `e2e_a.json` ——
    一个**文件名**。而 `--out` 的粒度是**目录**：`simulate` 从它读画像与
    知情映射，再往它下面的 `twitter_rounds.json` 写产出。传文件名进去，
    第 3 层会在读画像那一步 `FileNotFoundError`，整层跑不起来。

    所以这里断言两件事：`--out` 的取值**不是**产出文件本身
    （产出在它下面），而且它不能是个 `.json`。
    """
    out_dir = Path(tempfile.mkdtemp(prefix="weiran-test-")) / "e2e_a"
    argv = R.sim_argv(3, 5, out_dir)
    given = Path(argv[argv.index("--out") + 1])
    assert given.suffix != ".json", \
        f"--out 拿到的是文件名，而它要的是目录：{given}"
    assert given != R.SHIPPED_ROUNDS, "--out 被指到了产出文件本身"
    # 产出落在它下面 —— 这一条把「目录」与「文件」的关系钉死：
    # 若哪天 --out 的语义又变回文件，产出路径就会与 `_run_sim` 读的地方对不上。
    artifact = given / "twitter_rounds.json"
    assert artifact.parent == given and artifact.name == "twitter_rounds.json"
    assert artifact != R.SHIPPED_ROUNDS


def test_sim_argv_never_targets_the_shipped_artifact():
    """正面断言：输出目录**不在**入库产出上，也不是它所在的那个目录。"""
    out_dir = Path(tempfile.mkdtemp(prefix="weiran-test-")) / "e2e_a"
    argv = R.sim_argv(3, 5, out_dir)
    assert str(R.SHIPPED_ROUNDS) not in argv, \
        f"自检的推演输出指向了入库产出：{argv}"
    assert str(R.SHIPPED_DIR) not in argv, \
        f"自检的推演输出指向了入库产出所在的目录：{argv}"


def test_sim_argv_refuses_the_shipped_path_outright():
    """反面：真的把入库产出（或它所在的目录）传进去时必须**当场抛**。

    只靠「调用方记得传临时路径」是不够的 —— 那正是坏掉的那个假设。
    实现里留了一道兜底判断，这条测试确认它真的会响。

    **不做这个反面用例的话**，那段兜底代码就是一段没人验过的代码，
    而它在正常路径上永远不执行（所以看起来永远是对的）。

    **两个都要测**：目录那一格是第二次翻车之后补的 —— 第一版只防了
    「等于产出文件」，而真正的传入形式是「等于产出目录」。
    """
    for bad in (R.SHIPPED_ROUNDS, R.SHIPPED_ROUNDS.resolve(),
                R.SIM / "." / "twitter_rounds.json",
                R.SHIPPED_DIR, R.SHIPPED_DIR.resolve(), R.SIM / "."):
        try:
            R.sim_argv(3, 5, bad)
        except RuntimeError as exc:
            assert "twitter_rounds.json" in str(exc) or "目录" in str(exc), exc
        else:
            raise AssertionError(f"把入库产出的位置当输出传进去（{bad}）却没有报错")


# ---------------------------------------------------------------------------
# 1b. 运行目录真的能被造出来（不是「参数在场」，是「磁盘上有没有输入」）

def test_seed_run_dir_copies_every_input_a_run_needs():
    """**这一条是第二次翻车的正面检验。**

    上一次测试测的是「argv 里有没有 `--out`」，两条断言都真，
    而发出去的命令根本跑不通 —— 因为它要读的画像不在那个目录里。
    所以这里不看参数，看**磁盘**：一个刚造好的运行目录里，
    该有的输入必须有。
    """
    with tempfile.TemporaryDirectory(prefix="weiran-seed-") as tmp:
        dst = Path(tmp) / "run_a"
        copied = R._seed_run_dir(R.SHIPPED_DIR, dst)
        for name in ("twitter_profiles.csv", "actor_knowledge.json"):
            assert name in copied, f"没有复制 {name}（实得 {copied}）"
            assert (dst / name).is_file(), f"{name} 不在运行目录里"
        # 复制出来的必须是**同一份内容**，不是空文件占位。
        assert (dst / "twitter_profiles.csv").read_bytes() == \
            (R.SHIPPED_DIR / "twitter_profiles.csv").read_bytes()
        # 二次调用必须幂等（同一目录被两次运行轮流用过）。
        R._seed_run_dir(R.SHIPPED_DIR, dst)


def test_seed_run_dir_refuses_when_the_repo_is_incomplete():
    """缺必需输入时必须**抛**，不许「少复制一个然后让它去炸」。"""
    with tempfile.TemporaryDirectory(prefix="weiran-seed-") as tmp:
        tmp = Path(tmp)
        empty_src = tmp / "empty"
        empty_src.mkdir()
        try:
            R._seed_run_dir(empty_src, tmp / "run_a")
        except RuntimeError as exc:
            assert "twitter_profiles.csv" in str(exc), exc
        else:
            raise AssertionError("缺输入却没有报错")


def test_run_dir_is_where_the_artifact_lands():
    """把 `sim_argv` 和 `_seed_run_dir` 接起来看：
    argv 里的 `--out` 必须**就是**那个被喂饱了的目录。"""
    with tempfile.TemporaryDirectory(prefix="weiran-seed-") as tmp:
        run_dir = Path(tmp) / "run_a"
        R._seed_run_dir(R.SHIPPED_DIR, run_dir)
        argv = R.sim_argv(3, 5, run_dir)
        given = Path(argv[argv.index("--out") + 1])
        assert given.resolve() == run_dir.resolve(), \
            f"命令里的 --out（{given}）不是被喂饱的那个目录（{run_dir}）"
        assert (given / "twitter_profiles.csv").is_file(), \
            "--out 指向的目录里没有画像 —— 这一层跑起来会当场炸"


# ---------------------------------------------------------------------------
# 2. 第 3 层实际跑起来时落在哪
# ---------------------------------------------------------------------------

class _FakePayload(dict):
    """替身产出：形状够 `check_end_to_end` 走完，不需要真跑仿真。"""


def _fake_payload(shift: float) -> dict:
    dims = ("attention", "panic", "trust", "polarization", "risk", "stability")
    return {
        "meta": {"days_per_round": 1.0, "compressed": False},
        "rounds": [
            {"index": i,
             "behaviors": ["disclosure"] if i == 0 else ["discussion"],
             "state": {d: round(0.1 * (i + 1) + shift, 4) for d in dims}}
            for i in range(2)
        ],
    }


def test_run_sim_actually_goes_through_sim_argv():
    """**补上「谁真的把命令发出去了」这一段。**

    上面几条守的是 `sim_argv()`，而原先坏掉的是**它的调用方**（`_run_sim`
    自己拼了一份不带 `--out` 的 argv）。只测 `sim_argv` 的话，
    把 `_run_sim` 改回手拼命令、测试照样全绿 —— 那正是恒真检查的形状。

    这里拦下 `subprocess.run`，看它实际收到的 argv 是什么。
    """
    calls: list[list[str]] = []

    class _FakeProc:
        returncode = 0
        stdout = "完成：1 轮\n"
        stderr = ""

    def fake_subprocess_run(cmd, **kw):
        calls.append(cmd)
        return _FakeProc()

    real = R.subprocess.run
    R.subprocess.run = fake_subprocess_run
    try:
        with tempfile.TemporaryDirectory(prefix="weiran-sim-") as tmp:
            out_dir = Path(tmp) / "e2e_a"
            out_dir.mkdir()
            # 产出文件名由 simulate 定，`_run_sim` 读的是它。
            (out_dir / "twitter_rounds.json").write_text(
                '{"meta": {}, "rounds": []}', encoding="utf-8")
            R._run_sim(3, 5, out_dir)
    finally:
        R.subprocess.run = real

    assert len(calls) == 1, calls
    argv = calls[0]
    assert "--out" in argv, f"_run_sim 发出去的命令没有 --out：{argv}"
    assert argv[argv.index("--out") + 1] == str(out_dir)
    assert str(R.SHIPPED_ROUNDS) not in argv


# ---------------------------------------------------------------------------
# 2b. 子进程说的话，父进程读得动吗（第三次翻车）
# ---------------------------------------------------------------------------

def test_run_sim_reads_back_a_real_child_speaking_chinese():
    """**这条是拿真子进程测的，不是拿断言测的。**

    第三次翻车的形状：`_run_sim` 用 `encoding="utf-8"` 读子进程，而子进程在
    Windows 中文机器上走管道时说的是 GBK。解码发生在 `subprocess` 的读取
    线程里，失败不让 `run()` 抛，只让 `proc.stdout` 变成 `None`，
    于是报出来的是下一行的 `AttributeError: 'NoneType' object has no
    attribute 'splitlines'` —— 一个把方向指向「推演跑不起来」的报错。

    所以这里不构造替身进程（替身说话是不出声的，永远读得动），
    而是把 `sim_argv` 换成一个**真的会说话的真 Python 进程**，
    让它说一句中文，再走 `_run_sim` 完整的读法（解码 → splitlines →
    找 `"完成："` 那一行）。去掉 `env=child_env()` 这条就会红。

    子命令里带 `sys.executable` 而不是 `python`：测试要能在任意解释器下跑。
    """
    said = "完成：一句中文，编码不对就一个字都读不回来"
    real_argv = R.sim_argv
    R.sim_argv = lambda rounds, agents, out_dir: [
        sys.executable, "-c", f"print({said!r})"]
    # **先把父进程环境里的编码开关摘掉。** 不摘的话，只要跑测试的人自己
    # 设过 `PYTHONIOENCODING`（CI、`PYTHONUTF8=1`、或者我自己的命令行），
    # 子进程不靠 `child_env()` 也会说 UTF-8，这条测试就成了恒真检查 ——
    # 实测过一次：把 `env=child_env()` 整行删掉，它照样绿。
    # 摘掉之后，子进程只剩本机 locale（cp936）这一条路，开关的作用才被测到。
    saved = {k: R.os.environ.pop(k) for k in ("PYTHONIOENCODING", "PYTHONUTF8")
             if k in R.os.environ}
    try:
        with tempfile.TemporaryDirectory(prefix="weiran-sim-") as tmp:
            out_dir = Path(tmp) / "e2e_a"
            out_dir.mkdir()
            (out_dir / "twitter_rounds.json").write_text(
                '{"meta": {}, "rounds": []}', encoding="utf-8")
            payload, line = R._run_sim(3, 5, out_dir)
    finally:
        R.sim_argv = real_argv
        R.os.environ.update(saved)

    assert payload == {"meta": {}, "rounds": []}, payload
    assert line == said, (
        f"没把子进程的中文读回来：收到的是 {line!r} —— "
        "多半是 `_run_sim` 没把整体统一到 UTF-8"
    )


def test_child_env_overrides_only_the_encoding():
    """`env=` 是整份替换，不是增量 —— 只塞一个键会让子进程没了 PATH。

    所以除了「UTF-8 开关在」，还要「别的东西还在」：这条同时挡两种写法错误，
    一种是把开关丢了，一种是写成 `env={"PYTHONIOENCODING": "utf-8"}`。
    """
    env = R.child_env()
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert "PATH" in env and env["PATH"] == R.os.environ["PATH"]
    # 改的是副本，不是父进程的环境：`os.environ` 被就地改掉的话，
    # 这个自检脚本会把编码开关传染给它启动的**所有**后续子进程。
    assert env is not R.os.environ


def test_end_to_end_writes_into_the_directory_it_is_given():
    """用替身把第 3 层整条路径跑一遍，记下它把 `--out` 指到了哪。

    **同时检查那些目录被喂饱了** —— 第二次翻车的形状是「argv 看着对、
    发出去跑不通」，所以这里除了参数，还要看磁盘上有没有必需的输入。
    """
    #: 每次「发命令」的**当时**看到的磁盘状态。必须当场记：临时目录在
    #: `with` 退出时就没了，事后再断言只会看到「文件不存在」而分不清原因。
    seen: list[tuple[Path, bool, bool]] = []

    def fake_run_sim(rounds, agents, out):
        p = Path(out)
        seen.append((p, p.is_dir(), (p / "twitter_profiles.csv").is_file()))
        return _fake_payload(0.0), "完成：2 轮"

    real_run_sim = R._run_sim
    R._run_sim = fake_run_sim
    try:
        with tempfile.TemporaryDirectory(prefix="weiran-e2e-") as tmp:
            workdir = Path(tmp)
            R.check_end_to_end(3, 5, workdir)
    finally:
        R._run_sim = real_run_sim

    assert len(seen) == 2, f"第 3 层应当跑两次推演，实得 {len(seen)}"
    for p, is_dir, has_profile in seen:
        assert p.parent == workdir, f"输出落到了 {p}，不在给定的临时目录里"
        assert p.resolve() != R.SHIPPED_ROUNDS.resolve()
        assert is_dir, f"--out 拿到的是文件而不是目录：{p}"
        assert has_profile, \
            f"{p} 里没有画像 —— 这条命令发出去会当场 FileNotFoundError"


def test_end_to_end_does_not_touch_the_shipped_artifact():
    """**最终要的那一条。** 跑完一次第 3 层（替身版）之后，
    入库产出必须一个字节都没变。

    这条与上一条不重复：上一条看的是「参数传了什么」，这一条看的是
    「磁盘上发生了什么」—— 而只有后者是真正要守住的东西。
    """
    if not R.SHIPPED_ROUNDS.is_file():
        print("      （跳过：仓库里没有那份产出）")
        return
    before = R.SHIPPED_ROUNDS.read_bytes()
    before_stat = R.SHIPPED_ROUNDS.stat().st_mtime

    real_run_sim = R._run_sim
    R._run_sim = lambda rounds, agents, out: (_fake_payload(0.0), "完成：2 轮")
    try:
        with tempfile.TemporaryDirectory(prefix="weiran-e2e-") as tmp:
            R.check_end_to_end(3, 5, Path(tmp))
    finally:
        R._run_sim = real_run_sim

    assert R.SHIPPED_ROUNDS.read_bytes() == before, \
        "跑一次自检之后，仓库里那份产出被改动了"
    assert R.SHIPPED_ROUNDS.stat().st_mtime == before_stat


# ---------------------------------------------------------------------------
# 直接执行时的兜底 runner（不依赖 pytest）
# ---------------------------------------------------------------------------

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
