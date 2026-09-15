"""批次驱动的测试。**全部离线：不联网、不调 LLM、不跑真推演。**

这一套守的是「**起一批臂的纪律是不是由程序强制**」，不是「循环写得对不对」。

起一批臂本身只是一行循环；难的是「跑完之后这批臂**是**一批可比的臂」不靠
人记得。所以这里的重点全在**拒绝**上 —— 而且每一条拒绝都配了它自己的反例
（`--force`、`--allow-small-scale`）与一条「正常路径能过」的对照，否则一个
「什么都拒绝」的实现也能全绿。

**一条真推演都不跑。** `run_arm` 要起子进程，测试里把 `subprocess.run`
换成替身；替身只做两件事：写一份最小的产出、返回一个退出码。于是这一套
可以放进 CI，而它验的正是真正容易坏的那一半 —— 输入有没有落地、sha256
有没有核对、失败有没有停下。

    python backend/tests/test_run_seeds.py
    pytest backend/tests/
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_seeds as RS  # noqa: E402
from weiran.config import REPO_ROOT, ensure_console_encoding  # noqa: E402

SHIPPED = REPO_ROOT / "data" / "simulation" / "twitter_rounds.json"


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def _master(root: Path) -> Path:
    """一份底本：三个输入各一个文件，内容随意 —— 本套测的是「有没有按
    sha256 钉住」，不是内容对不对。"""
    master = root / "master"
    master.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(RS.INPUT_FILES):
        (master / name).write_bytes(f"content-{i}-{name}".encode())
    return master


def _options(seeds, **over) -> dict:
    base = dict(agents=27, rounds=15, chunking_guard=False,
                events_from="phases", platform="twitter", scenario="s",
                temperature=0.7, no_world_state=False, no_phases=False,
                no_knowledge=False, no_feedback=False)
    base.update(over)
    return {f"seed{s}": dict(base) for s in seeds}


def _plan(seeds, master: Path, root: Path, *, force=False, allow_small=False,
          agents=27, argv=None, options=None):
    return RS.plan_batch(list(seeds), root, argv or ["--agents", str(agents)],
                         options or _options(seeds), master=master,
                         allow_small=allow_small, agents=agents, force=force)


class _FakeRun:
    """`subprocess.run` 的替身。**不联网、不起真进程。**

    `fail_on` 指定第几次调用返回非零 —— 用来测「失败就停下」。
    """

    def __init__(self, *, fail_on: int | None = None, write_artifact=True):
        self.calls: list[list[str]] = []
        self.fail_on = fail_on
        self.write_artifact = write_artifact

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        index = len(self.calls)
        rc = 1 if index == self.fail_on else 0
        if rc == 0 and self.write_artifact:
            out = Path(cmd[cmd.index("--out") + 1])
            (out / "twitter_rounds.json").write_text(json.dumps({
                "meta": {"agents": 27, "rounds": 15, "platform": "twitter",
                         "days_per_round": 1.0, "compressed": False,
                         "events_from": "phases", "phases_on": True,
                         "knowledge_on": True, "feedback_on": True,
                         "world_state": True, "chunking_guard": False,
                         "temperature": 0.7},
                "rounds": [{"index": 0, "state": {}, "behaviors": []}],
            }, ensure_ascii=False), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, rc, stdout="ok", stderr="")


def _with_fake(fn, fake: _FakeRun):
    real = RS.subprocess.run
    RS.subprocess.run = fake
    try:
        return fn()
    finally:
        RS.subprocess.run = real


class _capture:
    """接住 stdout / stderr。

    **不用 pytest 的 `capsys`。** 本套必须与其余各套同形：既能被 pytest 跑，
    也能被文件末尾那个兜底 runner 直接跑。`capsys` 只有 pytest 给，用了它
    就等于把这一套锁死在 pytest 上 —— 而「逐套直接跑」正是本项目出过一次
    编码缺陷之后立下的规矩。
    """

    def __enter__(self):
        import contextlib
        import io
        self.out, self.err = io.StringIO(), io.StringIO()
        self._o = contextlib.redirect_stdout(self.out)
        self._e = contextlib.redirect_stderr(self.err)
        self._o.__enter__()
        self._e.__enter__()
        return self

    def __exit__(self, *exc):
        self._e.__exit__(*exc)
        self._o.__exit__(*exc)
        return False


# ---------------------------------------------------------------------------
# 1. 默认不花钱
# ---------------------------------------------------------------------------

def test_without_yes_it_spends_nothing_and_creates_nothing():
    """**默认状态是「不会花钱」。** 这条是整套的地基：一个忘了加 `--yes`
    的人不该付出任何代价。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-noyes-"))
    master = _master(root)
    fake = _FakeRun()
    out = root / "runs"

    with _capture():   # 干跑会把计划与估算印出来，接住它，别糊在测试日志里
        rc = _with_fake(lambda: RS.main([
            "--seeds", "11,22,33", "--agents", "27", "--rounds", "15",
            "--out", str(out), "--master", str(master)]), fake)

    assert rc == 0
    assert fake.calls == [], "没给 --yes 却起了进程"
    assert not out.exists(), "没给 --yes 却建了目录"


def _dry_run(root: Path, master: Path) -> str:
    """跑一次「不给 --yes」的干跑，拿回它印出来的字。"""
    with _capture() as cap:
        rc = RS.main(["--seeds", "11,22", "--agents", "27", "--rounds", "15",
                      "--out", str(root / "runs"), "--master", str(master)])
    assert rc == 0
    return cap.out.getvalue()


def test_the_dry_run_prints_an_estimate_derived_from_the_shipped_artifact():
    """不给 `--yes` 时要印出「将要跑什么、大概多少」。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-est-"))
    text = _dry_run(root, _master(root))
    assert "657.49" in text, "估算没有引用实测的那一次读数"
    assert "外推" in text


def test_the_estimate_never_invents_a_price():
    """**仓库里没有写死单价，所以一个金额都不许出现。**
    报一个编出来的数字比不报更坏。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-price-"))
    text = _dry_run(root, _master(root))
    for token in ("元", "$", "￥", "USD", "人民币", "美元", "花费"):
        assert token not in text, f"估算里出现了金额字样 {token!r}"


def test_the_estimate_says_it_is_an_extrapolation():
    """外推必须**标明**是外推 —— 它是一条直线，而真实值受服务端排队影响。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-ext-"))
    assert "不是测量" in _dry_run(root, _master(root))


def test_baseline_is_read_from_the_shipped_artifact():
    got = RS.measure_baseline(SHIPPED)
    assert got is not None
    assert got["agents"] == 27 and got["rounds"] == 15
    assert got["seconds"] and got["prompt_tokens"] > 0


def test_a_missing_baseline_says_so_instead_of_guessing():
    """底本不在时**不许猜**墙钟与 token。"""
    lines = RS.format_estimate(None, n_arms=2, agents=27, rounds=15)
    text = "\n".join(lines)
    assert "无法外推" in text
    assert not any(c.isdigit() and c != "2" for c in text.replace("27", "")
                   .replace("15", "")), text


# ---------------------------------------------------------------------------
# 2. 拒绝：不该跑的一律不跑
# ---------------------------------------------------------------------------

def test_small_scale_is_refused_without_the_explicit_flag():
    """相位读数只在 27 agent 上有效。小规模跑出来的臂**不能用来判相位**，
    拿它们汇总会得到一条看着有、其实是假的驱动峰。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-small-"))
    master = _master(root)
    try:
        _plan([11, 22], master, root / "runs", agents=12, allow_small=False)
    except RS.BatchRefused as exc:
        assert "27" in str(exc) and "相位" in str(exc)
    else:
        raise AssertionError("12 agent 没开口子却放过了")


def test_small_scale_is_allowed_with_the_flag():
    """反例：给了口子就放过。**没有这条，「什么都拒绝」也能全绿。**"""
    root = Path(tempfile.mkdtemp(prefix="weiran-small2-"))
    master = _master(root)
    plans = _plan([11, 22], master, root / "runs", agents=12, allow_small=True)
    assert len(plans) == 2


def test_writing_into_the_shipped_artifact_home_is_refused():
    """`data/simulation` 是入库那份产出的家，`repro_check.py` 第 2.5 / 2.6 层
    对它逐字锁定。臂一律不许写进去。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-ship-"))
    master = _master(root)
    for target in (RS.SHIPPED_OUT, RS.SHIPPED_OUT / "sub"):
        try:
            _plan([11], master, target)
        except RS.BatchRefused as exc:
            assert "simulation" in str(exc) or "入库" in str(exc)
        else:
            raise AssertionError(f"{target} 没被拦下")


def test_duplicate_seeds_are_refused():
    """同一个 seed 跑两次只会把跨 seed 离散往零拉。要验一致性是 --preflight。"""
    try:
        RS.parse_seeds("11,22,11")
    except RS.BatchRefused as exc:
        assert "重复" in str(exc) and "preflight" in str(exc)
    else:
        raise AssertionError("重复的 seed 没被拦下")


def test_a_non_integer_seed_is_refused():
    try:
        RS.parse_seeds("11,abc")
    except RS.BatchRefused as exc:
        assert "abc" in str(exc)
    else:
        raise AssertionError("非整数种子没被拦下")


def test_a_missing_master_input_is_refused():
    """缺一个输入就拒绝 —— 「臂之间输入同源」这句话需要三个都在。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-nomaster-"))
    master = _master(root)
    (master / RS.INPUT_FILES[0]).unlink()
    try:
        RS.hash_master(master)
    except RS.BatchRefused as exc:
        assert RS.INPUT_FILES[0] in str(exc)
    else:
        raise AssertionError("底本缺件没被拦下")


# ---------------------------------------------------------------------------
# 3. 拒绝：覆盖已有的臂
# ---------------------------------------------------------------------------

def test_an_existing_arm_with_a_different_seed_is_not_clobbered():
    """**覆盖是不可逆的** —— 那支臂花过钱，而 `simulate` 没有断点续跑。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-clob-"))
    master = _master(root)
    runs = root / "runs"
    (runs / "seed11").mkdir(parents=True)
    (runs / "seed11" / "arm.json").write_text(
        json.dumps({"label": "seed11", "seed": 99, "agents": 27}),
        encoding="utf-8")

    try:
        _plan([11, 22], master, runs)
    except RS.BatchRefused as exc:
        assert "--force" in str(exc)
    else:
        raise AssertionError("口径不符的已有臂被覆盖了")


def test_force_allows_re_running_that_arm():
    root = Path(tempfile.mkdtemp(prefix="weiran-force-"))
    master = _master(root)
    runs = root / "runs"
    (runs / "seed11").mkdir(parents=True)
    (runs / "seed11" / "arm.json").write_text(
        json.dumps({"label": "seed11", "seed": 99, "agents": 27}),
        encoding="utf-8")
    plans = _plan([11, 22], master, runs, force=True)
    assert [p.label for p in plans] == ["seed11", "seed22"]


def test_a_non_empty_directory_without_a_manifest_is_not_clobbered():
    """没有 `arm.json` 的非空目录**可能是别的什么东西** —— 不覆盖。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-junk-"))
    master = _master(root)
    runs = root / "runs"
    (runs / "seed11").mkdir(parents=True)
    (runs / "seed11" / "notes.txt").write_text("x", encoding="utf-8")
    try:
        _plan([11], master, runs)
    except RS.BatchRefused as exc:
        assert "arm.json" in str(exc)
    else:
        raise AssertionError("没有 arm.json 的非空目录被覆盖了")


def test_a_continuation_with_the_same_seed_and_scale_is_allowed():
    """反例：口径对得上就是续跑，不是覆盖。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-cont-"))
    master = _master(root)
    runs = root / "runs"
    (runs / "seed11").mkdir(parents=True)
    (runs / "seed11" / "arm.json").write_text(
        json.dumps({"label": "seed11", "seed": 11, "agents": 27}),
        encoding="utf-8")
    assert len(_plan([11], master, runs)) == 1


# ---------------------------------------------------------------------------
# 4. 拒绝：那不是一批
# ---------------------------------------------------------------------------

def test_arms_differing_in_chunking_guard_are_refused():
    """一批里混进一支 `--chunking-guard` 不同的臂，它的记忆会被切成
    1 token 一块、膨胀约 20 倍。**两条口径的曲线不可混用，而目录名上看不出来。**"""
    root = Path(tempfile.mkdtemp(prefix="weiran-mix-"))
    master = _master(root)
    options = _options([11, 22])
    options["seed22"]["chunking_guard"] = True
    try:
        _plan([11, 22], master, root / "runs", options=options)
    except RS.BatchRefused as exc:
        assert "chunking_guard" in str(exc)
    else:
        raise AssertionError("口径不一致的一批没被拦下")


def test_arms_differing_in_rounds_are_refused():
    root = Path(tempfile.mkdtemp(prefix="weiran-mix2-"))
    master = _master(root)
    options = _options([11, 22])
    options["seed22"]["rounds"] = 10
    try:
        _plan([11, 22], master, root / "runs", options=options)
    except RS.BatchRefused as exc:
        assert "rounds" in str(exc)
    else:
        raise AssertionError("轮数不一致的一批没被拦下")


def test_a_changed_master_since_the_last_batch_is_refused():
    """跨批次比较的前提是输入同源。**换过底本就得知道** ——
    否则新臂与旧臂的差异不止是 seed。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-batch-"))
    runs = root / "runs"
    runs.mkdir()
    (runs / "batch.json").write_text(json.dumps({
        "master": {RS.INPUT_FILES[0]: "deadbeef"}}), encoding="utf-8")
    warning = RS.check_against_previous_batch(
        runs, {RS.INPUT_FILES[0]: "aaa", RS.INPUT_FILES[1]: "b",
               RS.INPUT_FILES[2]: "c"})
    assert "换过" in warning
    assert RS.check_against_previous_batch(
        runs, {RS.INPUT_FILES[0]: "deadbeef"}) == ""


# ---------------------------------------------------------------------------
# 5. 顺序跑、不并行
# ---------------------------------------------------------------------------

def test_the_driver_has_no_parallel_option():
    """**顺序跑是刻意的**：墙钟是一个被记录的量，并行会把它变成一个
    没人能解释的数。所以这两个开关必须不存在。"""
    for flag in ("--jobs", "--parallel", "--workers"):
        with _capture():   # argparse 会把 usage 打到 stderr，接住它
            try:
                RS.main(["--seeds", "11", flag, "4"])
            except SystemExit as exc:
                assert exc.code == 2, flag
            else:
                raise AssertionError(f"{flag} 被接受了")


def test_the_driver_imports_no_concurrency_primitive():
    import ast
    src = Path(__file__).resolve().parents[1] / "run_seeds.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"threading", "multiprocessing", "concurrent", "asyncio", "joblib"}
    assert not (imported & banned), imported & banned


# ---------------------------------------------------------------------------
# 6. 每支臂：输入落地、指纹、--out
# ---------------------------------------------------------------------------

def test_every_arm_always_gets_an_explicit_out():
    """**这条是地基。** `repro_check.py` 那次事故的成因就是「没传 `--out`」——
    没传就走默认值，而默认值正好指向那份入库产出。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-out-"))
    master = _master(root)
    for plan in _plan([11, 22], master, root / "runs"):
        assert "--out" in plan.argv
        assert Path(plan.argv[plan.argv.index("--out") + 1]) == plan.out_dir
        assert plan.out_dir != RS.SHIPPED_OUT
        assert str(RS.SHIPPED_OUT) not in str(plan.out_dir)


def test_each_arm_gets_its_own_copy_of_all_three_inputs():
    """`simulate` 是从 `out_dir` 里读这三个输入的（`simulate.py:1092` /
    `:1121` / `:1276`），所以「逐臂各拷一份」不是洁癖 —— 那是唯一能把
    输入钉住的办法。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-copy-"))
    master = _master(root)
    fake = _FakeRun()
    plans = _plan([11, 22], master, root / "runs")

    for plan in plans:
        _with_fake(lambda p=plan: RS.run_arm(p, master, RS.hash_master(master)),
                   fake)
    for plan in plans:
        for name in RS.INPUT_FILES:
            assert (plan.out_dir / name).is_file(), name
            assert (plan.out_dir / name).read_bytes() == \
                (master / name).read_bytes()


def test_the_manifest_records_the_input_fingerprints_and_the_command():
    """`arm.json` 要能回答「这支臂是用哪份输入、哪条命令跑的」——
    缺了它，一支臂与另一支臂的差异就无从归因。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-manifest-"))
    master = _master(root)
    hashes = RS.hash_master(master)
    fake = _FakeRun()
    plan = _plan([11], master, root / "runs")[0]
    _with_fake(lambda: RS.run_arm(plan, master, hashes), fake)

    got = json.loads((plan.out_dir / "arm.json").read_text(encoding="utf-8"))
    assert got["inputs"] == hashes
    assert set(got["inputs"]) == set(RS.INPUT_FILES)
    assert got["seed"] == 11
    assert got["command"].startswith("python -m weiran.simulate")
    assert "--out" in got["command"]
    assert got["ok"] is True
    assert got["out_sha256"]


def test_the_manifest_copies_the_protocol_keys_from_the_artifact():
    """口径键抄进 `arm.json` 是为了让人一眼看出「驱动打算跑的」与
    「产出自己说的」是否一致。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-manifest2-"))
    master = _master(root)
    fake = _FakeRun()
    plan = _plan([11], master, root / "runs")[0]
    _with_fake(lambda: RS.run_arm(plan, master, RS.hash_master(master)), fake)
    got = json.loads((plan.out_dir / "arm.json").read_text(encoding="utf-8"))
    for key in ("agents", "rounds", "chunking_guard", "temperature"):
        assert key in got, key


def test_a_mismatched_copy_is_refused_before_the_run():
    """拷贝之后**逐份核对 sha256**，对不上就停。顺序不能反：先落地、再核对、
    才起进程。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-badcopy-"))
    master = _master(root)
    hashes = RS.hash_master(master)
    fake = _FakeRun()
    plan = _plan([11], master, root / "runs")[0]
    wrong = dict(hashes)
    wrong[RS.INPUT_FILES[0]] = "0" * 64
    try:
        _with_fake(lambda: RS.run_arm(plan, master, wrong), fake)
    except RS.BatchRefused as exc:
        assert "sha256" in str(exc)
    else:
        raise AssertionError("输入指纹对不上却还是跑了")
    assert fake.calls == [], "输入还没核对完就起了进程"


# ---------------------------------------------------------------------------
# 7. 一支坏了就停下
# ---------------------------------------------------------------------------

def _batch_run(root: Path, fake: _FakeRun, seeds: str = "11,22,33"):
    """跑一批（起进程那一步走替身），拿回退出码与两个流。"""
    master = _master(root)
    with _capture() as cap:
        rc = _with_fake(lambda: RS.main([
            "--seeds", seeds, "--agents", "27", "--rounds", "15",
            "--out", str(root / "runs"), "--master", str(master), "--yes"]),
            fake)
    return rc, cap.out.getvalue(), cap.err.getvalue()


def test_a_failed_arm_stops_the_batch():
    """**没有断点续跑**，所以一支坏臂不会自己好。闷头跑完只会得到一批
    从第二支开始就污染了的产出。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-fail-"))
    fake = _FakeRun(fail_on=2)
    rc, _, _ = _batch_run(root, fake)

    assert rc == 1
    assert len(fake.calls) == 2, f"第 2 支坏了却跑了 {len(fake.calls)} 支"
    assert not (root / "runs" / "seed33").exists(), "坏臂之后还在接着跑"


def test_a_failed_arm_reports_which_one_broke():
    root = Path(tempfile.mkdtemp(prefix="weiran-fail2-"))
    _, _, err = _batch_run(root, _FakeRun(fail_on=2))
    assert "seed22" in err
    assert "没跑" in err


def test_a_healthy_batch_runs_every_arm_and_writes_batch_manifest():
    """反例：都跑通时要跑满，并写 `batch.json`（跨批次输入同源的依据）。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-ok-"))
    fake = _FakeRun()
    rc, _, _ = _batch_run(root, fake)

    assert rc == 0
    assert len(fake.calls) == 3
    batch = json.loads((root / "runs" / "batch.json").read_text(encoding="utf-8"))
    assert batch["master"] == RS.hash_master(root / "master")
    assert batch["arms"] == ["seed11", "seed22", "seed33"]


def test_every_arm_invocation_passes_the_seed_and_the_out():
    """每一支臂真的把 seed 与 out 传下去了 —— 否则「多 seed」只是目录名。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-seed-"))
    master = _master(root)
    fake = _FakeRun()
    _with_fake(lambda: RS.main([
        "--seeds", "11,22", "--agents", "27", "--rounds", "15",
        "--out", str(root / "runs"), "--master", str(master), "--yes"]), fake)
    for cmd, seed in zip(fake.calls, (11, 22)):
        assert cmd[cmd.index("--seed") + 1] == str(seed)
        assert "--out" in cmd


# ---------------------------------------------------------------------------
# 8. preflight
# ---------------------------------------------------------------------------

def test_preflight_runs_the_same_seed_twice_at_a_tiny_scale():
    root = Path(tempfile.mkdtemp(prefix="weiran-pf-"))
    master = _master(root)
    fake = _FakeRun()
    tmp = root / "pf"
    tmp.mkdir()
    result = _with_fake(lambda: RS.run_preflight(
        master, RS.hash_master(master), seed=7, tmp_root=tmp), fake)
    assert len(fake.calls) == 2
    for cmd in fake.calls:
        assert cmd[cmd.index("--seed") + 1] == "7"
        assert cmd[cmd.index("--agents") + 1] == "1"
        assert cmd[cmd.index("--rounds") + 1] == "2"
    assert result["ok"] is True


def test_preflight_says_it_does_not_extrapolate_to_the_real_scale():
    """**它验的是一个当前只是假设的前提**，而结论推不到 27 agent：
    asyncio 的交错恰恰是播种覆盖不到的那一层，而 1 个 agent 几乎不触发它。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-pf2-"))
    master = _master(root)
    tmp = root / "pf"
    tmp.mkdir()
    result = _with_fake(lambda: RS.run_preflight(
        master, RS.hash_master(master), seed=7, tmp_root=tmp), _FakeRun())
    assert "这个规模" in result["caveat"] or "这个规模上" in result["caveat"]
    assert "27" in result["caveat"]


def test_preflight_says_what_to_do_when_it_fails():
    """不一致时的下一步必须写在结果里：**先别跑那批 27 agent 的臂。**"""
    root = Path(tempfile.mkdtemp(prefix="weiran-pf3-"))
    master = _master(root)
    tmp = root / "pf"
    tmp.mkdir()
    result = _with_fake(lambda: RS.run_preflight(
        master, RS.hash_master(master), seed=7, tmp_root=tmp), _FakeRun())
    assert "别跑" in result["if_inconsistent"]
    assert "花钱" in result["if_inconsistent"]


def test_preflight_does_not_need_yes_and_touches_no_batch_directory():
    """preflight 便宜到几乎免费，所以它**不需要 `--yes`** —— 而且它不许
    碰批次目录：它跑在临时目录里。"""
    root = Path(tempfile.mkdtemp(prefix="weiran-pf4-"))
    master = _master(root)
    runs = root / "runs"
    fake = _FakeRun()
    rc = _with_fake(lambda: RS.main([
        "--seeds", "11", "--agents", "27", "--rounds", "15",
        "--out", str(runs), "--master", str(master), "--preflight"]), fake)
    assert rc == 0
    assert len(fake.calls) == 2
    for cmd in fake.calls:
        out = cmd[cmd.index("--out") + 1]
        assert "preflight" in out
        assert str(runs) not in out
    assert not runs.exists()


def _run() -> int:
    # 兜底 runner 也要防这一条：**被测代码会往控制台印符号**（驱动印 ✓/✗
    # 与中文），Windows 中文控制台是 GBK，装不下时 print 会抛
    # UnicodeEncodeError —— 于是「有坏消息要报」的那次运行反而崩在报消息的
    # 路上，看起来像测试坏了。
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
