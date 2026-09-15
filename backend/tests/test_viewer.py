"""展示层的测试。

这一组守的是五件「页面看起来正常、其实是坏的」：

  1. **把「没记录」画成 0。** 老产出里没有缓存字段、没有在场人数。渲染时
     一个 `or 0` 就能让页面永远有值可显示 —— 而那个 0 与「真的发生过、量是
     0」在页面上长得一模一样。这一层要有唯一出口，且这个出口要被测到。
  2. **可选依赖变成必装。** flask 缺席时必须 `import weiran.viewer` 仍然成功、
     只有真起服务才报「装什么」。写成一个模块级 import，全仓的导入图就都
     挂上了 flask —— 而这件事在装了 flask 的开发机上看不出来。
  3. **只读变成可写。** 这一层的性质是「只读」：不起推演、不写文件。代码里
     一个 `write_text` 或一个 `requests.post` 就会让这句话变成假的。
  4. **页面联网。** 一个 CDN 引用在开发机上永远看不出问题（有网），在评委
     机器上可能是白屏。
  5. **页面与产物对不上。** 页面上的数必须是产物里的数，不能是渲染时算出来
     的另一个数。

1/3/4 三条用 **AST 静态检查**守 —— 这三件事都不是「某个函数写错了」，
是「某个东西一旦出现就错了」，运行时再测只能撞到已经写进去的那一处。

与前几套一样，**刻意不依赖 pytest**：普通 assert + 函数。
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weiran import viewer as V  # noqa: E402
from weiran.config import REPO_ROOT, ensure_console_encoding  # noqa: E402

PKG = Path(__file__).resolve().parents[1] / "weiran"

_cache: dict = {}


def _data() -> V.Data:
    if not _cache:
        _cache["d"] = V.Data(V.DEFAULT_BRIEF, V.DEFAULT_ROUNDS_FILE, V.DEFAULT_GOLD)
    return _cache["d"]


def _pages() -> dict[str, str]:
    if "pages" not in _cache:
        _cache["pages"] = V.render_all(_data())
    return _cache["pages"]


def _sources() -> list[Path]:
    return sorted(p for p in PKG.glob("*.py"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(tree: ast.Module) -> list[tuple[str, bool]]:
    """`(模块名, 是否在模块顶层)`。判「顶层 import」比判「有没有这个词」严格：
    延迟 import 是这一层的设计要求，不是疏漏。"""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name.split(".")[0],
                            getattr(node, "col_offset", 1) == 0))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                out.append((node.module.split(".")[0],
                            getattr(node, "col_offset", 1) == 0))
    return out


# ---------------------------------------------------------------------------
# 1. 可选依赖：只有这一层碰 flask，而且不能是顶层 import
# ---------------------------------------------------------------------------

def test_only_the_viewer_imports_flask():
    """全仓只有 `viewer.py` **导入** flask，而且它的导入是延迟的。

    这条原先数的是「哪些文件里出现过 flask 这个词」。那太糙了：`config.py`
    的注释里写着 `FLASK_HOST`，于是它被数进来，报成「依赖 flask 的模块」。
    **注释里提到一个词，和真的 import 它，是两回事** —— 被误报的后果是逼着
    大家别在注释里写键名，而少写一句说明正是本项目最不该有的那种代价。
    """
    importing = {}
    for p in _sources():
        tops = [top for mod, top in _imports(_tree(p)) if mod == "flask"]
        if tops:
            importing[p.name] = tops
    assert list(importing) == ["viewer.py"], (
        f"导入 flask 的模块不止展示层：{sorted(importing)}")
    # 非空转：上面那条在「谁都没导入」时也会通过，而这一层是**可选**依赖，
    # 本来就该有人导入。位置也一并钉死 —— 延迟导入是设计要求，不是巧合。
    assert importing["viewer.py"] == [False], (
        "viewer.py 里 flask 的导入位置不对（应为函数内延迟导入，不在模块顶层）："
        f"{importing['viewer.py']}")


def test_no_module_imports_flask_or_viewer_at_the_top_level():
    """模块顶层既不许 import flask，也不许 import viewer。

    后者更隐蔽：谁 `from . import viewer` 一下，那一条导入链就绑上了
    「必须装了 flask 才能 import」。测试与工具都只从命令行进这一层。
    """
    for p in _sources():
        for mod, top in _imports(_tree(p)):
            if not top:
                continue
            assert mod != "flask", f"{p.name} 在顶层 import 了 flask"
            if p.name != "viewer.py":
                assert mod != "viewer", f"{p.name} 在顶层 import 了 viewer"


def test_viewer_imports_without_flask_and_only_fails_when_serving():
    """**实测**：把 flask 挡住，`import weiran.viewer` 仍要成功，
    渲染五页仍要成功，只有 `build_app()` 才以非零码退出并说清怎么装。

    用真子进程测。父进程里已经在 sys.modules 缓存了的东西，改 `__import__`
    是挡不住的 —— 那样测出来的「成功」是假的。
    """
    script = """
import builtins, sys, io
# **两个流都要包**：缺 flask 的说明走 stderr，只包 stdout 的话这段中文会
# 被父进程按 UTF-8 解码成乱码，于是断言看到的是「找不到那几个字」——
# 一个编码问题被读成「提示语没写」。
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
real = builtins.__import__
def fake(name, *a, **k):
    if name.split(".")[0] in ("flask", "jinja2", "werkzeug"):
        raise ImportError("blocked for test: " + name)
    return real(name, *a, **k)
builtins.__import__ = fake
sys.modules.pop("flask", None)
import weiran.viewer as V
print("IMPORT_OK")
d = V.Data(V.DEFAULT_BRIEF, V.DEFAULT_ROUNDS_FILE, V.DEFAULT_GOLD)
print("RENDER_OK", len(V.render_all(d)))
try:
    V.build_app(d)
except SystemExit as e:
    print("EXIT", e.code)
"""
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(PKG.parent),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "IMPORT_OK" in out, f"flask 缺席时 import 就炸了：\n{out}"
    assert "RENDER_OK 5" in out, f"渲染不该依赖 flask：\n{out}"
    assert "EXIT 3" in out, f"缺 flask 时应以非零码退出，实得：\n{out}"
    assert "pip install flask" in out, "退出时没说清怎么装"
    assert "可选依赖" in out, "没说明它只是可选依赖"


# ---------------------------------------------------------------------------
# 2. 只读：不写文件、不发网络请求、不调 LLM
# ---------------------------------------------------------------------------

def test_viewer_never_writes_or_calls_out():
    """AST 上不许出现写文件与出网的调用。

    「只读」是这一层的性质：一个 `open(..., "w")` 或一个 `requests.post`
    就能把它变成「看一眼就会改产物」。静态检查在这里比运行时有用 ——
    运行时只能撞到已经发生的那一次。
    """
    banned_attrs = {"write_text", "write_bytes", "mkdir", "unlink", "rmdir",
                    "post", "put", "delete", "urlopen", "urlretrieve"}
    banned_mods = {"requests", "urllib", "httpx", "socket", "shutil",
                   "weiran.simulate", "simulate"}
    for node in ast.walk(_tree(PKG / "viewer.py")):
        if isinstance(node, ast.Attribute):
            assert node.attr not in banned_attrs, \
                f"viewer.py 调用了 {node.attr} —— 这一层是只读的"
        elif isinstance(node, ast.Name):
            assert node.id not in {"open", "eval", "exec"}, \
                f"viewer.py 用了 {node.id}"
    for mod, _top in _imports(_tree(PKG / "viewer.py")):
        assert mod not in banned_mods, f"viewer.py 导入了 {mod}"


def test_only_get_routes_are_registered():
    """路由只注册 GET。**POST 一律 405**，且 405 的正文要说清这是设计。"""
    src = (PKG / "viewer.py").read_text(encoding="utf-8")
    assert 'methods=["GET"]' in src
    for verb in ('methods=["POST"]', 'methods=["PUT"]', 'methods=["DELETE"]'):
        assert verb not in src, f"注册了 {verb} —— 这一层只读"
    assert "只接受 GET" in src


def test_the_trigger_is_present_but_disabled():
    """预留的接口必须在页面上看得见，且**灰着**、带代价。"""
    page = _pages()["/"]
    assert V.PLANNED_TRIGGER_ROUTE in page, "预留路由没印在页面上"
    assert "<button disabled" in page, "按钮没有被灰置"
    assert "657" in page and "4377107" in page, "没把实测代价写在按钮旁边"
    assert "没有接任何启动逻辑" in page, "没说清这一轮它不工作"


# ---------------------------------------------------------------------------
# 3. 不联网、不引前端构建
# ---------------------------------------------------------------------------

def test_pages_have_no_external_reference():
    """页面上不许有任何外部引用：没有 script、没有 CDN、没有外链。"""
    for path, body in _pages().items():
        low = body.lower()
        assert "<script" not in low, f"{path} 里有脚本 —— 本层不引 JS"
        assert "http://" not in low and "https://" not in low, \
            f"{path} 里有外部 URL"
        assert "cdn" not in low, f"{path} 里出现了 cdn"
        assert "@import" not in low and "<link" not in low, \
            f"{path} 引了外部样式表"
        assert "src=" not in low, f"{path} 里有外链资源"


def test_charts_are_server_rendered_svg():
    """图必须是服务端生成的内联 SVG —— 不是 canvas（那要 JS），也不是图片。"""
    curves = _pages()["/curves"]
    assert curves.count("<svg") >= 6, "六维应当各有一张图"
    assert "<polyline" in curves and "<circle" in curves
    assert "<canvas" not in curves.lower()
    assert "<img" not in curves.lower()


# ---------------------------------------------------------------------------
# 4. 「没记录」不许变成 0
# ---------------------------------------------------------------------------

def test_unrecorded_fields_render_as_unrecorded_not_zero():
    """**核心用例**：把字段设成 None，页面上必须是「未记录」。

    这条产出里缓存字段与逐轮人数都是 None（老产出没有这些键）。页面若写成
    `r.get(k) or 0`，就会显示 0 —— 而「端点没报缓存」与「报了、一次都没命中」
    是两件事，前者要去查端点，后者要去查提示词前缀。
    """
    rounds = _pages()["/rounds"]
    assert rounds.count("未记录") >= 10, \
        f"逐轮页只出现 {rounds.count('未记录')} 处「未记录」—— 像是把空值当 0 了"
    assert '<span class="unrecorded">未记录</span>' in rounds


def test_none_is_rendered_by_the_single_exit():
    """`_v(None)` 与 `_v(0)` 必须是两个不同的字串 —— 这是那个唯一出口。"""
    assert "未记录" in V._v(None)
    assert "未记录" not in V._v(0)
    assert V._v(0) == "0"
    assert V._v(False) == "否", "布尔要显示成是/否，不能显示成 0/1"
    # 浮点按四位小数印 —— 与产物里 `round(v, 4)` 的位数一致，页面上的数
    # 因此能逐字回到 JSON 里去找。**不在这里挑舍入规则**：先按同一套格式化
    # 比一遍，再单点确认「0.0000 不会被吞成 0」，那才是这一层要守的事。
    for x in (0.0, 0.44475, 0.6843, 1.0, 0.123456789):
        assert V._v(x) == f"{x:.4f}", f"{x} 的印法与产物不一致"
    assert V._v(0.0) == "0.0000", "贴界值 0.0 要印成 0.0000，与「没记录」分得开"


def test_missing_artifact_says_so_instead_of_showing_a_blank_page():
    """缺产物时页面要说「显示不了」，不能是空白，也不能假装有数。"""
    with tempfile.TemporaryDirectory() as tmp:
        d = V.Data(Path(tmp) / "no_brief.json", V.DEFAULT_ROUNDS_FILE,
                   Path(tmp) / "no_gold.json")
        pages = V.render_all(d)
        for path in ("/", "/curves", "/rounds", "/windows"):
            assert "这一页显示不了" in pages[path], f"{path} 缺产物时没有说明"
        assert "python -m weiran.brief" in pages["/"], "没说怎么生成"
        gold = pages["/gold"]
        assert "这一页显示不了" in gold
        assert "python -m weiran.gold_check" in gold
        # 缺的是 brief 与 gold，rounds 仍在 —— 那块说明里不该出现 rounds。
        # 只切出那段来看：页脚一直列着三份文件的名字，拿整页去断言
        # 会把「说对了」测成「说错了」。
        block = pages["/"].split("这一页显示不了", 1)[1].split("</section>", 1)[0]
        assert "twitter_rounds.json" not in block, \
            "缺的是 brief 与 gold，说明里却提到了 rounds"
        assert "brief.json" in block, "没说是哪一份缺"
        # 概览页不读金标对照表，所以它也不该在这里提金标。
        assert "gold_check.json" not in block, \
            "概览页不读金标对照表，缺它不该在这一页报出来"


def test_gold_page_missing_does_not_break_the_other_pages():
    """金标对照表是后加的产物：它缺席时其余各页必须照常渲染。"""
    with tempfile.TemporaryDirectory() as tmp:
        d = V.Data(V.DEFAULT_BRIEF, V.DEFAULT_ROUNDS_FILE, Path(tmp) / "x.json")
        pages = V.render_all(d)
        for path in ("/", "/curves", "/rounds", "/windows"):
            assert "这一页显示不了" not in pages[path], f"{path} 被金标缺席带崩了"


# ---------------------------------------------------------------------------
# 5. 页面上的数是产物里的数
# ---------------------------------------------------------------------------

def test_displayed_values_come_from_the_artifact():
    """抽查：页面上的数必须是产物里的原值，不是渲染时另算的一个。"""
    b = json.loads(V.DEFAULT_BRIEF.read_text(encoding="utf-8"))
    curves = _pages()["/curves"]

    # 逐轮原值表：每一轮的每一维都要以原样出现在页面上。
    missing = []
    for r in b["rounds"]:
        for dim, val in r["state"].items():
            if f"{val:.4f}" not in curves:
                missing.append((r["index"], dim, val))
    assert not missing, f"这些原值没出现在页面上：{missing[:5]}"

    # 窗口表：实际路径与分支的信任终值。
    win = _pages()["/windows"]
    for w in b["windows"]:
        assert f"{w['branch_a']:.4f}" in win, f"{w['phase']} 的分支读数没出现"
        assert f"{w['actual_from_fork']:.4f}" in win, f"{w['phase']} 的实际读数没出现"


def test_gold_page_reports_all_three_verdicts_and_the_not_a_score_line():
    """金标页要把三态都印出来，且「这不是分数」那句话必须带出处。"""
    g = json.loads(V.DEFAULT_GOLD.read_text(encoding="utf-8"))
    page = _pages()["/gold"]
    for r in g["assertions"]:
        assert r["id"] in page, f"{r['id']} 没出现在页面上"
    for verdict, cls in V.VERDICT_CLASS.items():
        assert f'class="tag {cls}"' in page, f"{verdict} 没有对应的样式"
    assert "这不是分数" in page
    assert g["not_a_score"]["source"] in page, "没印出处"
    assert "不可采信" in page or "不能当证据" in page, "没说明「不采信」是什么意思"


def test_degradations_get_a_warning_box():
    """降级项非空时必须显眼（红框），而不是一行小字。"""
    b = json.loads(V.DEFAULT_BRIEF.read_text(encoding="utf-8"))
    page = _pages()["/"]
    assert b["degradations"], "这份产出本该有降级项"
    assert page.count('class="warn"') >= len(b["degradations"])
    assert "降级项" in page
    for d in b["degradations"]:
        assert d["kind"] in page, f"降级项 {d['kind']} 没显示"
    assert "贴界" in page and "1.31%" in page, "两条降级项的文字没印出来"


def test_footer_carries_provenance_on_every_page():
    """每一页的页脚都要能追回产生它的那次运行。"""
    b = json.loads(V.DEFAULT_BRIEF.read_text(encoding="utf-8"))
    prov = b["provenance"]
    for path, body in _pages().items():
        assert prov["rounds_sha256"] in body, f"{path} 缺产出指纹"
        assert prov["scenario_sha256"] in body, f"{path} 缺金标指纹"
        assert prov["command"] in body, f"{path} 缺生成命令"


def test_text_is_escaped():
    """产物里的文字要转义后再放进 HTML。

    产出里现在是中文正常文本，但「注入内容会进页面」这条路径是真的
    （`windows[].trigger` 来自场景文件）。一个未转义的 `<` 就能把页面结构
    打散 —— 这条测试用一份合成的产物把这件事钉住。
    """
    brief = json.loads(V.DEFAULT_BRIEF.read_text(encoding="utf-8"))
    brief["windows"][0]["question"] = '<script>alert(1)</script>&"'
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "brief.json"
        p.write_text(json.dumps(brief, ensure_ascii=False), encoding="utf-8")
        d = V.Data(p, V.DEFAULT_ROUNDS_FILE, V.DEFAULT_GOLD)
        page = V.render_all(d)["/windows"]
    assert "<script>alert(1)</script>" not in page, "没转义，脚本原样进了页面"
    assert "&lt;script&gt;" in page and "&amp;" in page


# ---------------------------------------------------------------------------
# 6. 命令行：自检这一个入口
# ---------------------------------------------------------------------------

def test_check_mode_passes_and_needs_no_flask():
    """`--check` 要能跑通并返回 0 —— 它是没有 flask 的机器上唯一的验证入口。"""
    assert V.main(["--check"]) == 0


def test_check_mode_does_not_read_the_config_at_all():
    """自检路径**不读配置** —— 这是刻意的，不是碰巧。

    读配置意味着自检要求 `.env` 存在、且 `FLASK_HOST` 合法。而「我的产物是
    不是真的」这个问题的使用场景，正是**刚 clone 下来、还没配 `.env`** 的
    机器。这里把一个**非法**的 `FLASK_HOST` 放进环境：起服务时它会立刻报错
    停下（见 `test_config.py`），自检却必须照常返回 0 —— 说明两者走的是
    两条路。
    """
    saved = os.environ.get("FLASK_HOST")
    os.environ["FLASK_HOST"] = "0.0.0.0"
    try:
        assert V.main(["--check"]) == 0, "自检因为一个只与起服务有关的配置项而失败了"
    finally:
        if saved is None:
            os.environ.pop("FLASK_HOST", None)
        else:
            os.environ["FLASK_HOST"] = saved


def test_check_mode_announces_a_missing_artifact_as_not_measured():
    """缺产物时自检**不能报通过**：那是「没测到」。"""
    with tempfile.TemporaryDirectory() as tmp:
        code = V.main(["--check", "--brief", str(Path(tmp) / "x.json"),
                       "--gold", str(Path(tmp) / "y.json")])
    assert code == 1, "缺产物却返回 0"


def test_every_route_has_a_nav_entry():
    """导航里的每一项都必须真的有那一页，反之亦然 —— 一条死链就是一个
    「点开是 404」的演示事故。"""
    pages = _pages()
    for path, _title in V._NAV:
        assert path in pages, f"导航里的 {path} 没有对应页面"
    assert {p for p, _ in V._NAV} == set(pages), "有页面不在导航里"


# -- 简易 runner（与其余测试文件保持一致）----------------------------------

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
