"""使用手册的测试：**手册里写的每一句可核的话，都要能被机器核。**

**为什么给一份 markdown 写测试。** 这个项目已经吃过一次同类的亏：
`README.md` 里写着「重写上游前端」，而仓库里没有前端；资源清单里的行数与
测试条数过期了几个月；清单声称某个键已落盘，入库产出里却没有那个键。
**文档不会自己发现自己在说谎** —— 它只会安静地漂走，然后被评委当成
「这个人不知道自己在说什么」。

所以这份测试守四件事：

  1. **命令真的存在**：手册里出现的每个 `python -m weiran.X` 都真有那个模块，
     每个 `--flag` 都真的被那个模块的 argparse 接受 —— 抄错一个字母，
     手册里那条命令就是一条**照着做会失败**的指令。
  2. **数字真的是那些数**：测试条数、降级项种类数、标记数、金标三态读数，
     全部**从源头重算**再比对。写死的数字不在此列 —— 那种测试会在数字变了
     之后才红，而它红的时候没人知道是文档错了还是代码错了。
  3. **`.env` 的键名齐了、值一个都没有**：这是红线。键名不全 → 手册不完整；
     值出现 → 密钥进作品材料。
  4. **手册不会随代码一起漂**：标记表、降级项表、三态表都是从源码里抽出来
     比对的，改了一处而没改手册，这里会红。

**全部离线。不联网、不调 LLM、不 import oasis / camel。**

    python backend/tests/test_handbook.py
    pytest backend/tests/
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weiran import brief as B                      # noqa: E402
from weiran.config import REPO_ROOT, ensure_console_encoding                # noqa: E402

HANDBOOK = REPO_ROOT / "docs" / "使用手册.md"
README = REPO_ROOT / "README.md"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
WEIRAN = REPO_ROOT / "backend" / "weiran"
TESTS = REPO_ROOT / "backend" / "tests"


def _text() -> str:
    return HANDBOOK.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 工具：从源码里抽事实
# ---------------------------------------------------------------------------

def _module_flags() -> dict[str, set[str]]:
    """每个模块的 argparse 开关集合 —— 从 AST 读，不 import（import 会跑模块）。

    只收以 `-` 开头的第一个参数：`add_argument("--out", ...)` 是开关，
    而位置参数不是。
    """
    out: dict[str, set[str]] = {}
    for p in sorted(WEIRAN.glob("*.py")):
        flags: set[str] = set()
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_argument"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and node.args[0].value.startswith("-")):
                flags.add(node.args[0].value)
        if flags:
            out[p.stem] = flags
    return out


def _degradation_table() -> set[str]:
    """§3.5 那张降级项表里列的 kind。

    **必须只取这一节**：手册里还有别的表格，`.env` 键名表的第一列同样是
    「反引号包着的词」，天真的全文正则会把 `LLM_API_KEY` 之类一起收进来，
    于是「手册写了源码里没有的降级项」这条会报出一串环境变量名。
    范围由那一节的标题划死；标题改了，这里会红 —— 那是好事。
    """
    text = _text()
    start = text.find("### 3.5")
    assert start >= 0, "手册里找不到 §3.5 —— 降级项那一节的编号变过了"
    end = text.find("\n###", start + 1)
    section = text[start:end if end > 0 else len(text)]
    return set(re.findall(r"^\|\s*`(\w+)`\s*\|", section, re.M))


def _test_counts() -> dict[str, int]:
    """每套测试里 `def test_` 的条数，直接数源码。

    不调 pytest：这一条要能在这个文件被**单独运行**时也成立
    （本项目每套测试都要求不带 pytest 也能跑）。
    """
    counts = {}
    for p in sorted(TESTS.glob("test_*.py")):
        counts[p.stem] = len(re.findall(r"^def (test_\w+)",
                                        p.read_text(encoding="utf-8"), re.M))
    return counts


def _env_keys() -> list[str]:
    """`.env.example` 里的键名。**只取键名，永不取值。**"""
    keys = []
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.append(line.partition("=")[0].strip())
    return keys


#: 名字里带这些词的环境变量 = 凭据。**只有它们的值才是红线。**
#:
#: 为什么不把「所有非空值」都当成红线：`.env.example` 里的
#: `LLM_SEND_THINKING_PARAM=1` 的值就是一个 `1`，而手册里到处都有 `1`
#: （`1.31%`、`1e-4`）—— 那样的断言只能靠把手册写残来满足，是**噪声**。
#: 端点的根地址也不是凭据：README 本来就把它当例子写着，方便读者照抄。
#: 真正必须一个字符都不许外泄的是**密钥**，所以按名字认，认错了也不会漏 ——
#: 只要名字里有 KEY/TOKEN/SECRET/PASSWORD，值就不许出现。
_CREDENTIAL_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _credential_values() -> list[tuple[str, str]]:
    """`(键名, 值)`，**只取凭据类**。

    两份文件都读：`.env.example` 是模板（值多半是空的或占位的），
    **`.env` 才是真的那份** —— 那份里的值一个都不许进文档。
    读不到 `.env` 时（CI、干净 clone）就只查模板，不报错。
    """
    out = []
    for path in (ENV_EXAMPLE, REPO_ROOT / ".env"):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if val and any(h in key.upper() for h in _CREDENTIAL_HINTS):
                out.append((key, val))
    return out


# ---------------------------------------------------------------------------
# 1. 命令真的存在
# ---------------------------------------------------------------------------

def test_every_module_the_handbook_names_really_exists():
    """手册里每个 `python -m weiran.X` 都得有那个文件。

    抄错一个字母，手册里那条命令就是一条**照着做会失败**的指令 ——
    而它看起来完全正常。
    """
    text = _text()
    named = set(re.findall(r"python -m weiran\.(\w+)", text))
    assert named, "手册里一条 `python -m weiran.X` 都没有？那它没在讲怎么用"
    have = {p.stem for p in WEIRAN.glob("*.py")}
    missing = sorted(named - have)
    assert not missing, f"手册点名了不存在的模块：{missing}（现有：{sorted(have)}）"


def test_every_flag_in_the_handbook_is_accepted_by_the_module_it_is_written_next_to():
    """手册里的 `--flag` 必须被**它所在那一行**点名的模块接受。

    只查「全仓某个模块有没有这个开关」是不够的：`--agents` 存在于
    `repro_check.py`，把它写到 `simulate` 那一行上照样是一条跑不通的命令。
    所以按行解析，逐行核对。
    """
    flags = _module_flags()
    bad: list[str] = []
    for lineno, line in enumerate(_text().splitlines(), 1):
        m = re.search(r"python -m weiran\.(\w+)(.*)", line)
        if not m:
            continue
        module, rest = m.group(1), m.group(2)
        for flag in re.findall(r"(--[a-z][a-z0-9-]*)", rest):
            if module not in flags:
                bad.append(f"L{lineno}: 模块 {module} 根本没有 argparse")
            elif flag not in flags[module]:
                bad.append(f"L{lineno}: {module} 不接受 {flag}"
                           f"（它接受：{sorted(flags[module])}）")
    assert not bad, "手册里有跑不通的命令：\n  " + "\n  ".join(bad)


def test_handbook_covers_every_module_that_has_a_command_line():
    """反过来也要守：**有命令行的模块，手册都得提一次**。

    只守「提到的都存在」会漏掉另一半 —— 新加了一个入口而手册没写，
    读者不会知道它存在。这条会把新入口逼进手册。
    """
    named = set(re.findall(r"python -m weiran\.(\w+)", _text()))
    have = set(_module_flags())
    missing = sorted(have - named)
    assert not missing, (
        f"这些模块有命令行但手册没提：{missing} —— "
        "要么补进手册，要么它们本来就不该有入口")


def test_repro_check_is_documented_with_its_own_flags():
    """`repro_check.py` 不在 `weiran/` 包里，上面几条按行解析时看不见它。

    单独守一次 —— 它是**唯一会真的把推演 CLI 拉起来**的检查，
    手册要是把它的参数写错了，验收那一层就废了。
    """
    text = _text()
    assert "python repro_check.py" in text, "手册没写怎么跑可复现性自检"
    src = (REPO_ROOT / "backend" / "repro_check.py").read_text(encoding="utf-8")
    # `[a-z0-9-]` 而不是 `[a-z-]`：`--skip-e2e` 里有个数字，
    # 少了它这条会**静默地**少收一个开关，然后在校验一行本来正确的命令时报红。
    real = set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', src))
    assert len(real) >= 3, f"只从 repro_check 里抽出 {len(real)} 个开关 —— 抽取失效"
    # 只扫**命令**行。散文里也会出现 `repro_check.py` 和 `--out`
    # （「第 3 层一度传了个文件名给 --out」），那不是一条要照着敲的命令。
    for line in text.splitlines():
        if "python repro_check.py" not in line:
            continue
        for flag in re.findall(r"(--[a-z][a-z0-9-]*)", line):
            assert flag in real, f"手册写了 repro_check {flag}，而它不接受（接受：{sorted(real)}）"


# ---------------------------------------------------------------------------
# 2. 数字真的是那些数
# ---------------------------------------------------------------------------

def test_documented_test_counts_match_the_files():
    """手册与 README 里逐套写的条数，必须与源码里 `def test_` 的条数一致。

    这些数字是本项目「有多少条测试在盯着」的唯一对外读数。
    它们过期了，读者不会知道；但它过期这件事本身就是文档漂移。

    **两个文件都要查，是因为 README 里有两处写条数的地方**（快速开始那段
    可复制的命令、与「目录结构」那棵树）。只查一处的话，另一处会安静地
    停在上一版的数字上 —— 这正是这次实际发生的事：改完树、漏了快速开始。
    """
    counts = _test_counts()
    bad = []
    for path in (HANDBOOK, README):
        text = path.read_text(encoding="utf-8")
        for name, n in sorted(counts.items()):
            # 两种写法都要认：命令块里的 `# 14/14`、树里的 `14 条`。
            # 树里那个不带 `tests/` 前缀 —— 漏了这个可选前缀，就等于
            # 没查那棵树（实测过：把树里的 `33 条` 改成 `30 条`，测试不红）。
            hits = re.findall(rf"tests/{name}\.py\s*#\s*(\d+)", text)
            hits += re.findall(rf"(?:tests/)?{name}\.py\s+(\d+)\s*条", text)
            if not hits:
                bad.append(f"{path.name}: 没写 {name} 的条数")
            for h in hits:
                if int(h) != n:
                    bad.append(f"{path.name}: {name} 写 {h}，实际 {n}")
    assert not bad, "文档里的测试条数对不上：\n  " + "\n  ".join(bad)


def test_every_standalone_runner_guards_the_console_encoding():
    """手册承诺「十二套都能不装 pytest 直接跑」，所以**每一套的兜底 runner
    都得先设好控制台编码策略**。

    这不是凭空加的要求，是一次实测：`test_repro_check.py` 单独跑时报 11/13，
    而 pytest 下 13/13 —— 两条「失败」是 `UnicodeEncodeError: 'gbk' codec
    can't encode character '\\u2705'`。被测代码（`check_end_to_end`）往控制台
    印 ✅，Windows 中文控制台装不下，于是 print 抛异常。

    **崩溃的是「有坏消息要报」的那次运行** —— 它崩在报消息的路上，
    看起来像测试坏了。这正是 `config.ensure_console_encoding()` 存在的理由。

    这条只守机制还在（有没有调用那个函数）；「调了之后真的不崩」由
    `test_config.py` 自己测。
    """
    missing = []
    for p in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        runs = [n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_run"]
        if not runs:
            missing.append(f"{p.name}: 没有兜底 runner")
            continue
        calls = {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
                 for n in ast.walk(runs[0]) if isinstance(n, ast.Call)}
        if "ensure_console_encoding" not in calls:
            missing.append(f"{p.name}: _run() 没调 ensure_console_encoding()")
    assert not missing, (
        "这些套件照手册单独跑会崩在编码上：\n  " + "\n  ".join(missing))


def test_documented_total_matches_the_sum():
    """总数必须等于逐套之和 —— 两处各写一个数字，迟早有一处忘了改。

    手册与 README 一起查（README 里也写了两处，见上一条的说明）。
    """
    counts = _test_counts()
    total = sum(counts.values())
    bad = []
    for path in (HANDBOOK, README):
        text = path.read_text(encoding="utf-8")
        stated = (re.findall(r"(\d+)\s*条离线测试", text)
                  + re.findall(r"#\s*(\d+)\s*条", text)
                  + re.findall(r"(\d+)\s*条测试全绿", text)
                  + re.findall(r"共\s*(\d+)\s*条", text))
        if not stated:
            bad.append(f"{path.name}: 没写总条数")
        for s in stated:
            if int(s) != total:
                bad.append(f"{path.name}: 写 {s} 条，逐套加起来是 {total} 条")
    assert not bad, "\n  ".join(bad)


def test_documented_degradation_kinds_match_the_source():
    """§3.5 那张表列的 kind，必须与 `brief.py` 里真的会产出的完全一致。

    两个方向都查：手册漏写的（读者不知道它存在）、手册编出来的
    （照着表去核对产物会找不到），都是错的。
    """
    real: set[str] = set()
    for node in ast.walk(ast.parse(
            (WEIRAN / "brief.py").read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "append"
                and node.args and isinstance(node.args[0], ast.Dict)):
            d = node.args[0]
            for k, v in zip(d.keys, d.values):
                if (isinstance(k, ast.Constant) and k.value == "kind"
                        and isinstance(v, ast.Constant)):
                    real.add(v.value)
    assert len(real) >= 10, f"只抽出 {len(real)} 个 kind —— 抽取逻辑可能失效了"

    documented = _degradation_table()
    missing = sorted(real - documented)
    extra = sorted(documented - real)
    assert not missing, f"手册漏了这些降级项：{missing}"
    assert not extra, f"手册写了源码里没有的降级项：{extra}"


def test_documented_kind_count_matches():
    """「一共定义 N 种」里的 N 也要对上 —— 一个数错的总数比没有总数更糟。"""
    documented = _degradation_table()
    stated = re.search(r"一共定义 \*\*(\d+) 种 kind\*\*", _text())
    assert stated, "手册没写降级项一共几种"
    assert int(stated.group(1)) == len(documented), (
        f"手册写 {stated.group(1)} 种，表里列了 {len(documented)} 种")


def test_documented_markers_match_the_source():
    """7 个标记的**原文**必须逐字出现在手册里。

    标记是机检的抓手（`check_brief` 断言它们出现），手册里写成近义词
    就失去意义了 —— 读者会按手册去产物里找，然后找不到。
    """
    text = _text()
    for name, marker in B.MARKERS.items():
        assert marker in text, f"手册里没有标记 {name} 的原文 {marker!r}"
    stated = re.search(r"那 (\d+) 个方括号标记", text)
    assert stated, "手册没写标记的个数"
    assert int(stated.group(1)) == len(B.MARKERS), (
        f"手册写 {stated.group(1)} 个，MARKERS 里有 {len(B.MARKERS)} 个")


def test_documented_gold_counts_match_the_committed_table():
    """手册里金标三态的读数必须与**入库的那张表**一致。

    这三个数（通过 / 否决 / 不可判定）是手册里最容易被读者记住的一组，
    而它来自一个会随产出变化的产物 —— 一旦产出重跑而手册没改，
    读者会拿旧读数去对新表。
    """
    gold = json.loads((REPO_ROOT / "data" / "simulation" / "gold_check.json")
                      .read_text(encoding="utf-8"))
    counts = gold["summary"]["counts"]
    text = _text()
    stated = re.search(r"通过 (\d+) · 否决 (\d+) · 不可判定 (\d+)", text)
    assert stated, "手册没写金标三态的读数"
    got = {"通过": int(stated.group(1)), "否决": int(stated.group(2)),
           "不可判定": int(stated.group(3))}
    assert got == counts, f"手册写 {got}，入库表里是 {counts}"
    # 可采信的那几条也要对上 —— 它是这张表最容易被断章取义的数字
    n_trusted = len(gold["summary"]["trusted_ids"])
    assert re.search(rf"只有 (\d+) 条可采信", text), "手册没写可采信的条数"
    assert int(re.search(rf"只有 (\d+) 条可采信", text).group(1)) == n_trusted, (
        f"手册写的可采信条数与表里的 {n_trusted} 不一致")


def test_the_four_assertions_blamed_on_one_defect_are_the_real_ones():
    """「四条否决指向同一条相位缺陷」——这四个人是谁，必须与表里一致。

    手册单独点了名（AS-2/AS-3/AS-4/AS-8）。这是个**会变的集合**：
    产出重跑后也许只剩三条。写死在手册里而不同步，读者就会去找
    一条根本不在否决列表里的断言。
    """
    gold = json.loads((REPO_ROOT / "data" / "simulation" / "gold_check.json")
                      .read_text(encoding="utf-8"))
    # 表里那一句点名的正是「四条否决」。**不能拿整句里的全部 AS-N 去比** ——
    # 那一句后半还提了 AS-10，但说的是「它是另一件事」，不在四条之列。
    # 天真的全文抽取会把 AS-10 收进来，然后逼着手册也把它列进那四条里。
    note = gold["summary"].get("one_defect_note", "")
    m0 = re.search(r"(AS-[\d\s/,AS-]+?)四条否决", note)
    assert m0, "入库表的 one_defect_note 里没点名哪几条否决同源 —— 抽取逻辑失效了"
    real = set(re.findall(r"AS-\d+", m0.group(1)))

    m = re.search(r"其中 \*\*([^*]+?)四条否决", _text())
    assert m, "手册没点名是哪几条否决指向同一条缺陷"
    stated = set(re.findall(r"AS-\d+", m.group(1)))
    assert stated == real, f"手册写 {sorted(stated)}，表里写的是 {sorted(real)}"


# ---------------------------------------------------------------------------
# 3. .env：键名齐了，值一个都没有（红线）
# ---------------------------------------------------------------------------

def test_handbook_lists_every_env_key():
    """`.env.example` 里的每个键，手册的配置表里都要有一行。

    漏一个键的后果是具体的：读者按手册配完，以为配全了。
    """
    text = _text()
    keys = _env_keys()
    assert keys, "没从 .env.example 里读到任何键 —— 抽取逻辑失效了"
    missing = [k for k in keys if k not in text]
    assert not missing, f"手册的配置表漏了这些键：{missing}"


def test_handbook_contains_no_credential_value():
    """**红线**：凭据类环境变量的值，一个都不许出现在手册或 README 里。

    手册只写「键名 + 含义」。这条不靠人去逐条审，靠机器扫 ——
    **而且扫的是真的那份 `.env`**，不只是模板。

    **失败消息里只报键名，不报值。** 这条断言要是红了，异常信息会被
    打印到终端、贴进 issue、录进屏幕 —— 一条把密钥打进报错里的红线检查，
    比没有这条检查更糟。
    """
    creds = _credential_values()
    assert creds, (
        "没读到任何凭据类变量 —— 要么 .env.example 的键名改过了，"
        "要么抽取逻辑失效了；无论哪种，这条断言现在都是**恒真的**")
    for path in (HANDBOOK, README):
        text = path.read_text(encoding="utf-8")
        for key, val in creds:
            assert val not in text, (
                f"{path.name} 里出现了 {key} 的值（值本身不在此打印）—— "
                "文档只许写键名与含义")


def test_dotenv_is_gitignored():
    """`.env` 真的在 `.gitignore` 里。

    上面那条守的是「文档里没有密钥」，这条守的是「密钥不进 git」。
    两件事缺一不可，而后者更硬：进了 git 就是进了历史，删不干净。
    """
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = [l.strip() for l in ignored.splitlines()]
    assert ".env" in lines, (
        f".gitignore 里没有单独的 `.env` 一行（现有：{[l for l in lines if l and not l.startswith('#')]}）")


def test_no_documented_command_prints_a_secret():
    """手册里那些命令，一律不许带 `echo $XXX_KEY` 这类写法。

    这条是上一条的补强：值不写死也有可能被「打出来」。手册是给人照抄的，
    一条会把密钥打到终端的命令抄进去，密钥就进了录屏。
    """
    text = _text()
    for bad in ("echo $", "cat .env", "type .env", "printenv", "env |"):
        assert bad not in text, f"手册里出现了会泄漏配置的写法：{bad!r}"


# ---------------------------------------------------------------------------
# 4. 手册与 README 的分工确实写清了
# ---------------------------------------------------------------------------

def test_handbook_states_its_division_of_labour_with_the_readme():
    """手册开头必须自己说清与 README 的分工。

    `README.md` 那次「声称重写了前端而仓库里没有前端」的漂移，
    根因就是两份文档的边界没写下来 —— 谁都可以在任一份里写任何话。
    """
    head = "\n".join(_text().splitlines()[:12])
    assert "README.md" in head, "手册开头没有说清与 README 的分工"
    assert "怎么把它跑起来" in head or "怎么跑" in head, (
        "手册开头没有说清自己负责回答什么")


def test_readme_points_at_the_handbook():
    """反过来也要有路：README 读者要能找到手册。

    只写单向链接的话，评委读完 README 会以为「快速开始」那一节就是全部，
    而它不含产出的字段解读与已知降级项清单。
    """
    text = README.read_text(encoding="utf-8")
    assert "使用手册" in text, "README 里没有指向使用手册的链接"
    assert "docs/使用手册.md" in text, "README 里的手册链接路径不对"


def test_handbook_names_the_known_degradations():
    """坦白清单必须真的在，且点到了那几条已声明未修复的性质。

    这一条不是查完整性，是查**它们没有被删掉**。手册里最容易在
    「让文档好看一点」时被删的就是这一节，而它恰恰是这个项目最不该丢的部分。
    """
    text = _text()
    for must in ("相位", "1.31%", "767", "证伪", "待决", "chunking-guard"):
        assert must in text, f"坦白清单里少了 {must!r} —— 这一节不许被删薄"


#: 从一节的开头扫到下一个三级标题。写成常量而不是内联字符串，是因为
#: 它**必须**是一个真换行 —— 本节其余地方的 `\n` 都在正则里，混在一起
#: 很容易把这一处也写成转义序列而它偏偏不能是。
_NEXT_HEADING = "\n###"


def _variance_table() -> set[str]:
    """§4.8 那张「会变的那五处」表的 key 列。

    同样**只取那一节** —— 全文正则会连 `.env` 键名表一起收进来。
    """
    text = _text()
    start = text.find("### 4.8")
    assert start >= 0, "手册里找不到 §4.8 —— 稳定性那一节的编号变过了"
    end = text.find(_NEXT_HEADING, start + 1)
    section = text[start:end if end > 0 else len(text)]
    return set(re.findall(r"^\|\s*`(\w+)`\s*\|", section, re.M))


def _root_script_flags(name: str) -> set[str]:
    """`backend/` 根下那几个脚本（不在 `weiran/` 包里）的 argparse 开关。

    它们按行解析时看不见 —— 手册里写的是 `python run_seeds.py`，不是
    `python -m weiran.X`。所以单独抽一次。
    """
    src = (REPO_ROOT / "backend" / f"{name}.py").read_text(encoding="utf-8")
    return set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', src))


def test_documented_variance_sources_match_the_source():
    """§4.8 那张表列的随机性来源，必须与 `stability.py` 的 `VARIANCE_SOURCES`
    完全一致。

    两个方向都查，与降级项那张表同款做法：手册漏写的（读者以为没有这一层）、
    手册编出来的（照着表去核对代码会找不到），都是错的。

    这一条是**第 2 块剩下的那一半**：那张表在代码里是唯一出处，消费者是
    汇总器与手册。谁哪天在代码里加一层随机性而没写进手册，这里会红。
    """
    from weiran import stability as S

    real = {v.key for v in S.VARIANCE_SOURCES}
    assert len(real) >= 5, f"只抽到 {len(real)} 个来源 —— 表被删薄了？"

    documented = _variance_table()
    assert documented == real, (
        f"手册漏了：{sorted(real - documented)}；"
        f"手册写了代码里没有的：{sorted(documented - real)}")


def test_the_variance_table_states_a_total_that_matches():
    """「五处」这个数也要对上 —— 一个数错的总数比没有总数更糟。"""
    from weiran import stability as S

    stated = re.search(r"会变的那\*\*?五处\*\*?|会变的那五处", _text())
    assert stated, "手册没写一共几处会变"
    assert len(S.VARIANCE_SOURCES) == 5, (
        f"代码里现在是 {len(S.VARIANCE_SOURCES)} 处，手册的「五处」要改")


def test_the_unseedable_layer_is_still_marked_as_unseedable():
    """**asyncio 交错那一行不许被写成「能收窄」。** 它是这份清单存在的理由：
    播了种之后仍然收不回来的那一层要说清楚，否则读者会把 `--seed` 读成
    「可复现」。
    """
    from weiran import stability as S

    row = next(v for v in S.VARIANCE_SOURCES if v.key == "asyncio_interleaving")
    assert "不能" in row.seedable, row.seedable
    text = _text()
    assert "播种覆盖不到" in text, "手册没写清播种覆盖不到哪一层"


def test_the_handbook_says_the_batch_has_not_been_run():
    """工具就位、**批次待跑** —— 且要写出为什么待跑。

    这一条防的是两种走样：把「工具就位」写成「已经做完了」（那是在
    虚报），以及把待决原因删掉（读者会以为现在就能跑一批）。
    """
    text = _text()
    assert "工具已经就位" in text
    assert "但一批还没跑" in text or "一批还没跑" in text
    for must in ("run_seeds.py", "stability_report.py"):
        assert must in text, f"手册没提 {must}"


def test_the_stability_tools_are_documented_with_their_own_flags():
    """手册里 `python run_seeds.py ...` / `python stability_report.py ...`
    那几行用的开关，必须真被那两个脚本接受。

    它们不在 `weiran/` 包里，上面那条按行解析的测试看不见它们 —— 而一条
    「照着敲会失败」的命令，看起来和一条正确的命令一模一样。
    """
    text = _text()
    for script in ("run_seeds", "stability_report"):
        real = _root_script_flags(script)
        assert len(real) >= 3, f"只从 {script} 里抽出 {len(real)} 个开关 —— 抽取失效"
        seen = 0
        for line in text.splitlines():
            if f"python {script}.py" not in line:
                continue
            seen += 1
            for flag in re.findall(r"(--[a-z][a-z0-9-]*)", line):
                assert flag in real, (
                    f"手册写了 {script} {flag}，而它不接受（接受：{sorted(real)}）")
        assert seen, f"手册里没有 `python {script}.py` 的命令行"


def test_the_preflight_command_is_not_filed_under_free():
    """**`--preflight` 会真发请求。** 它跑的是 1 agent × 2 轮的真推演，
    只是便宜 —— 手册不能把它列在「都不花钱」那一块下面，那会把「便宜」
    洗成「免费」，而这一整节的立足点就是不洗这类话。
    """
    text = _text()
    # **只看 §5 那个速查块。** §4.8 的散文里也会提到 preflight，而它在
    # 「都不花钱」之前 —— 那不是在分栏归类，不能算进来。
    start = text.find("## 5. 命令速查")
    end = text.find("## 6.", start + 1)
    assert start > 0 and end > start
    lines = text[start:end].splitlines()

    free_at = next(i for i, l in enumerate(lines) if "# 都不花钱" in l)
    paid_at = next(i for i, l in enumerate(lines) if "# 花钱" in l)
    assert free_at < paid_at
    pf = [i for i, l in enumerate(lines) if "python run_seeds.py" in l
          and "--preflight" in l]
    assert pf, "速查块里没有 preflight 的命令"
    for i in pf:
        assert i > paid_at, (
            "preflight 被列在「都不花钱」下面了 —— 它是要发请求的")
    # 同一次核对：dry-run 那一行**必须**留在免费那栏（它真的不发请求）。
    dry = [i for i, l in enumerate(lines) if "python run_seeds.py" in l
           and "--preflight" not in l]
    for i in dry:
        assert free_at < i < paid_at, (
            "批次计划那一行是免费的（不加 --yes 不发请求），不该列在花钱栏")


# -- 资源清单里那几个数字（行数、套数、条数）也要是实测的 --------------------

MANIFEST = REPO_ROOT / "docs" / "开源及第三方资源使用清单.md"

#: 模块行 → 它该对哪个套件。只列**一对一**的那些；场景装载 / 角色画像 /
#: 金标校验 / 连通性冒烟没有自己的套件（由别的套件覆盖），行里本来也没写条数。
_SUITE_OF_MODULE = {
    "config.py": "test_config",
    "llm.py": "test_llm",
    "world_state.py": "test_world_state",
    "store.py": "test_store",
    "stance.py": "test_stance",
    "perception.py": "test_perception",
    "simulate.py": "test_simulate",
    "gold_check.py": "test_gold_check",
    "viewer.py": "test_viewer",
    "brief.py": "test_brief",
    "stability.py": "test_stability",
    "repro_check.py": "test_repro_check",
    "run_seeds.py": "test_run_seeds",
    "stability_report.py": "test_stability_report",
}


def _manifest_rows() -> list[tuple[str, int, int | None]]:
    """清单第三节表格的每一行 → `(文件路径, 行数, 测试条数或 None)`。

    **只认带 `（N 行）` 的行**，因为只有那种行在声明一个可核的数。
    """
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*[^|]+\|\s*`(backend/[\w/]+\.py)`（([\d,]+) 行）", line)
        if not m:
            continue
        t = re.search(r"（(\d+) 条测试）", line)
        rows.append((m.group(1), int(m.group(2).replace(",", "")),
                     int(t.group(1)) if t else None))
    return rows


def test_the_manifest_row_counts_are_measured_not_remembered():
    """清单自称「行数为实测（`wc -l`）」，那就当场实测一遍。

    这条不是形式主义：这份清单的合计与它自己那几行**曾经相加对不上**
    （差 165 行），而它是参赛必交附件。数字漂了没人会知道 ——
    文档不会自己发现自己在说谎，只会安静地漂走。
    """
    rows = _manifest_rows()
    assert len(rows) >= 18, (
        f"只认出 {len(rows)} 行 —— 解析器多半失配了（这是防恒真的下限，不是行数上限）")
    counts = _test_counts()
    bad = []
    for path, stated, tcount in rows:
        p = REPO_ROOT / path
        if not p.exists():
            bad.append(f"{path}: 清单里有、仓库里没有")
            continue
        real = len(p.read_text(encoding="utf-8").splitlines())
        if real != stated:
            bad.append(f"{path}: 清单写 {stated} 行，实际 {real} 行")
        suite = _SUITE_OF_MODULE.get(p.name)
        if suite is not None and tcount is not None and suite in counts:
            if tcount != counts[suite]:
                bad.append(f"{path}: 清单写 {tcount} 条测试，"
                           f"{suite}.py 里是 {counts[suite]} 条")
    assert not bad, "资源清单的行数/条数对不上实测：\n  " + "\n  ".join(bad)


def test_the_manifest_totals_are_the_sum_of_its_own_rows():
    """合计必须等于逐行相加 —— 两处各写一个数，迟早有一处忘了改。"""
    text = MANIFEST.read_text(encoding="utf-8")
    m = re.search(r"\*\*合计\*\*：(\d+) 个自研模块共 ([\d,]+) 行 \+ (\d+) 套离线测试共 "
                  r"([\d,]+) 行 = \*\*(\d+) 条测试全绿\*\*", text)
    assert m, "清单里找不到那句合计（格式变了就要一起改）"

    rows = _manifest_rows()
    n_mod, n_lines = int(m.group(1)), int(m.group(2).replace(",", ""))
    n_suite, s_lines = int(m.group(3)), int(m.group(4).replace(",", ""))
    n_tests = int(m.group(5))

    suites = sorted(TESTS.glob("test_*.py"))
    assert n_mod == len(rows), f"合计写 {n_mod} 个模块，表里有 {len(rows)} 行"
    assert n_lines == sum(r[1] for r in rows), (
        f"合计写 {n_lines} 行，逐行加起来是 {sum(r[1] for r in rows)} 行")
    assert n_suite == len(suites), f"合计写 {n_suite} 套，实际 {len(suites)} 套"
    got_lines = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in suites)
    assert s_lines == got_lines, f"合计写 {s_lines} 行测试，实际 {got_lines} 行"
    total = sum(_test_counts().values())
    assert n_tests == total, f"合计写 {n_tests} 条，逐套加起来是 {total} 条"


_ZH_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}


def _zh_number(s: str) -> int:
    """「十五」→ 15。只处理到 99 —— 这个用法只需要这个范围。"""
    if "十" not in s:
        return _ZH_DIGITS.get(s, -1)
    head, _, tail = s.partition("十")
    tens = _ZH_DIGITS.get(head, 1) if head else 1
    ones = _ZH_DIGITS.get(tail, 0) if tail else 0
    return tens * 10 + ones


def test_the_documented_suite_count_matches_the_directory():
    """「十五套测试也都能不装 pytest 直接跑」——那个数得真的是目录里套数。

    **只认当前那句**：两份文档里都还有一句「当时**九套**测试全绿」，
    那是在讲一次历史事件（第 3 层那个静默覆盖缺陷），不是在报今天的套数。
    所以这里不扫全文，只找「N 套测试…也都能」那一句。
    """
    n = len(list(TESTS.glob("test_*.py")))
    bad = []
    for path in (HANDBOOK, README):
        text = path.read_text(encoding="utf-8")
        hits = re.findall(r"([一二三四五六七八九十]+)套测试\*{0,2}也都能", text)
        if len(hits) != 1:
            bad.append(f"{path.name}: 找不到唯一那句「N 套测试…也都能」（找到 {len(hits)} 处）")
            continue
        if _zh_number(hits[0]) != n:
            bad.append(f"{path.name}: 写 {hits[0]}套，目录里是 {n} 套")
    assert not bad, "\n  ".join(bad)


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
