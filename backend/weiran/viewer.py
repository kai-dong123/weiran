"""展示层：把**已有的产物**演给人看。只读、只 GET、不改任何东西。

这一层解决的是「没有产品形态」：产出里本来就有六维逐轮曲线、5 个决策窗口、
逐轮证据量、降级项、金标对照，但它们都在 JSON 里，而 JSON 不是给别人看的
形态。这里把它们渲染成页面。

**四条设计约束，每条都有理由：**

1. **不触发任何 LLM 调用。** 页面上的每个数都来自已经躺在盘上的产出文件。
   一次推演是 27 agent × 15 轮、实测 657s 墙钟、4377107/44478 token
   （百元级），「打开页面看一眼」绝不该产生这个代价。页面底部因此一直挂着
   数据文件的路径与 sha256 —— 看到的每个数都能追回产生它的那次运行。

2. **Flask 是可选依赖，且只在这里被 import。** 延迟到 `build_app()` 里才
   import：没有它时 `import weiran.viewer` 仍然成功，只有真的起服务才报
   「装什么、怎么装」。核心运行时（`requests` / `numpy`）刻意保持轻，少一个
   依赖就少一个「评委装不上」的失败点。`tests/test_viewer.py` 里有一条 AST
   检查守着「全仓只有这一个文件提到 flask」。

3. **不引 CDN、不引前端构建步骤。** 图表是**服务端生成的内联 SVG**，页面里
   没有一行 JavaScript、没有一个外部请求。离线可跑是本项目的取向（`.env`
   之外的网络请求都要能解释），一个要联网加载图表库的页面在评委的机器上
   可能是白屏。

4. **「没记录」绝不显示成 0。** 老产出里没有 `cost.paths.direct`、没有
   `cache_hit_tokens`、没有 `n_participants` —— 一律显示「未记录」。一个编
   出来的 0 比一个空值危险得多，它与「真的发生过、量是 0」在页面上长得一样。
   这一条与 `brief.py` 的 `failures_recorded`、`llm.py` 的 `cache_recorded`
   是同一条纪律的第三次复制。

**这一轮只读。** 页面上「触发推演」的位置与代价说明都摆出来了，但按钮是
**灰置**的、`PLANNED_TRIGGER_ROUTE` 没有任何处理器 —— 接口留在那里，是为了
让「下一步接什么」看得见，而不是假装已经有了。

    python -m weiran.viewer               # 起服务，地址取 .env（默认 127.0.0.1:8000）
    python -m weiran.viewer --check       # 不起服务，把每个页面渲一遍并自检
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

# `_sha256` 从 `brief` 引，**不在这里另抄一份**：指纹要按文本算（行尾是 git
# 的合法产物，不是内容的改动），同一件东西抄两份，迟早有一份漏改 —— 金标
# 对照表当初也是这么从 `brief` 取同一个函数的。
from .brief import DEFAULT_ROUNDS_FILE, _sha256
from .config import (
    REPO_ROOT,
    ConfigError,
    ViewerConfig,
    ensure_console_encoding,
    load_config,
)

DEFAULT_BRIEF = REPO_ROOT / "data" / "simulation" / "brief.json"
DEFAULT_GOLD = REPO_ROOT / "data" / "simulation" / "gold_check.json"

#: 六维的展示次序与中文名。与 `world_state.DIMENSIONS` 同集合 —— 中文名
#: 取自 `brief.json` 的 `dimension_desc`，不在这里另起一套译名。
DIM_ORDER = ("attention", "panic", "trust", "polarization", "risk", "stability")

#: 「触发推演」将来的落点。**现在没有任何处理器**，只印在 HTML 的
#: `data-planned-route` 上。接口先放出来，是为了让下一步接什么看得见。
PLANNED_TRIGGER_ROUTE = "/simulate"

#: 那个按钮灰着的原因，连同代价一起写在页面上 —— 把代价写在按钮旁边，
#: 比写在文档里更容易被看见。
TRIGGER_NOTE = (
    "本轮**只读**：这个按钮是预留的接口，没有接任何启动逻辑。"
    "一次推演是 27 agent × 15 轮、实测 657s 墙钟、prompt 4377107 / completion "
    "44478 token（百元级），且需要一个可用的端点与密钥（只在服务端读，不进页面）。"
)

#: 三态配色。**同一个词在三处（表、摘要、逐条）必须同色** —— 颜色在这里是
#: 判据结果的载体，不是装饰。
VERDICT_CLASS = {"通过": "ok", "否决": "bad", "不可判定": "undecided"}


# ---------------------------------------------------------------------------
# 取数：只读，缺失就如实说「没有」
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict | None:
    """读不到就返回 None。**不抛** —— 一份产物缺失是「这个页面显示不了」，
    不是「这个工具坏了」；页面自己会说缺哪一份。"""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class Data:
    """三份产物 + 它们的溯源。页面渲染只读这个对象，不再碰磁盘。"""

    def __init__(self, brief_path: Path, rounds_path: Path, gold_path: Path):
        self.paths = {"brief": brief_path, "rounds": rounds_path,
                      "gold": gold_path}
        self.brief = _read_json(brief_path)
        self.rounds = _read_json(rounds_path)
        self.gold = _read_json(gold_path)
        # 文件级指纹（启动时算一次）。与 `_provenance` 里那个是**两回事**：
        # 那个是「产出它时的那份输入」，这个是「现在盘上那份」，两者不一致
        # 说明文件被换过。
        self.sha = {k: (_sha256(p) if p.is_file() else None)
                    for k, p in self.paths.items()}

    @property
    def missing(self) -> list[str]:
        return [k for k, v in (("brief", self.brief), ("rounds", self.rounds),
                               ("gold", self.gold)) if v is None]

    def meta(self) -> dict:
        return (self.brief or {}).get("meta") or {}


def _v(value, *, unit: str = "") -> str:
    """**一个值 → 页面上的字。这是「不许把没记录显示成 0」的唯一出口。**

    所有数字都从这里过：`None` 一律渲染成「未记录」。散在几十处 f-string
    里各写各的，就一定有一处会把 `None` 印成 `0` 或 `None`。
    """
    if value is None:
        return '<span class="unrecorded">未记录</span>'
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.4f}{unit}"
    return f"{value}{unit}"


def _txt(value) -> str:
    """纯文本的同一个出口（用于 SVG 里的标签，那里不能放 HTML）。"""
    if value is None:
        return "未记录"
    return str(value)


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _pct(part, whole) -> str:
    if not whole:
        return _v(None)
    return f"{100.0 * part / whole:.1f}%"


# ---------------------------------------------------------------------------
# 图：服务端生成的内联 SVG（无 JS、无 CDN）
# ---------------------------------------------------------------------------

def _svg_lines(series: list[tuple], *, w: int = 360, h: int = 150,
               y_max: float = 1.0, x_labels: list[tuple] | None = None,
               title: str = "", bands: list[tuple] | None = None) -> str:
    """多序列折线图。`series` 是 `[(名字, 颜色, [值...])]`。

    `bands` 是 `[(起, 止, 说明)]` 的底纹区间，用来标出**贴界轮**那种
    「这段读数不是读数」的区域 —— 曲线图上最容易被看成本来如此的，
    正是这种由夹逼决定的形状。
    """
    pad_l, pad_r, pad_t, pad_b = 34, 8, 14, 22
    n = max(len(s[2]) for s in series) if series else 0
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b

    def X(i: int) -> float:
        return pad_l + (iw * i / (n - 1) if n > 1 else iw / 2)

    def Y(v: float) -> float:
        return pad_t + ih * (1.0 - max(0.0, min(y_max, v)) / y_max)

    out = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" '
           f'aria-label="{_esc(title or "曲线")}">']
    for frac in (0.0, 0.5, 1.0):
        y = Y(frac * y_max)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" '
                   f'y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{pad_l - 4}" y="{y + 3:.1f}" class="tick" '
                   f'text-anchor="end">{frac * y_max:g}</text>')
    for lo, hi, why in (bands or []):
        x0, x1 = X(lo), X(hi)
        out.append(f'<rect x="{x0:.1f}" y="{pad_t}" width="{max(1.0, x1 - x0):.1f}" '
                   f'height="{ih}" class="band"><title>{_esc(why)}</title></rect>')
    for name, color, vals in series:
        pts = " ".join(f"{X(i):.1f},{Y(float(v)):.1f}"
                       for i, v in enumerate(vals) if v is not None)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                   f'stroke-width="1.8"/>')
        for i, v in enumerate(vals):
            if v is None:
                continue
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(float(v)):.1f}" r="2" '
                       f'fill="{color}"><title>{_esc(name)} 轮{i} = '
                       f'{float(v):.4f}</title></circle>')
    for idx, label in (x_labels or []):
        out.append(f'<text x="{X(idx):.1f}" y="{h - 6}" class="tick" '
                   f'text-anchor="middle">{_esc(label)}</text>')
    out.append("</svg>")
    return "".join(out)


def _svg_stack(rows: list[tuple], kinds: list[str], *, w: int = 620, h: int = 190,
               title: str = "") -> str:
    """逐轮行为构成的堆叠柱。`rows` 是 `[(轮, 阶段, {类别: 条数})]`。"""
    pad_l, pad_r, pad_t, pad_b = 34, 10, 12, 26
    n = len(rows)
    totals = [sum(rc.values()) for _, _, rc in rows] or [1]
    top = max(totals) or 1
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    bw = iw / max(1, n) * 0.62
    colors = ["#2b6cb0", "#2f855a", "#b7791f", "#805ad5", "#c53030", "#0987a0",
              "#718096", "#975a16"]

    def Y(v: float) -> float:
        return pad_t + ih * (1.0 - v / top)

    out = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" '
           f'aria-label="{_esc(title or "行为构成")}">']
    for frac in (0.0, 0.5, 1.0):
        y = Y(frac * top)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" '
                   f'class="grid"/>')
        out.append(f'<text x="{pad_l - 4}" y="{y + 3:.1f}" class="tick" '
                   f'text-anchor="end">{frac * top:.0f}</text>')
    for i, (idx, phase, rc) in enumerate(rows):
        cx = pad_l + iw * (i + 0.5) / max(1, n)
        y = pad_t + ih
        for k in kinds:
            c = rc.get(k, 0)
            if not c:
                continue
            hgt = ih * c / top
            col = colors[kinds.index(k) % len(colors)]
            out.append(f'<rect x="{cx - bw / 2:.1f}" y="{y - hgt:.1f}" '
                       f'width="{bw:.1f}" height="{hgt:.1f}" fill="{col}" '
                       f'fill-opacity="0.85"><title>轮{idx} {phase} '
                       f'{_esc(k)} = {c}</title></rect>')
            y -= hgt
        if phase:
            out.append(f'<text x="{cx:.1f}" y="{h - 12}" class="tick" '
                       f'text-anchor="middle">{_esc(phase)}</text>')
        out.append(f'<text x="{cx:.1f}" y="{h - 2}" class="tick" '
                   f'text-anchor="middle">R{idx}</text>')
    out.append("</svg>")
    legend = "".join(
        f'<span class="key"><i style="background:'
        f'{colors[i % len(colors)]}"></i>{_esc(k)}</span>'
        for i, k in enumerate(kinds))
    return "".join(out) + f'<div class="legend">{legend}</div>'


# ---------------------------------------------------------------------------
# 页面：五个视图
# ---------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light; }
body { font: 14px/1.6 "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
       margin: 0; color: #1a202c; background: #f7fafc; }
header { background: #1a365d; color: #fff; padding: 12px 20px; }
header h1 { margin: 0 0 4px; font-size: 17px; font-weight: 600; }
header .sub { font-size: 12px; opacity: .8; }
nav { background: #2a4365; padding: 0 20px; }
nav a { display: inline-block; color: #cbd5e0; text-decoration: none;
        padding: 9px 13px; font-size: 13px; }
nav a:hover { background: #2c5282; color: #fff; }
nav a.on { background: #f7fafc; color: #1a365d; font-weight: 600; }
main { padding: 18px 20px 60px; max-width: 1120px; }
section { background: #fff; border: 1px solid #e2e8f0; border-radius: 5px;
          padding: 14px 16px; margin-bottom: 16px; }
h2 { font-size: 15px; margin: 0 0 10px; }
h3 { font-size: 13px; margin: 14px 0 6px; color: #2d3748; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border: 1px solid #e2e8f0; padding: 5px 8px; text-align: left;
         vertical-align: top; }
th { background: #edf2f7; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.chart { width: 100%; height: auto; }
.chart .grid { stroke: #e2e8f0; stroke-width: 1; }
.chart .tick { font-size: 9px; fill: #718096; }
.chart .band { fill: #c53030; fill-opacity: .07; }
.grid6 { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
         gap: 14px; }
.card { border: 1px solid #e2e8f0; border-radius: 4px; padding: 8px 10px; }
.card .t { font-size: 12px; color: #4a5568; margin-bottom: 2px; }
.unrecorded { color: #975a16; background: #fefcbf; padding: 0 3px;
              border-radius: 2px; font-size: 12px; }
.tag { display: inline-block; padding: 0 6px; border-radius: 9px;
       font-size: 12px; font-weight: 600; }
.ok { background: #c6f6d5; color: #22543d; }
.bad { background: #fed7d7; color: #822727; }
.undecided { background: #e2e8f0; color: #4a5568; }
.warn { border: 1px solid #fc8181; border-left: 4px solid #c53030;
        background: #fff5f5; border-radius: 4px; padding: 10px 12px;
        margin-bottom: 10px; }
.warn h3 { margin-top: 0; color: #822727; }
.trigger { border: 1px dashed #a0aec0; background: #edf2f7; border-radius: 5px;
           padding: 12px 16px; color: #4a5568; }
.trigger button { font-size: 13px; padding: 6px 14px; border-radius: 4px;
                  border: 1px solid #cbd5e0; background: #e2e8f0; color: #a0aec0;
                  cursor: not-allowed; }
footer { padding: 0 20px 30px; color: #718096; font-size: 12px;
         max-width: 1120px; }
footer code { background: #edf2f7; padding: 1px 4px; border-radius: 3px; }
.legend { font-size: 12px; color: #4a5568; margin-top: 4px; }
.legend .key { margin-right: 12px; }
.legend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px;
            margin-right: 4px; }
.note { color: #4a5568; font-size: 12.5px; }
details { margin: 6px 0; }
summary { cursor: pointer; font-size: 13px; }
"""

_NAV = [("/", "概览"), ("/curves", "六维曲线"), ("/rounds", "逐轮"),
        ("/windows", "决策窗口"), ("/gold", "金标对照")]


def _page(d: Data, path: str, title: str, body: list[str]) -> str:
    nav = "".join(
        # 当前页高亮。用 `format` 而不是 f-string：3.11 的 f-string 表达式里
        # 不许出现反斜杠，而那个 `class="on"` 需要转义引号。
        '<a href="{}"{}>{}</a>'.format(
            _esc(p), ' class="on"' if p == path else "", _esc(t))
        for p, t in _NAV)
    m = d.meta()
    sub = (f"{_v(m.get('agents'))} agent × {_v(m.get('rounds'))} 轮 · "
           f"{_esc(m.get('platform') or '—')} · "
           f"压缩：{_v(m.get('compressed'))} · 总耗时 "
           f"{_v(m.get('total_seconds'))}s")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)} · 未然</title><style>{_CSS}</style></head>
<body>
<header><h1>「未然」推演产出</h1><div class="sub">{sub}</div></header>
<nav>{nav}</nav>
<main>{''.join(body)}</main>
{_footer(d)}
</body></html>"""


def _footer(d: Data) -> str:
    """页脚挂溯源。**每个页面都有**：页面上的数要能追回产生它的那次运行。"""
    b = d.brief or {}
    p = b.get("provenance") or {}
    rows = []
    if p:
        rows.append(f"<div>推演产出 <code>{_esc(Path(str(p.get('rounds_file', ''))).name)}</code> "
                    f"sha256:<code>{_esc(p.get('rounds_sha256', '—'))}</code> · "
                    f"金标 <code>{_esc(Path(str(p.get('scenario_file', ''))).name)}</code> "
                    f"sha256:<code>{_esc(p.get('scenario_sha256', '—'))}</code> · "
                    f"命令 <code>{_esc(p.get('command', '—'))}</code></div>")
    now = " · ".join(f"{k} <code>{_esc(Path(str(v)).name)}</code>"
                     f"{'（缺）' if d.sha[k] is None else f' sha256:{d.sha[k]}'}"
                     for k, v in d.paths.items())
    rows.append(f"<div>本页读的三份文件：{now}</div>")
    rows.append('<div class="note">这一层只读：不触发任何 LLM 调用，也不写文件。</div>')
    return f"<footer>{''.join(rows)}</footer>"


def _need(d: Data, *keys: str) -> list[str] | None:
    """缺产物时给一块说明，而不是空白页。**「没有」要写出来。**"""
    if not d.missing:
        return None
    names = {"brief": "决策简报 brief.json", "rounds": "推演产出 twitter_rounds.json",
             "gold": "金标对照表 gold_check.json"}
    want = [k for k in keys if k in d.missing]
    if not want:
        return None
    how = {"brief": "python -m weiran.brief", "rounds": "（随仓库入库）",
           "gold": "python -m weiran.gold_check"}
    items = "".join(f"<li>{_esc(names[k])} —— 缺。生成：<code>{_esc(how[k])}</code></li>"
                    for k in want)
    return [f'<section><div class="warn"><h3>这一页显示不了</h3>'
            f"<ul>{items}</ul></div></section>"]


def view_index(d: Data) -> str:
    body: list[str] = []
    miss = _need(d, "brief", "rounds")
    if miss:
        return _page(d, "/", "概览", miss)

    b = d.brief
    m = d.meta()
    cov = b.get("coverage") or {}
    ev = b.get("evidence") or {}
    pr = ev.get("presence") or {}

    claim = b.get("core_claim") or {}
    body.append(
        '<section><h2>这一份要证的那句话</h2>'
        f'<p>{_esc(claim.get("statement") or "（产出里没有 core_claim）")}</p>'
        f'<p class="note">可证伪：{_v(claim.get("falsifiable"))} · '
        f'判据：{_esc(claim.get("measured_by") or "—")}</p>'
        f'<p class="note">{_esc(claim.get("implication") or "")}</p>'
        '</section>')
    body.append(
        '<section><h2>这一份的规模与状态</h2>'
        '<table><tr><th>项</th><th>值</th></tr>'
        f'<tr><td>规模</td><td>{_v(m.get("agents"))} agent × {_v(m.get("rounds"))} 轮</td></tr>'
        f'<tr><td>是否压缩模式</td><td>{_v(m.get("compressed"))}'
        + ('' if m.get("compressed") else
           '（轮 = 天，与金标 `phases[].day` 逐位对齐）') + '</td></tr>'
        f'<tr><td>总动作数</td><td>{_v(m.get("total_actions"))}</td></tr>'
        f'<tr><td>总耗时</td><td>{_v(m.get("total_seconds"), unit="s")}</td></tr>'
        f'<tr><td>上下文截断</td><td>{_v(m.get("truncations"))} 次</td></tr>'
        f'<tr><td>决策窗口覆盖</td><td>未覆盖 '
        f'{_v(cov.get("n_uncovered"))} · 共享 {_v(cov.get("n_shared"))}</td></tr>'
        '</table></section>')

    # -- 降级项：红框，非空就显眼 -----------------------------------------
    degs = b.get("degradations") or []
    if degs:
        inner = "".join(
            f'<div class="warn"><h3>降级项 · {_esc(x.get("kind"))}</h3>'
            f'<p>{_esc(x.get("text"))}</p></div>' for x in degs)
        body.append(f'<section><h2>降级项（{len(degs)} 条）—— '
                    "这些前提不成立时，上面的数不能照读</h2>" + inner + "</section>")

    # -- 证据量 -----------------------------------------------------------
    row = ev.get("leanest") or {}
    body.append(
        '<section><h2>证据量</h2>'
        '<p class="note">每轮六维状态是由**几条行为**算出来的 —— 这一栏此前不在'
        '产出里，于是「27 个 agent 吵了 50 条」与「3 个 agent 说了 3 句话」在'
        '文件里长得一模一样。</p>'
        '<table><tr><th>项</th><th class="num">值</th></tr>'
        f'<tr><td>最少行为的一轮</td><td class="num">R{_v(row.get("round"))}'
        f'（{_esc(row.get("phase") or "—")}）</td></tr>'
        f'<tr><td>那一轮行为条数</td><td class="num">{_v(row.get("n_behaviors"))}</td></tr>'
        f'<tr><td>那一轮发言人数 / 在场人数</td><td class="num">'
        f'{_v(row.get("n_speakers"))} / {_v(row.get("n_participants"))}</td></tr>'
        f'<tr><td>零行为轮</td><td class="num">'
        f'{_v(len(ev.get("no_evidence_rounds") or []))}</td></tr>'
        f'<tr><td>在场人数范围</td><td class="num">'
        f'{_v(pr.get("participants_min"))} ~ {_v(pr.get("participants_max"))}'
        f'（首 {_v(pr.get("participants_first"))} / 末 {_v(pr.get("participants_last"))}）</td></tr>'
        f'<tr><td>发言人数范围</td><td class="num">'
        f'{_v(pr.get("speakers_first"))} → {_v(pr.get("speakers_last"))}</td></tr>'
        f'<tr><td>贴界轮</td><td class="num">'
        f'{_esc(ev.get("railed_rounds") or "无")}</td></tr>'
        f'<tr><td>贴界维度</td><td class="num">'
        f'{_esc("、".join(ev.get("railed_dims") or []) or "无")}</td></tr>'
        '</table></section>')

    # -- 成本：两条路径分列 ------------------------------------------------
    cost = b.get("cost") or {}
    paths = cost.get("paths") or {}
    cam = paths.get("camel") or {}
    direct = paths.get("direct") or {}
    body.append(
        '<section><h2>成本（两条调用路径分列）</h2>'
        f'<p class="note">{_esc(cost.get("summary") or "")}</p>'
        '<table><tr><th>路径</th><th class="num">调用</th><th class="num">入 token</th>'
        '<th class="num">出 token</th><th>前缀缓存命中</th></tr>'
        f'<tr><td>camel 驱动的 agent</td><td class="num">{_v(cam.get("calls"))}</td>'
        f'<td class="num">{_v(cam.get("prompt_tokens"))}</td>'
        f'<td class="num">{_v(cam.get("completion_tokens"))}</td>'
        f'<td>{_v(cam.get("cache_hit_pct"), unit="%") if cam.get("cache_recorded") else _v(None)}</td></tr>'
        f'<tr><td>直调（归类器 / 画像）</td>'
        + (f'<td class="num">{_v(direct.get("calls"))}</td>'
           f'<td class="num">{_v(direct.get("prompt_tokens"))}</td>'
           f'<td class="num">{_v(direct.get("completion_tokens"))}</td><td>—</td></tr>'
           if direct.get("recorded") else
           '<td colspan="4">未记录在本产出里 —— <b>这不是 0</b>'
           '（这笔钱只在服务商账单上看得见）</td></tr>')
        + '</table>'
        f'<p class="note">{_esc(paths.get("note") or "")}</p></section>')

    # -- 预留的触发接口：灰置 + 代价 --------------------------------------
    body.append(
        f'<section><div class="trigger" data-planned-route="{_esc(PLANNED_TRIGGER_ROUTE)}">'
        '<button disabled title="本轮只读，未接启动逻辑">触发一次推演</button>'
        f'<p class="note">{_esc(TRIGGER_NOTE)}</p>'
        '</div></section>')
    return _page(d, "/", "概览", body)


def view_curves(d: Data) -> str:
    miss = _need(d, "brief")
    if miss:
        return _page(d, "/curves", "六维曲线", miss)
    b = d.brief
    rows = b.get("rounds") or []
    ev = b.get("evidence") or {}
    railed = ev.get("railed_rounds") or []
    desc = b.get("dimension_desc") or {}

    x_labels = [(r["index"], r.get("phase_label"))
                for r in rows if r.get("phase_label")]
    cells = []
    for dim in DIM_ORDER:
        vals = [(r.get("state") or {}).get(dim) for r in rows]
        if all(v is None for v in vals):
            cells.append(f'<div class="card"><div class="t">{_esc(dim)}</div>'
                         f'<p class="note">{_v(None)}（这份产出里没有这一维）</p></div>')
            continue
        # 贴界轮画一条底纹：那段曲线的形状由 `_clamp` 决定，不由行为决定。
        bands = []
        if railed and dim in (ev.get("railed_dims") or []):
            lo = min(railed)
            bands = [(lo, max(railed), f"R{lo}~R{max(railed)}：这一维贴到了"
                                        "夹逼边界，形状由夹逼决定")]
        svg = _svg_lines([(dim, "#2b6cb0", vals)], x_labels=x_labels,
                         bands=bands, title=f"{dim} 逐轮")
        last = next((v for v in reversed(vals) if v is not None), None)
        cells.append(
            f'<div class="card"><div class="t">{_esc(desc.get(dim) or dim)}'
            f'（{_esc(dim)}）· 末轮 {_v(last)}</div>{svg}</div>')
    body = [
        '<section><h2>六维 15 轮</h2>'
        f'<p class="note">横轴是 0 基轮号（R0~R{len(rows) - 1}），标出阶段标签的'
        '那几轮就是金标 `phases[].day` 的落点。红底纹是**贴界轮**：'
        '值恰好等于 0.0000 / 1.0000，那是 `_clamp(·, 0, 1)` 的边界产物，'
        '不是读数。</p>'
        f'<div class="grid6">{"".join(cells)}</div></section>',
        '<section><h2>逐轮原值</h2><table><tr><th>轮</th><th>阶段</th>'
        + "".join(f'<th class="num">{_esc(x)}</th>' for x in DIM_ORDER)
        + '<th>贴界</th></tr>'
        + "".join(
            f'<tr><td class="num">R{_v(r.get("index"))}</td>'
            f'<td>{_esc(r.get("phase_label") or "—")}</td>'
            + "".join(f'<td class="num">{_v((r.get("state") or {}).get(x))}</td>'
                      for x in DIM_ORDER)
            + f'<td>{_esc("、".join(r.get("railed") or []) or "—")}</td></tr>'
            for r in rows)
        + '</table></section>',
    ]
    return _page(d, "/curves", "六维曲线", body)


def view_rounds(d: Data) -> str:
    miss = _need(d, "brief")
    if miss:
        return _page(d, "/rounds", "逐轮", miss)
    b = d.brief
    rows = b.get("rounds") or []
    kinds: list[str] = []
    for r in rows:
        for k in (r.get("behavior_counts") or {}):
            if k not in kinds:
                kinds.append(k)
    stacked = [(r.get("index"), r.get("phase_label") or "",
                r.get("behavior_counts") or {}) for r in rows]
    svg = _svg_stack(stacked, kinds, title="逐轮行为构成")

    body = [
        '<section><h2>行为构成</h2>'
        '<p class="note">柱高是**行为条数**（也就是喂进引擎的那个列表的长度），'
        '按行为类别分色。条数之外还要看人数：本条产出里在场人数全程恒定，'
        '而发声人数从 '
        f'{_v((b.get("evidence") or {}).get("presence", {}).get("speakers_first"))} 掉到 '
        f'{_v((b.get("evidence") or {}).get("presence", {}).get("speakers_last"))} —— '
        '少的是产出内容的人，不是在场的人。只看条数会把这两件事读成一件。</p>'
        f'<div class="card">{svg}</div></section>',
        '<section><h2>逐轮计数</h2><table>'
        '<tr><th>轮</th><th>阶段</th><th class="num">行为</th><th class="num">动作</th>'
        '<th class="num">发言动作</th><th class="num">在场</th><th class="num">发声</th>'
        '<th class="num">声势权重</th><th class="num">调用</th>'
        '<th class="num">入 tok</th><th class="num">出 tok</th>'
        '<th class="num">最慢 s</th><th>缓存命中/未命中</th><th>事件</th></tr>'
        + "".join(
            f'<tr><td class="num">R{_v(r.get("index"))}</td>'
            f'<td>{_esc(r.get("phase_label") or "—")}</td>'
            f'<td class="num">{_v(r.get("n_behaviors"))}</td>'
            f'<td class="num">{_v(r.get("n_actions"))}</td>'
            f'<td class="num">{_v(r.get("n_talk_actions"))}</td>'
            f'<td class="num">{_v(r.get("n_participants"))}</td>'
            f'<td class="num">{_v(r.get("n_speakers"))}</td>'
            f'<td class="num">{_v(r.get("volume"))}</td>'
            f'<td class="num">{_v(r.get("calls"))}</td>'
            f'<td class="num">{_v(r.get("prompt_tokens"))}</td>'
            f'<td class="num">{_v(r.get("completion_tokens"))}</td>'
            f'<td class="num">{_v(r.get("slowest"))}</td>'
            f'<td>{_v(r.get("cache_hit_tokens"))} / {_v(r.get("cache_miss_tokens"))}</td>'
            f'<td>{_esc("、".join(e.get("kind", "") for e in (r.get("events") or [])) or "—")}</td>'
            '</tr>' for r in rows)
        + '</table>'
        '<p class="note">缓存两列显示「未记录」是正常的：这一版端点没有打开'
        '前缀缓存报告，那个「未记录」与「报了但一次都没命中」是两件事，'
        '不能合并成 0。</p></section>',
    ]
    return _page(d, "/rounds", "逐轮", body)


def view_windows(d: Data) -> str:
    miss = _need(d, "brief")
    if miss:
        return _page(d, "/windows", "决策窗口", miss)
    b = d.brief
    rows = b.get("windows") or []
    scale = b.get("intervention_scale")
    body = [
        '<section><h2>决策窗口</h2>'
        '<p class="note">每个窗口给的是：当时的问题、实际选择、以及**在同一个'
        '分叉点上换一种处置**之后的信任终值。注意分支是**离线**算的 —— '
        '「后续行为不变」，所以它不是一次真正的反事实重跑，'
        'agent 的反应并没有跟着变。</p>'
        f'<table><tr><th>阶段</th><th>第几天</th><th>当时的问题</th><th>实际选择</th>'
        f'<th class="num">实际路径信任</th><th class="num">干预分支信任</th>'
        f'<th class="num">差值</th><th>说明</th></tr>'
        + "".join(
            f'<tr><td>{_esc(r.get("phase") or "—")}'
            f'{"（拐点）" if r.get("is_turning_point") else ""}</td>'
            f'<td class="num">{_v(r.get("gold_day"))}'
            + (f' <span class="note">（轮 {_v(r.get("round_index"))}）</span>'
               if r.get("round_index") is not None else "") + '</td>'
            f'<td>{_esc(r.get("question") or "—")}</td>'
            f'<td>{_esc(r.get("actual_choice") or "—")}</td>'
            f'<td class="num">{_v(r.get("actual_from_fork"))}</td>'
            f'<td class="num">{_v(r.get("branch_a"))}</td>'
            f'<td class="num">{_v(r.get("delta_a"))}</td>'
            f'<td class="note">{_esc(r.get("note") or "")}</td></tr>'
            for r in rows)
        + '</table>'
        f'<p class="note">干预倍数 `intervention_scale = {_v(scale)}`'
        '（沿用离线校验的约定值，不是本次调出来的）。</p></section>',
        '<section><h2>拐点信号</h2>'
        + _turning_point_html(b) + '</section>',
    ]
    return _page(d, "/windows", "决策窗口", body)


def _turning_point_html(b: dict) -> str:
    tp = b.get("turning_point") or {}
    gold = tp.get("gold") or {}
    sig = tp.get("signal_series") or []
    out = ['<table><tr><th>项</th><th>值</th></tr>'
           f'<tr><td>金标拐点</td><td>{_esc(gold.get("phase") or "—")}'
           f'（第 {_v(gold.get("day"))} 天）· {_esc(gold.get("event") or "—")}</td></tr>'
           f'<tr><td>金标说可检测的信号</td><td>{_esc(gold.get("detectable_signal") or "—")}</td></tr>'
           f'<tr><td>检出与金标一致</td><td>{_v(tp.get("signal_agrees_with_gold"))}</td></tr>'
           f'<tr><td>极化首次超过关注增速</td><td>'
           f'轮 {_v(tp.get("first_round_polarization_exceeds_attention"))}</td></tr>'
           '</table>']
    if sig:
        out.append('<details><summary>逐轮信号增速（金标判据的两个量）</summary>'
                   '<table><tr><th>轮</th><th>阶段</th><th class="num">Δ关注</th>'
                   '<th class="num">Δ极化</th><th>极化超过关注</th></tr>'
                   + "".join(
                       f'<tr><td class="num">R{_v(s.get("round"))}</td>'
                       f'<td>{_esc(s.get("phase") or "—")}</td>'
                       f'<td class="num">{_v(s.get("d_attention"))}</td>'
                       f'<td class="num">{_v(s.get("d_polarization"))}</td>'
                       f'<td>{_v(s.get("polarization_exceeds_attention"))}</td></tr>'
                       for s in sig)
                   + '</table></details>')
    for u in tp.get("unimplemented_signals") or []:
        out.append(f'<div class="warn"><h3>未实现的信号 · {_esc(u.get("name"))}</h3>'
                   f'<p>{_esc(u.get("reason"))}</p></div>')
    return "".join(out)


def _railed_cell(hits) -> str:
    """「这条判据读了哪个贴界值」。空列表与「没记录」在这里是同一件事：
    这一栏只说明**为什么降级**，没有降级就没有内容，写成「—」而不是 0。"""
    if not hits:
        return "—"
    return "、".join(f"{_esc(h.get('dimension'))}@R{_v(h.get('round'))}"
                    for h in hits)


def _interpretations_html(items) -> str:
    """`summary.interpretations` —— **「哪几条否决其实是同一件事」**。

    这一节不是装饰：它是把一行行的判据读成一句话的唯一地方（比如「AS-2 /
    AS-3 / AS-4 / AS-8 四条否决指向同一条相位缺陷」）。少了它，读者会把**一条
    根因数成四个独立问题**。

    它此前读的是 `s.get("one_defect_note")` —— 那个键在 `gold_check.json` 的
    `summary` 里**从来没有存在过**（真正的键是 `interpretations`）。于是它一直
    渲染成一个空段落：页面看着完全正常，那句话一次都没上过页面。所以这里
    逐条渲染**产物里真有的字段**，并且 `applies` 为假时把「为什么不适用」
    一起印出来 —— 「不适用」是一个结论，不是「没内容」。
    """
    if not items:
        return ""
    out = ['<h3>怎么读这一页</h3>']
    for it in items:
        observed = "、".join(f"{k}={v}" for k, v in (it.get("observed") or {}).items())
        out.append(f'<p><code>{_esc(it.get("id"))}</code> {_esc(it.get("claim"))}</p>')
        out.append(f'<p class="note">前提：{_esc(it.get("requires"))}'
                   + (f'；本页读数：{_esc(observed)}' if observed else "")
                   + "</p>")
        if it.get("applies") is False:
            out.append('<p class="note">**本页上不适用**：'
                       f'{_esc(it.get("not_applicable_because"))}</p>')
    return "".join(out)


def view_gold(d: Data) -> str:
    miss = _need(d, "gold")
    if miss:
        return _page(d, "/gold", "金标对照", miss)
    g = d.gold
    s = g.get("summary") or {}
    counts = s.get("counts") or {}
    nas = g.get("not_a_score") or {}
    ra = g.get("railing") or {}

    def tag(v: str) -> str:
        return f'<span class="tag {VERDICT_CLASS.get(v, "undecided")}">{_esc(v)}</span>'

    body = [
        '<section><h2>这不是分数</h2>'
        f'<p>{_esc(nas.get("statement") or "")}</p>'
        f'<p class="note">出处：<code>{_esc(nas.get("source") or "—")}</code></p>'
        f'<p class="note">{_esc(nas.get("why") or "")}</p>'
        f'<p class="note">{_esc(nas.get("excluded_from_judgement") or "")}</p></section>',

        '<section><h2>合计</h2>'
        f'<p>{_v(s.get("total"))} 条 —— 通过 {tag("通过")} {_v(counts.get("通过"))} · '
        f'否决 {tag("否决")} {_v(counts.get("否决"))} · '
        f'不可判定 {tag("不可判定")} {_v(counts.get("不可判定"))}。</p>'
        f'<p class="note">判得出来的 {_v(s.get("judged"))} 条里，只有 '
        f'{_esc(s.get("trusted_ids") or [])} 可采信 —— 其余的结果都不作数：'
        '不是「结果反了」，是「这个结果不能当证据」。</p>'
        + _interpretations_html(s.get("interpretations")) + '</section>',

        '<section><h2>逐条</h2><table><tr><th>断言</th><th>金标判据</th>'
        '<th>读数</th><th>结果</th><th>采信</th><th>贴界</th></tr>'
        + "".join(
            f'<tr><td><b>{_esc(r.get("id"))}</b><br>'
            f'<span class="note">{_esc(r.get("claim") or "")}</span></td>'
            f'<td><code>{_esc(r.get("check") or "")}</code></td>'
            f'<td class="note">{_esc(r.get("detail") or "")}</td>'
            f'<td>{tag(str(r.get("verdict")))}</td>'
            f'<td>{_v(r.get("trusted")) if r.get("trusted") is not None else "—"}</td>'
            f'<td>{_railed_cell(r.get("tainted_by_railing"))}</td></tr>'
            for r in g.get("assertions") or [])
        + '</table></section>',

        '<section><h2>贴界的判定</h2>'
        f'<p class="note">{_esc(ra.get("rule") or "")}</p>'
        f'<p>贴界轮 {_esc(ra.get("railed_rounds") or "无")} · '
        f'维度 {_esc("、".join(ra.get("railed_dims") or []) or "无")}</p>'
        '<table><tr><th>阶段</th><th class="num">本表轮号（0 基）</th>'
        '<th class="num">金标 day</th><th>逐位相等</th></tr>'
        + "".join(
            f'<tr><td>{_esc(r.get("phase"))}</td>'
            f'<td class="num">{_v(r.get("round_index"))}</td>'
            f'<td class="num">{_v(r.get("gold_day"))}</td>'
            f'<td>{_v(r.get("match"))}</td></tr>'
            for r in (g.get("alignment") or {}).get("rows") or [])
        + '</table>'
        f'<p class="note">{_esc((g.get("alignment") or {}).get("note_1based") or "")}</p>'
        '</section>',
    ]
    return _page(d, "/gold", "金标对照", body)


_ROUTES = {"/": ("概览", view_index), "/curves": ("六维曲线", view_curves),
           "/rounds": ("逐轮", view_rounds), "/windows": ("决策窗口", view_windows),
           "/gold": ("金标对照", view_gold)}


# ---------------------------------------------------------------------------
# 服务：延迟 import flask，只绑本机，绝不开 debug
# ---------------------------------------------------------------------------

FLASK_MISSING = (
    "展示层需要 Flask，而当前环境里没有装它。\n"
    "\n"
    "  pip install flask\n"
    "\n"
    "（Flask 是**可选依赖**：核心推演与简报都不需要它。\n"
    "  不想装也可以直接看 data/simulation/brief.md 与 gold_check.md，\n"
    "  那两份文字与这里的页面是同一批数。）"
)


def build_app(data: Data):
    """装配 WSGI 应用。**flask 只在这里被 import。**

    没有 flask 时抛 `SystemExit` 而不是 `ImportError`：这是一条「怎么装」
    的指引，不是程序错误。而且 `import weiran.viewer` 本身不会走到这里 ——
    其它模块因此永远不受这个可选依赖的影响。
    """
    try:
        from flask import Flask, Response, abort
    except ImportError:
        print(FLASK_MISSING, file=sys.stderr)
        raise SystemExit(3) from None

    app = Flask(__name__)

    @app.after_request
    def _no_store(resp):
        # 这是一份**快照**：产出换了页面就该换，缓存只会让两处对不上。
        resp.headers["Cache-Control"] = "no-store"
        return resp

    for path, (title, fn) in _ROUTES.items():
        def make(path=path, title=title, fn=fn):
            def route():
                return Response(fn(data), mimetype="text/html; charset=utf-8")
            route.__name__ = "view_" + (path.strip("/") or "index")
            return route
        app.add_url_rule(path, view_func=make(), methods=["GET"])

    @app.errorhandler(404)
    def _404(_e):
        body = ['<section><h2>没有这一页</h2><p class="note">'
                "本展示层只有这几页：" + "、".join(
                    f'<a href="{_esc(p)}">{_esc(t)}</a>' for p, (t, _) in _ROUTES.items())
                + "</p></section>"]
        return Response(_page(data, "", "没有这一页", body), status=404,
                        mimetype="text/html; charset=utf-8")

    @app.errorhandler(405)
    def _405(_e):
        # **只读**是这一层的性质，不是疏忽：所以 POST/PUT/DELETE 一律 405，
        # 并且明确说出来。预留的触发接口（`PLANNED_TRIGGER_ROUTE`）将来如果
        # 要接，也应当另起一个单独的入口去写文件，而不是让这一层变成可写。
        return Response("本展示层是只读的：只接受 GET。"
                        f"预留的触发接口 {PLANNED_TRIGGER_ROUTE} 尚无实现。",
                        status=405, mimetype="text/plain; charset=utf-8")

    return app


def render_all(data: Data) -> dict[str, str]:
    """把每一页渲一遍。**不需要 flask** —— 渲染是纯函数，服务只是外壳。
    所以 `--check` 在没有 flask 的机器上也能跑。"""
    return {path: fn(data) for path, (_, fn) in _ROUTES.items()}


def _check(data: Data) -> int:
    """离线自检：起不了服务也能验页面。

    **缺产物算失败，不算跳过。** 这一层的自检问的是「页面上的数是不是真的」，
    而一份缺失的产物会让每一页都渲染成「显示不了」—— 结构检查会全绿，于是
    自检报出一个「5/5 渲染正常」，而它实际上一页数据都没显示。这与
    `repro_check.py` 把「没测到」记成 None 而不是 True 是同一条纪律。
    """
    print("展示层自检（不起服务，不引 flask，不发任何网络请求）")
    bad = 0
    for k, p in data.paths.items():
        present = p.is_file()
        bad += 0 if present else 1
        print(f"  {'✅' if present else '❌'} {k}: {p}"
              + (f"  sha256:{data.sha[k]}" if data.sha[k]
                 else "  （缺 —— 这不是「跳过」，是没测到）"))
    pages = render_all(data)
    for path, body in sorted(pages.items()):
        ok = "<!doctype html>" in body and "</html>" in body
        header = body.split("<main>", 1)[-1][:200]
        empty = len(header.strip()) == 0
        blank = "这一页显示不了" in body
        good = ok and not empty and not blank
        bad += 0 if good else 1
        print(f"  {'✅' if good else '❌'} {path:<10} {len(body):>7} 字符"
              + ("（缺产物，页面显示「显示不了」）" if blank else ""))
    # 未记录必须**以「未记录」出现**，不能变成一个编出来的 0。
    if "未记录" not in pages["/rounds"]:
        print("  ❌ 逐轮页里一处「未记录」都没有 —— 缓存字段本该未记录；"
              "要么产出里已经有那个数（好事），要么这一层把 None 印成了 0（坏事）")
        bad += 1
    print(f"  → {len(_ROUTES) - min(bad, len(_ROUTES))}/{len(_ROUTES)} 页显示真数据")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ensure_console_encoding()
    ap = argparse.ArgumentParser(
        description="展示层：把已有产物渲染成页面（只读，不触发 LLM）")
    ap.add_argument("--host", default=None,
                    help="监听地址。只接受回环（默认取 .env 的 FLASK_HOST / 127.0.0.1）")
    ap.add_argument("--port", type=int, default=None,
                    help="监听端口（默认取 .env 的 FLASK_PORT / 8000）")
    ap.add_argument("--brief", default=None, help="决策简报 JSON")
    ap.add_argument("--rounds", default=None, help="推演产出 JSON")
    ap.add_argument("--gold", default=None, help="金标对照表 JSON")
    ap.add_argument("--check", action="store_true",
                    help="不起服务：把每页渲一遍并自检（不需要 flask）")
    args = ap.parse_args(argv)

    def resolve(v, default: Path) -> Path:
        p = Path(v) if v else default
        return p if p.is_absolute() else REPO_ROOT / p

    data = Data(resolve(args.brief, DEFAULT_BRIEF),
                resolve(args.rounds, DEFAULT_ROUNDS_FILE),
                resolve(args.gold, DEFAULT_GOLD))

    if args.check:
        return _check(data)

    # 配置**只在这一条路径上**读。上面 `--check` 刻意不读：它要能在没有
    # .env、没装 flask 的机器上跑（那正是「我的产物是不是真的」这个问题的
    # 使用场景，问的人常常是刚 clone 下来的）。
    # 命令行显式传的值优先于 .env —— 与 `load_config` 里「已存在的环境变量
    # 不被 .env 覆盖」是同一条取向。两次构造之间会重新过一遍回环校验，
    # 所以 `--host 0.0.0.0` 绕不过去。
    try:
        conf = load_config(require_llm=False).viewer
        if args.host is not None:
            conf = ViewerConfig(host=args.host, port=conf.port)
        if args.port is not None:
            conf = ViewerConfig(host=conf.host, port=args.port)
    except ConfigError as exc:
        print(f"配置有问题：\n{exc}")
        return 2

    app = build_app(data)
    print("「未然」展示层 —— 只读，不触发任何 LLM 调用")
    for k, p in data.paths.items():
        print(f"  {k:<7} {p}"
              + (f"  sha256:{data.sha[k]}" if data.sha[k] else "  （缺，对应页会显示「显示不了」）"))
    if data.missing:
        print(f"  ⚠ 缺 {'、'.join(data.missing)} —— 页面会说明缺哪一份，不假装有数")
    src = ("命令行" if (args.host is not None or args.port is not None)
           else ".env / 默认值")
    print(f"  监听 http://{conf.host}:{conf.port}/  （来源：{src}）")
    # 只绑回环、绝不开 debug：页面读的是本地文件，debug 的调试器是一个
    # 可以在服务端执行代码的面，而这一层不需要它（渲染是纯函数，出错就是
    # Python 回溯，直接在终端看）。
    app.run(host=conf.host, port=conf.port, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
