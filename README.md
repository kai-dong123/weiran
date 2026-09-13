# 「未然」—— 校园舆情推演与决策辅助系统

> 全球校园人工智能算法精英大赛（AIC）·「AI+开源」· 方向（一）开源赋能的 AI 应用创新
> 状态：**开发中**（骨架与场景已就绪；世界状态引擎完成，**六维闭环已接入逐轮循环**）

面对校园突发事件，管理者往往在两难中做决定：公开信息会引发舆情，不公开则损耗信任。
**「未然」把这场两难提前演练一遍**——用多智能体仿真重放事件的舆论演化，逐轮追踪
关注、恐慌、信任、极化、风险、稳定六个维度的状态，识别风险拐点，并在你真正开口之前
给出可比较的处置方案。

**闭环是怎么闭上的。** 每一轮：读 agent 的动作 → 归类成行为信号 → 推进六维状态 →
**把状态重新注回下一轮每个 agent 的输入**。回注的不是数值而是定性档位与方向
（「信任偏高、下滑中」），原因是数值会被 agent 原样复述进帖文，曲线就成了回声。
回注的还有**差异化感知**：同一时刻，家长只知道落实率，而当事人知道核实率只有
三成且对外不披露——事实按角色逐轮注入，而不是在提示词里写死一段背景故事。
三个阶段事件（P1–P5）同样从金标按轮号注入，且**以帖子的形式进入时间线**，
六维的变化经由 agent 的反应发生，而不是直接给维度加激励。

---

## 快速开始

### 1. 环境

Python **3.10+**（本项目在 3.11.9 上开发验证）。
需要 SQLite 带 **FTS5** 支持——Python 官方发行版自带的 `sqlite3` 通常已满足；
若报「未编译 FTS5 支持」，换用官方 python.org 的发行版即可。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> `requirements.txt` 里的仿真内核（`camel-oasis==0.2.5` / `camel-ai==0.2.78`）是
> **刻意锁死的**——上游驱动脚本对 oasis 内部做了 monkey-patch，版本一变就碎。
> 请勿单独升级。

### 3. 配置

```bash
cp .env.example .env
```

然后编辑 `.env`，至少填入：

| 变量 | 说明 |
|---|---|
| `LLM_API_KEY` | 服务商密钥 |
| `LLM_BASE_URL` | OpenAI 兼容端点根地址，形如 `https://api.deepseek.com/v1` |
| `LLM_MODEL_NAME` | 对话模型 id |

`.env` 已在 `.gitignore` 中，**不会入库**。提交前自检：`git status` 里不应出现 `.env`。

### 4. 验证（**不需要 API key**）

下面两条命令完全不联网，可以在没有任何密钥的情况下确认环境正常：

```bash
cd backend
python -m pytest tests/             # 108 条，一次跑完（推荐）

# 五套测试也都能**不装 pytest** 直接跑（各自带兜底 runner）：
python tests/test_store.py          # 14/14
python tests/test_world_state.py    # 17/17
python tests/test_llm.py            # 19/19
python tests/test_stance.py         # 26/26
python tests/test_perception.py     # 32/32
python -m weiran.scenario --reset
python -m weiran.validate           # 世界状态引擎离线重放校验
```

预期输出：

```
本次写入  episodes=5 chunks=33 entities=27
库内合计  {'episodes': 5, 'entities': 27, 'edges': 0, 'chunks': 33}
```

> `weiran.validate` 的输出里有一个 MAE 数字。**它不是分数。**
> 它是模型输出与**作者手写预期**之间的偏离量，只作诊断用。
> 把它当准确率，就是给评测注水——脚本开头会再提醒一次。

> **注意工作目录**：`weiran` 包位于 `backend/` 下，因此上述命令需先 `cd backend`。
> 若希望从任意目录调用，可执行一次 `pip install -e .`。

### 5. 跑一次推演（**需要 API key**）

```bash
cd backend
python -m weiran.simulate --agents 5 --rounds 15    # 轮 = 天，推荐口径
python -m weiran.simulate --agents 3 --rounds 3     # 省钱冒烟（会压缩，见下）
```

闭环的三个开关**默认全开**，都可以单独关掉做消融对照：

| 开关 | 关掉之后 |
|---|---|
| `--no-phases` | 不注入 P1–P5 阶段事件，时间线只剩 `--seed-text` |
| `--no-feedback` | 不回注六维态势 |
| `--no-knowledge` | 不回注该角色的知情范围 |

> **轮数少于 15 会被压缩**（每轮代表多于一天），此时六维曲线**与「轮 = 天」
> 不可比** —— 弛豫项可以复合，但激励项按步累加，同样 6 天拆成 6 步和合成 1 步
> 是两条不同的曲线。落盘的 `meta` 里带 `compressed` 与 `comparable_to_round_day`，
> 启动时也会打印警告，免得日后被当成「轮 = 天」的结果引用。

---

## 目录结构

```
backend/
  weiran/
    config.py       配置装载（缺项一次性列全，不静默降级）
    llm.py          OpenAI 兼容客户端 + 用量账本
    store.py        SQLite + FTS5 + 向量列
    scenario.py     场景装载（不依赖 LLM，可离线跑）
    world_state.py  六维耦合弛豫引擎（不依赖 LLM，可离线跑）
    stance.py       行为归类器：自由文本 → 六维行为信号
    profiles.py     金标角色 → OASIS profile + 知情映射
    perception.py   感知层：态势与知情范围 → 逐轮注入文本（不依赖 LLM）
    simulate.py     推演驱动：读动作 → 归类 → 推进状态 → 回注 → 落盘
    validate.py     金标校验（离线重放 + 可机检断言）
    smoke.py        连通性冒烟（端点 / JSON / 中文 / 推理开关成本）
  repro_check.py    可复现性分级自检（三层各测一遍）
  tests/
    test_store.py        14 条
    test_world_state.py  17 条
    test_llm.py          19 条
    test_stance.py       26 条
    test_perception.py   32 条
    —— 共 108 条，`python -m pytest tests/` 一次跑完
benchmark/
  scenarios/
    employment_trust_crisis/     虚构场景：某大学《就业质量报告》信任危机
      case_overview.md           五阶段时间线、立场群体、争议核心
      seed_materials/01..05.md   种子材料（合成）
      reference_data.json        金标：期望状态、事件顺序、角色、可机检断言
docs/
  开源及第三方资源使用清单.md      参赛必交附件，随开发持续更新
进度.md                          实时进度日志
```

---

## 设计取向

几条贯穿全项目的取舍，都不是随手决定的：

- **评委能一条命令跑起来** 优先于技术上的先进。存储用 SQLite 单文件而非图数据库，
  LLM 客户端只用 `requests` 而非官方 SDK，`.env` 自己解析而非引入 `python-dotenv`——
  每少一个依赖，就少一个「装不上」的失败点。
- **不静默降级**。配置缺失、向量维度不匹配、FTS5 不可用，一律立刻报错并说清楚原因。
  这类项目最常见的失败不是算法错，而是跑到一半发现某个值是空的。
- **能离线验证的部分必须可真离线验证**。场景装载与存储层不依赖任何 LLM，
  因此可以进 CI、可以复跑、可以作为复现的起点。
- **不做没有收益的复杂度**。语料仅 33 个 chunk 的规模下，混合检索（RRF）不会带来
  可测量的召回提升，因此只保留词法与向量两条独立通路，不做融合。
  这一点会在技术报告中如实写明，而不是假装做了。
- **代价也写进文档，不只写收益**。六维回注会经 agent 记忆逐轮累积，
  实测每轮 prompt 比不回注高得越来越多（3 轮里从 +12% 拉到 +100%）。
  所以闭环**不是**上下文增长的解药，它现在是加重项——这句话写在
  [`进度.md`](进度.md) 里，也会原样写进技术报告。
---

## 上游与许可

本项目基于开源项目 [MiroFish](https://github.com/666ghj/MiroFish) 二次开发，依据
**AGPL-3.0** 发布。上游版本、修改范围与自研边界见 [`NOTICE`](NOTICE) 与
[`docs/开源及第三方资源使用清单.md`](docs/开源及第三方资源使用清单.md)。

AGPL-3.0 第 13 条：本项目若以网络服务形式对外提供，须向使用者提供完整对应源码。
本项目公开发布于代码托管平台，满足该义务。

---

## 场景声明

本项目推演的场景为**完全虚构**：学校名称、人物、机构、事件、数据均为合成，
不映射任何真实个人、机构或事件。仓库内不包含真实当事人的个人信息。
