# 「未然」—— 校园舆情推演与决策辅助系统

> 全球校园人工智能算法精英大赛（AIC）·「AI+开源」· 方向（一）开源赋能的 AI 应用创新
> 状态：**开发中**（当前处于第 1 步：骨架与首次可运行）

面对校园突发事件，管理者往往在两难中做决定：公开信息会引发舆情，不公开则损耗信任。
**「未然」把这场两难提前演练一遍**——用多智能体仿真重放事件的舆论演化，逐轮追踪
关注、恐慌、信任、极化、风险、稳定六个维度的状态，识别风险拐点，并在你真正开口之前
给出可比较的处置方案。

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
| `LLM_BASE_URL` | OpenAI 兼容端点根地址，形如 `https://api.siliconflow.cn/v1` |
| `LLM_MODEL_NAME` | 对话模型 id |

`.env` 已在 `.gitignore` 中，**不会入库**。提交前自检：`git status` 里不应出现 `.env`。

### 4. 验证（**不需要 API key**）

下面两条命令完全不联网，可以在没有任何密钥的情况下确认环境正常：

```bash
cd backend
python tests/test_store.py      # 应当 14/14 通过
python -m weiran.scenario --reset
```

预期输出：

```
本次写入  episodes=5 chunks=33 entities=27
库内合计  {'episodes': 5, 'entities': 27, 'edges': 0, 'chunks': 33}
```

> **注意工作目录**：`weiran` 包位于 `backend/` 下，因此上述命令需先 `cd backend`。
> 若希望从任意目录调用，可执行一次 `pip install -e .`。

---

## 目录结构

```
backend/
  weiran/
    config.py      配置装载（缺项一次性列全，不静默降级）
    llm.py         OpenAI 兼容客户端 + 用量账本
    store.py       SQLite + FTS5 + 向量列
    scenario.py    场景装载（不依赖 LLM，可离线跑）
  tests/
    test_store.py  14 条测试，直接 python 运行，无需 pytest
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
