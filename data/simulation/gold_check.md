# 金标对照表

> **这不是分数。** 它不含「准确率」「得分」「命中率」这类量，也不该被算成任何这类量。

本表逐条核对金标自己写的断言「在这**一次**运行里发生了没有」，**不是模型有多准**。它没有分母，也没有被比较的基准 —— 所以它既不能被加总成一个「得分」，也不能与另一次运行的读数比出「提高了多少」。

出处：`benchmark/scenarios/*/reference_data.json → gold_status`。

## 一、先说对齐口径（这一步错过一次，所以写进产物）

阶段 → 轮号取自产出里 `rounds[*].phase_id` **非空**的那几轮（`phase_id` 记录的是「这一轮注入了哪个阶段」，只有少数轮有）。本表一律用产出的 0 基 `index`；`gold_day` 是金标 `phases[].day`。两者在 `days_per_round = 1.0`（轮 = 天）时应逐位相等 —— `match` 列为 null 表示本产出不是轮=天，天与轮不可直接比。

| 阶段 | 本表轮号（0 基） | 金标 `day` | 逐位相等 |
|---|---|---|---|
| P1 | 0 | 0 | 是 |
| P2 | 2 | 2 | 是 |
| P3 | 5 | 5 | 是 |
| P4 | 8 | 8 | 是 |
| P5 | 14 | 14 | 是 |

**别与 1 基的 R 号混用。** 2026-09-14 那批表是 0 基（R0~R14），后来的日志用 1 基（R1~R15）。本表全部 0 基，如需对照 1 基加 1。

## 二、贴界：读数的前提，不是附注

一个值恰好等于 0.0000 或 1.0000 即为贴界（`_clamp(·, 0, 1)` 的夹逼产物）。**判据读过的那个 (维, 轮) 单元格只要本身踩在贴界上，该条即「不采信」** —— 不是「结果作废」，是「这个结果不能当证据」。**按单元格判，不按「沾没沾到贴界轮」判，也不按「轮集合 × 维集合」的乘积判** —— 后两者会把干净的操作数判成贴界，把整张表染红。`railed_rounds` / `railed_dims` 是两张给人看的汇总清单，**不参与判定**；判定只看 `railed_cells`。

本次运行的贴界轮：**[9, 10, 11, 12, 13, 14]**；贴界维度：**polarization, risk, stability, trust**。

## 三、逐条

| 断言 | 金标 check 原文 | 读数 | 结果 | 采信 | 贴界 |
|---|---|---|---|---|---|
| AS-1 | `trust(P3) < trust(P2) < trust(P1)` | trust(P1)=0.6843；trust(P2)=0.6191；trust(P3)=0.4448 | **通过** | 是 | — |
| AS-2 | `argmax_phase(attention) == P3` | attention(P3)=0.3021；argmax_attention=0.3465；argmax_phase=P4 | **否决** | 是 | — |
| AS-3 | `risk(P3) >= 0.70` | risk(P3)=0.4736；threshold=0.7 | **否决** | 是 | — |
| AS-4 | `argmax_phase(polarization) == P3` | polarization(P3)=0.3308；argmax_polarization=1.0；argmax_phase=P5 | **否决** | **否** | polarization@R14 |
| AS-5 | `attention(P3)-attention(P5) > 3 * (trust(P5)-trust(P3))` | left=0.0184；right=-1.3344；attention(P3)-attention(P5)=0.0184；3*(trust(P5)-trust(P3))=-1.3344 | **通过** | **否** | trust@R14 |
| AS-6 | `trust(P5) < trust(P1) - 0.10` | trust(P5)=0.0；trust(P1)=0.6843；trust(P1)-0.10=0.5843 | **通过** | **否** | trust@R14 |
| AS-7 | `seq(劝删) in (seq(口径质疑), seq(家长介入))` | — | **不可判定** | — | — |
| AS-8 | `warning_issued_at <= P2` | warning_issued_at=7；P2_round=2 | **否决** | 是 | — |
| AS-9 | `argmax_group(role_conflict) == 辅导员` | — | **不可判定** | — | — |
| AS-10 | `trust_branch(P5) > trust_actual(P5)` | trust_branch(P5)=0.0；trust_actual(P5)=0.0；delta=0.0 | **否决** | **否** | trust@R14 |

**合计：10 条 —— 通过 3 · 否决 5 · 不可判定 2。**其中**判得出来**的 8 条，而判得出来的里面只有 **4 条可采信**（AS-1, AS-2, AS-3, AS-8）—— 其余的结果都不作数：不是「结果反了」，是「这个结果不能当证据」。

**AS-2 / AS-3 / AS-4 / AS-8 四条否决指向同一条相位缺陷** —— 模型的状态峰值落在 P3 之后一到两轮（见 进度.md 的相位定案）。不要读成四个独立问题：它们是同一个成因的四个落点。

AS-10 是另一件事（贴界 + 离线分支不是真反事实）。

**不采信的条目**（不是「结果作废」，是「不能当证据」）：

- `AS-4`（否决）—— 贴界
- `AS-5`（通过）—— 判据恒真、贴界
- `AS-6`（通过）—— 贴界
- `AS-10`（否决）—— 贴界

**不可判定的条目，一条都不许算成通过或失败**（它们是这张表最该补的地方）：

- `AS-7` —— 缺两样：(a) 产出侧的事件 id，(b) 一条能表达「排在某两者之间」的判据
- `AS-9` —— 缺逐条行为 → 角色的归属（产出里没有这个映射），以及该指标本身的实现

## 四、逐条说明

### AS-1 · direction · 通过

- 声称：trust 在 P1→P3 单调下降
- 金标判据：`trust(P3) < trust(P2) < trust(P1)`
- trust(P1)：0.6843
- trust(P2)：0.6191
- trust(P3)：0.4448
- 读数：0.4448 < 0.6191 < 0.6843
- 受贴界影响：否

### AS-2 · ordering · 否决

- 声称：attention 在 P3 见顶
- 金标判据：`argmax_phase(attention) == P3`
- attention(P3)：0.3021
- argmax_attention：0.3465
- argmax_phase：P4
- argmax_all_15_rounds：`{"phase": "10", "value": 0.3467}`
- 读数：argmax = P4（0.3465）；P3 是 0.3021。
- 受贴界影响：否
- argmax_rule：`argmax_phase` 有两个口径：只在 5 个**阶段轮**上取，或在全部 15 轮上取。本表按「阶段轮」取，因为金标写的是阶段（P3），而阶段只在被注入的那一轮上有落点；另一个口径的读数同时印出来供核对。

### AS-3 · threshold · 否决

- 声称：risk 在 P3 越过 0.7
- 金标判据：`risk(P3) >= 0.70`
- risk(P3)：0.4736
- threshold：0.7
- 读数：risk(P3) = 0.4736，金标阈值 0.70 —— 差 0.2264
- 受贴界影响：否
- threshold_from：金标 check 原文

### AS-4 · ordering · 否决（不采信）

- 声称：polarization 在 P3 见顶
- 金标判据：`argmax_phase(polarization) == P3`
- polarization(P3)：0.3308
- argmax_polarization：1.0
- argmax_phase：P5
- argmax_all_15_rounds：`{"phase": "11", "value": 1.0}`
- 读数：argmax = P5（1.0000）；P3 是 0.3308。 剔掉贴界轮后 argmax = P4（0.6310）
- 受贴界影响：**是** —— 判据读过 polarization@R14
- 不采信的原因：贴界
- argmax_rule：`argmax_phase` 有两个口径：只在 5 个**阶段轮**上取，或在全部 15 轮上取。本表按「阶段轮」取，因为金标写的是阶段（P3），而阶段只在被注入的那一轮上有落点；另一个口径的读数同时印出来供核对。
- variant：`{"scope": "剔掉 `polarization` 自己贴界的阶段轮", "argmax_phase": "P4", "value": 0.631, "rounds_used": [0, 2, 5, 8]}`

### AS-5 · asymmetry · 通过（不采信）

- 声称：信任恢复显著弱于关注回落（核心断言）
- 金标判据：`attention(P3)-attention(P5) > 3 * (trust(P5)-trust(P3))`
- left：0.0184
- right：-1.3344
- attention(P3)-attention(P5)：0.0184
- 3*(trust(P5)-trust(P3))：-1.3344
- 读数：左端 0.0184，右端 -1.3344。 **右端为负（信任下降 -0.4448），这条判据在该数据上恒真** —— 它与「不对称」无关：把信任换成任何一个下降得更快的量，结论一样。
- 受贴界影响：**是** —— 判据读过 trust@R14
- 不采信的原因：判据恒真、贴界

### AS-6 · irreversibility · 通过（不采信）

- 声称：P5 信任未回到 P1 基线
- 金标判据：`trust(P5) < trust(P1) - 0.10`
- trust(P5)：0.0
- trust(P1)：0.6843
- trust(P1)-0.10：0.5843
- 读数：trust(P5) = 0.0000，基线减 0.10 是 0.5843
- 受贴界影响：**是** —— 判据读过 trust@R14
- 不采信的原因：贴界

### AS-7 · event_order · 不可判定

- 声称：劝删事件被检出且排在口径质疑之后、父母介入之前
- 金标判据：`seq(劝删) in (seq(口径质疑), seq(家长介入))`
- 读数：产出里**没有「叙事事件 → 检出信号」的映射**。金标 `event_order` 是 17 条有序叙事事件，产出有逐轮行为分类与引擎自报的事件（`suppression_detected` 等），但没有任何字段把两者连起来 —— 「劝删被检出」这句话在产出里无法机检。
- 受贴界影响：否
- **为什么不可判定**：缺两样：(a) 产出侧的事件 id，(b) 一条能表达「排在某两者之间」的判据
- **还缺什么才能判**：① 产出：`injected` 现在只存「actor → 块文本的 sha1 前 12 位」，注入事件时把它的事件 id 一并落盘，事件顺序才可机检。② 金标：`check` 原文写的是 `seq(劝删) in (seq(口径质疑), seq(家长介入))`，这是「属于其中某一个」，而 `claim` 说的是「排在两者之间」—— 表达式与声称**不等价**，按字面求值永远不会成立（除非三者序号相等）。**本表不替金标改写判据**：改写判据就是替金标做决定。

### AS-8 · prediction · 否决

- 声称：风险预警在 P3 之前触发（「未然」的存在理由）
- 金标判据：`warning_issued_at <= P2`
- warning_issued_at：7
- P2_round：2
- risk_alert_rounds：`[7]`
- 读数：首个 `risk_alert` 在轮 7（0 基），金标要求 ≤ P2 那一轮（2） —— 晚了 5 轮。**注意别把这条与另一条混起来**：产出在 P2（轮 2）确实有一条 `polarization_surge`，而金标 `turning_point.detectable_signal` 说的正是「极化增速超过关注增速」；但 AS-8 判的是 `risk` **预警线**，不是这条信号。把两者混起来，一条否决会被读成通过。
- 受贴界影响：否
- definition_used：**本表的定义**（金标只给了 `warning_issued_at` 一个名字，没给判据）：产出里首个 `risk_alert` 事件所在的轮。依据是引擎自己把这条事件的消息写成「风险越过预警线 0.70」（`world_state.WorldStateEngine.RISK_ALERT = 0.70`，判据是**穿越** `before < 0.70 <= after`）—— 这是产出里唯一自称「预警」的量。**换一个定义可以改变结论**，所以定义写在这里。
- threshold_crossed：risk(P3) >= 0.70 用的是同一个 0.70

### AS-9 · role · 不可判定

- 声称：辅导员群体的角色冲突指标为全群体最高
- 金标判据：`argmax_group(role_conflict) == 辅导员`
- 读数：该指标**在代码里根本没有实现**，不是「这次没测到」。产出自己声明了这件事：「辅导员群体的角色冲突指标」—— 需要逐角色「公开立场 vs 私下立场」的情感极性差；当前产出里 `injected` 只存块摘要（sha1 前 12 位），不含立场文本。宁可空着，也不报一个编出来的数。
- 受贴界影响：否
- **为什么不可判定**：缺逐条行为 → 角色的归属（产出里没有这个映射），以及该指标本身的实现
- **还缺什么才能判**：需要逐角色「公开立场 vs 私下立场」的极性差。要判它，先得让产出落盘「哪一条行为出自哪个角色、立场是什么」—— 现在只有 actor id 与行为类别。**在实现出来之前，这一条不许被算成通过，也不许被算成失败。**

### AS-10 · counterfactual · 否决（不采信）

- 声称：若在 P2 主动公开分专业数据，P5 信任终值高于实际路径
- 金标判据：`trust_branch(P5) > trust_actual(P5)`
- trust_branch(P5)：0.0
- trust_actual(P5)：0.0
- delta：0.0
- 读数：分支 0.0000 对配对实际 0.0000 —— **两边都贴在 0.0 下界**，判据是 0.0000 > 0.0000。这条否决与它可能的「通过」一样没有内容：贴界把两个数都压到地板上，比较结果由夹逼决定。这一条与 AS-5 / AS-6 正好构成对偶 —— 那两条因贴界而「通过得毫无内容」，这一条因贴界而「否决得毫无内容」。**贴界对判据的破坏是双向的。**
- 受贴界影响：**是** —— 判据读过 trust@R14
- 不采信的原因：贴界
- branch_source：`branch_a` / `actual_from_fork` 来自 `brief.build_windows` 的离线分支：同一个分叉点、同一批后续行为，只把那一轮的动作换成干预版。**agent 的反应不变** —— 所以它不是一次真正的反事实重跑，金标自己的 `note` 也写着「此项需要运行干预分支」。
- gold_note：此项需要运行干预分支，是「干预建议」功能的评测依据

## 五、这份表的来源

- 推演产出：`twitter_rounds.json` sha256:ab4a3e0070a41aef（27 agent × 15 轮）
- 金标：`reference_data.json` sha256:9f14e8578aa4bd45
- 命令：`python -m weiran.gold_check`
- 复现：`python -m weiran.gold_check`（确定性，同输入同输出）

复现校验：`replay_max_deviation = 4.924954769952583e-05`（>1e-4 说明这份曲线不是本引擎产生的，那样整张表都不成立）
