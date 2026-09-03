# DDaTR 复现后创新点调研报告（最终版）

> 日期：2026-09-02 ｜ 三个并行子代理调研 + 本人一手文献核对与仓库数据实测
> 目标：在完整复现 DDaTR（IEEE TMI 2025）后，找到**能切实提升指标**（CE F1 为首，BLEU/METEOR/ROUGE 与 ReXrank 口径为辅）的创新点，投一区（TMI/MedIA 级别）。
> 配套文件：`LRRG_文献调研报告_2024-2026.md`（LRRG 赛道全景+撞车清单）、`handoff/innovation_research.md`（跨领域 11 方向+模块设计）、本文件（最终合成建议）。

---

## 1. 结论速览（先看这个）

**2026 年 9 月的局面**：纵向报告生成（LRRG）赛道已被 2025-2026 年论文高密度占据——TIM(CVPR'26)、BiOTPrompt(CVPR'26)、MARE(AAAI'26)、Diff-RRG(MICCAI'25)、HC-LLM(AAAI'25)、MLRG(CVPR'25)、PriorRG(AAAI'26) 等。**但全部满足三个条件：LLM 重模型、只在 Longitudinal-MIMIC 一个数据集上报数、对比表普遍不列 DDaTR。** DDaTR 的"紧凑专精模型 + MIMIC-CXR 官方全量 split（macro CE）+ IU 零样本 + ReXrank + RadCliQ 1.14 全场最佳"阵地是竞品盲区，也是你复现资产所在的阵地。

**推荐的创新组合（"DDaTR+"，三模块全部插进现有架构，不动数据管线主结构）**：

| # | 模块 | 打的弱点 | 一行卖点 | 预期增益 | 成本 |
|---|---|---|---|---|---|
| **M1** | **Temporal Registration（编码器前置轻量形变配准）** | 先验/当前图体位视角错位污染像素差（综述点名全领域无人做配准） | "warp-then-difference"：配准后坐标系里算差，只留真实变化 | CE F1 ↑（P/R 双升，recall 主升） | 中（~5M 参数，自监督 NCC，零标注） |
| **M2** | **Interval-Conditioned Dynamic Fusion（时间间隔条件化门控）** | 完全无视检查间隔（1天 vs 1年语义完全不同）；TiBiX/HERGen 只做了浅层输入编码 | Δt 可学习嵌入调制 DDAM/DFAM 门控与差分量级 | CE +1~2%（HERGen 实证量级）+ 进展句 BLEU/ROUGE | **低**（<50 行，数据现成） |
| **M3** | **Progression-Event Supervision + recall 加权（进展事件监督）** | recall 是领域公认最短板（DDaTR 自己消融证明动态融合只买 precision；CXRMate-2 放射科盲评"recall 是主要剩余障碍"） | 用先验/当前 CheXbert 标签自动构造 14×3 进展事件（新发/消退/稳定），进展头监督 + 软进展门控 + 非对称损失 | recall 0.465→0.49+，CE F1 直接抬 | 低-中（真实数据：87.5% 纵向对含新发/消退事件） |
| **M4（推理期，可选叠加）** | **CheXbert-in-the-loop 临床反馈回路** | hallucination/遗漏（论文案例自证） | 草稿→CheXbert 提取标签↔模型分类头比对→定位漏报/误报→纠错 prompt 二次解码 | CE 推理端即提，零训练成本 | 低（纯推理工程） |

**可选二期（CE 最大杠杆，成本最高）**：GRPO/DPO 强化微调，奖励 = CheXbert F1 + F1-RadGraph + 进展一致性（EditGRPO 式句子级编辑），参考 EditGRPO 平均 +3.4%/OOD +5.9%。

**不建议做**（撞车/高成本低回报）：纯差分解码（已被 DDaTR/BiOT/Diff-RRG/ALTER 占满）、多视图对比学习（MLRG 已占且去掉对比学习 F1 掉 1.2 点）、检索增强（TMI 2025 已做纵向 RAG）、frozen-LLM 路线（LLM-CXR 四榜垫底已死）、把解码器换成大 LLM 作为主创新（工程量大且偏离现有资产，只能作二期或第二篇）。

---

## 2. 领域版图（谁在 2025-2026 做了什么）

### 2.1 直接竞品（纵向 RRG）

| 论文 | 出处 | 方法核心 | Longitudinal-MIMIC 关键数字 |
|---|---|---|---|
| HERGen | ECCV 2024 | 组因果 Transformer（≤5 次就诊）+ date embedding + 课程学习 | B-4 0.117 / R-L 0.282 |
| MLRG | CVPR 2025 | 多视图×纵向对比预训练 + 缺失先验 tokenization + INDICATION | MIMIC F1 0.364ma/0.505mi，B-4 0.158 |
| HC-LLM | AAAI 2025 | time-shared/specific 特征分解 + 多模态对比/结构约束，LLM | B-4 0.142 / F1 0.357ma |
| Diff-RRG | MICCAI 2025 | 病种级 patch 差引导 LLM（BiomedCLIP+BioMistral） | F1 0.474mi |
| STREAM | TMI 2025 | 时空（近两次先验）+ 检索增强（SwinT+TinyLlama） | F1 0.478mi，B-4 0.139 |
| **TIM** | **CVPR 2026** | 静态病理/动态进展双分支解耦 + 先验-当前报告互精修（LLM） | B-1 0.430 / B-4 0.124 / MTR 0.185 / R-L 0.287 / F1 0.511mi |
| **BiOTPrompt** | **CVPR 2026** | 双向最优传输，非对称 transport plan 定位新出现/消退区域→LLM prompt | B-4 0.126 / F1 0.417ma（对比表未列 DDaTR） |
| **MARE** | **AAAI 2026** | 类比推理（Abduction-Mapping-Induction）+ ARA 区域对齐 + 双演化一致性损失（LLM） | B-4 0.133 / R-L 0.291 / F1 0.375mi |
| PriorRG | AAAI 2026 | 先验引导对比预训练 + 粗到细解码 | F1 0.376ma/0.511mi |
| CURV | NeurIPS 2025 | 不确定性感知 VLM（单图） | — |

### 2.2 关键结论（来自综述 [arXiv:2510.12444](https://arxiv.org/abs/2510.12444) 与逐篇核对）

1. **两套协议不可混比**：MIMIC-CXR 用 **macro** CE（RECAP/DDaTR 协议），Longitudinal-MIMIC 用 **micro** CE（LR2Gen 协议）。新论文普遍只报一个。
2. **DDaTR 的资产**：RadCliQ **1.14 是纵向赛道全场最低（最好）**；CE macro 0.441 接近 RADAR 0.460；但 **BLEU-4 0.113 被 RADAR 0.262 / Libra 0.245 / MAIRA-2 0.231 甩开 100%+**——瓶颈在 BERT 解码器端，不在纵向前瞻立意。
3. **综述点名的五大局限 = 你的创新缺口**：① 无配准（"迄今没有任何 LRRG 方法做配准"）；② 时间间隔建模不足；③ 多先验冗余无路由；④ 跨期配准不足（同 1）；⑤ 缺纵向专用评测。
4. **ReXrank 实时榜单（2026 快照）**：前段被 Deepwise-RG/CARE-X/UniRG-CXR/CheXOne-R1/MedGemma 大 VLM 占领；但**小模型未死**（MoERad-IU 在 IU 第 2、Cvt2distilgpt2 中游、GREEN 0.51+），frozen-LLM 已死（LLM-CXR GREEN 0.044 垫底）。指标主口径已从 BLEU 迁移到 RadCliQ/CRIMSON/GREEN/RadGraph。

---

## 3. 我实测的本仓库数据证据（2026-09-02，零标注成本）

用 `data/mimic_cxr/mimic_cxr_ddatr.json` 直接统计：

| 统计项 | 数值 | 意义 |
|---|---|---|
| 有先验样本 | train 54%（146,061）/ test 82%（3,153） | 与论文一致 |
| **含新发(neg→pos)事件**的标签对 | **8.91%** | M3 监督信号充足 |
| **含消退(pos→neg)事件**的标签对 | **8.58%** | 同上 |
| **≥1 个新发/消退事件的 study 对** | **87.5%**（131,543/150,395） | 进展描述是任务核心 |
| 先验-当前间隔（time 字段，150,395 对可解析） | min 1 / p25 6 / **median 114** / p90 10,796 / max 50,001 | 间隔分布极度偏斜（同日复查→数年），M2 数据条件完美 |
| UNC(3) 标签 | 173,716 个（~2.3%） | M4/不确定校准有真实信号 |
| ViewPosition | PA/AP/Lat 等，逐条存在 | 配准/视图条件可用 |

注：`time` 字段单位非直观（疑似分钟级时间戳，21800506 → 2011-06-14 05:46），取差值用即可；先验时间需用 study_id 查表（`context_image` 路径含先验 study_id，JSON 无 context_time 字段，需一次性预处理）。M1/M2/M3 全部零新标注。

---

## 4. 推荐创新点详细设计（DDaTR+ 三模块 + 一期推理回路 + 二期 RL）

### M1：Temporal Registration Module（编码器前置形变配准）
- **动机**：DDaTR 的 DDAM 实现（`resnet.py` `FeatureFusion`）实际是**通道维全局池化的标量差**（`AvgPool→Sigmoid`），对 AP/PA/侧位混合、体位变化、胸廓膨胀极其敏感——真实临床变化被体位噪声淹没。放射学界的 temporal subtraction（经典 warping 系列）证明配准后差值的读片敏感度显著提升，但**现代配准从未接入 LRRG 生成器**。
- **设计**：轻量 RegNet（共享下采样 → U 形 4 层 conv → 28×28 形变场 + 可选 6-DoF 仿射头，~5M 参数）+ STN 双线性 warp，只 warp 先验图到当前坐标系；自监督 `L_reg = −λ1·NCC(x1, warped(x2)) + λ2·‖∇φ‖²`，无需标注。建议仿射+形变两阶段（stage0/1 像素级、stage2/3 特征级），形变正则先强后弱 warm-up。
- **差异化**：MARE 用 perceiver 只对齐显著区域 token；BiOTPrompt 用 OT 全局匹配但无解剖一致性约束；我们是**稠密空间形变 + 显式几何一致性 + 配准场本身作为新临床特征**（如肺容积变化率）。
- **插入位置**：`resnet.py` forward 开头（`self.embeding` 之前或 stage0 入口）。
- **预期**：precision（差特征更干净）+ recall（新发异常不再被体位噪声冲淡）双升。

### M2：Interval-Conditioned Dynamic Fusion（时间间隔条件化门控）
- **动机**：DDaTR 完全无视间隔语义（1 天 vs 1 年的"变化"含义完全不同）。TiBiX（间隔 token）、HERGen（date vocabulary）只做到输入编码；**无人做"间隔→差分门控/进展监督"联动**（综述 §7.1.2 点名）。
- **设计**：`Δt_emb = MLP(log(1+Δt))`（无先验时 Δt=0），在 `resnet.py` L103/L107 把门控改为 `alpha * has_progress * sigmoid(g_Δt)`，并给 DFAM 的 key/value 加间隔 bias；可选辅助头做 Δt 桶分类（软标签，w=0.1）。`dataset/medical_dataset.py` 增加返回 time_interval 字段。
- **临床直觉**：短间隔 = 变化多为管理决策（重症监护），长间隔 = 慢性演变；间隔编码应**缩放差分支增益**（短间隔抑制伪变化、长间隔放大真实演变）。
- **预期**：CE +1~2%（HERGen date embedding 实证 NLG +5.9%/CE +2.0% 的量级）、复发-缓解型发现的进展句 BLEU/ROUGE 提升。

### M3：Progression-Event Supervision + Recall-Weighted Loss（进展事件监督）
- **动机**：领域级结论——所有模型 recall < precision；DDaTR 自己消融证明动态融合只买 precision（w/o DF 时 recall 0.465 纹丝不动）；CXRMate-2 放射科盲评宣布 recall 是主要剩余障碍；**2024-2026 没有任何 LRRG 论文以 recall/完整性为主目标**。
- **设计**：
  1. 监督构造：对训练样本用现有 CheXbert（`checkpoints/stanford/chexbert/chexbert.pth` 已在仓库）打先验/当前 14 类标签 → 进展三元组 `prog_k = sign(p_cur − p_prior) ∈ {新发/加重, 稳定, 消退/减轻}`（14×3）。数据实测：87.5% 纵向对含事件→信号充足。
  2. `ProgressHead: Linear(1024→14×3)` 挂在 `avg_feats`，`loss_prog = CE` 加入总损失；
  3. **软进展门控**：预测的进展强度 σ 缩放每 stage 门控（与 M2 的间隔门控相乘）；
  4. **recall 加权**：分类头 CE 换 CheXpert 类别加权 focal/asymmetric loss（stable-neg 占 74.7%，极端长尾），对 new/worse 事件上调权重。
- **差异化**：RECAP 的进展监督用外部 ChestImaGenome 标注（排除侧位、需外部数据）；我们是**全量 split 上 CheXbert 自动标签转移，零外部数据**，且与间隔（M2）、门控（软进展）联动。
- **预期**：recall 0.465→0.49+，CE F1 直接抬（macro/micro 双协议都受益）。

### M4（推理期，可选）：CheXbert-in-the-loop 临床反馈回路
- 生成草稿 → CheXbert 提取 14 标签 → 与模型当前分类头预测比对 → 定位漏报（先验有/模型分类头有但草稿没写）与幻觉（草稿写了但分类头无证据）→ 构造纠错 prompt（漏报病种强制 [POS]/[NEG] 前缀）→ 二次 beam 解码。
- 直击论文案例研究的 hallucination/遗漏；零训练成本、与 M1-M3 正交；可作为论文"应用/推理"章节 + 与 M3 的 recall 叙事呼应。

### 二期（可选，CE 最强杠杆）：GRPO/DPO 强化微调
- 奖励 `r = CheXbert F1(y) + β·F1-RadGraph(y) + γ·进展一致性(y, r_prior)`；EditGRPO 式句子级编辑（FP 句替换、FN 句补入）构造偏好对；冻结视觉编码器或 LoRA，只微调解码器。
- 证据：EditGRPO 平均 +3.4%、OOD +5.9%；CLARIFID 证明 CheXbert F1 当奖励有效。注意 BLEU 不必然升——以 CE 为准（符合 DDaTR "CE 优先" 定位）。
- 成本：训练时间 ×2 量级；建议作为二期或投稿前的最后一击。

**组合预期**：M1+M2+M3 编码端协同 + （可选）M4 推理端 → CE F1 0.441→0.455~0.47、recall 0.465→0.49+、BLEU/METEOR 持平或微升；二期 RL 后可再 +2~4%。该幅度对应 Longitudinal-MIMIC micro 协议可冲 0.54+。

---

## 5. 投稿策略（一区必备）

### 5.1 基准与对比（关键！）
- **必须两套协议双高**：MIMIC-CXR 官方 split（macro CE，对标 DDaTR 0.441）+ Longitudinal-MIMIC（micro CE，对标 0.527）+ IU-Xray 零样本（0.262）+ ReXrank 提交分。
- **对比对象**：DDaTR（你自己的复现基线）+ 尽可能复跑新竞品（TIM/MARE/BiOTPrompt/Diff-RRG/MLRG 代码均已开源，GitHub 已确认 200 可访问）——**统一口径复跑是 DDaTR 论文自己都没做、而审稿人最想看到的事**；至少也要引用其自报数字并注明协议。
- **注意口径陷阱**：BiOTPrompt 报 Long-MIMIC macro 0.417 却自称 SOTA（DDaTR 的 micro 0.527 不在其表）；TIM 的 F1 0.511 低于 DDaTR 0.527。**写清楚 macro/micro 与是否含首诊，是防止审稿人质疑的第一道防线。**

### 5.2 指标矩阵（TMI/MedIA 硬门槛）
- lexical：BLEU-1/4、METEOR、ROUGE-L（pycocoevalcap 口径，与 DDaTR 一致）
- 临床：CE P/R/F1（macro 与 micro 双列）+ **F1-RadGraph + RadCliQ**（ReXrank 口径）
- 纵向专用：**GREEN 的 e（幻觉比较既往）/f（漏比较既往）子类** + 可选 TEM/CLEAR——你的 M1-M3 卖点必须用这些指标验证，否则审稿人会说"指标没测到你 claim 的东西"
- 统计：bootstrap 95% CI / 显著性（比赛 p<0.05 口径），多 seed
- 案例：对比句高亮（progressions）+ Grad-CAM + 错误分析（漏报/误报/幻觉各举 2-3 例）
- 效率：参数量/FLOPs/推理时延 vs TIM/MARE/BiOTPrompt（都是 LLM 重模型）——**你的紧凑模型效率优势是 TMI 愿意听的叙事**

### 5.3 送审死法清单（避坑）
- ❌ 只报 BLEU/ROUGE（词面指标已边缘化）
- ❌ 只对比 2023-2024 基线（审稿人会拿 RADAR/Libra/MAIRA-2/MedGemma 打你）
- ❌ 纵向前瞻不报 GREEN e-f/TEM/CLEAR
- ❌ 用非官方测试拆分（ReXrank/官方 2,347~3,858 拆分）
- ❌ 换解码器/分辨率不谈机制与临床语义绑定
- ❌ 声称"超越 DDaTR"却不用同协议（先跑通 DDaTR 官方 eval 脚本再说话）

### 5.4 论文叙事建议（避免被当"又一个注意力模块"）
- 故事主线：**"从'看到差异'到'读懂变化'：对齐、定时、定事件的纵向报告生成"**——M1 解决"差异在哪"（配准），M2 解决"差异多紧要"（间隔），M3 解决"差异是什么"（事件监督），推理回路解决"写对了没"（临床反馈）。
- 定位：紧凑专精模型 vs LLM 重模型的效率/临床双赢（延续 DDaTR 与 RadFM/LLaVA-Med 对比的叙事，但现在对手换成 TIM/MARE/BiOTPrompt）。
- 每个模块一个消融 + 间隔/视角/先验有无的纵向因素消融（综述明确这是纵向评审重点）。

---

## 6. 实施路线（建议 6-8 周）

| 阶段 | 内容 | 交付物 |
|---|---|---|
| W0（现在） | 训练跑完（Epoch 7/10，约剩 12h）→ 换权重 → 全套基线指标（MIMIC 官方 split + Long-MIMIC + IU 零样本 + ReXrank） | 基线表（含 papier 目标对照） |
| W1 | **M2 时间间隔门控**（改 dataset + resnet.py 门控，~50 行） | Δt 版 vs 基线：CE F1 对比 |
| W2-3 | **M3 进展事件监督**（CheXbert 标注构造 → ProgressHead + 软门控 + focal 加权） | 消融：w/o prog head / w/o 软门控 / w/o focal |
| W4-5 | **M1 配准前置**（RegNet 联合训练，先单训配准 1 epoch warm-up） | 消融：w/wo warp、仿射 vs 形变 |
| W6 | **M4 推理回路**（CheXbert 校验 + 纠错再解码） | 推理端 CE/GREEN 对比 |
| W7-8 | 二期可选 RL（GRPO/DPO）+ 全套指标 + 显著性 + 案例可视化 + ReXrank 提交 | 投稿初稿 |

**先验建议**：以 M2 为第一刀（成本最低、HERGen 有实证、失败风险小）；M3 为主菜（recall 是数字上最确定的增量）；M1 为差异化核心（物理上最干净、综述点名空白）；M4 为论文增色。若时间紧，M2+M3 已足以构成一篇 TMI 投稿（两个空白 + 一个公认弱点），M1 作为强扩展。

---

## 7. 风险与预案

| 风险 | 预案 |
|---|---|
| M1 配准在 CXR 上收敛差（肋骨投影变化、低纹理） | 仿射优先 + 限制形变正则 + 只在 stage0/1 像素级；若无效退化为"特征级 soft 对齐"（correlation volume）+ 循环一致性，仍区别于 MARE/BiOT |
| M3 的进展监督与 RECAP 撞车质疑 | 强调三点差异：零外部标注（CheXbert 自动转移）、全量 split 含侧位、间隔×事件×门控联动（RECAP 无） |
| M2 与 TiBiX/HERGen 输入编码撞车 | 明确卖点是"间隔条件化差分门控/进展监督"而非"间隔进输入"；消融里做对比（输入编码 vs 门控调制） |
| 只报 CE 被质疑"NLG 不行" | BLEU 持平即可（DDaTR 自身定位就是 CE 优先）；用 GREEN-e/f 与进展句案例证明临床价值；二期 RL 或轻量解码升级（Transition-Aware best-of-N 免训练采样、对比解码）可补 NLG |
| 审稿人拿 RADAR/Libra 词面差距攻击 | 定位清楚：本文目标是 compact model + CE/临床口径；RADAR/Libra 是 384-518 分辨率 + LLM 解码的另一个复杂度级别；报参数效率对比表 |
| 2026 下半年又出竞品 | 投稿前再审一遍 arXiv（LRRG 榜），必要时把 M1-M3 组合的"联动"作为核心卖点（单模块容易被跟，三模块联动 + 推理回路难被复刻） |

---

## 8. 关键参考链接

- DDaTR：[arXiv:2505.03401](https://arxiv.org/abs/2505.03401) ｜ 代码：[xmed-lab/DDaTR](https://github.com/xmed-lab/DDaTR)
- LRRG 综述（5 大局限 + 全对比表）：[arXiv:2510.12444](https://arxiv.org/abs/2510.12444)
- TIM (CVPR'26) 代码：[yihengd/TIM](https://github.com/yihengd/TIM) ｜ MARE (AAAI'26) 代码：[gaoqingqing77/MARE](https://github.com/gaoqingqing77/MARE) ｜ BiOTPrompt (CVPR'26) 代码：[TengfeiLiu966/BiOTPrompt](https://github.com/TengfeiLiu966/BiOTPrompt) ｜ Diff-RRG (MICCAI'25) 代码：[ku-milab/Diff-RRG](https://github.com/ku-milab/Diff-RRG)
- MLRG (CVPR'25)：[arXiv:2502.20056](https://arxiv.org/abs/2502.20056) ｜ HC-LLM (AAAI'25)：[arXiv:2412.11070](https://arxiv.org/abs/2412.11070) ｜ PriorRG (AAAI'26)：[arXiv:2508.05353](https://arxiv.org/abs/2508.05353)
- HERGen（date embedding 实证 +5.9%/+2.0%）：[arXiv:2407.15158](https://arxiv.org/abs/2407.15158) ｜ TiBiX：[arXiv:2403.13343](https://arxiv.org/abs/2403.13343)
- Transition-Aware best-of-N（免训练纵向采样）：[arXiv:2606.28393](https://arxiv.org/abs/2606.28393)
- EditGRPO（RL +3.4%/+5.9%）：[arXiv:2509.22812](https://arxiv.org/abs/2509.22812) ｜ CLARIFID：[arXiv:2507.17234](https://arxiv.org/abs/2507.17234) ｜ CXRMate-2（GRPO，recall 结论）：[arXiv:2604.18967](https://arxiv.org/abs/2604.18967)
- 配准：VoxelMorph [arXiv:1809.05231](https://arxiv.org/abs/1809.05231) ｜ 无监督 temporal CXR 配准 [SPIE'20](https://journals.spiedigitallibrary.org/conference-proceedings-of-spie/11313/113132X/Unsupervised-learning-based-deformable-registration-of-temporal-chest-radiographs-to/10.1117/12.2549211.short) ｜ 经典 temporal subtraction 观察者研究 [Radiology 1997](https://pubmed.ncbi.nlm.nih.gov/9015072/)
- 不确定性：CURV (NeurIPS'25) [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/86a99b74adac7c77998902371f53850d-Abstract-Conference.html) ｜ semantic-consistency UQ [NAACL-F'25](https://aclanthology.org/2025.findings-naacl.95/)
- 纵向专用评测：GREEN [项目页](https://stanford-aimi.github.io/green.html) ｜ LLM 标注纵向基准 [arXiv:2601.16753](https://arxiv.org/abs/2601.16753)
- ReXrank：[rexrank.ai](https://rexrank.ai) ｜ 榜单论文 [arXiv:2411.15122](https://ar5iv.labs.arxiv.org/html/2411.15122) ｜ CRIMSON：[arXiv:2603.06183](https://ar5iv.labs.arxiv.org/html/2603.06183)