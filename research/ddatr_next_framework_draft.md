# DDaTR-NEXT：BLEU×CE 双指标协同提升框架设计（工作稿）

> 2026-09-02 ｜ 用户新硬性要求：**BLEU 与 CE F1 必须同时提升**（优于当前所有新方法——无人在 DDaTR 相同协议上双超它）
> 状态：工作稿，两个专项子代理（双指标归因、48G 可行性）返回后定稿。

## 0. 核心判断（已自行验证的证据）

1. **BLEU 的最大杠杆是解码器**：所有 LLM 解码器方法 BLEU-4 ≥0.139（STREAM 224 输入 TinyLlama 0.139 / HC-LLM 0.142 / LLM-RG4 0.144），而 BERT 解码器方法基本 ≤0.12（DDaTR 0.113、Diff-RRG 0.120）。**同 224 分辨率下 LLM 解码器仍净赚 +0.03~0.04 BLEU-4。**
2. **进展建模组件双指标齐涨（TIM 消融实证）**：DPM 动态进展分支 F1 0.404→0.427 且 BLEU-1 0.396→0.414；DRG 双时相生成抬 B-4/F1；Lspr 抬 F1，Ldpm 抬 B-4——**进展句是 GT n-gram 匹配的主要来源，M2/M3 类模块天然双收益**。
3. **LLM-RG4 证明双高可行**：LLM 解码器 + 自适应 token 融合 + **token 级损失加权（重视 positive/uncertain 描述）** → CE 与 NLG 同时 SOTA（其 BLEU-4 0.144 / F1 micro 0.588，协议与 DDaTR 不同但方向成立）。
4. **RADAR 证明"上下文+结构+解码参数"是 BLEU 增益集**：384 分辨率 + SwinV2-large + Perceiver(128 token) + Phi-3-mini + 临床上下文（Indication/History/Comparison/Technique/Prior Findings）+ **length penalty 2.0** → BLEU-4 0.262（frontal 协议）；其 PF（先生成观察再生成报告）结构与 DDaTR 的 diagnosis prompt 同源。
5. **51% 上限风险**：MRG-R1 的语义 RL 明确"token 级似然只奖励表面匹配"——**纯 CE 奖励的 RL 可能伤 BLEU**，必须混合奖励或分段放弃，证据待子代理回归。
6. **数据缺口**：仓库 JSON 无 INDICATION 字段（RADAR/MLRG 的 BLEU 大杠杆），需下载 MIMIC-CXR 元数据 CSV（PhysioNet 凭据）补进管线——**值得做**。

## 1. 框架草稿：DDaTR-NEXT（Plan-then-Generate + Lightweight-LLM + Mixed-Reward）

```
┌────────────────────────── 编码端（继承 DDaTR + M1/M2/M3）──────────────────────────┐
│ 当前图 ──► SwinT-B ──┐                                                              │
│ 先验图 ──► [M1 配准] ─┴─► stage×4（DFAM/DDAM 门控受 M2 间隔调制 + M3 软进展门控）      │
│ 先验报告 ─► BERT(冻结) ─► 文本特征 ────────────────────────────────► 纵向视觉特征 F   │
│ 先验/当前 CheXbert 标签 ─► [M3 ProgressHead] ─► 进展事件三元组(14×3) + 软门控         │
│ Δt ─► [M2 IntervalEmbed] ─► 门控调制                                                   │
└────────────────────────────────┬──────────────────────────────────────────────────┘
                                 ▼
┌────────────────────────── 规划端（新，BLEU+CE 核心新模块）─────────────────────────┐
│ T1. 诊断 prompt（继承）：[BLA/POS/NEG/UNC]×14（当前）                                 │
│ T2. 进展 prompt（新）：由 M3 进展事件 → 主题槽规划：                                  │
│     {topic_1: 'pleural effusion', status:'new', interval:'short', ...}               │
│     主题排序策略：新发/恶化 > 消退/好转 > 稳定阳性 > 阴性（对应 GT 报告的行文顺序）     │
│     每个主题槽附加进展词表约束（“newly appeared/decreased/increased/unchanged…”）     │
└────────────────────────────────┬──────────────────────────────────────────────────┘
                                 ▼
┌────────────────────────── 生成端（新）──────────────────────────────────────────────┐
│ 轻量 LLM 解码器（TinyLlama 1.1B / Phi-3-mini 3.8B，LoRA/QLoRA，bf16）                 │
│ 输入：纵向视觉特征 F（Perceiver/交叉注意力压缩为 64-128 token）                       │
│      + 规划 prompt（诊断+进展+间隔）+ 先验报告（可截断）                              │
│ 输出：Findings 段落（plan-then-generate：先主题句骨架，再逐槽填充）                   │
└────────────────────────────────┬──────────────────────────────────────────────────┘
                                 ▼
┌────────────────────────── 优化端（新）──────────────────────────────────────────────┐
│ Phase1 SFT：LM + 分类双头 + M3 进展头（token 级损失加权：positive/uncertain ↑）       │
│ Phase2 混合奖励 RL（GRPO，EditGRPO 式句子级编辑）：                                    │
│   r = λ1·CE-F1(CheXbert) + λ2·F1-RadGraph + λ3·进展一致性 + λ4·BLEU/ROUGE(保底)      │
│ Phase3 推理端：Transition-Aware best-of-N 采样 + length penalty 调参 + （可选）        │
│        CheXbert-in-the-loop 校验回路（M4）                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**各组件对双指标的预期贡献（标注证据来源）**：
| 组件 | BLEU | CE F1 | 证据 |
|---|---|---|---|
| 轻量 LLM 解码器 | **+0.03~0.13**（B-4） | 0~-0.02（需纵向模块补） | STREAM/HC-LLM/LLM-RG4/RADAR |
| M2 间隔调制 | +0.005~0.01 | +0.01~0.02 | HERGen date embedding +5.9%NLG/+2.0%CE |
| M3 进展事件+软门控+token加权 | +0.005~0.01 | +0.015~0.03（recall 主升） | TIM DPM/Lspr/Ldpm 消融、LLM-RG4 token 加权 |
| M1 配准 | ~0 | +0.01~0.02（P/R 双升） | 综述"零配准"空白、temporal subtraction 文献 |
| 主题槽规划+进展词表 | +0.01~0.02 | +0.005~0.015 | RADAR PF 结构、ICLR'26 topic-guided（待证） |
| 混合奖励 RL（含保底 BLEU） | ±0~0.01 | +0.02~0.04 | EditGRPO +3.4%、MRG-R1（须混合） |
| 免训练采样（best-of-N/对比解码） | +0.005~0.02 | +0~0.01 | Transition-Aware best-of-N |
| 高分辨率 384 + INDICATION | +0.01~0.03 | +0~0.01 | RADAR/Libra/MLRG（48G 预算受限，可选） |

**预期总账（DDaTR 协议）**：MIMIC macro：BLEU-4 0.113→**0.14~0.16**，F1 0.441→**0.455~0.475**；Long-MIMIC micro：BLEU-4 0.105→**0.12~0.14**，F1 0.527→**0.53~0.55**。→ **"首个在 DDaTR 双协议上同时超越其 BLEU 与 F1 的方法"**（待子代理确认无先例）。

## 2. 待子代理确认的问题
- [ ] MRG-R1/CLARIFID/EditGRPO 消融中 RL 前后 BLEU 变化（纯 CE 奖励是否伤 BLEU）
- [ ] 有没有已存在的方法在完全同协议上双超 DDaTR（BLEU-4>0.115 且 F1macro>0.445 / Long-MIMIC B-4>0.11 且 F1micro>0.53）
- [ ] topic-guided（ICLR'26）/self-critiquing/Phrase-grounded APO 的具体数字
- [ ] TinyLlama/Phi-3-mini 在单卡 24G（RTX 3090）的 LoRA/全参训练显存与时间
- [ ] 对比解码/best-of-N/自一致性在 RRG 的量化效果
- [ ] 224→384 的显存与 BLEU 增益平衡

## 3. 本地可行性实测（2026-09-02，比交接文档更新的硬约束）

| 项 | 实测 | 对框架的影响 |
|---|---|---|
| **GPU** | **RTX 3090 24G**（nvidia-smi 实测；交接文档写的是 RTX 5880 Ada 48G，已不符） | LLM 解码器只能 **QLoRA/LoRA**；Phi-3-mini 需 4bit；全参微调 1.1B+ 不可行；384 分辨率基本放弃（除非牺牲 batch）；**现有 DDaTR 训练占 22.1G/24G，跑完前不能并行开新训练** |
| 环境 | torch 2.1.1+cu118 / transformers 4.25.0；**peft/bitsandbytes/accelerate/flash_attn 均未装** | LoRA 路线需新 conda env（clone ddatr + 装 peft/bnb；transformers 升级到 ≥4.33 以支持 TinyLlama chat 模板）——**不要动正在训练的环境** |
| **HF 缓存** | **TinyLlama-1.1B-Chat-v1.0 已在** ~/.cache/huggingface/hub | LLM 路线早有准备；1.1B QLoRA 在 24G 上可行（~12-16G，batch 4-8） |
| 磁盘 | 5.6T 空闲 | 无约束 |
| M3 事件信号质量 | 抽查报告文本：raw 标签转移有噪声（CheXbert 措辞误判、UNC 翻转）；**UNC 无关的"干净"事件仍有 26.2 万+（训练集内 new 6.48% + resolved 6.35%，占全部转移的 73.6%）** | M3 设计规则：UNC-free 转移做硬目标、UNC 参与做软目标/忽略；软门控来自模型自身分类头而非纯 GT |
| 解码器现状 | `blip.py` generate：beam=3、rep_penalty=1.0、无 length_penalty、min_len=10/max_len=100 | 免费杠杆：length_penalty≈2.0（RADAR）、beam 4-5、rep_penalty 1.05、no_repeat_ngram——一行改动，先测基线收益 |