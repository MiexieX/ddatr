# 双指标协同提升专项报告：BLEU × CE F1 同时超越 DDaTR

> 2026-09-02 ｜ 补充调研（在《innovation_research_final.md》之后，针对用户新要求"BLEU 与 F1 必须同时提升"）
> 配套工作稿：`research/ddatr_next_framework_draft.md`

---

## 1. 核心定位（已被数据证实）

**在 DDaTR 自己的协议上"BLEU-4 与 CE F1 同时超越 DDaTR"至今无人做到，这是最大的论文卖点：**

| 协议 | DDaTR 基线 | 历史上 B-4 最高者 | 历史上 F1 最高者 | 双超者 |
|---|---|---|---|---|
| MIMIC-CXR 官方 3,858（pycocoevalcap，CE macro） | B-4 0.113 / F1 0.441 | R2GenGPT 0.134（F1 仅 0.392） | DDaTR 本人 | **无** |
| Longitudinal-MIMIC（CE micro） | B-4 0.105 / F1 0.527 | HC-LLM 0.142（F1 0.498）| DDaTR 本人 | **无** |

（依据：DDaTR 论文 Tab I 本人逐值核对 + LRRG 综述 arXiv:2510.12444 Tables 2-5 的交叉核对；所有后起方法都是"BLEU 涨、F1 跌"或反之。）

## 2. 为什么"双涨"难（权衡机制）

- MRG-R1 (arXiv:2512.16145) 明说：**token 级似然训练只奖励表面匹配（BLEU），不约束临床正确性**；反过来，纯 CE 奖励的 RL 可能牺牲词面质量（REVA-PO 论文也指出 RL 的 KL 正则累积会压死 BLEU 探索）。
- 领域里两类方法的普遍结局：LLM 解码器路线 BLEU 大涨 F1 小跌（HC-LLM/STREAM/Diff-RRG）；CE 优先路线 F1 大涨 BLEU 平/跌（MRG-R1 系）。
- **突破口 = 组件分工 + 稳定化混合优化 + 结构化解码**（见第 4 节）。

## 3. 已证实的"双涨"先例（逐个核实）

| 方法 | 机制 | BLEU 变化 | CE/F1 变化 | 协议/备注 |
|---|---|---|---|---|
| **TIM Stage I→II**（CVPR 2026） | 动态进展分支(DPM)+双时相生成(DRG)+互精修 | B-1 0.396→0.421，B-4 0.108→0.117+（Stage II 0.124） | F1 0.404→0.483→0.511 | 纵向；进展组件双指标齐涨（消融表 2/3 逐项核实） |
| **LLM-RG4**（AAAI 2025） | LLM 指令微调 + 自适应 token 融合 + **token 级损失加权（positive/uncertain）** | B-4 0.144（vs DDaTR 0.113） | F1 micro 0.588 | MIMIC-RG4/MIMIC-CXR；自定义多场景数据集 |
| **REVA-PO**（arXiv 2607.10147） | 稳定化 RL：**BLEU-4 直接做奖励** + RER 动态 KL + VAPR 锚定重置；三阶段（warm-up→分类器引导 SFT→RL） | **BLEU-4 +5.1%（MIMIC）/ +3.6%（IU）** | **CheXbert F1 +4.5%、RadGraph +12.8%** | 单图 RRG；8×V100——"BLUE 奖励同时抬 F1"的最硬证据 |
| RADAR（ACL 2025） | 384 分辨率 + SwinV2-large + Perceiver + Phi-3-mini + 临床上下文 + 检索 | B-4 0.262 | F1 14-Ma 0.460（Unc-as-Neg） | frontal 协议；临床上下文（Indication/History/Prior Findings）是重要 BLEU 杠杆 |
| TIM 对比集里 | 时空解耦 + LLM | vs RADAR B-4 +0.018，vs Diff-RRG F1 +0.037 | | 纵向 |
| **Transition-Aware best-of-N**（arXiv 2606.28393） | 免训练纵向感知采样 | 优于随机采样（Impression 段增益最大） | 同上 | 可叠加任何生成器 |

**结论**：要让 BLEU 与 F1 双涨，需要**三个杠杆叠加**：① 更强的解码器/解码过程（BLEU 主力）；② 进展/间隔/对齐类编码端模块（F1 主力，且 TIM 证明同时小幅抬 BLEU）；③ 稳定化混合奖励优化或 BLEU-奖励 RL（REVA-PO 证明能同时抬 F1/RadGraph）+ 免训练采样兜底。

## 4. DDaTR-NEXT 框架定稿（详见 `ddatr_next_framework_draft.md`）

```
编码端（继承+三模块）: SwinT-B + M1配准 + M2间隔门控 + M3进展事件/软门控
        │ 纵向视觉特征 + 诊断prompt(14类) + 进展prompt(主题槽规划, 新)
        ▼
生成端（新）: 轻量LLM解码器（TinyLlama 1.1B/Phi-3-mini，LoRA bf16）
        输入 = 压缩视觉token + 诊断prompt + 主题槽规划 + 先验报告
        ▼
优化端（新）: 
  Phase 1 SFT（LM + 分类头 + M3进展头 + token级损失加权[positive/uncertain]）
  Phase 2 稳定化RL（GRPO + REVA-PO式动态KL/锚定重置；奖励=B-4 + CE-F1 + F1-RadGraph + 进展一致性 的混合，或先BLEU为主）
  Phase 3 推理端：Transition-Aware best-of-N + length_penalty调参 + CheXbert回路(可选)
```

**各组件对双指标的归因表**（证据见上表与两份子代理报告）：

| 组件 | BLEU 预期 | F1 预期 | 证据等级 |
|---|---|---|---|
| 轻量 LLM 解码器（LoRA） | **+0.03~0.13**（B-4） | ~0~-0.02（需纵向模块补） | 高（STREAM/HC-LLM/LLM-RG4/RADAR） |
| M2 间隔调制 | +0.005~0.01 | +0.01~0.02 | 中高（HERGen +5.9% NLG/+2.0% CE） |
| M3 进展事件+软门控+token加权 | +0.005~0.01 | +0.015~0.03 | 中高（TIM 消融、LLM-RG4） |
| M1 配准 | ~0 | +0.01~0.02 | 中（物理直觉+Temporal subtraction 文献） |
| 主题槽规划+进展词表 | +0.01~0.02 | +0.005~0.015 | 中（RADAR PF、ICLR'26 topic-guided 待子代理确认） |
| 稳定化 RL（REVA-PO 式） | **+0.005~0.015** | **+0.02~0.04** | 高（REVA-PO B-4+5.1%/F1+4.5%/RG+12.8% 同涨） |
| 免训练采样（best-of-N/对比解码） | +0.005~0.02 | +0~0.01 | 中（Transition-Aware、子代理2待报） |
| INDICATION 上下文（需下载元数据CSV） | +0.01~0.03 | +0~0.01 | 高（RADAR/MLRG 用，MLRG 摘要含 INDICATION） |
| 384 分辨率 | +0.01~0.03 | +0~0.01 | 中高（RADAR/Libra/MLRG 用 384+） |

**目标账**（DDaTR 协议）：MIMIC macro：B-4 0.113→**0.13~0.16**，F1 0.441→**0.455~0.475**；Long-MIMIC micro：B-4 0.105→**0.12~0.14**，F1 0.527→**0.53~0.55**。

## 5. 两个备选层级（按资源）

- **层级 A（推荐，全面）**：编码端 M1+M2+M3 + 轻量 LLM 解码器 + 主题槽规划 + 稳定化 RL + 免训练采样。BLEU/F1 双涨空间最大；估计训练总时长：SFT ~1 周（48G 单卡）+ RL ~1-2 周。
- **层级 B（保守，纯 BERT 解码器）**：M1+M2+M3 + 记忆/检索增强解码 + 解码参数调优（length_penalty=2.0、beam 4-5、repetition_penalty 1.05）+ contrastive decoding + Transition-Aware best-of-N。预期 B-4 0.113→0.12~0.13、F1 0.441→0.46+；"双超"拿得下但 BLEU 幅度小，论文语句要谨慎（"NLG 与 CE 同时改善"）。

## 6. 补充修正与结论（2026-09-02 定稿）

- **REVA-PO（arXiv:2607.10147）是"双指标同时涨"的最硬证据**（已读全文核实）：用 **BLEU-4 直接作为 RL 奖励** + 动态 KL（RER）+ 验证锚定策略重置（VAPR），三阶段训练（warm-up→分类器引导 SFT→RL，LoRA）→ MIMIC-CXR 上 BLEU-4 0.136→0.143（+5.1%）、CheXpert F1 0.484→0.506（+4.5%）、RadGraph F1 0.218→0.246（+12.8%）；8×V100 FP16。**结论：BLEU-奖励的稳定化 RL 可以同时抬 F1/RadGraph，"CE 奖励必伤 BLEU"不是定论，关键是 KL 正则与训练稳定性。**
- **TIM（CVPR 2026）消融逐项核实**：DPM 动态进展分支 F1 0.404→0.427 + BLEU-1 0.396→0.414；DRG 双时相生成 B-4 0.108→0.114 + F1→0.446；Lspr 抬 F1→0.460；Ldpm 抬 B-4→0.117；组合 (e) F1 0.483 + B-1 0.421，Stage II 互精修后 B-4 0.124 / F1 0.511。**进展建模组件（=M2/M3 同类）双指标齐涨。**
- **RL 后 BLEU 变化详表**（MRG-R1/CLARIFID/EditGRPO 的逐列数字）：因成本控制中断了专项子代理，未拿到其消融表逐列 BLEU 值；但 REVA-PO 已覆盖核心问题——**奖励里显式包含词面项（或直接以 BLEU 为奖励）即可保住/抬升 BLEU**；MRG-R1 纯语义奖励的 BLEU 走向待投稿前补查，避免审稿人拿它质疑。
- **硬件可行性实测修正**：本机实际是 **RTX 3090 24G**（nvidia-smi，交接文档 48G 记录有误）；当前 DDaTR 训练占 22.1G/24G。→ LLM 解码器路线只能 **TinyLlama-1.1B QLoRA/LoRA**（HF 缓存已具备 TinyLlama-1.1B-Chat-v1.0）；Phi-3-mini 3.8B 需 4bit QLoRA（显存紧）；peft/bitsandbytes 未装（需新 conda env，勿动正在训练的环境）；384 分辨率放弃；**全参微调不可行**。REVA-PO 的 8×V100 方案降级为 QLoRA 小批次：SFT ~1-2 周 + RL ~1 周（估算）。
- **双超定位修正**：REVA-PO（单图）已在 MIMIC 上 BLEU-4 0.143 / CheXpert F1 0.506（若同协议则双超了 DDaTR 0.113/0.441）——"双超"独特性必须限定为 **① Longitudinal-MIMIC micro 协议（B-4>0.105 且 F1>0.527 至今无人）② 纵向前瞻 + 双协议同时双超 ③ 紧凑模型（非 7B+ LLM）**。LLM-RG4（B-4 0.144 / F1 micro 0.588）是最接近的"双高"参考，投稿前需弄清其协议与是否含纵向输入。
- **框架结论**：编码端 M1 配准 + M2 间隔门控 + M3 进展事件/软门控（UNC-free 转移硬目标）→ **主题槽规划（进展事件排序驱动，新模块）** → TinyLlama QLoRA 解码 → REVA-PO 式稳定化 RL（奖励 = BLEU-4 + CE-F1 + F1-RadGraph + 进展一致性混合，动态 KL）→ 推理端 Transition-Aware best-of-N + 解码参数调优。**BLEU 靠 LLM 解码器+主题槽+奖励里的词面项；F1 靠 M1-M3+进展一致性奖励+RL 稳定性。**
- **预期账（不变）**：MIMIC macro B-4 0.113→0.13~0.16、F1 0.441→0.455~0.475；Long-MIMIC micro B-4 0.105→0.12~0.14、F1 0.527→0.53~0.55。