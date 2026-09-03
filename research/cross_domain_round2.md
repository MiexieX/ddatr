# 第二轮跨领域创新点调研：BLEU×F1 双指标 + 适合 DDaTR 紧凑框架

> 2026-09-02 ｜ 本轮的筛选条件：① 其他领域成熟 ② LRRG 无人做过 ③ 能同时抬 BLEU 与 CE F1 ④ 放得进 DDaTR（3090 24G、不换大 LLM 也能做）
> 与第一轮（`handoff/innovation_research.md`，11 方向）不重复，聚焦新发现的 7 个候选。

---

## 0. 一句话结论

**最强的三个新候选**：
1. **进展句定位 + 主题槽规划**（用 arXiv:2601.16753 的 95k LLM 标注造"进展规划器"，主题槽按解剖+进展事件排序）——ICLR 2026 LLaVA-TA 已证明"主题分解生成"在单图设置同时抬 BLEU-4(29.4→34.3) 与 CheXpert F1(39.5→62.4)，纵向版没人做，且**数据已就绪**；
2. **跨模态差分一致性**（共享空间里约束"图像差分 ≈ 报告差分"的循环对齐）——单模块、加损失即可，差异化 MARE 的模态内 DEC；
3. **变化模式范例检索**（按 CheXbert 标签转移向量检索相似"变化病例"注入解码）——RAG 的纵向新变体，BLEU 杠杆明确。

中远期大坑（第二篇）：**多访视 EHR 轨迹 + 纵向 CXR 联合**（ACL 2025 只做了单次 ED 数据，纵向空白）。

---

## 1. 新候选清单（每条含：来源领域 SOTA → LRRG 空白确认 → 落地方案 → 预期）

### C1. 进展句定位 + 主题槽规划（Progress-Sentence Planning）⭐ 最推荐
- **来源领域**：NLG 的句子级规划/受控生成；摘要领域的"重要句定位"；[LLaVA-TA: Rethinking RRG: From Narrative Flow to Topic-Guided Findings（ICLR 2026）](https://en.papernotes.org/ICLR2026/medical_imaging/rethinking_radiology_report_generation_from_narrative_flow_to_topic-guided_findi/)：报告分解为按**解剖区域组织**的独立主题，每个主题基于全图+对应**解剖掩码**生成一句发现 → 报告级 BLEU-4 29.4→34.3、CheXpert F1-14 39.5→62.4（MIMIC-CXR，单图、VLM 框架——机制可迁移）。
- **LRRG 空白**：该文是单图 RRG；**"按进展事件排序的主题槽"纵向版无人做**。且 [arXiv:2601.16753](https://ar5iv.labs.arxiv.org/html/2601.16753) 已发布 **95,169 条 MIMIC-CXR 的"纵向信息/疾病进展句"LLM 标注**（本来只用于评估）——**没人用它训练生成器**，直接可做"进展句定位器"的监督。
- **落地方案**（插进 DDaTR-NEXT 的规划端）：
  1. 用 2601.16753 标注（或先用 OpenAI/本地 Qwen 给训练集重新标注一遍进展句）训练轻量**进展句定位器**（BERT 级句子分类器，或直接从 M3 的 14×3 进展事件预测"哪些病种要写进展句"）；
  2. **主题槽规划**：槽 = 14 病种∩解剖区，排序规则 = 新发/加重 > 消退/好转 > 稳定阳性 > 阴性（与放射科行文习惯一致，BLEU 直接吃 GT 句子顺序的 n-gram 红利）；
  3. 每槽生成一句（LLaVA-TA 的掩码条件可用 M1 配准后的解剖掩码，或简化版只做槽排序不做掩码）。
- **预期**：BLEU-4 +0.01~0.03（结构对齐）、F1 +0.01~0.02（覆盖）；**双涨机制与 LLaVA-TA 同构**。成本：低-中（一个 BERT 分类器 + 解码 prompt 改造）。

### C2. 跨模态差分一致性（Cross-Modal Delta Consistency）
- **来源领域**：多模态对齐里的循环一致性/最优传输（[Bridging the Modality Gap in Medical VLMs: Hybrid Contrastive-OT](https://nchr.elsevierpure.com/en/publications/bridging-the-modality-gap-in-medical-vision-language-models-a-hyb/)）；对比学习的共享空间设计。
- **LRRG 空白**：MARE 的 DEC 是**模态内**双路径（视觉进展路径自洽 + 文本进展路径自洽）；**显式的"图像差分 ↔ 文本差分"跨模态循环对齐没人做**（BiOTPrompt 只做图-图匹配）。
- **落地方案**：把 DDAM 的差分特征 F_da（M1 配准后更干净）投影到与 CheXbert 标签转移向量（M3 的 14×3 进展事件）共享的语义空间，加对齐损失：
  `L_delta = ‖Proj_img(F_da) − Proj_txt(Δlabel)‖²`（Δlabel 用 CheXbert 先验/当前标签差，软目标）。
  可选再加"变化句级"对齐（用 C1 进展句的 embedding 做 target，而不是裸标签）。
- **预期**：进展描述准确性（进展句的 BLEU n-gram）+ 事件召回（F1）双受益；成本**最低**（一行损失，可作 M3 的姊妹模块）。
- **差异化命门**：写论文时强调"跨模态差分对齐" vs MARE"模态内演化一致性"，并消融对比。

### C3. 变化模式范例检索（Change-Pattern Exemplar Retrieval）
- **来源领域**：IR/RAG；[Spatio-Temporal and Clinical Conditioning for Fine-Grained Radiology Report Retrieval（BMVC 2026，arXiv:2607.02024）](https://arxiv-org.ezproxy.obspm.fr/html/2607.02024v1) 证明时空/临床条件检索可行（但任务是"检索报告"，不是"用检索结果生成"）；STREAM 的检索是通用相似检索。
- **LRRG 空白**：**按"变化模式"（标签转移向量）检索训练集范例再生成**——没人做；与 STREAM/RAG 的"视觉相似"检索是不同维度。
- **落地方案**（推理期即可先试）：对每个测试样本，用 (先验标签, 当前标签) 的转移向量（14×3）在训练集索引里 KNN 检索 top-k 相似"变化病例"，把其报告(或进展句)作为解码上下文/exemplar 输入。训练端可做"检索即增强"（retrieval-augmented training）。
- **预期**：BLEU +（范例句 n-gram 复用）、F1 +（变化覆盖）；**推理期零训练成本先行验证**（Retrieval 建索引一次，KDTree 秒级）；可与 C1 组合（检索进展句模板填入主题槽）。

### C4. 患者时间轴 token 化（Patient Timeline Tokenization / Temporal Summarization）
- **来源领域**：临床 NLP 的纵向笔记时序摘要 [CLIN-SUMM（medRxiv 2025）](https://www.medrxiv.org/content/10.64898/2025.11.28.25341233v1)；EHR 轨迹建模（TrajGPT、EHR2Path）。
- **LRRG 空白**：HERGen 的 time vocabulary 是**每个图像的**相对日期编码；**把患者全部历史压缩成"时间轴摘要 token 序列"喂解码器**（不是堆更多图像）没人做。HIST-AID 是堆 11 个报告（冗余），"压缩摘要"是不同思路。
- **落地方案**：先验报告 → 轻量摘要器（BERT 抽取句子→时间轴槽位，间隔标注）→ 生成 `[TL] date, event1; date, event2 ... [TL]` 格式的时间轴 token 注入解码 prompt。训练时用 CheXbert 转移事件自动构造"事件时间轴"（零标注）；推理时用 M3 预测事件+间隔。
- **预期**：历史一致性 F1（不遗忘旧发现）+ BLEU（时间轴措辞与 GT 的"compared to prior"句匹配）；成本中（需摘要器，但可复用 C1 的句子模型）。

### C5. 解剖掩码引导差分（Anatomy-Mask-Gated Difference）
- **来源领域**：CXR 解剖分割已成熟（[Morphology-aware hierarchical MoE for CXR anatomy segmentation（Eng. Appl. AI 2026）](https://www.sciencedirect.com/science/article/abs/pii/S0952197626006469)）；LLaVA-TA 的掩码条件生成证明解剖约束有效。
- **LRRG 空白**：CheXRelNet 做解剖感知的纵向**分类**追踪；**用解剖掩码门控差分特征（只在肺野/心脏区算变化、压制肩胛/肋骨影）**在 LRRG 无人做。
- **落地方案**：离线用公开分割模型给训练集打解剖掩码（一次成本）→ DDAM 差分乘掩码门控（每 stage 或只在 stage0/1）；同时给 M1 配准做解剖重叠约束（NCC 只在掩码内算）。
- **预期**：precision 升（伪变化少）+ recall 升（真变化更突出）；与 LLaVA-TA 叙事呼应。成本低（掩码是离线资源+一个乘法）。

### C6. 纵向 LLM 教师蒸馏（Decisive-Token Distillation, 纵向版）
- **来源领域**：[DIVE: Dynamic In-context Vector Distillation with Decisive-Token Supervision（arXiv:2605.27194）](https://arxiv-org.ezproxy.obspm.fr/html/2605.27194v1)——决定性 token（病理 token/EOS）加权 KL 蒸馏，长文本医学报告；单图。
- **LRRG 空白**：**纵向蒸馏**（教师也吃先验上下文，蒸馏给学生）没人做。
- **落地方案**：用开源强 VLM（MedGemma/CARE-X 级，走 opencode/云端一次性推理）给 MIMIC 纵向对生成教师报告（或只用教师输出概率），DIVE 式决定性 token 加权蒸馏进紧凑 DDaTR-NEXT；学生保留纵向模块但措辞模仿教师。
- **预期**：BLEU +0.02~0.05（最接近 LLM 的措辞质量）、F1 持平或微升（教师临床准确 + 学生纵向模块兜底）；成本：教师推理一次（订阅/配额）+ 蒸馏训练一次。

### C7（中远期/第二篇）：多访视 EHR 轨迹 + 纵向 CXR 联合生成
- **现状**：[The Impact of Auxiliary Patient Data on CXR Report Generation（ACL 2025）](https://aclanthology.org/2025.acl-long.9/) 用 MIMIC-IV-ED 的**单次** vitals/medicines 条件化单图报告；**跨访视 EHR 轨迹 + 纵向影像联合**没人做（MIMIC-IV↔MIMIC-CXR 需另下载 + 链接）。
- **落地方案**：vitals/labs 时序 → 小编码器 → 与 M2 间隔门控合并调制解码；工程量大、数据许可待办，**建议作第二篇或论文的"多模态扩展"章节**。

---

## 2. 创新空白矩阵（本轮 vs 第一轮合并）

| 候选 | 来源领域 | LRRG 是否有人做 | BLEU 增益证据 | F1 增益证据 | 成本 | 备注 |
|---|---|---|---|---|---|---|
| C1 进展句定位+主题槽 | NLG 规划/ICLR'26 LLaVA-TA | **否**（单图有） | 高（29.4→34.3） | 高（39.5→62.4） | 低-中 | 数据就绪（2601.16753） |
| C2 跨模态差分一致性 | 多模态对齐 | **否** | 中（进展句 n-gram） | 中-高（事件召回） | **最低** | 可与 M3 合并 |
| C3 变化模式范例检索 | IR/RAG | **否**（通用检索已有） | 中-高 | 中 | 低（推理端先行） | 与 STREAM 差异化 |
| C4 时间轴 token 化 | 临床 NLP 摘要 | **否**（HERGen 是图级 date 编码） | 中（compared-to-prior 句） | 中 | 中 | 替代"堆历史" |
| C5 解剖掩码门控差分 | CXR 分割 | **否**（分类追踪有） | 中 | 中-高（precision/recall） | 低 | 与 M1 互补 |
| C6 纵向蒸馏 | LLM 蒸馏 | **否**（单图 DIVE） | **高** | 中（持平） | 中（教师推理一次） | BLEU 最大杠杆之一 |
| C7 多访视 EHR+影像 | 多模态 EHR | **否**（单图 ED 有） | 中 | 高（临床上下文） | **高** | 第二篇/扩展 |

---

## 3. 推荐组合（更新 DDaTR-NEXT）

**双指标引擎的三级结构**（全部适配 3090 24G、不需要 7B 级模型）：

```
阶段 0（数据）: CheXbert 前后标签 → 14×3 进展事件 + 时间间隔（零标注，已实测）
              + 2601.16753 进展句标注（或自标注）→ 进展句定位器训练数据
              + 离线解剖分割掩码（公开模型，一次成本）
阶段 1（编码端，继承第一轮）: M1 配准（+C5 掩码门控）→ M2 间隔门控 → M3 进展事件/软门控（+C2 差分一致性损失）
阶段 2（规划端，本轮新增核心）: C1 主题槽规划器
              —— 槽序列 = f(M3 进展事件排序, C5 解剖, M2 间隔, C1 进展句定位)
              —— 输出 = 有序主题槽 + 每槽进展词表约束
阶段 3（生成端）: 槽位条件化解码（BERT 解码器或 TinyLlama QLoRA）
              + C3 变化范例检索注入（先推理端，后训练端）
阶段 4（优化端）: C2 损失 + LLM-RG4 式 token 加权 + （可选）REVA-PO 式 RL（BLEU+CE 混合奖励）
              +（可选 C6）纵向蒸馏预训练
```

**预期总账（不变目标）**：MIMIC macro B-4 0.113→0.14~0.17、F1 0.441→0.46~0.48；Long-MIMIC micro B-4 0.105→0.13~0.15、F1 0.527→0.54+。
**论文故事**："从看懂变化到写出进展——对齐(M1)、定时(M2)、定事(M3+C2)、定序(C1)、取证(C3) 的纵向报告生成"。**C1 是标题级创新点**（进展句规划，数据混着评估基准没人反过来用），C2+C5 是支撑机制，C3 是推理端加分项，C6 可作效率对照章节。

## 4. 关键链接
- 进展句标注基准：[arXiv:2601.16753](https://ar5iv.labs.arxiv.org/html/2601.16753)
- 主题分解生成（单图，双涨证据）：[LLaVA-TA / ICLR 2026 笔记](https://en.papernotes.org/ICLR2026/medical_imaging/rethinking_radiology_report_generation_from_narrative_flow_to_topic-guided_findi/)
- 时空+临床条件报告检索：[arXiv:2607.02024](https://arxiv-org.ezproxy.obspm.fr/html/2607.02024v1)
- 纵向临床笔记时序摘要：[CLIN-SUMM](https://www.medrxiv.org/content/10.64898/2025.11.28.25341233v1)
- CXR 解剖分割（形态学 MoE）：[Eng. Appl. AI 2026](https://www.sciencedirect.com/science/article/abs/pii/S0952197626006469)
- 决定性 token 蒸馏：[DIVE arXiv:2605.27194](https://arxiv-org.ezproxy.obspm.fr/html/2605.27194v1)
- 辅助病人数据（单图 ED）：[ACL 2025](https://aclanthology.org/2025.acl-long.9/)
- 跨模态混合对比-OT 对齐：[Elsevier Pure](https://nchr.elsevierpure.com/en/publications/bridging-the-modality-gap-in-medical-vision-language-models-a-hyb/)