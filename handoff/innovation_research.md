# 跨领域创新点调研报告：面向 DDaTR（纵向胸部 X 光报告生成）

**调研目标**：为 DDaTR（IEEE TMI 2025，Dynamic Difference-aware Temporal Residual Network for Longitudinal Radiology Report Generation）寻找"其他领域成熟、尚未迁移到纵向胸部 X 光报告生成"的创新点，必须能提升 CE F1（CheXbert 临床指标）与 BLEU/METEOR/ROUGE，特别关注 recall（漏报新发异常）问题。

**DDaTR 现状速览**（已核对代码 `models/resnet.py` / `models/blip.py` / `dataset/medical_dataset.py`）：
- 双 SwinT-B 编码器（当前+先验图），冻结 BERT 编码先验报告；4 个 stage 各含 DFAM（先验图↔先验文对齐，PWAM）+ DDAM（像素级差值 + LDConv + 双线性门控）；门控输出乘二值 `has_progress`（代码第 103/107 行）。
- 报告生成：BERT LM 解码器 + 诊断 prompt（`[BLA]/[POS]/[NEG]/[UNC]`），beam search（num_beams=3）。
- 损失：`L = L_LM + w*(L_CE_cur + L_CE_prior)`，w=4。
- 指标：MIMIC-CXR CE F1 **0.441**（P 0.438 / R **0.465**）；Longitudinal-MIMIC F1 0.527（P 0.539 / R 0.584）；IU-Xray F1 0.262；ReXrank GREEN 0.293。

**DDaTR 已知弱点**（对照用户提供，全部与代码吻合）：
1. 先验与当前图无配准（体位/AP/PA/胸廓膨胀直接污染像素差）；
2. 无时间间隔建模；
3. 只用最近一次先验；
4. 无确定性 vs probable 不确定性建模（虽 token 有 UNC）；
5. recall 低——动态融合主要提 precision；
6. 解码器无检索增强/事实核查；
7. 无显式进展事件（better/worse/stable）监督。

---

## 方向 1：胸部 X 光 temporal subtraction / 双片配准（经典放射学 → 深度学习）

**(a) 领域 SOTA**
- 经典 temporal subtraction：Ishida 等的图像 warping 系列（"Application of temporal subtraction for detection of interval changes on chest radiographs: improvement of subtraction images using automated initial image matching"，[Semantic Scholar](https://www.semanticscholar.org/paper/1074f23910d474b09acc7f66961454897bf8fe5e)；"Iterative image warping technique for temporal subtraction of sequential chest radiographs"，[Semantic Scholar](https://www.semanticscholar.org/paper/Iterative-image-warping-technique-for-temporal-of-Ishida-Katsuragawa/da6a2075cd1f45909ded510bef6170852303ff14)）。观察者研究证实其读片检测提升（[Radiology 1997, 202(2):447-52](https://pubmed.ncbi.nlm.nih.gov/9015072/)）。
- 现代深度版：无监督可变形配准 temporal chest radiographs（[SPIE Medical Imaging 2020](https://journals.spiedigitallibrary.org/conference-proceedings-of-spie/11313/113132X/Unsupervised-learning-based-deformable-registration-of-temporal-chest-radiographs-to/10.1117/12.2549211.short)）；3D-CNN 在经时差分像上做结节检测（[J-STAGE 医学画像情報学会](https://www.jstage.jst.go.jp/article/mii/36/2/36_77/_article/-char/ja/)）；Dark-field CXR 可变形配准用于局部肺信号变化评估（[arXiv 2501.10757](https://arxiv-org.ezproxy.obspm.fr/html/2501.10757v1)）；"A Spatially Enhanced CNN and Multiscale Transformer Fusion Approach for Chest Radiograph Registration"（[Springer](https://link-proxy.springer.com/chapter/10.1007/978-981-96-2684-7_28)）。

**(b) 指标提升证据**：经典文献表明 temporal subtraction 提高放射科医生对间隔变化（尤其结节）的检出敏感度；纯深度学习差分/配准方案在间隔变化检测任务上稳定高于无配准差值（SPIE 2020；3D-CNN 结节检测）。**注意：没有直接证据链到报告生成指标**——这正是创新空白点。

**(c) 是否已用到医学纵向 X 光**：为"检测/读片辅助"用途做过（无配准的差值在 DDaTR 的 DDAM 中已用），但**没有任何论文把现代可变形配准集成进纵向报告生成器**用于净化像素差。

**(d) 迁移方案**：详见"创新点 2"——在视觉编码器最前面插入轻量 2D 形变场配准（VoxelMorph 式，28×28 场 + STN warp），把先验图 warp 到当前图坐标系后再进 context_stage，消除 AP/PA/体位/胸廓膨胀污染；配准用自监督 NCC + 平滑正则联合训练，无额外标注。

---

## 方向 2：遥感图像变化描述 / 变化检测（change captioning / change detection）

**(a) 领域 SOTA**
- **Changen2: Multi-Temporal Remote Sensing Generative Change Foundation Model**（TPAMI 2025，[arXiv 2406.17998](https://arxiv.org/abs/2406.17998)，[PubMed](https://pubmed.ncbi.nlm.nih.gov/39388323/)）：双流（BiConvLSTM 时相编码 + SAM 视觉编码）生成式变化基础模型，自监督合成时序数据预训练 + SCST 强化微调，支持 zero-shot 变化检测与变化描述。
- 双流 Transformer 变化描述："Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset"（[IEEE TGRS](https://ieeexplore.ieee.org/document/9934924)）；SCNet（[IEEE](https://ieeexplore.ieee.org/abstract/document/11397416)）；Learnable Descriptive Convolution（LDCNet）正是 DDaTR LDConv 的来源（[IEEE](https://www.semanticscholar.org/paper/Difference-aware-Fusion)）。
- 变化检测 + 视觉语言：ChangeCLIP（[NSTL 收录](https://wlmq.nstl.gov.cn/paper_detail.html?id=f7014a4b546e777d91e0e44e21d6d3b2&ad_check=1)）。

**(b) 指标提升证据**：Changen2 在 LEVIR-CC / DUBAI-CCD / SECOND 上刷新 SOTA（BLEU-4、ROUGE-L、CIDEr、零样本 F1），论文报告系统性超越 MCCFormer、DUDA、Satlas 等；SCST 多 token 奖励是变化描述指标大幅提升的关键手段。

**(c) 是否已用到医学**：**没有**。LEVIR-CC 式"双时相图 → 变化描述"任务与纵向 CXR 报告本质同构（双时相图 + 相对先验的"变化语言"），但医学侧仅有高度相关的 Diff-RRG（MICCAI 2025，"longitudinal disease-wise patch difference as guidance"，[MICCAI 页面](https://papers.miccai.org/miccai-2025/0234-Paper4138.html)，[GitHub](https://github.com/ku-milab/Diff-RRG)）——它做 patch 级差值引导 LLM，但没有遥感界的"差分描述融合 transformer + SCST 奖励"的完整方法论。

**(d) 迁移方案**：把 DDAM 升级为"变化描述式"——差分特征不仅乘回原特征，而是被显式编码成变化 token 注入解码器（模仿双流 Transformer 变化胶囊）；再叠加 SCST/CMM-RL 式多 token 临床奖励（见方向 8）。**这是 recall 提升的强候选**：变化 token 直接把"新发/恶化区域"显式喂给解码器。

---

## 方向 3：视频理解中的时序建模（TSN/TRN/TSM/SlowFast/TDN/Non-local）

**(a) 领域 SOTA**
- **TDN: Temporal Difference Networks for Efficient Action Recognition**（CVPR 2021，[OpenAccess](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.html)，[GitHub](https://github.com/MCG-NJU/TDN)）：在每 stage 用帧差（Temporal Difference）增强当前帧特征——**与 DDaTR 的"差分区余"思想同源但形式化、可复用**（差分采样 + 差分调制）。
- SlowFast（快慢双路径）、TSM（[Temporal Shift Module](https://cloud.baidu.com/article/3344468)）、Temporal Difference Networks 的高阶变体（TDN v2）、Motion meets Attention: Video Motion Prompts（[arXiv 2407.03179](https://ui.adsabs.harvard.edu/abs/2024arXiv240703179C/abstract)）。

**(b) 指标提升证据**：TDN 相对 TSN 在 Something-Something/EPIC-Kitchens 上显著提升 top-1 并降 FLOPs；SlowFast/TSM 是视频理解标配；Motion Prompts 免训练提升下游 SOTA。

**(c) 是否已用到医学**：TDN/TSM/SlowFast 类差分模块**未直接用于纵向 CXR**；但思想已部分内化于 DDaTR 的 DDAM（LDConv 差分）与 MRID（"Modeling Radiological Image Differences... Multi-Task Self-Supervision"，[MDPI Electronics 2025](https://www.mdpi.com/2079-9292/15/5/997/xml)）。

**(d) 迁移方案**：在 stage 内把当前/先验特征按 TDN 的"差分调制"方式组织（差分不是只做一次，而是每 stage 用轻量 Temporal Difference Block 显式建模一阶/二阶差分）；或直接在 Swin 窗口注意力里加一个时序 shift 维度（TSM 式），造出"通道移位 + 差分"的低成本时序建模。

---

## 方向 4：医学影像可变形配准 + 解剖感知注意力

**(a) 领域 SOTA**
- **VoxelMorph**（[arXiv 1809.05231](https://arxiv.org/abs/1809.05231)）：无监督形变配准，NCC + 平滑正则，2D/3D 通用。
- **GaussianMorph: deformable medical image registration with Gaussian noise constraints**（2024，[PubMed 39781058](https://pubmed.ncbi.nlm.nih.gov/39781058/)）：高斯噪声约束下配准更稳。
- 解剖感知注意力：AnatSwin（[Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231224001504)），VerFormer（脊椎感知分割，[PMC11393940](https://pmc.ncbi.nlm.nih.gov/articles/PMC11393940/)），CheXRelNet（解剖感知的纵向 CXR 关系追踪，[2022](https://scite.ai/reports/chexrelnet-an-anatomy-aware-model-for-tracking-wmG2kjJM)），Finding-Aware Anatomical Tokens（[Springer](https://www.springerprofessional.de/finding-aware-anatomical-tokens-for-chest-x-ray-automated-report/26182010)）。

**(b) 指标提升证据**：GaussianMorph 等在多个配准基准上刷新 Dice/TRE SOTA；配准后对齐特征做差是经典 temporal subtraction 技术的主干（方向 1）。解剖感知提示（anatomical tokens / anatomy-aware attention）在 CXR 分类与报告生成中普遍带来 CE 指标小幅稳定提升。

**(c) 是否已用到医学纵向 X 光**：配准本身在 CXR 上有人做（SPIE 2020、Dark-field 2025），CheXRelNet 做了解剖感知纵向关系；但**"配准→净化差值→喂纵向报告生成"整条链路没人做**。Dissimilar：Diff-RRG 的 patch difference 无显式配准。

**(d) 迁移方案**：见创新点 2（配准前置模块）+ 在 DDAM 的差特征上叠解剖先验注意力（用现成肺/心脏分割图或 CheXRelNet 式解剖位置编码约束差特征的空间配准性）。

---

## 方向 5：时序间隔编码（temporal interval embedding / time-aware transformer）

**(a) 领域 SOTA**
- **HERGen**（ECCV 2024，"History Enhanced Radiology Report Generation"，[arXiv 2407.15158](https://arxiv.org/abs/2407.15158)）：**date-aware sequential temporal embedding**——把相对检查日期做成可学习 embedding 与纵向序列一起进 group causal transformer，是纵向 RRG 里唯一系统化建模时间间隔的工作。
- 非影像医学事件时序：TrajGPT（[arXiv 2410.02133](https://ar5iv.labs.arxiv.org/html/2410.02133)）、ChronoFormer（[arXiv 2504.07373](https://ar5iv.labs.arxiv.org/html/2504.07373)）、XTSFormer（AAAI 2025，跨时间尺度 Transformer，[ACL](https://dl.acm.org/doi/abs/10.1609/aaai.v39i27.35073)）、INTERVenE（时间抽象间隔，[Semantic Scholar](https://www.semanticscholar.org/paper/80f8b3a44b29c41709d695affcd718314068fe35)）、conditional latent diffusion for irregularly spaced longitudinal radiological data（MICCAI 2025，[页面](https://papers.miccai.org/miccai-2025/0164-Paper2656.html)）。

**(b) 指标提升证据**：HERGen 论文报告比次优（CvT-21v2+DistilGPT2）**NLG 均值 +5.9%、CE recall/F1 +2.0%**（[arXiv 2407.15158](https://arxiv.org/abs/2407.15158)）；TrajGPT/ChronoFormer 在 EHR 事件预测上提升显著。

**(c) 是否已用到医学纵向 X 光**：**只有 HERGen 做了（date embedding），DDaTR 完全没有**。时间间隔是用户点名弱点 2，且 DDaTR 数据集（MIMIC-CXR）带 study date，数据现成。

**(d) 迁移方案**：见创新点 1——Δt（天）→ 可学习 embedding（或 log 压缩 MLP），条件化 DFAM/DDAM 的双门控（不是加在特征上而是调制门控强度），可以再加一个轻量"间隔感知"辅助损失。

---

## 方向 6：不确定性感知生成（uncertainty-aware generation / calibration / epistemic uncertainty）

**(a) 领域 SOTA**
- **CURV: Coherent Uncertainty-Aware Reasoning in Vision-Language Models for X-Ray Report Generation**（NeurIPS 2025，[proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/86a99b74adac7c77998902371f53850d-Abstract-Conference.html)，[GitHub](https://github.com/wwwadx/CURV)）：不确定性建模（教模型表达合适置信度）+ 结构化推理 + 推理一致性奖励，三阶段训练（uncertainty-aware SFT → reasoning init → RL）。
- **Semantic Consistency-Based Uncertainty Quantification for Factuality in Radiology Report Generation**（NAACL Findings 2025，[ACL](https://aclanthology.org/2025.findings-naacl.95/)）：用回答自洽性做零样本事实性不确定性。
- Uncertainty-aware report generation by variational topic inference（Najdenkoska，MedIA 2022，[PubMed](https://pubmed.ncbi.nlm.nih.gov/36116297/)）。
- DDaTR 现状：tokenizer 有 `[UNC]`，CE 评估把 uncertain 映射为 positive（论文 Section IV.A：positive+uncertain→positive）。

**(b) 指标提升证据**：CURV 报告显著优于 prior methods（临床相关性 + 不确定性表达质量）；变分主题不确定性方法比确定性 baseline 提升 CE；**对 CE F1 的特别意义**：uncertain→positive 的映射使"恰当表达不确定"可以直接抬 recall（不确定不是漏报），同时抑制确定性误报可抬 precision。

**(c) 是否已用到医学纵向 X 光**：CURV/NAACL-2025 都是单时相 CXR/通用 RRG；**结合纵向先验做不确定性建模——没有人做**。

**(d) 迁移方案**：见创新点 4——在 DDaTR 的分类头与解码之间加显式不确定性分支 + 校准（ECE 惩罚/温度缩放/focal 式不确定加权），并用先验报告的一致性（先验确定 + 当前不确定 = 新发怀疑）作为 uncertainty 先验信号。

---

## 方向 7：检索增强生成（RAG）用于医学报告

**(a) 领域 SOTA**
- **Spatio-Temporal and Retrieval-Augmented Modeling for Chest X-Ray Report Generation**（**IEEE TMI 2025**，[Semantic Scholar](https://www.semanticscholar.org/paper/Spatio-Temporal-and-Retrieval-Augmented-Modeling-Yang-You/cb312c611595ae9af6a29de36247c949580c171a)）：**已经把时空建模 + 检索增强合进纵向 CXR 报告生成**——与 DDaTR 目标重叠度最高的竞品式工作。
- RA-RRG: Multimodal Retrieval-Augmented RRG with Key Phrase Extraction（[ACL Findings](https://aclanthology.org/2026.findings-acl.247/)）；RRG-RAG（[bytez](https://bytez.com/docs/arxiv/2504.07415)）；CBM-RAG（概念瓶颈 + 多智能体 RAG，[arXiv 2504.20898](https://fub-hagen.digibib.net/search/eds/record/edsarx:edsarx.2504.20898)）；Semantic-Aware CXR RRG with Domain-Specific Lexicon and Diversity-Controlled Retrieval（MICCAI 2025，[页面](https://papers.miccai.org/miccai-2025/0821-Paper3102.html)）；RGRG（retrieval-guided，2023）；LaB-RAG（label-boosted，[GitHub](https://github.com/uc-cdis/label-boosted-RAG-for-RRG)）。

**(b) 指标提升证据**：检索增强在 RRG 上普遍带来 BLEU/F1-RadGraph 提升（RGRG、RRG-RAG、RA-RRG 均报告超越无检索 baseline）；TMI 2025 时空+检索在纵向设置下有效。

**(c) 是否已用到医学纵向 X 光**：**是**（TMI 2025 已做时空+检索；Hist-AID 用历史报告外存，[arXiv 2411.10684](https://arxiv.org/abs/2411.10684)；HC-LLM 用历史约束，AAAI 2025）——**新颖性弱**，但收益明确、工程成本低，适合作为二级方案或与前述创新点组合（如检索相似历史病例强化时间间隔建模）。

**(d) 迁移方案**：给 DDaTR 解码器加"检索记忆"——训练集里按 CheXbert 标签检索相似先验病例的 (图,报告) 对，把检索到的 report 片段作为额外 prompt/上下文 token 与 beam search 集成（也可做先验报告重排序）；或采用 TMI 2025 的粗到细检索设计。注意规避与已发表工作的重复度。

---

## 方向 8：强化学习 / 偏好优化（RL / DPO / GRPO）用于报告生成

**(a) 领域 SOTA**
- **EditGRPO**（IJCNLP 2025，"RL with Post-Rollout Edits for Clinically Accurate Chest X-Ray Report Generation"，[ACL](https://aclanthology.org/2025.ijcnlp-short.26/)，[GitHub](https://github.com/taokz/EditGRPO)，[arXiv 2509.22812](https://arxiv.org/abs/2509.22812)）：on-policy GRPO + off-policy 句子级编辑注入（用参考报告替换假阳性句子、补漏报句子），规避探索困难。
- **RRG-DPO**（MICCAI 2025，"Direct Preference Optimization for Clinically Accurate RRG"，[页面](https://papers.miccai.org/miccai-2025/0795-Paper1273.html)）；RadHiera（分层 RL，[UST-HK](https://lbnx03.ust.hk/ir/Record/1783.1-171455)）；RadVLM-GRPO（[GitHub](https://github.com/uzh-dqbm-cmi/RadVLM-GRPO)）；CLARIFID、CXRMate-2（tractable RL，[scirate](https://scirate.com/arxiv/2604.18967)）；早期 CMM-RL（2022，DDaTR 对照表已含）。

**(b) 指标提升证据**：**EditGRPO 报告平均 +3.4%（CheXbert/GREEN/RadGraph/RATEScore 四数据集平均），未见数据集 OOD +5.9%**（[arXiv 2509.22812](https://arxiv.org/abs/2509.22812)）；DPO/GRPO 系方法普遍比 SFT 提升 CE 与 RadGraph F1。

**(c) 是否已用到医学纵向 X 光**：所有上述工作都是**单时相 RRG**；**没有纵向 RRG 用 RL/DPO，且没有把"进展正确性"放进奖励**。CMM-RL 在 DDaTR 对照表里但未进纵向链路。

**(d) 迁移方案**：见创新点 5——对 DDaTR 做两阶段：SFT（现状）→ 奖励微调。奖励 = 当前报告 CE F1（CheXbert）+ 先验-当前进展一致性/F1-RadGraph；用 EditGRPO 式句子级编辑构造偏好对（漏报补句、误报换句），GRPO/DPO 更新解码器（冻结/低秩更新视觉编码器）。**这是直接提 CE F1 与 recall 的最强杠杆之一**。

---

## 方向 9：Mamba / 状态空间模型（时序或多模态融合）

**(a) 领域 SOTA**
- **R2Gen-Mamba**（[arXiv 2410.18135](https://www.alphaxiv.org/abs/2410.18135)）：Mamba 用于 RRG 编码/解码。
- **RRGMambaFormer**（hybrid Transformer-Mamba for RRG，2025，[Semantic Scholar](https://www.semanticscholar.org/paper/RRGMambaFormer%3A-A-hybrid-Transformer-Mamba-for-Li-Liu/c77596bb9f7d630854239ab0ddc00626074e95d3)）。
- **VMRA-MaR: Asymmetry-Aware Temporal Framework for Longitudinal Breast Cancer Risk Prediction**（MICCAI 2025，[arXiv 2506.17412](https://arxiv.org/abs/2506.17412)）：**Mamba 时序 + 非对称感知**的纵向医学影像建模——与 DDaTR 结构高度互补。
- SegMaST（Mamba spatio-temporal 纵向疾病检测，[MIDL](https://mlanthology.org/midl/2026/varma2026midl-segmast/)）；Star-vmamba（遥感变化检测，[S2](https://www.semanticscholar.org/paper/bd8ca319423dfe8a587fdb4181e73f2b3d5ae643)）。

**(b) 指标提升证据**：R2Gen-Mamba / RRGMambaFormer 在 MIMIC-CXR 上超越 R2Gen 系/纯 Transformer；VMRA-MaR 在纵向乳腺 MRI 风险预测上超越 SOTA；Mamba 线性复杂度利于长序列历史。

**(c) 是否已用到医学纵向 X 光**：**没有**。R2Gen-Mamba/RRGMambaFormer 是单时相；VMRA-MaR 是乳腺 MRI 且任务非报告生成；纵向 CXR 报告生成 + Mamba 是空白。

**(d) 迁移方案**：把 DDaTR 的 stage 级 gate 融合换成/叠加一个轻量 Mamba 时序块：输入序列 = [先验对齐特征 tokens ; 当前特征 tokens]（HW×C），Mamba 选择性扫描输出调制后的当前特征；或仿 VMRA-MaR 做非对称时序注意。低成本融入现前向（一个 block，训练无新损失）。

---

## 方向 10：时序对比学习（temporal contrastive learning）

**(a) 领域 SOTA**
- **MLRG: Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation**（**CVPR 2025**，[OpenAccess](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Enhanced_Contrastive_Learning_with_Multi-view_Longitudinal_Data_for_Chest_X-ray_CVPR_2025_paper.html)，[arXiv 2502.20056](https://arxiv.org/abs/2502.20056)）：多视角（当前多视图）+ 纵向（时间）双分支对比学习，**tokenized absence encoding** 处理缺失先验知识。
- **CoCa-CXR: Contrastive Captioners Learn Strong Temporal Structures for Chest X-Ray VLU**（MICCAI 2025，[arXiv 2502.20509](https://arxiv.org/abs/2502.20509)）。
- Time-Equivariant Contrastive Learning for Degenerative Disease Progression in Retinal OCT（[arXiv 2405.09404](https://www.semanticscholar.org/paper/882f843f58ceed5f7c25e2e271f52bb05c0f526a)）；EVOKE（多视角对比 + 患者特异性知识，[arXiv 2411.10224](https://arxiv.org/abs/2411.10224)）；PriorRG（先验引导对比预训练，[arXiv 2508.05353](https://arxiv.org/abs/2508.05353)）；Contrastive Scene Change Representation（TGRS）。

**(b) 指标提升证据**：**MLRG 报告 MIMIC-CXR BLEU-4 +2.3%、MIMIC-ABN F1 +5.5%、Two-view F1-RadGraph +2.7%**（CVPR 2025 摘要）；CoCa-CXR 证明对比式 captioner 能学到强时序结构。

**(c) 是否已用到医学纵向 X 光**：**是——这是竞争最激烈的方向**（MLRG、CoCa-CXR、EVOKE、PriorRG 都已经做纵向 CXR 对比学习）。新颖性中等偏弱，但对 DDaTR 的"先验-当前特征一致性"仍可直接获益（DDaTR 无任何对比目标）。

**(d) 迁移方案**：给 DDaTR 加时序 InfoNCE：同患者 (前一时间图, 当前图) 为正对（以先验报告为锚的 soft 正对），跨患者为负对；损失挂到 avg_feats/avg_feat2 上，w 小（0.1~0.3）；可选 tokenized absence encoding 直接解决 has_progress=0 的零先验分支（现为零张量，可升级为可学习 absence token）。

---

## 方向 11：长历史 / 多实例学习（multi-visit / MIL）

**(a) 领域 SOTA**
- **Transition-Aware best-of-N sampling for Longitudinal Chest X-ray Reports**（[arXiv 2606.28393](https://arxiv.org/abs/2606.28393)）：首个**训练免费**的纵向 aware 采样：把 (先验→当前) 报告差异编码为方向向量，用 set-to-set 距离（mean-shift / novelty residual / directed-Hausdorff / OT）从缓存库检索最一致候选，推理期直接提分——报告"全面优于随机选择，Impression 段增益最大"。
- SMILE: Sequential Multi-Instance Learning with Multiple CT Scans for lung cancer screening（[IEEE](https://ieeexplore.ieee.org/document/10960342)）；SDPL: Symptom Disentanglement for Fine-grained Progression Learning（MICCAI 2024，[页面](https://papers.miccai.org/miccai-2024/751-Paper3074.html)）；HERGen 的 group causal transformer（多 visit）；Hist-AID / HC-LLM（AAAI 2025）。
- Review：**A Review of Longitudinal Radiology Report Generation: Dataset Composition, Methods, and Performance Evaluation**（[arXiv 2510.12444](https://arxiv.org/abs/2510.12444)）——首个 LRRG 综述，总结 5 大局限与未来方向（伪对数生成的纵向增强、结局预测、多先验融合、跨期对齐）。

**(b) 指标提升证据**：Transition-Aware best-of-N 在多个生成器 + 多种 prompt 下系统优于随机采样（尤其 Impression）；SMILE 相对单实例在筛查任务显著提升；HERGen multi-visit + contrastive + curriculum 报告全面 SOTA。

**(c) 是否已用到医学纵向 X 光**：**部分**——多 visit 建模（HERGen/Hist-AID/HC-LLM）与推断期过渡aware 采样（best-of-N）已出现；但 **MIL 式"历史 bag + 注意力聚合"在纵向 CXR 报告生成仍新鲜**，且 DDaTR 只用最近一次先验。

**(d) 迁移方案**：两档——(i) 低档：把 DDaTR 线性扩展到最近 K 次先验（K=2~3），stage 级多先验注意力聚合（轻量，增参少）；(ii) 高档：MIL bag 池化全部历史 visit 特征 + 时间间隔加权（结合创新点 1），推理再加 best-of-N 式过渡重排（免训练，立即可测）。

---

## 创新空白矩阵

| # | 方向 | 是否已被用于纵向胸部X光报告生成 | 指标提升证据强度 | 实施成本 | 对 DDaTR 的互补性 / 备注 |
|---|------|-------------------------------|------------------|---------|--------------------------|
| 1 | temporal subtraction / 双片配准（深度版） | **否**（检测/读片有，生成器无） | 中（经典+配准文献强，无直接 RRG 证据） | 中 | 直击弱点 1，净化像素差 → recall 潜力大 |
| 2 | 遥感变化描述（change captioning） | 否（Diff-RRG 近似但不完整） | 中-高（Changen2 SOTA、任务同构） | 中 | 变化 token 注入解码器 + SCST 奖励，提 recall |
| 3 | 视频时序建模（TDN/TSM/SlowFast） | 否（思想部分内化于 DDAM） | 中（动作识别强，医学未验证） | 低-中 | 差分块显式化/窗口内时序移位，低成本 |
| 4 | 医学配准 + 解剖感知注意力 | 配准**否**；解剖提示部分有 | 中-高（GaussianMorph SOTA；解剖提示稳定提 CE） | 中 | 配准净化 + 解剖先验约束差值空间位置 |
| 5 | 时序间隔编码 | **仅 HERGen（date embedding）**，DDaTR 无 | **高**（HERGen NLG +5.9%、CE +2.0%） | **低** | 直击弱点 2，数据现成（study date） |
| 6 | 不确定性感知生成 | 否（单时相有 CURV/NAACL-25） | 中-高（CURV 显著优于 prior） | 中 | 直击弱点 4；uncertain→positive 映射可抬 recall |
| 7 | RAG 检索增强 | **是（IEEE TMI 2025 已做**） | 高 | 中 | 新颖性弱，收益明确；适合组合或二级方案 |
| 8 | RL / DPO / GRPO | **否（纵向没有，且进展奖励无人做）** | **高**（EditGRPO +3.4%/+5.9% OOD） | **高** | 直击弱点 5/6；最强 CE 提升杠杆之一 |
| 9 | Mamba / 状态空间时序融合 | **否**（单时相 R2Gen-Mamba；纵向是乳腺 MRI） | 中-高（R2Gen-Mamba/RRGMambaFormer SOTA；VMRA-MaR） | 低-中 | 换/叠加 stage 级 gate，低成本高新颖 |
| 10 | 时序对比学习 | **是（MLRG/CoCa-CXR/EVOKE 已做）** | **高**（MLRG BLEU-4 +2.3%、F1 +5.5%） | 低-中 | 竞争激烈但 DDaTR 零对比目标，直接可加 |
| 11 | 多 visit / MIL / 过渡采样 | 部分（HERGen 多 visit；best-of-N 采样） | 中-高（best-of-N 全面优于随机；SMILE） | 低（免训练采样）/高（全历史 MIL） | 弱点 3；免训练 best-of-N 立即可测 |

**图例**：证据强度=该方向在自身领域的公开量化增益；成本=迁移到 DDaTR 的工程量（数据/训练/推理）。

---

## 重点推荐：DDaTR 上最可行、最可能提升 CE F1 与 recall 的 5 个创新点

**选择逻辑**：优先 (i) 直击 DDaTR 七弱点中当前完全没有堵住的（弱点 1/2/4/5/6/7 全部未解决），(ii) 证据强度高，(iii) 实施成本可控（A40 训练、不换 backbone、不改数据管线主结构），(iv) 与 2024-2025 已发表工作保持差异化。

### ⭐ 创新点 1：时间间隔条件化门控（Temporal-Interval-Conditioned Gating）——弱点 2，成本最低
- **来源**：方向 5（HERGen date embedding、ChronoFormer、XTSFormer）。
- **输入**：`Δt`（当前 study date − 先验 study date，天；MIMIC-CXR 元数据现成）。无先验时 Δt=0（等价 has_progress=0 分支）。
- **模块**：
  ```
  IntervalEmbed: log(1+Δt) → Linear(1→d) → ReLU → Linear(d→2d)  # d=vision_width=1024
      → 拆成 g_ctx, g_cur ∈ R^{B,d}
  ```
  或 HERGen 式可学习相对日期 vocabulary（更简单，但需设桶数，推荐 log 压缩 MLP）。
- **插入位置**：`models/resnet.py` 第 103/107 行——把门控从 `alpha*has_progress` 改为
  ```
  alpha1 = context_gate[i](x2_residual) * has_progress.u(-1).u(-1) * sigmoid(g_ctx).u(-1).u(-1)
  alpha2 = gate[i](x1_residual) * has_progress.u(-1).u(-1) * sigmoid(g_cur).u(-1).u(-1)
  ```
  并把 `g_ctx/g_cur` 同时加到 DFAM 的 key/value（`x2_residual`）上作为 bias。`forward/generate` 签名增加 `time_interval` 张量；`dataset/medical_dataset.py` 返回该字段。
- **损失改动**：主损失不变；可选加辅助头 `Δt 桶分类`（软标签，权重 0.1）做隐式正则。**预期**：模拟 HERGen 量级，CE F1 +1~2%，且不改解码器、不增推理成本。
- **为什么排第一**：数据零成本、改动 <50 行、直接补上"1 天 vs 1 年"的临床语义；单独一项就足以写成 ablation。

### ⭐ 创新点 2：编码器前置 2D 形变配准（Registration-Guided Difference）——弱点 1，recall 提升主攻
- **来源**：方向 1/4（VoxelMorph、GaussianMorph、SPIE 2020 temporal CXR 配准）。
- **输入**：当前图 x1、先验图 x2（均为 224×224）；输出 warp 后先验图 x̃2 = STN(x2; φ)。
- **模块**（轻量版，~5M 参数）：
  ```
  RegNet: 下采样两图（共享小 ResNet，3 层 stride2）→ 拼接到 28×28×S →
          U 形 4 层 conv（带跳连）→ 输出 φ ∈ R^{2×28×28}（可变形场）+ 可选 6-DoF 仿射头
  Warp: SpatialTransformer（双线性采样，forward 与 backward 一致）
  ```
- **插入位置**：`models/resnet.py` `forward` 开头、`embeding` 之前或 stage0 输入处：`x2 = warp(x2); x1 = warp(x1)`（以当前为 fixed、先验为 moving，只 warp 先验更稳妥；仿射与形变串联）。配准后的 x̃2 才进 `context_stage` 与 DDAM 差值。
- **损失改动**：加自监督配准损失
  ```
  L_reg = λ1·(-NCC(x1, x̃2)) + λ2·||∇φ||² + λ3·(可选)肺占比一致性
  L_total = L_LM + w·(L_CE_cur+L_CE_prior) + λ_reg·L_reg
  ```
  无需配准标注；λ_reg 从 0.01 起 warm-up（先用两图训练配准 1 epoch，再联合）。
- **预期**：AP/PA、体位、胸廓膨胀差异被吸收进形变场 → DDAM 像素差只剩"真实变化"→ **recall 直接受益（新发异常不再被体位噪声冲淡）**，precision 同样受益。这是 7 弱点里"物理上最干净"的一刀。
- **风险与对策**：2D 胸部配准固有挑战（肋骨投影变化）；对策：限制形变正则强度 + 只在 stage0/1 用像素级配准、stage2/3 用特征级仿射（两阶段配准，仿照 SPIE 2020 两阶段思路）。

### ⭐ 创新点 3：显式进展事件监督 + 软进展门控（Progress-Supervised Gating）——弱点 7，直接打 recall
- **来源**：方向 3/11（TDN 差分思想；SDPL 症状解耦；Anatomy-specific progression classification in CXR，Radiology AI 2024，[PMC11427915](https://pmc.ncbi.nlm.nih.gov/articles/PMC11427915/)）。
- **监督构造**：对训练样本用现有 CheXbert（`checkpoints/stanford/chexbert/chexbert.pth` 已在库）给先验报告与当前报告打 14 类标签 p_prior, p_cur；定义**进展三元组** per finding：
  ```
  prog_k = sign(p_cur_k − p_prior_k) ∈ {+1 新发/加重, 0 稳定, −1 消退/减轻}
  ```
  共 14×3（或 14 维 soft）。无需人工标注。
- **模块**：`ProgressHead: Linear(1024→14*3)` 挂在 `resnet.py` 输出的 `avg_feats`（当前）旁；并在 stage 融合处把**预测的进展强度**作为软门控替换/缩放 `has_progress`：
  ```
  soft_prog = σ(ProgressHead→14×3→max-over 진행维度 → scalar) → clamp(0,1)
  alpha2 = gate[i](x1_residual) * has_progress * soft_prog  # 每 stage 共享或独立
  ```
- **插入位置**：`resnet.py` return 前；`blip.py` `forward` 加
  ```
  loss_prog = CE(ProgressHead(avg_feats), prog_labels)   # 14×3
  loss = loss_lm + w*(loss_cls+ctx_loss_cls) + λ_prog·loss_prog
  ```
- **预期**：进展监督让模型在编码阶段就感知"哪些病种变了"，门控强度随真实进展浮动——直接修正弱点 7，并系统性抬高 recall（新发/恶化病种被 DPAM 差分放大）。该想法在纵向 RRG 中**无人做过**（SDPL/解剖进展分类都不生成报告）；配合 `has_progress` 二值开关不破坏现有 zero-prior 分支。
- **收益预估**：三分类辅助任务在 LRRG 上通常可模拟 +1~3% CE F1；recall 增益大概率超过 precision。

### ⭐ 创新点 4：不确定性感知解码与校准（Uncertainty-Aware Decoding）——弱点 4，precision/recall 双效
- **来源**：方向 6（CURV、NAACL-2025 semantic-consistency UQ、MedIA 2022）。
- **现状基础**：DDaTR 的 prompt 已有 `[UNC]` token（`models/blip.py` SCORES），但未被显式训练/校准；CE 评估把 uncertain→positive。
- **模块**：
  1. **不确定性分支**：`UNCHead: Linear(1024→14)` 与 cls_head 并联，训练目标 = CheXbert 的 uncertain 标注（MIMIC 标签含 uncertain 类，数据现成 `base_probs.json`），focal 加权缓解类别失衡；
  2. **生成期策略**：`cls_preds` 对高不确定 finding 输出 `[UNC]`（阈值或 ECE 校准后），CER 映射下即正——**把"不敢说但该提"的发现保留为正例** → 抬 recall；
  3.（进阶）推理时对低自洽候选做语义一致性重排（NAACL-2025 法）。
- **损失改动**：
  ```
  loss_unc = BCE(UNCHead(avg_feats), unc_labels)          # per-finding uncertain
  loss += λ_unc·loss_unc + λ_ece·ECE_logit(calibration)
  ```
- **插入位置**：`blip.py` cls_head 区域与 `generate()` 的 prompt 构建处（第 122-125 行）。
- **预期**：uncertain→positive 映射 + 显式校准 → 减少"漏报型不确定"（recall+）同时减少确定性误报（precision+）；与 CURV 的 reasoning-coherence reward 结合可作二期优化。
- **差异化**：CURV 是单时相；本方案用**先验一致性的变化**（先验确定→当前不确定 = 高度怀疑新发）作为 uncertainty 先验，纵向独有。

### ⭐ 创新点 5：CE 奖励强化微调（GRPO/DPO + 句子级编辑）——弱点 5/6，CE 指标最强杠杆
- **来源**：方向 8（EditGRPO +3.4%/+5.9% OOD、RRG-DPO、CMM-RL/SCST）。
- **两阶段管线**：
  1. **SFT**：现有 DDaTR 训练（不动）；
  2. **RL 微调**：冻结视觉编码器（或 LoRA 低秩更新），只微调 BERT 解码器：
  ```
  奖励 r(y|x1,x2,r_prior) = CheXbert CE F1(y) + β·F1-RadGraph(y) + γ·进展一致性(y, r_prior)
    进展一致性 = 用 CheXbert 打 y 与 r_prior，比较 14 类变化方向与 (p_cur−p_prior) 的余弦/符号一致率
  ```
  用 EditGRPO 式训练循环：采样 rollouts → 用参考报告做句子级编辑（FP 句替换、FN 句补入）构造 off-policy 对 → GRPO/DPO 更新。
- **插入位置**：`main_train.py` 之后加第二阶段脚本；复用 `modules/trainer.py` 的 `generate`（beam search 已有）；推理不变。
- **成本**：训练时间翻倍量级（GRPO 需多次 rollout）；A40 可承受（模型 ~200M）。收益上限最高：**直接以 CE 为优化目标**，且句子级编辑专门针对 FN（= recall 漏报）与 FP（= precision 误报）机制性修复弱点 5。
- **预期**：参照 EditGRPO 量级，CE F1 +2~4%、F1-RadGraph 同步升；BLEU 因 RL 不必然升（以 CE 为准，符合 DDaTR 自身"CE 优先"定位）。

---

## 实施优先级建议（一个 6~8 周增量路线）

| 阶段 | 内容 | 交付物 |
|------|------|--------|
| W1 | 创新点 1（时间间隔门控） | MIMIC-CXR 复现基线上 +Δt 版，对比 CE F1 |
| W2-3 | 创新点 3（进展监督） | 新增 progress head/soft gate，ablation has_progress |
| W4-5 | 创新点 2（配准前置） | RegNet 联合训练；配准消融（w/wo warp） |
| W6 | 创新点 4（不确定解码） | UNCHead + 校准 + [UNC] 决策策略 |
| W7-8 | 创新点 5（GRPO/DPO 微调） | 二期训练脚本 + 四指标表（CE F1/BLEU/METEOR/ROUGE + ReXrank） |

**组合增益预期**：1+2+3 协同作用于编码端（间隔感知 + 对齐差分 + 进展门控），4+5 作用于解码端（不确定校准 + CE 奖励）；联合后 CE F1 目标幅度 **+3~6%（相对 0.441 → 0.455~0.467）**，recall 0.465 → 0.49+，BLEU 系保持或微升——与 Reviewer 讨论"CE 优先、NLG 持平即可"的定位一致。

---

## 关键文献链接汇总

- DDaTR 论文：[arXiv 2505.03401](https://arxiv.org/abs/2505.03401)（本仓库含 PDF）
- LRRG 首篇综述（5 大局限）：[arXiv 2510.12444](https://arxiv.org/abs/2510.12444)
- HERGen（date embedding，ECCV'24）：[arXiv 2407.15158](https://arxiv.org/abs/2407.15158)
- Changen2（TPAMI'25）：[arXiv 2406.17998](https://arxiv.org/abs/2406.17998)
- MLRG（CVPR'25，多视角+纵向对比）：[OpenAccess](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Enhanced_Contrastive_Learning_with_Multi-view_Longitudinal_Data_for_Chest_X-ray_CVPR_2025_paper.html)
- CoCa-CXR（MICCAI'25）：[arXiv 2502.20509](https://arxiv.org/abs/2502.20509)
- EditGRPO（IJCNLP'25）：[arXiv 2509.22812](https://arxiv.org/abs/2509.22812)
- RRG-DPO（MICCAI'25）：[MICCAI 页面](https://papers.miccai.org/miccai-2025/0795-Paper1273.html)
- CURV（NeurIPS'25）：[proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/86a99b74adac7c77998902371f53850d-Abstract-Conference.html)
- Semantic-consistency UQ（NAACL-F'25）：[ACL](https://aclanthology.org/2025.findings-naacl.95/)
- TDN（CVPR'21）：[OpenAccess](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.html)
- VoxelMorph：[arXiv 1809.05231](https://arxiv.org/abs/1809.05231)；GaussianMorph：[PubMed](https://pubmed.ncbi.nlm.nih.gov/39781058/)
- 无监督 temporal CXR 配准（SPIE'20）：[SPIE](https://journals.spiedigitallibrary.org/conference-proceedings-of-spie/11313/113132X/Unsupervised-learning-based-deformable-registration-of-temporal-chest-radiographs-to/10.1117/12.2549211.short)
- 经典 temporal subtraction（Radiology'97 观察者研究）：[PubMed 9015072](https://pubmed.ncbi.nlm.nih.gov/9015072/)
- R2Gen-Mamba：[arXiv 2410.18135](https://arxiv.org/abs/2410.18135)；RRGMambaFormer：[S2](https://www.semanticscholar.org/paper/RRGMambaFormer%3A-A-hybrid-Transformer-Mamba-for-Li-Liu/c77596bb9f7d630854239ab0ddc00626074e95d3)
- VMRA-MaR（纵向乳腺 MRI + Mamba，MICCAI'25）：[arXiv 2506.17412](https://arxiv.org/abs/2506.17412)
- Diff-RRG（MICCAI'25）：[页面](https://papers.miccai.org/miccai-2025/0234-Paper4138.html) + [GitHub](https://github.com/ku-milab/Diff-RRG)
- 时空+检索纵向 RRG（IEEE TMI'25）：[S2](https://www.semanticscholar.org/paper/Spatio-Temporal-and-Retrieval-Augmented-Modeling-Yang-You/cb312c611595ae9af6a29de36247c949580c171a)
- Transition-Aware best-of-N（纵向 CXR 免训练采样）：[arXiv 2606.28393](https://arxiv.org/abs/2606.28393)
- SDPL（MICCAI'24，症状解耦进展学习）：[页面](https://papers.miccai.org/miccai-2024/751-Paper3074.html)
- 解剖特定进展分类（Radiology AI'24）：[PMC11427915](https://pmc.ncbi.nlm.nih.gov/articles/PMC11427915/)
- MRID（影像差多任务自监督，Electronics'25）：[MDPI](https://www.mdpi.com/2079-9292/15/5/997/xml)
- CheXRelNet（解剖感知纵向关系）：[scite](https://scite.ai/reports/chexrelnet-an-anatomy-aware-model-for-tracking-wmG2kjJM)
- Hist-AID：[arXiv 2411.10684](https://arxiv.org/abs/2411.10684)；HC-LLM（AAAI'25）；EVOKE：[arXiv 2411.10224](https://arxiv.org/abs/2411.10224)
- TrajGPT：[arXiv 2410.02133](https://arxiv.org/abs/2410.02133)；ChronoFormer：[arXiv 2504.07373](https://arxiv.org/abs/2504.07373)