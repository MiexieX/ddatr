# HANDOFF5.md — 2026-09-03（合并版：凌晨窗口 + 白天调研窗口）

> 新窗口开工第一步：依次读 `handoff/HANDOFF1.md`~`HANDOFF5.md`，再执行"1. 当前状态"里的检查命令确认环境与产物。

## 1. 当前状态（最重要）

- **训练已完成、复现成功**（10 epochs，best = epoch 7，按 val_ce_f1=0.4262 择优；epoch 8/9/10 val_ce_f1 连续回落 → 过拟合，别加 epochs）。
- **复现权重位置（注意目录已改名）**：`results/基线训练权重/model_best.pth`（epoch 7，md5 9ef26bb…）
- **测试集复现指标（与论文 Tab I 差距 ≤±0.003）**：
  - MIMIC 官方 split（3,858）：BLEU-1 0.4024 / B-4 0.1099 / METEOR 0.1596 / ROUGE-L 0.2700 / CE P 0.4271 / R 0.4734 / **F1 0.4393**（论文 0.401/0.113/0.162/0.272/0.438/0.465/0.441）；最大缺口 CE Prec（0.438 vs 0.427）
  - IU-Xray 零样本（4,168）：B-4 0.1070 / METEOR 0.1605 / ROUGE-L 0.3094 / CE F1 0.2583（论文 0.262）
  - Longitudinal-MIMIC（2,058，rexrank 口径）：B-4 0.1045 / METEOR 0.1586 / ROUGE-L 0.2658 / CE F1 0.4437（macro，非论文 micro 0.527 口径）
  - 分数日志存档：`results/test/基线测试分数/`（test_mimic/iu/rexrank.log + summary.txt）
- **权重所有权（纪律）**：`checkpoints/ddatr/model_best.pth` = 官方权重（md5 cbdcfcc…，勿动勿覆盖）；`checkpoints/ddatr/model_best_official.pth` = 官方备份（同指纹）；复现权重只放 `results/`
- GPU 硬事实：**RTX 3090 24G**（nvidia-smi 实测；HANDOFF4 的 48G 记录有误）；训练 batch16 占 ~21.4G/24.6G；**训练与其它 GPU 任务不能并行**
- HF 缓存已有 `TinyLlama-1.1B-Chat-v1.0`；torch 2.1.1+cu118 / transformers 4.25.0；**peft/bitsandbytes/accelerate/flash_attn 未装**（LoRA 需新 conda env）

## 2. 任务概述

复现 DDaTR（IEEE TMI 2025）✅ → 现进入论文阶段：**问题驱动的创新点方案 ProgressPlan**，硬性要求 **BLEU 与 CE F1 双协议同时提升**，目标 IEEE TMI / MedIA 一区。方案文档见 `research/FINAL_PLAN.md`（先读这个）。

## 3. 论文调研成果（全部已落盘）

按重要度排序：
1. **`research/FINAL_PLAN.md`** ← 最终方案（问题驱动版）：P1 看不见要写的变化→M3 进展事件监督+C1 进展句规划/主题槽+C2 跨模态差分一致性；P2 摆位当变化→M1 形变配准；P3 1天当1年→M2 间隔条件化门控；P4 写不出句式→解码升级+C3 变化模式检索+档B TinyLlama(QLoRA)。含 10 周路线、预期指标账（MIMIC macro B-4 0.110→0.125~0.16 / F1 0.439→0.455~0.48；Long micro B-4 0.105→0.115~0.15 / F1 0.527→0.53~0.56）、评审预案、不做清单
2. `research/innovation_research_final.md`：第一轮主报告（M1/M2/M3 三模块初版 + 竞品全景）
3. `research/bleu_f1_dual_report.md`：BLEU×F1 双指标专项（REVA-PO 用 BLEU-4 当 RL 奖励同时抬 F1+4.5%/RadGraph+12.8%；TIM 消融进展组件双涨；**单图赛道 REVA-PO 已双超 DDaTR，我们主战场限定 Long-MIMIC micro 双超**）
4. `research/cross_domain_round2.md`：第二轮跨领域 C1~C7（C1 进展句规划⭐数据就绪/ C2 差分一致性/ C3 变化范式检索/ C4 时间轴token/ C5 解剖掩码/ C6 蒸馏/ C7 EHR 留第二篇）
5. `research/ddatr_next_framework_draft.md`：框架工作稿（含 24G 可行性）
6. `LRRG_文献调研报告_2024-2026.md`（根目录）：竞品全景+双协议陷阱+撞车清单+空白排序 A-E
7. `handoff/innovation_research.md`：跨领域 11 方向（含 6-8 周实施路线）

**关键竞品事实（写论文对比用，全部要统一口径重跑，代码全公开）**：TIM(CVPR'26) Long-MIMIC F1 0.511、BiOTPrompt(CVPR'26) 双向OT macro F1 0.417、MARE(AAAI'26)、Diff-RRG(MICCAI'25)、HC-LLM(AAAI'25) B-4 0.142、MLRG(CVPR'25)、LLaVA-TA(ICLR'26) topic分解 BLEU+F1 双涨（C1 实证）、REVA-PO(arXiv:2607.10147)、LLM-RG4(AAAI'25)。**注意：TIM/BiOTPrompt 对比表都没列 DDaTR，口径混乱（macro/micro）——自己重跑才能作数**。纵向进展句标注基准：arXiv:2601.16753（95k 条 LLM 标注，可反用做 C1 训练数据）。

## 4. 平台成本修复（重要，别回退）

- **联网搜索原来走官方 DeepSeek API 按量计费**（web-search-deepseek 插件硬编码 api.deepseek.com/anthropic/v1 + DEEPSEEK_API_KEY；本窗口核实过一个会话 46 次搜索）。
- 已修复：`~/.dsh/settings.yaml` 保持以下配置（热更新生效，实测 zen/go 支持原生 web_search_20250305）：
  ```yaml
  web-search-deepseek:
    baseURL: https://opencode.ai/zen/go/v1   # 必须带 /v1（provider 会拼 /messages）
    apiKeyEnv: OPENCODE_GO_API_KEY
    model: deepseek-v4-flash
  ```
- 主 agent LLM 通道本来就是 opencode-go 订阅（settings.yaml `agent-default-model`）；密钥在 `~/.dsh/.credentials.yaml`（OPENCODE_GO_API_KEY）。
- 教训：**少开子代理**（推理+搜索耗配额/费钱，用户成本敏感）；搜索合并查询、控制轮次。本窗口曾中断两个子代理（aa3e0bbb/c52fdaf1），剩余专项内容见「待补查」。

## 5. 代码改造清单（未 commit，不要擅自 commit）

| 文件 | 改动 | 状态 |
|---|---|---|
| `main_test.py` | 新增 `--save_reports --report_dir --report_examples --report_note_chars`；`save_reports()` 输出 **CSV**：`真实报告.csv`（医生GT+先验+进展事件）/ `模型生成报告.csv`（模型文本+诊断prompt）/ `评估得分.csv` / `病例对照.md`（前24例）；目录 = `--report_dir`，两份 CSV 按 index/id 对齐 | ✅ 用户自跑 |
| `modules/tester.py` | `test_blip()` 返回第 4 值 `test_cls_preds`（分类 prompt）；指标路径未动 | ✅ |
| `test_mimic_cxr.sh` / `test_iu_xray.sh` / `test_rexrank.sh` | 带 `--save_reports --report_dir ./results/test/{mimic\|iu\|rexrank}`；修正原 `CUDA_VISIBLE_DEVICES=0,` 逗号 | ✅ |
| `train_mimic_cxr.sh` | `SAVE_DIR=./results/mimic_cxr_train_$(date +%Y%m%d_%H%M%S)` 时间戳独立目录；日志建议 `train_${TIMESTAMP}.txt` | ✅ |
| `modules/trainer.py` | 每 epoch 末存 `model_last.pth`（当前权重）；训练完自动写 `best_scores.txt`（best 全套 val/test 得分）；`model_best.pth` 逻辑不变 | ✅ |
| `tools/generate_diagnostic_reports.py` | **已删除**（与主流程重复的临时入口） | ✅ |
| `parse_train_log.py` / `run_all_tests.sh` | 凌晨窗口新增（解析训练日志 / 一键三测试 → `results/test/<时间戳>[_备注]/`），保留 | 凌晨窗口 |
| `modules/trainer.py`（凌晨打印重构） | `>>> NEW best` 打印、epoch 结果块、末尾 Best 块标注 | 凌晨窗口 |

## 6. 目前卡在哪 / 注意

- 无硬阻塞。**用户明确偏好：训练/测试/生成等 GPU 任务由用户自己跑，不代跑**（凌晨窗口代跑被叫停过）。
- 测试脚本无进度条是正常的（每 10 batch 打印 + stdout 块缓冲），看进度用 `nvidia-smi` 或 `ps`。

## 7. 下一步计划（论文主线 ProgressPlan，research/FINAL_PLAN.md 详版）

1. **用户侧（快）**：跑三个带 --save_reports 的测试脚本 → `results/test/{mimic,iu,rexrank}/` 出 CSV；跑 `train_mimic_cxr.sh` → 时间戳目录（model_best/model_last/best_scores.txt）
2. **W1 问题证据（零训练，直接出论文 Intro 素材）**：按间隔分桶 BLEU/CE 曲线（数据统计：间隔 1→50,001 极度偏斜）+ 同/异视角分层（ViewPosition 逐条存在）+ 解码升级网格（length_penalty≈1.5-2.0/beam 4-5/rep_penalty 1.05）+ C3 变化模式检索原型（推理端 KNN，按 CheXbert 标签转移向量检索）
3. **W2-3 M3 进展事件监督**（recall 主菜；标签构造规则：UNC-free 硬目标/UNC 参与软目标；数据实测 new 8.91%/resolved 8.58%，87.5% study 对含事件，干净事件 26万+）→ **W3-4 M2 间隔门控**（resnet.py L103/L107）→ **W4-5 M1 配准**（RegNet+NCC，先仿射后形变）→ **W5-6 C1 规划端**（反用 2601.16753 进展句标注）→ **W7-8 消融/竞品复跑/显著性** → **W9-10 档B TinyLlama 或 REVA-PO RL + ReXrank 提交 + 成稿**
4. **投稿前待补查（一次搜索解决一个，成本收敛）**：MRG-R1 纯语义 RL 后 BLEU 是否下降；LLM-RG4 确切协议；ICLR'26 LLaVA-TA 数字（papernotes 链接在 cross_domain_round2.md）

## 8. 踩过的坑（绝对不要再踩）

1. **实际 GPU 是 RTX 3090 24G**，别再说 48G；训练占 22G，任何 GPU 任务前先 `nvidia-smi`
2. **不要代跑用户的训练/测试/生成**（用户明确要求；只给命令等用户执行）
3. **checkpoints/ 只放官方权重**（当前 model_best.pth 与 model_best_official.pth 同指纹，md5 cbdcfcc…，勿覆盖）；复现/实验权重一律 results/
4. **联网搜索已切 opencode go 订阅**（settings.yaml 已配），别改回 DEEPSEEK_API_KEY；少开子代理、合并查询
5. 训练脚本现在每次独立时间戳目录，别在旧目录找新权重；`best_scores.txt` 只在全部 epoch 跑完后写，中途停训只有 `model_last.pth`
6. 测试 `--save_reports` 只影响落盘不影响指标；CSV 为 utf-8-sig；两份 CSV 按 index/id 对齐（dataloader 顺序 = annotation 顺序）
7. 论文对比双协议陷阱：MIMIC 全量 macro CE / Long-MIMIC micro CE；新竞品口径混乱，所有"超越"须自己统一口径重跑
8. 时间间隔：JSON 无 context_time，需 study_id 查表（context_image 路径含先验 study_id）；label 编码 0=BLA 1=POS 2=NEG 3=UNC；事件标签有噪声（CheXbert 误判/UNC 翻转）
9. 沿用 HANDOFF4 第 6 节：trainer return_gates 不要加回、nohup 对 torchrun 无效（必须 setsid）、pkill 限定 main_train.py、测试前 HF_HUB_OFFLINE=1、test_rexrank.sh ann_path 保持官方原样

## 9. 环境速查

- 工作目录：`/home/sjx/work/project/DDaTR`；conda env：`ddatr`；GPU：RTX 3090 24G（CUDA_VISIBLE_DEVICES=0）
- 权重：复现 best = `results/基线训练权重/model_best.pth`；官方 = `checkpoints/ddatr/model_best.pth`（勿动）
- 测试（用户自跑）：`bash test_mimic_cxr.sh` / `test_iu_xray.sh` / `test_rexrank.sh` → `results/test/{mimic,iu,rexrank}/`；一键版 `bash run_all_tests.sh`（凌晨窗口，结果 `results/test/<时间戳>/`）
- 训练：`bash train_mimic_cxr.sh`（setsid 方式启动，日志 `train_<时间戳>.txt`）；日志解析 `python parse_train_log.py train.txt`
- 数据速查：MIMIC test 3,858 / val 2,130 / train 270,790；Long 2,058；IU 4,168
- 关键数字：MIMIC B-4 0.1099 / F1 0.4393（宏）；Long B-4 0.1045 / F1 0.4437（宏）；IU B-4 0.1070 / F1 0.2583

## 10. git 状态（不要擅自 commit）

- M：`main_test.py`、`modules/tester.py`、`modules/trainer.py`、`test_mimic_cxr.sh`、`test_iu_xray.sh`、`test_rexrank.sh`、`train_mimic_cxr.sh`
- ??：`research/`（6 份文档）、`LRRG_文献调研报告_2024-2026.md`、`handoff/`（含 innovation_research.md、HANDOFF5.md）、`paper_cache/`、`checkpoints/`、`results/`（基线训练权重/基线测试分数）、data 标注 json、`rexrank_mimic_test_ddatr.json` 等（正常）
- 残留可清理（用户确认后再动）：`results/experiments/20260903_013734_trained_ep7/`（凌晨窗口代跑遗留）、`run_tests.nohup.log`