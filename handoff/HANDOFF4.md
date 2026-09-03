# HANDOFF4.md — 2026-08-13（上午）窗口交接

> 新窗口开工第一步：依次读 `handoff/HANDOFF1.md`、`handoff/HANDOFF2.md`、`handoff/HANDOFF3.md`、`handoff/HANDOFF4.md`，再执行"1. 当前状态"里的检查命令确认训练进度。

## 1. 当前状态（最重要）

- **MIMIC-CXR 训练正在跑**（用户用 setsid 命令自行启动；未走 `train_bg.sh`，所以没有 `train.pid`）：
  - 启动时间：2026-08-13 约 01:22；启动命令（用户认可的固定姿势）：
    ```bash
    cd /home/sjx/work/project/DDaTR
    conda activate ddatr
    setsid --fork bash train_mimic_cxr.sh > train.txt 2>&1 < /dev/null &
    ```
  - 日志：仓库根目录 `train.txt`（tqdm 进度条）；监控：`tail -f train.txt` 或 `tail -1 train.txt`
  - **稳定速度 ~1.3-1.35 it/s（≈80 步/分钟）**；开头几百步可能显示 2.5 it/s（缓存红利，会回落，别被骗）
  - 单 epoch：训练 ~3.5h + 每 epoch 末 val/test 评估（val 2,130 / test 3,858）
  - 交接时进度：Epoch 3 约 9%（~1560/16924）；Epoch 1 约 04:50 完成、Epoch 2 约 08:20 完成
  - 预计总时长 35~40h → **约 8/14 中午~下午跑完**
- 已产出：`results/mimic_cxr_train/model_best.pth`（按 val_ce_f1 择优，交接时是 epoch 2 版本，val_ce_f1=0.3669）
- 进程识别：`ps -eo pid,cmd | rg main_train`（agent + worker + 4 个 dataloader）；GPU：`nvidia-smi`（worker 占 ~22.8G/48G，利用率 100%）
- 指标轨迹（test 集；论文 Tab. I MIMIC 目标：BLEU-1 0.401 / BLEU-4 0.113 / METEOR 0.162 / ROUGE-L 0.272 / CE F1 0.441）：
  - Epoch 1：BLEU-1 0.390 / BLEU-4 0.103 / METEOR 0.149 / ROUGE-L 0.262 / CE Prec 0.381 / Recall 0.420 / F1 0.364
  - Epoch 2：BLEU-1 0.398 / BLEU-4 0.110 / METEOR 0.156 / ROUGE-L 0.270 / CE Prec 0.423 / Recall 0.390 / F1 0.374
  - 文本指标已贴近目标；CE 差距最大（主要 recall 偏低）。CE 是模型选择标准，后面 epoch 一般继续涨，先观察

## 2. 任务概述

复现论文 **DDaTR: Dynamic Difference-aware Temporal Residual Network for Longitudinal Radiology Report Generation**（IEEE TMI 2025, xmed-lab）：跑通 MIMIC-CXR / Longitudinal-MIMIC / IU-Xray / ReXrank 测试 → 训练复现 → 对比论文 Tab. I / V。
论文纯文本：`/tmp/ddatr_paper.txt`（离线查指标用）。

## 3. 已完成工作（HANDOFF1~4 汇总）

- 环境：`ddatr` conda 环境可用（torch 2.1.1+cu118、transformers 4.25.0、numpy 1.24.3、tokenizers 0.13.3）；数据/官方权重/chexbert/HF 缓存（bert-base-uncased、swin-base）齐全
- `modules/trainer.py`：删掉 `return_gates=False`（models/blip.py 的 generate() 没有该参数）+ tqdm 进度条 + epoch 起止/评估打印
- 测试：`test_rexrank.sh` 已用官方权重跑通（纵向 2,058 条）：
  - BLEU-1 0.3934 / BLEU-2 0.2370 / BLEU-4 0.1083 / METEOR 0.1612 / ROUGE-L 0.2690；CE(macro) Prec 0.4422 / Recall 0.4866 / F1 0.4544
  - 论文 Tab. V 的 ReXrank 数字（BLEU-2 0.210 / F1-RadGraph 0.198 / BertScore 0.362 / SembScore 0.465 / GREEN 0.293）是官方 rexrank 评估器口径，与本仓库 pycocoevalcap 不可比
  - `test_rexrank.sh` 保持官方原样：`--ann_path ./data/mimic_cxr/longitudinal_mimic_ddatr.json`
- `train_mimic_cxr.sh`：一行改多行、去掉脚本内后台重定向、`--save_dir ./results/mimic_cxr_train`、头部加 HF 离线变量
- 训练启动方式排查（本窗口重点，已实测）：
  - `nohup` 对 torchrun 无效：torch elastic 启动器自己注册 SIGHUP 处理器（`api.py:237` `signal.signal(signal.SIGHUP, _terminate_process_handler)`），覆盖 nohup 的忽略；SSH 断线/关窗口时 SIGHUP 送达即整组退出。实验：torchrun+nohup 收 SIGHUP 必死，普通 python+nohup 存活
  - 解决：`setsid --fork` 启动（训练进独立会话、无控制终端，挂断信号送不到）
  - 备选脚本 `train_bg.sh`（仓库根目录：setsid 包装 + 防重复启动 + train.pid 记录）；用户选择直接用 setsid 命令
- 训练复现已启动（见第 1 节）

## 4. 当前卡在哪

- 无硬阻塞：唯一在等训练跑完（35~40h 长任务，预计 8/14 中午~下午）
- **训练中途没有 checkpoint**（只在每 epoch 末按 val_ce_f1 存 best）→ 别轻易杀/重启训练，重启 = 从头开始
- trainer.py 等代码改动对运行中的训练不生效，必须重启才生效

## 5. 下一步计划（训练完成后）

1. 换新权重流程（HANDOFF3 第 5 节）：
   - 备份官方：`cp checkpoints/ddatr/model_best.pth checkpoints/ddatr/model_best_official.pth`
   - 覆盖：`cp results/mimic_cxr_train/model_best.pth checkpoints/ddatr/model_best.pth`
   - 每个测试脚本前先 `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
   - 依次跑：`bash test_mimic_cxr.sh`（MIMIC 3,858）→ `bash test_iu_xray.sh`（IU 零样本）→ `bash test_rexrank.sh`（纵向 2,058）
   - 结果对比论文 Tab. I（MIMIC/Longitudinal/IU 列）；想恢复官方：把备份拷回去
2. （待确认，建议不做）`main_train.py` 的 `find_unused_parameters=True` 改 False：已确认安全、可提速；但现在训练已跑 3+ epoch，不值得为它重启
3. （可选）CE 纵向列 micro 口径对齐（改 `modules/metrics_clinical.py`；论文纵向列 CE 用 micro，MIMIC/IU 列用 macro）
4. （可选）用官方 rexrank 评估器重跑 ReXrank 指标对齐论文 Tab. V

## 6. 踩过的坑（绝对不要再踩）

1. **trainer.py 的 `return_gates=False` 不要加回**（models/blip.py 的 generate() 没有该参数，验证阶段 TypeError）
2. **nohup 对 torchrun 无效**：torch elastic 自己注册 SIGHUP 处理器，覆盖 nohup 的忽略；SSH 断线/关窗口必挂（已实测）。必须 `setsid --fork`（或 train_bg.sh）。普通单进程 python 训练用 nohup 没事
3. **训练启动前必须激活 ddatr 环境**：`conda activate ddatr` 后 `which python` 确认是 `/home/sjx/miniconda3/envs/ddatr/bin/python`；base python 无 torch（`ModuleNotFoundError: No module named torch`），启动即失败
4. **不要用 `pkill -u sjx -9 -f python`**：会杀掉所有 python（其他项目、Jupyter 等）；停训练用 `pkill -f main_train.py`（默认 SIGTERM，torch 优雅退出）；强制才加 -9，且仍限定 main_train.py
5. **pkill/pgrep -f 不要匹配自己命令行里含有的字符串**（会误杀自己的 shell）；用 bracket 写法 `[m]ain_train.py`，或启动时 `$!` 拿 PID
6. **训练速度别被开头骗了**：tqdm 前几百步可能显示 2.5 it/s（缓存红利），稳定值 ~1.3 it/s
7. 沙箱 bwrap 报错（`bwrap: loopback: Failed RTM_NEWADDR`）间歇性：重试即可；写文件失败就请用户手动执行；escalate 的 auto-review 服务可能 401 不可用，被拒后改用沙箱内可用命令或请用户手动
8. 测试前必须 `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`（huggingface.co 不可达）；可用代理 `127.0.0.1:7897`；`.bashrc` 的 setproxy:12345 不可用
9. 训练产物在 `results/mimic_cxr_train/`，测试只读 `checkpoints/ddatr/model_best.pth`：换权重前先备份官方权重，别丢
10. MIMIC 图片嵌套软链路径（`data/mimic_cxr/mimic-cxr-jpg/mimic-cxr-jpg_2.0.0/files -> CAMANet data`）勿动；CAMANet/CMCRL 数据只读勿删
11. 分布式端口 `--master_port=12361`，冲突时换端口
12. `test_rexrank.sh` 的 ann_path 保持官方原样（longitudinal_mimic_ddatr.json），不要改成 `rexrank_mimic_test_ddatr.json`（那是另一批 2,058 条，不是官方测试口径）
13. conda clone 后必须 `which python pip` 验证（克隆失败会留半成品环境，只能 rm -rf 重建）

## 7. 环境速查

- 工作目录：`/home/sjx/work/project/DDaTR`；conda env：`ddatr`；GPU：单卡 RTX 5880 Ada 48G（`CUDA_VISIBLE_DEVICES=0`）
- 看训练：`tail -f train.txt`；看进程：`ps -eo pid,cmd | rg main_train`；看 GPU：`nvidia-smi`；停训练：`pkill -f main_train.py`
- 数据文件：
  - `data/mimic_cxr/longitudinal_mimic_ddatr.json`：纵向（test 2,058）
  - `data/mimic_cxr/mimic_cxr_ddatr.json`：MIMIC 全量（train 270,790 / val 2,130 / test 3,858）
  - `data/iu_xray/iu_annotation_promptmrg.json`：IU 标注
  - `checkpoints/ddatr/model_best.pth`：官方权重（勿丢）
  - `checkpoints/stanford/chexbert/chexbert.pth`：CE 评估
- 论文关键数字（Tab. I）：MIMIC Prec 0.438 / Recall 0.465 / F1 0.441、BLEU-1 0.401 / BLEU-4 0.113；Longitudinal F1 0.527、BLEU-1 0.396 / BLEU-4 0.105（CE 用 micro）；IU F1 0.262、BLEU-1 0.421 / BLEU-4 0.103

## 8. 未提交改动（git status 摘要，不要擅自 commit）

- M `modules/trainer.py`（tqdm/epoch 打印；return_gates=False 已删）
- M `train_mimic_cxr.sh`（多行格式 / save_dir / 离线变量）
- 未跟踪 `train_bg.sh`（备选启动脚本）
- D `data/iu_xray/README.md`（来源不明，如不需要可忽略）
- 其他大量未跟踪（checkpoints/、data 标注 json、handoff/、__pycache__ 等）为正常状态
- 用户偏好：训练由用户自己启动/管理，不要代跑
