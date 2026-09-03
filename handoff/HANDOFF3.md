# HANDOFF3.md — 2026-08-12（夜）窗口交接

> 新窗口开工第一步：依次读 `handoff/HANDOFF1.md`、`handoff/HANDOFF2.md`、`handoff/HANDOFF3.md`，再执行"1. 当前状态"里的检查命令。

## 1. 当前状态
- MIMIC-CXR 训练正在跑（用户终端启动，nohup 后台）：
  - 启动命令：`nohup bash train_mimic_cxr.sh > train.txt 2>&1 &`
  - 日志：仓库根目录 `train.txt`；监控用 `tail -f train.txt` 或 `sed -n '$p' train.txt`
  - 日志格式（当前跑的是旧 trainer）：每 10 步打印 `7190/16924 loss: 5.39..., loss_lm: ..., loss_cls: ...`
  - 实测速度 ~72-79 步/分钟（约 0.8s/步）；总量 16,924 步/epoch × 10 epoch ≈ 169,240 步
  - 23:11 时进度 7190/16924（epoch 0，约 42%）；预计总时长 ~37-39 小时（含每 epoch 末 val/test 评估），约 8/14 上午~中午跑完
  - 估算公式：剩余步数 / 75 ≈ 剩余分钟
- 训练产出：每轮 epoch 末按 `val_ce_f1` 择优保存 `results/mimic_cxr_train/model_best.pth`（trainer 逻辑见 `modules/trainer.py` train()）
- 当前进程是旧版 trainer.py（无进度条）；进度条/其他代码改动必须重启训练才生效

## 2. 环境与执行前提
- 工作目录：`/home/sjx/work/project/DDaTR`；所有命令都从这里执行
- conda 环境：`ddatr`（`conda activate ddatr`）
- GPU：单卡 RTX 5880 Ada 48G（`CUDA_VISIBLE_DEVICES=0`），`nvidia-smi` 确认可用
- 网络：huggingface.co 不可达（训练/测试都要离线变量）：
  - `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
  - 训练脚本已内置；测试脚本需手动 export
  - 若需联网：代理 `http://127.0.0.1:7897` 可用；`.bashrc` 的 setproxy:12345 不可用
- 分布式端口：`--master_port=12361`（冲突时改端口）

## 3. 本窗口完成的工作

### 3.1 ReXRank 测试跑通 + 对不上论文的口径核对
- `bash test_rexrank.sh` 跑通纵向测试（2,058 条样本）结果：
  - BLEU-1 0.3934 / BLEU-2 0.2370 / BLEU-4 0.1083 / METEOR 0.1612 / ROUGE-L 0.2690
  - CE（macro）：Prec 0.4422 / Recall 0.4866 / F1 0.4544
- `test_rexrank.sh` 恢复官方原样：`--ann_path ./data/mimic_cxr/longitudinal_mimic_ddatr.json`，不要再改成 `rexrank_mimic_test_ddatr.json`
- 已确认 `longitudinal_mimic_ddatr.json` 的 test split（2,058 条）与仓库根目录 `rexrank_mimic_test_ddatr.json`（2,058 条）不是同一批样本（首条 id 互不在对方文件里）；官方脚本就是用 longitudinal 的 test，能跑通即可
- 对不上论文的根因（已从本地 PDF 文本确认，`/tmp/ddatr_paper.txt`）：
  - 论文 Tab. V ReXRank（DDaTR）：BLEU-2 0.210 / F1-RadGraph 0.198 / BertScore 0.362 / SembScore 0.465 / GREEN 0.293 —— 这是官方 rexrank 评估器口径，与本仓库本地 pycocoevalcap 不可比
  - 论文 Tab. I Longitudinal-MIMIC 列（复现目标）：Prec 0.539 / Recall 0.584 / F1 0.527 / BLEU-1 0.396 / BLEU-4 0.105 / METEOR 0.156 / ROUGE-L 0.266
  - 表注：纵向列 CE 用 MICRO（LR2Gen 口径），MIMIC/IU 列用 MACRO；本仓库 `modules/metrics_clinical.py` 算的是 MACRO（`precision_class.mean()`）→ 纵向 CE 对不上是主因
  - 论文 Longitudinal-MIMIC = LR2Gen 官方（266 patients / 2,058 samples）；MIMIC test = 3,858；MIMIC 中的纵向子集 = 3,153，不是 2,058

### 3.2 train_mimic_cxr.sh 改造（用户要求）
- 原来一整行 → 拆成多行参数，格式与其他 sh 一致
- 去掉脚本内的 `> ddatr.out 2>&1 &`，由用户自己 `nohup bash xxx > train.txt 2>&1 &`
- `--save_dir ./results/mimic_cxr_train`（不再用 checkpoints/ddatr_train）
- 头部加 `export HF_HUB_OFFLINE=1` 和 `export TRANSFORMERS_OFFLINE=1`

### 3.3 训练排障
- GPU 驱动故障：内核模块 580.159.03 vs NVML 580.173 → 重启机器解决（重启后 SSH 即可，无需图形化登录；需要手动重新起训练）
- huggingface.co 不可达 → 离线变量解决（见第 2 节）
- 首次启动失败（distributed init exit 1）就是驱动问题，重启后正常

### 3.4 trainer.py 进度条（用户要求，未生效）
- `modules/trainer.py`：主进程且 tqdm 可用时显示 tqdm 进度条
  - desc `Epoch x/10`，postfix 显示 loss/lm/cls，mininterval=1.0
  - epoch 起止有打印行；evaluating on val/test 前有提示
  - 无 tqdm 时回退原每 10 步打印逻辑
- tqdm 4.70.0 已在 env `ddatr` 确认存在
- 当前跑着的训练进程是旧代码；重启后才有进度条

## 4. 代码改动清单（未 commit，git diff 已核实；不要擅自 commit）
| 文件 | 改动 |
|---|---|
| M `modules/trainer.py` | tqdm 进度条 + epoch 起止/评估提示；generate 调用去掉 `return_gates=False` |
| M `train_mimic_cxr.sh` | 一行改多行、去后台重定向、save_dir 改 `results/mimic_cxr_train`、加离线变量 |

## 5. 测试脚本与换权重流程（训练完成后执行）
三个测试脚本都加载 `./checkpoints/ddatr/model_best.pth`（官方权重），跑之前先 `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`：

| 脚本 | ann_path | save_dir | 说明 |
|---|---|---|---|
| `test_mimic_cxr.sh` | `data/mimic_cxr/mimic_cxr_ddatr.json` | `results/mimic_cxr` | MIMIC test 3,858 |
| `test_iu_xray.sh` | `data/iu_xray/iu_annotation_promptmrg.json` | `results/iuxray` | IU 零样本 |
| `test_rexrank.sh` | `data/mimic_cxr/longitudinal_mimic_ddatr.json` | `results/rexrank` | 纵向 2,058 |

换新训练权重流程：
1. 备份官方：`cp checkpoints/ddatr/model_best.pth checkpoints/ddatr/model_best_official.pth`
2. 覆盖：`cp results/mimic_cxr_train/model_best.pth checkpoints/ddatr/model_best.pth`
3. 依次跑三个测试脚本，结果与论文对照（Tab. I Longitudinal-MIMIC 列 / Tab. V）
4. 想恢复官方：把备份拷回去

## 6. 下一步工作
1. 等训练跑完 → 按第 5 节流程测三个数据集 → 对比论文 Tab. I
2. （待确认）`main_train.py` 的 `find_unused_parameters=True` 改 False：已确认安全（`models/resnet.py` 的 has_progress 是张量门控、无未使用参数，torch 警告也确认过），可提速且不影响结果；用户尚未确认。当前才 epoch 0，若决定改，建议趁早重启
3. 若重启训练（改 find_unused_parameters 或启用进度条）：`pkill -f torch.distributed.run; pkill -f main_train.py` → `nohup bash train_mimic_cxr.sh > train.txt 2>&1 &`；重启后日志会出现 tqdm 进度条
4. （可选）CE 纵向列对齐论文：改 `modules/metrics_clinical.py` 支持 micro 聚合（用户之前说"不用了"，如要对齐 Tab. I 纵向列再做）
5. （可选）用官方 rexrank 评估器重跑 ReXRank 指标（F1-RadGraph/BertScore/SembScore/GREEN），对齐论文 Tab. V

## 7. 注意事项 / 踩坑记录
- 不要加回 `return_gates=False`（trainer.py 里原本该行已删）
- 图片软链引用 CAMANet 数据，只读勿删
- 所有命令从仓库根目录执行；conda env `ddatr`
- 沙箱 bwrap 报错（`bwrap: loopback: Failed RTM_NEWADDR`）是间歇性的，失败重试即可；写入文件也常失败，必要时让用户手动执行
- 数据文件说明：
  - `data/mimic_cxr/longitudinal_mimic_ddatr.json`：LR2Gen 纵向格式（test 2,058）
  - `rexrank_mimic_test_ddatr.json`（仓库根目录）：另一批 2,058 样本，与上面不是同一批
  - `data/mimic_cxr/mimic_cxr_ddatr.json`：MIMIC 全量（train 270,790 / val 2,130 / test 3,858）
  - `data/mimic_cxr/base_probs.json`：疾病先验，训练用
  - `checkpoints/ddatr/model_best.pth`：官方权重（勿丢，已计划备份）
  - `checkpoints/stanford/chexbert/chexbert.pth`：CE 评估用