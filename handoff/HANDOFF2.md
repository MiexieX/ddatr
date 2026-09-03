# DDaTR 复现任务交接文档（2026-08-12）

## 1. 任务

复现 DDaTR: Dynamic Difference-aware Temporal Residual Network for Longitudinal Radiology Report Generation（IEEE TMI 2025）。

目标：跑通 MIMIC-CXR / Longitudinal-MIMIC / IU-Xray / ReXrank 测试，再训练复现。

## 2. 已完成

- `ddatr` conda 环境已重建并验证可用。
- torch `2.1.1+cu118`，CUDA True，transformers `4.25.0`，tokenizers `0.13.3`，numpy `1.24.3`。
- `modules/trainer.py` 已删掉 `return_gates=False`，不要再加回去。
- DDaTR `data/` 已取消整目录软链，只保留本项目需要的本地文件。
- 图片仍通过单目录软链引用：
  - `data/mimic_cxr/mimic-cxr-jpg/mimic-cxr-jpg_2.0.0/files -> /home/sjx/work/project/CAMANet/data/mimic_cxr/images`
  - `data/iu_xray/images -> /home/sjx/work/project/CAMANet/data/iu_xray/images`
- 数据文件已齐：MIMIC 标注、ReXrank、Longitudinal-MIMIC、base_probs、IU 标注。
- 权重已齐：`checkpoints/ddatr/model_best.pth`、`checkpoints/stanford/chexbert/chexbert.pth`。
- HF 缓存已齐：`bert-base-uncased`、`microsoft/swin-base-patch4-window7-224`。
- 验证通过：
  ```text
  model_ok 1562
  chexbert_ok 3
  HF_LOAD_OK
  ```

## 3. 当前卡在哪

- 测试还没有跑。
- 环境、数据、权重、HF 缓存都已就绪，无已知阻塞。

## 4. 下一步

```bash
cd /home/sjx/work/project/DDaTR

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

bash test_mimic_cxr.sh
bash test_iu_xray.sh
bash test_rexrank.sh
```

先跑 MIMIC，再跑 IU 和 ReXrank。

后台运行：

```bash
nohup bash test_mimic_cxr.sh > test_mimic.log 2>&1 &
tail -f test_mimic.log
```

训练：

```bash
bash train_mimic_cxr.sh
cp checkpoints/ddatr_train/model_best.pth checkpoints/ddatr/model_best.pth
```

测试只读 `checkpoints/ddatr/model_best.pth`，训练完必须复制。

## 5. 验收基准

- MIMIC-CXR：Prec `0.438` / Recall `0.465` / F1 `0.441`；BLEU-1 `0.401`、BLEU-4 `0.113`、METEOR `0.162`、ROUGE-L `0.272`
- ReXrank：BLEU-2 `0.210`、F1-RadGraph `0.198`、BertScore `0.362`、SembScore `0.465`、GREEN `0.293`
- IU-Xray：F1 `0.262`；BLEU-1 `0.421`、BLEU-4 `0.103`

## 6. 踩过的坑，绝对不要再踩

1. `return_gates=False` 不能加回 `modules/trainer.py`。
2. conda clone 后必须验证 `which python pip`。
3. `.bashrc` 的 `setproxy` 指向 `127.0.0.1:12345`，当前没服务，不要用。
4. 当前可用代理是 `127.0.0.1:7897`。
5. gdown 断线时可能把半成品改名成最终文件，文件存在不代表下载完成。
6. 大文件下载后必须用大小或 `torch.load` 验证。
7. `huggingface-cli` 走 xet 会卡在 0 字节，必须设置 `HF_HUB_DISABLE_XET=1`。
8. `hf-mirror.com` 本机直连超时，走代理会跳回 `huggingface.co`，没有加速作用。
9. HF 缓存只有 config/tokenizer 时测试会尝试联网，可能报 `pytorch_model.bin` 相关错误。
10. 所有命令必须从 `/home/sjx/work/project/DDaTR` 根目录跑。
11. 训练产物在 `checkpoints/ddatr_train/`，测试只读 `checkpoints/ddatr/model_best.pth`。
12. MIMIC 图片必须保持嵌套软链路径。
13. CAMANet / CMCRL 是其他项目的原始数据，只读引用，不要删改。
14. 重新下载前先清理半成品。
15. 测试前设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。
