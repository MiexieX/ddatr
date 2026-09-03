# DDaTR 复现任务交接文档

更新日期：2026-08-12

## 1. 我们在做什么任务

复现论文 **DDaTR: Dynamic Difference-aware Temporal Residual Network for Longitudinal Radiology Report Generation**（IEEE TMI 2025，HKUST xmed-lab）。

- 代码仓库：`/home/sjx/work/project/DDaTR`（GitHub: xmed-lab/DDaTR）
- 论文 PDF：`/home/sjx/work/project/DDaTR/DDaTR Dynamic Difference-aware Temporal Residual Network for Longitudinal Radiology Report Generation.pdf`
- 论文纯文本提取件：`/tmp/ddatr_paper.txt`（查指标用这个，不用联网）
- 目标：搭好环境 → 下载权重/标注 → 跑通 MIMIC-CXR / Longitudinal-MIMIC / IU-Xray / ReXrank 测试 → 训练复现

## 2. 项目关键路径

/home/sjx/work/project/DDaTR/
├── data/
│   ├── mimic_cxr/
│   │   ├── mimic-cxr-jpg/mimic-cxr-jpg_2.0.0/files/   # MIMIC 图片（软链到 CMCRL）
│   │   ├── mimic_cxr_ddatr.json                       # 待下载（GD）
│   │   ├── longitudinal_mimic_ddatr.json              # 已下载 186MB
│   │   ├── rexrank_test.json                          # 待下载（GD，文件名需核对）
│   │   └── base_probs.json                            # 已有
│   └── iu_xray/
│       ├── images/                                    # 软链到 CMCRL
│       └── iu_annotation_promptmrg.json               # 待下载（GD/PromptMRG）
├── checkpoints/
│   ├── ddatr/model_best.pth                           # 待下载（官方权重）
│   ├── ddatr_train/model_best.pth                     # 训练产出
│   └── stanford/chexbert/chexbert.pth                 # 待下载
├── modules/trainer.py                                 # 已修 bug（见踩坑 1）
├── test_mimic_cxr.sh / test_iu_xray.sh / test_rexrank.sh
└── train_mimic_cxr.sh

数据集源（只读引用，勿删源数据）：
- MIMIC 图片：`/home/sjx/work/project/CMCRL-main/data/mimic_cxr/images`（573G，p10~p19，与 DDaTR 标注路径一致）
- IU 图片：`/home/sjx/work/project/CMCRL-main/data/iu_xray/images`（1.4G，CXRxxx/0.png 结构）
- CMCRL 的 `annotation.json`/`labels_all_one.json`/CSV 等是 CMCRL 格式，DDaTR 不会读，仅供目录完整

## 3. 已经完成的工作

1. **仓库已克隆**到 `/home/sjx/work/project/DDaTR`（GitHub 当时 443 超时，后来成功）
2. **修了一个必踩的 bug**：`modules/trainer.py` 第 163 行 `model.module.generate()` 调用里删掉了 `return_gates=False,` 参数（`models/blip.py` 的 `generate()` 没有这个参数，不删的话训练到验证阶段必报 TypeError）。已用 `python -m py_compile` 验证，**不要再加回去**
3. **目录已建好**：`data/mimic_cxr/mimic-cxr-jpg/mimic-cxr-jpg_2.0.0/`、`checkpoints/ddatr/`
4. **已下载**：`data/mimic_cxr/longitudinal_mimic_ddatr.json`（186MB）、`base_probs.json` 已有
5. **图片软链已执行**：
   - `data/mimic_cxr/mimic-cxr-jpg/mimic-cxr-jpg_2.0.0/files -> CMCRL-main/data/mimic_cxr/images`
   - `data/iu_xray/images -> CMCRL-main/data/iu_xray/images`
6. **"整目录链接"方案已给出**（用户要求把数据集文件夹的其他东西也链接过来）：把 DDaTR 的两个 json mv 进 CMCRL 源目录 → rm -rf 旧 data/mimic_cxr → 整目录 `ln -s` → 在 CMCRL 目录里补嵌套的 `mimic-cxr-jpg/mimic-cxr-jpg_2.0.0/files -> images`；IU 同理。**此步骤是否已执行，接手时先确认**（执行了的话 `data/mimic_cxr` 应是指向 CMCRL 的软链，且 CMCRL 目录里有 `longitudinal_mimic_ddatr.json`）
7. **环境分析完成**：
   - `stream` 环境（8.1G，python3.10，torch 2.1.1+cu118，cv2、gdown、nltk、timm、scipy、pandas、sklearn 齐全）是理想的克隆母本
   - HF 缓存已有 `bert-base-uncased`、`swin-base-patch4-window7-224`，无需再下

## 4. 当前卡在哪（阻塞项）

**`ddatr` conda 环境损坏，必须重建。**

现象：`conda create -n ddatr --clone stream` 克隆**不完整**（ddatr 只有 5.1G，stream 有 8.1G；`ddatr/bin/` 里只有 gdown/nltk 等脚本，**没有 python/pip**），导致：
- `pip install ...` → `找不到命令 pip`
- `conda install -y python=3.10 pip` → `DirectoryNotACondaEnvironmentError: ... not a conda environment`（conda-meta 没写完整，conda 不认这个目录）

**接手后先执行：**
```bash
conda deactivate
df -h /home/sjx        # 剩余空间必须 >= ~10G，否则克隆还会失败
rm -rf /home/sjx/miniconda3/envs/ddatr      # 半成品只能手动删，conda remove 也会报错
conda create -n ddatr --clone stream        # 纯本地复制，不需要网络
conda activate ddatr
which python pip       # 必须两个都能打印出来，这是克隆成功的唯一判据
python -V && pip -V
若磁盘不足：先清理（du -sh ~/.cache/pip、du -sh ~/.cache/huggingface 等）。
若第二次克隆仍失败（排除磁盘后）：改用全新安装：
conda create -n ddatr python=3.10 pip -y
conda activate ddatr
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gdown transformers==4.25.0 tokenizers==0.13.3 numpy==1.24.3 timm scipy pandas scikit-learn opencv-python nltk
（注意：PyPI 上的 torch==2.1.1 是 CPU 版，必须走 pytorch 的 cu118 index）
5. 下一步计划
5.1 修环境（见上，当前阻塞）
装包版本必须严格：transformers==4.25.0 tokenizers==0.13.3 numpy==1.24.3（仓库 requirements.txt 要求；不要用 stream 自带的 transformers 4.41.2 / numpy 1.26.4）。
5.2 确认/完成数据链接
确认整目录链接是否已执行（见"已完成 6"）；没有就按该方案执行
验证：ls data/mimic_cxr/mimic-cxr-jpg/mimic-cxr-jpg_2.0.0/files/ | head 应看到 p10~p19
5.3 下载缺失文件（base 环境没有 gdown，用 conda run -n stream gdown 或装好 ddatr 后用 ddatr 的 gdown）
cd /home/sjx/work/project/DDaTR
conda run -n stream gdown 1UZFGA8FXuYfnN23eV3ksxoBgRwWKfPZR -O data/mimic_cxr/mimic_cxr_ddatr.json
conda run -n stream gdown 1erDVnUJ-xJ84PkrcFp_hHj_m5EP_DhBs -O data/mimic_cxr/rexrank_test.json
conda run -n stream gdown 1zV5wgi5QsIp6OuC1U95xvOmeAAlBGkRS -O data/iu_xray/iu_annotation_promptmrg.json
conda run -n stream gdown 1zd4rEtDKbzEx0bGAWu4CtuQvRsNJaEMg -O checkpoints/ddatr/model_best.pth
文件	来源	放置
mimic_cxr_ddatr.json	GD 1UZFGA8FXuYfnN23eV3ksxoBgRwWKfPZR	data/mimic_cxr/
ReXrank 测试	GD 1erDVnUJ-xJ84PkrcFp_hHj_m5EP_DhBs	data/mimic_cxr/（下载后核对文件名）
iu_annotation_promptmrg.json	GD 1zV5wgi5QsIp6OuC1U95xvOmeAAlBGkRS（PromptMRG）	data/iu_xray/
model_best.pth	GD 1zd4rEtDKbzEx0bGAWu4CtuQvRsNJaEMg	checkpoints/ddatr/
chexbert.pth	Box stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9 或 HF StanfordAIMI/RRG_scorers（国内加 HF_ENDPOINT=https://hf-mirror.com）	checkpoints/stanford/chexbert/

GD/Box 均需代理（用户 GitHub 443 超时）；实在不行浏览器手动下载放到对应位置，文件名保持不变。
5.4 测试验证（都必须在仓库根目录跑）
conda activate ddatr
bash test_mimic_cxr.sh    # 先验证环境和官方权重
bash test_iu_xray.sh
bash test_rexrank.sh
5.5 训练
bash train_mimic_cxr.sh   # 10 epochs, batch16, lr5e-5, cls_weight4, seed456789; 日志在 ddatr.out
tail -f ddatr.out
cp checkpoints/ddatr_train/model_best.pth checkpoints/ddatr/model_best.pth   # 必须复制！测试脚本只认这个路径
显存不够 OOM 时把 --batch_size 16 调小。
6. 验收基准（论文 Table I/II/V，数字来自 /tmp/ddatr_paper.txt）
数据集	关键指标
MIMIC-CXR	Prec 0.438 / Recall 0.465 / F1 0.441；BLEU-1 0.401、BLEU-4 0.113、METEOR 0.162、ROUGE-L 0.272
Longitudinal-MIMIC	F1 0.527；BLEU-1 0.396、BLEU-4 0.105
IU-Xray	F1 0.262；BLEU-1 0.421、BLEU-4 0.103
ReXrank	BLEU-2 0.210、F1-RadGraph 0.198、BertScore 0.362、SembScore 0.465、GREEN 0.293

7. 踩过的坑 —— 绝对不要再踩
trainer.py 的 return_gates=False 不能加回去：models/blip.py 的 generate() 没有这个参数，训练到验证阶段直接 TypeError。已修复在 modules/trainer.py 第 163 行。
conda create --clone 必须以 which python pip 验证：克隆失败会静默留下半成品（bin 里没有 python/pip，conda-meta 不完整）。克隆期间不要中断、不要开第二个终端动同一个环境。
半成品环境不能用 conda remove / conda install 修复（报 DirectoryNotACondaEnvironmentError），只能 rm -rf /home/sjx/miniconda3/envs/ddatr 后重建。
base 环境没有 gdown、pip 也可能不在 PATH：下载一律 conda run -n stream gdown ...，或装好 ddatr 后用 ddatr 的 gdown。
不要用 stream 的包版本直接跑 DDaTR：transformers 必须 4.25.0、numpy 1.24.3（transformers 4.41.2 / numpy 1.26.4 不匹配 requirements）。
所有命令必须从 /home/sjx/work/project/DDaTR 根目录执行：脚本硬编码 ./checkpoints/stanford/chexbert/chexbert.pth、./data/...、configs/bert_config.json。
训练产物在 checkpoints/ddatr_train/，测试只读 checkpoints/ddatr/model_best.pth：训练完必须 cp 过去，否则测试加载不到权重。
MIMIC 图片路径脚本硬编码 mimic-cxr-jpg/mimic-cxr-jpg_2.0.0/files/，而源目录结构是 images/：必须做嵌套软链，别把图片直接铺到 data/mimic_cxr/ 下。
整目录软链时，DDaTR 自己的 json 要先 mv 进源目录再删旧目录：否则 rm -rf data/mimic_cxr 会把已下载的 longitudinal_mimic_ddatr.json（186MB）和 base_probs.json 一起删掉。
网络：GitHub / Google Drive / Box 直连超时（443），必须开代理；HF 用 hf-mirror.com 镜像；pip 用清华源。gdown 报 "Access denied" 或卡住就是网络问题。
CMCRL-main 是别的项目的数据：只读引用，别删别改它的原始文件；整目录软链后 CMCRL 目录里会多出 DDaTR 的文件（两个 json、mimic-cxr-jpg 嵌套链接、以后下载的标注），清理时别误删。
沙箱环境：bwrap: loopback: Failed RTM_NEWADDR 是沙箱间歇性故障，重试同一命令即可；网络受限命令需要用户批准/代理。
8. 常用命令速查
conda env list                                  # 环境列表（a3net/camanet/mrg/mvketr/stream/test/ddatr）
conda run -n stream gdown <GD_ID> -O <输出路径>
du -sh /home/sjx/miniconda3/envs/ddatr          # 5.1G=克隆失败；应与 stream 的 8.1G 接近
tail -f /home/sjx/work/project/DDaTR/ddatr.out  # 训练日志