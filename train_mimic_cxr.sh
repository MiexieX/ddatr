export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 每次运行独立目录，避免覆盖上一次训练：results/mimic_cxr_train_YYYYmmdd_HHMMSS
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR=./results/mimic_cxr_train_${TIMESTAMP}
echo ">> 本次训练保存目录: ${SAVE_DIR}"
echo ">> 日志: 启动时请重定向到 train_${TIMESTAMP}.txt（setsid 方式启动，勿覆盖旧日志）"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
--nproc_per_node=1 \
--master_port=12361 \
main_train.py \
--image_dir ./data/mimic_cxr/mimic-cxr-jpg/mimic-cxr-jpg_2.0.0/files/ \
--ann_path ./data/mimic_cxr/mimic_cxr_ddatr.json \
--dataset_name mimic_cxr \
--gen_max_len 150 \
--gen_min_len 100 \
--batch_size 16 \
--epochs 10 \
--seed 456789 \
--init_lr 5e-5 \
--min_lr 5e-6 \
--warmup_lr 5e-7 \
--weight_decay 0.05 \
--warmup_steps 2000 \
--cls_weight 4 \
--beam_size 3 \
--save_dir ${SAVE_DIR}