#!/bin/bash
# run_all_tests.sh — run MIMIC-CXR / IU-Xray / ReXrank tests with a timestamped result dir.
#
# Usage:
#   bash run_all_tests.sh [备注]
#   nohup bash run_all_tests.sh [备注] > run_tests.nohup.log 2>&1 &   # 后台、断线不中断
#
# - Result dir: results/test/<YYYYmmdd_HHMMSS>[_备注]/
# - WEIGHT below selects which checkpoint is tested (default: the locally
#   trained weights in results/mimic_cxr_train; edit to test another one,
#   e.g. ./checkpoints/ddatr/model_best.pth for the official weights).
# - This script NEVER writes into checkpoints/ — official weights stay
#   untouched. No weight snapshot is made; the tested weight path+mtime is
#   printed to the log for traceability.
# - Logs + an extracted metric summary are saved per test; a failing test
#   does not stop the others.
set -u
cd "$(dirname "$0")" || exit 1

# Which weights to test — edit here if you want another checkpoint.
WEIGHT=./results/mimic_cxr_train/model_best.pth

NOTE="${1:-}"
TS=$(date +%Y%m%d_%H%M%S)
EXP_ROOT=results/test
if [ -n "$NOTE" ]; then
    EXP_DIR="$EXP_ROOT/${TS}_${NOTE}"
else
    EXP_DIR="$EXP_ROOT/${TS}"
fi
mkdir -p "$EXP_DIR" || { echo "cannot create $EXP_DIR"; exit 1; }

echo "==> experiment dir: $EXP_DIR"

# ---- which weights are under test (info only, no snapshot) ----
if [ ! -f "$WEIGHT" ]; then
    echo "ERROR: $WEIGHT not found"; exit 1
fi
echo "==> weight tested : $WEIGHT ($(stat -c %y "$WEIGHT" | cut -d. -f1))"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SUMMARY="$EXP_DIR/summary.txt"
true > "$SUMMARY"

run_test() {
    local name="$1"; shift
    local logf="$EXP_DIR/$name.log"
    echo ""
    echo "==> [$(date '+%H:%M:%S')] running $name ... log: $logf"
    "$@" > "$logf" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "==> $name done (rc=0)"
    else
        echo "==> $name FAILED (rc=$rc)"
        echo "FAILED $name (rc=$rc)" >> "$SUMMARY"
    fi
    # extract metric lines into summary
    {
        echo "===== $name ====="
        grep -a 'test_' "$logf" || echo "(no metric lines)"
        echo ""
    } >> "$SUMMARY"
}

# ---- test 1: MIMIC-CXR (test 3,858) ----
run_test test_mimic \
    env CUDA_VISIBLE_DEVICES=0, \
    python main_test.py \
    --n_gpu 1 \
    --image_dir ./data/mimic_cxr/mimic-cxr-jpg/mimic-cxr-jpg_2.0.0/files/ \
    --ann_path ./data/mimic_cxr/mimic_cxr_ddatr.json \
    --dataset_name mimic_cxr \
    --gen_max_len 150 \
    --gen_min_len 100 \
    --batch_size 16 \
    --save_dir ./results/experiments \
    --seed 456789 \
    --beam_size 3 \
    --load_pretrained "$WEIGHT"

# ---- test 2: IU-Xray (zero-shot, 1,906) ----
run_test test_iu \
    env CUDA_VISIBLE_DEVICES=0, \
    python main_test.py \
    --n_gpu 1 \
    --image_dir ./data/iu_xray/images/ \
    --ann_path ./data/iu_xray/iu_annotation_promptmrg.json \
    --dataset_name iu_xray \
    --gen_max_len 110 \
    --gen_min_len 60 \
    --batch_size 16 \
    --save_dir ./results/experiments \
    --seed 456789 \
    --beam_size 3 \
    --load_pretrained "$WEIGHT"

# ---- test 3: ReXrank (longitudinal, 2,058; official ann_path) ----
run_test test_rexrank \
    env CUDA_VISIBLE_DEVICES=0, \
    python main_test.py \
    --n_gpu 1 \
    --image_dir ./data/mimic_cxr/mimic-cxr-jpg/mimic-cxr-jpg_2.0.0/files/ \
    --ann_path ./data/mimic_cxr/longitudinal_mimic_ddatr.json \
    --dataset_name mimic_cxr \
    --gen_max_len 150 \
    --gen_min_len 100 \
    --batch_size 16 \
    --save_dir ./results/experiments \
    --seed 456789 \
    --beam_size 3 \
    --load_pretrained "$WEIGHT"

echo ""
echo "==> ALL DONE in $EXP_DIR"
echo "==> summary: $SUMMARY"