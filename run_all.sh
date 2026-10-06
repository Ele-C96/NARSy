#!/bin/bash


export TF_XLA_FLAGS=--tf_xla_auto_jit=0
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit

echo "Starting experiments..."


source .venv/bin/activate



# =========================
# BASELINE
# =========================
python 1_baseline.py --dataset NPHA
python 1_baseline.py --dataset amazon
python 1_baseline.py --dataset adult
python 1_baseline.py --dataset loan
python 1_baseline.py --dataset diabetes
python 1_baseline.py --dataset irrigation
python 1_baseline.py --dataset pollution
python 1_baseline.py --dataset studentdespression
python 1_baseline.py --dataset student_performance
python 1_baseline.py --dataset diabetes012

# =========================
# CARS
# =========================

python 2_CAR.py --dataset NPHA
python 2_CAR.py --dataset amazon
python 2_CAR.py --dataset adult
python 2_CAR.py --dataset loan
python 2_CAR.py --dataset diabetes
python 2_CAR.py --dataset irrigation
python 2_CAR.py --dataset pollution
python 2_CAR.py --dataset studentdespression
python 2_CAR.py --dataset student_performance
python 2_CAR.py --dataset diabetes012


# =========================
# CAR ACCURACY
# =========================

python 3_CAR_accuracy.py --dataset NPHA --accuracy 0.82
python 3_CAR_accuracy.py --dataset amazon --accuracy 0.72
python 3_CAR_accuracy.py --dataset adult --accuracy 0.83
python 3_CAR_accuracy.py --dataset diabetes --accuracy 0.61
python 3_CAR_accuracy.py --dataset diabetes012 --accuracy 0.90
python 3_CAR_accuracy.py --dataset irrigation --accuracy 0.53
python 3_CAR_accuracy.py --dataset loan --accuracy 0.70
python 3_CAR_accuracy.py --dataset pollution --accuracy 0.89
python 3_CAR_accuracy.py --dataset studentdespression --accuracy 0.90
python 3_CAR_accuracy.py --dataset student_performance --accuracy 0.57

# =========================
# CAR TOP-K
# =========================

python 4_CAR_top_k.py --dataset NPHA --topk 1039
python 4_CAR_top_k.py --dataset adult --topk 428
python 4_CAR_top_k.py --dataset amazon --topk 6
python 4_CAR_top_k.py --dataset diabetes --topk 9
python 4_CAR_top_k.py --dataset diabetes012 --topk 1368
python 4_CAR_top_k.py --dataset irrigation --topk 1
python 4_CAR_top_k.py --dataset loan --topk 36
python 4_CAR_top_k.py --dataset pollution --topk 15
python 4_CAR_top_k.py --dataset studentdespression --topk 195
python 4_CAR_top_k.py --dataset student_performance --topk 11

# # =========================
# # SPLIT ACCURACY
# # =========================
python 5_CAR_split_accuracy.py --dataset NPHA
python 5_CAR_split_accuracy.py --dataset adult
python 5_CAR_split_accuracy.py --dataset loan
python 5_CAR_split_accuracy.py --dataset amazon
python 5_CAR_split_accuracy.py --dataset diabetes
python 5_CAR_split_accuracy.py --dataset irrigation
python 5_CAR_split_accuracy.py --dataset pollution
python 5_CAR_split_accuracy.py --dataset studentdespression
python 5_CAR_split_accuracy.py --dataset student_performance
python 5_CAR_split_accuracy.py --dataset diabetes012

# # =========================
# # SPLIT TOPK
# # # =========================
python 6_CAR_split_topk.py --dataset NPHA
python 6_CAR_split_topk.py --dataset adult
python 6_CAR_split_topk.py --dataset loan
python 6_CAR_split_topk.py --dataset amazon
python 6_CAR_split_topk.py --dataset diabetes
python 6_CAR_split_topk.py --dataset irrigation
python 6_CAR_split_topk.py --dataset pollution
python 6_CAR_split_topk.py --dataset studentdespression
python 6_CAR_split_topk.py --dataset student_performance
python 6_CAR_split_topk.py --dataset diabetes012

# # =========================
# # SPLIT ACC + LIFT
# # =========================
python 7_CAR_split_acc_lift.py --dataset NPHA
python 7_CAR_split_acc_lift.py --dataset adult
python 7_CAR_split_acc_lift.py --dataset loan
python 7_CAR_split_acc_lift.py --dataset amazon
python 7_CAR_split_acc_lift.py --dataset diabetes
python 7_CAR_split_acc_lift.py --dataset irrigation
python 7_CAR_split_acc_lift.py --dataset pollution
python 7_CAR_split_acc_lift.py --dataset studentdespression
python 7_CAR_split_acc_lift.py --dataset student_performance
python 7_CAR_split_acc_lift.py --dataset diabetes012

# # =========================
# # SPLIT TOPK + LIFT
# # =========================

python 8_CAR_split_topk_lift.py --dataset NPHA
python 8_CAR_split_topk_lift.py --dataset adult
python 8_CAR_split_topk_lift.py --dataset loan
python 8_CAR_split_topk_lift.py --dataset amazon
python 8_CAR_split_topk_lift.py --dataset diabetes
python 8_CAR_split_topk_lift.py --dataset irrigation
python 8_CAR_split_topk_lift.py --dataset pollution
python 8_CAR_split_topk_lift.py --dataset studentdespression
python 8_CAR_split_topk_lift.py --dataset student_performance
python 8_CAR_split_topk_lift.py --dataset diabetes012


echo "All experiments completed."