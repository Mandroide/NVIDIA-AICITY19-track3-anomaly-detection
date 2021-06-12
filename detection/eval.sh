#!/bin/bash
# Usage:
# ./eval.sh TASKNAME PARTITION WEIGHTS IMAGE_DIR TEST_IMAGE_LIST
#
# Example:
# ./eval.sh test Test model.caffemodel "/mnt/lustre/weijiayi/AIC19/aic19-track3/" \
# "/mnt/lustre/weijiayi/AIC19/aic19-track3/test_bg.txt"

set -x
set -e

# Slurm Parameters
readonly TASKNAME="$1"
readonly PARTITION="$2"

# Caffe Parameters
readonly WEIGHTS="$3"
readonly IMAGE_DIR="$4"
readonly TEST_IMAGE_LIST="$5"
readonly MODEL="./test.prototxt"
readonly CONFIG="/config.json"
readonly OUT_DIR="./vis_res"
TIMESTAMP="$(date +%Y-%m-%d-%H-%M-%S)"

# Temp environment variables
MV2_USE_CUDA=1
MV2_ENABLE_AFFINITY=0
MV2_SMP_USE_CMA=0

srun --partition="${PARTITION}" --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name="${TASKNAME}" \
  caffe test \
  --model="${MODEL}" --weights="${WEIGHTS}" \
  --config="${CONFIG}" \
  --test_image_list="${TEST_IMAGE_LIST}" \
  --image_dir="${IMAGE_DIR}" \
  --vis_img=1 --out_dir="${OUT_DIR}" 2>&1 | tee ./log/log_"${TASKNAME}"
