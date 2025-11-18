#!/bin/bash
# ===== 基础配置 =====
cd /inspire/ssd/project/qproject-fundationmodel/public/yiwei/megatron-workspace/source/Megatron-LM
source /inspire/ssd/project/qproject-fundationmodel/public/yiwei/megatron-workspace/env/megatron/bin/activate

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_CONFIG_FILE="$CURRENT_DIR/dataset_config.json"

echo "📁 配置文件路径: $DATASET_CONFIG_FILE"
if [ -f "$DATASET_CONFIG_FILE" ]; then
    echo "✅ 配置文件存在"
else
    echo "❌ 配置文件不存在: $DATASET_CONFIG_FILE"
    exit 1
fi

MODEL_NAME="Qwen3-30B-A3B"
DATASET_NAME="pt_11_18_6TB"
STAGE="stage_1_1_1"
EXPERIMENT_NAME="${MODEL_NAME}_${DATASET_NAME}_${STAGE}"
OUTPUT_PATH="/inspire/hdd/project/qproject-fundationmodel/public/yiwei/megatron-workspace/experiments/pt_30b_a3b_from_scratch_11_18"
TOKENIZER_MODEL="/inspire/hdd/project/qproject-fundationmodel/public/yiwei/ckpts/Qwen3-30B-A3B-Instruct-2507"

# ===== 训练配置 =====
# 15B tokens 计算: 15B / (GBS * seq_length) = 15B / (1024 * 4096) = ~3,578 steps
TRAIN_ITERS=100
EVAL_INTERVAL=100

SEQ_LENGTH=4096
GBS=1024
MBS=4
MODEL_NAME="qwen25_3b"

# ===== 并行配制 =====
SEQUENCE_PARALLEL="false"
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1

# ===== 准备数据 =====
python scripts/prepare_gptdataset.py \
    --dataset_config_file $DATASET_CONFIG_FILE \
    --output_path $OUTPUT_PATH \
    --experiment_name $EXPERIMENT_NAME \
    --tokenizer_model $TOKENIZER_MODEL \
    --global_batch_size $GBS \
    --mbs $MBS \
    --seq_length $SEQ_LENGTH \
    --train_iters $TRAIN_ITERS \
    --eval_interval $EVAL_INTERVAL \
    --num_dataset_builder_threads 16 \
    --tensor_model_parallel_size $TENSOR_MODEL_PARALLEL_SIZE