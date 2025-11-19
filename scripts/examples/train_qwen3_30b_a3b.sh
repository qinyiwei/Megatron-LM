#!/bin/bash
source /inspire/ssd/project/qproject-fundationmodel/public/yiwei/megatron-workspace/env/megatron/bin/activate

# ===== 分布式训练配制 =====
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=8
NUM_NODES=${PET_NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${PET_NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

echo "==========================="
echo "NUM_NODES:${NUM_NODES}"
echo "MASTER_ADDR:${MASTER_ADDR}"
echo "MASTER_PORT:${MASTER_PORT}"
echo "NODE_RANK:${NODE_RANK}"
echo "WORLD_SIZE:${WORLD_SIZE}"


# ===== 基础配制 =====
MODEL_NAME="Qwen3-30B-A3B"
DATASET_NAME="pt_11_18_6TB"
STAGE="stage_1_1_1"
EXPERIMENT_NAME="${MODEL_NAME}_${DATASET_NAME}_${STAGE}"
OUTPUT_PATH="/inspire/hdd/project/qproject-fundationmodel/public/yiwei/megatron-workspace/experiments/examples/$EXPERIMENT_NAME"
CHECKPOINT_PATH="$OUTPUT_PATH/checkpoints"
ORIG_CHECKPOINT_PATH=$CHECKPOINT_PATH
TENSORBOARD_LOGS_PATH="$OUTPUT_PATH/tensorboard_logs"
TOKENIZER_MODEL="/inspire/hdd/project/qproject-fundationmodel/public/yiwei/ckpts/Qwen3-30B-A3B-Instruct-2507"
DATA_CACHE_PATH="${OUTPUT_PATH}/dataset_cache"

PRETRAIN_SCRIPT_PATH="/inspire/ssd/project/qproject-fundationmodel/public/yiwei/megatron-workspace/source/Megatron-LM/pretrain_gpt.py"
LOG_DIR="${OUTPUT_PATH}/logs"

# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"
mkdir -p "$DATA_CACHE_PATH"
mkdir -p "$LOG_DIR"

# ===== 数据集配制 =====
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="$CURRENT_DIR/dataset_config.json"

echo "📁 配置文件路径: $DATA_PATH"
if [ -f "$DATA_PATH" ]; then
    echo "✅ 配置文件存在"
else
    echo "❌ 配置文件不存在: $DATA_PATH"
    exit 1
fi

# ===== 并行配制 =====
TP_SIZE=1             
EP_SIZE=4              
PP_SIZE=1              
CP_SIZE=1              
MICRO_BATCH_SIZE=1     
GLOBAL_BATCH_SIZE=1024   
SEQ_LENGTH=4096   

NLAYERS=48
FIRST_K_DENSE_REPLACE=0

arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done

printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"


MODEL_ARGS=(

    --disable-bias-linear
    --qk-layernorm ##
    --group-query-attention
    --num-attention-heads 32
    --num-query-groups 4
    --kv-channels 128 ## default should be 64
    --num-layers 48                   
    --hidden-size 2048                 
    --ffn-hidden-size 6144

    --normalization RMSNorm
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --rotary-percent 1.0
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 151936       

    --max-position-embeddings 262144 ##
    --rotary-base 10000000

    --moe-ffn-hidden-size 768
    --moe-router-score-function softmax ##
    --moe-token-dispatcher-type alltoall
    --moe-router-topk 8
    --moe-layer-freq $MOE_LAYER_FREQ ##
    --num-experts 128  
    --moe-grouped-gemm
    --moe-token-drop-policy probs ##
    --moe-router-dtype fp32
    --moe-permute-fusion
    --moe-aux-loss-coeff 0 ##

    
    # 【初始化】
    # --init-method-std 0.02
)



TRAINING_ARGS=(
    # 【批次配置】
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    
    # 【训练时长】
    --train-iters 100 #TODO

    --seq-length $SEQ_LENGTH

    --attention-dropout 0.0
    --hidden-dropout 0.0
    
    # 【学习率】MoE模型通常用较小的学习率
    --lr 1e-4      #TODO                 
    --min-lr 1e-5 #TODO
    --lr-decay-style cosine #TODO
    --lr-warmup-iters 10 #TODO
    
    # 【优化器】
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
    --weight-decay 0.1
    
    # 【混合精度】
    --bf16
    
    # 【性能优化】
    --cross-entropy-loss-fusion
    --empty-unused-memory-level 0
    --transformer-impl transformer_engine
    --attention-backend flash
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --expert-model-parallel-size $EP_SIZE    # 专家并行！
    --context-parallel-size $CP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    #--sequence-parallel                      # TP>1时必须启用
)

DDP_ARGS=(
    --use-distributed-optimizer              # 分布式优化器：节省显存
    --overlap-grad-reduce                    # 梯度规约重叠
    --overlap-param-gather                   # 参数gather重叠
    --ddp-bucket-size 200000000
)

# Data arguments
DATA_ARGS=(
    # 【数据路径】
    --data-path $DATA_PATH
    
    # 【Tokenizer】
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
    
    # 【数据分割】
    --split 99,1,0
    
    # 【数据加载】
    --data-cache-path ${DATA_CACHE_PATH}
    --num-workers 8
    --dataloader-type cyclic
    --no-mmap-bin-files
    --num-dataset-builder-threads 16
    
    # 【内存优化】
    --no-create-attention-mask-in-dataloader

    # 【文档间attention】
    # --reset-attention-mask     
    # --reset-position-ids   
)

PROFILER_ARGS=(
    --profile
    --use-pytorch-profiler
    --profile-ranks 0              # 指定要profile的rank,默认是[0]
    --profile-step-start 10        # 从第10步开始profile
    --profile-step-end 11          # 到第11步结束profile
)

EVAL_AND_LOGGING_ARGS=(
    # 【日志频率】
    --log-interval 1
    --log-throughput
    --log-timers-to-tensorboard
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    
    # 【验证】
    --eval-iters 32
    --eval-interval 100 #TODO
    
    # 【检查点】
    --save-interval 100 #TODO
    --ckpt-format torch_dist
    --save "$CHECKPOINT_PATH"
    --load "$ORIG_CHECKPOINT_PATH"
    # --finetune

    --no-load-optim # 不加载优化器状态（因为是新训练）
    --no-load-rng   # 不加载随机数状态
    
    # 【超时保护】
    --distributed-timeout-minutes 60
)


RECOMPUTE_ARGS=(
    # --recompute-granularity selective
    # --recompute-method uniform
    # --recompute-num-layers 12            # 重计算部分层以节省显存
)

# Ensure pretrain_gpt.py exists
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    exit 1
fi

echo "=========================================="
echo "Training Qwen3-30B-A3B MoE Model"
echo "=========================================="

export LOG_DIR

# 创建 Python 包装脚本（而不是 bash 脚本）
WRAPPER_SCRIPT="${OUTPUT_PATH}/run_with_log.py"
cat > "$WRAPPER_SCRIPT" << 'EOF'
#!/usr/bin/env python
import sys
import os
import subprocess

node_id = os.environ.get('PET_NODE_RANK', '0')
rank = os.environ.get('RANK', '0')
log_dir = os.environ.get('LOG_DIR', './logs')

# 创建node文件夹
node_log_dir = os.path.join(log_dir, f'node_{node_id}')
os.makedirs(node_log_dir, exist_ok=True)

# 日志文件路径：logs/node_X/rank_Y.log
log_file = os.path.join(node_log_dir, f'rank_{rank}.log')

# 打开日志文件
with open(log_file, 'w') as f:
    # 运行 Python 脚本，重定向输出到日志文件
    subprocess.run(
        ['python'] + sys.argv[1:],
        stdout=f,
        stderr=subprocess.STDOUT
    )
EOF
chmod +x "$WRAPPER_SCRIPT"

python -m torch.distributed.run \
    ${DISTRIBUTED_ARGS[@]} \
    "$WRAPPER_SCRIPT" \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DDP_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]}
    #${PROFILER_ARGS[@]}  \
