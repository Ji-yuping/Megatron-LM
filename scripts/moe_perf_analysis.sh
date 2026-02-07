#!/bin/bash

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}

# Paths and Arguments
# CHECKPOINT_PATH=${1:-"checkpoints/mixtral_8x7b"}
# TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/mixtral_8x7b"}
TOKENIZER_ARG=${3:-"data_wpf/llama3_data/llama3_token/"} # Path to tokenizer model, or "MOCK"
DATA_ARG=${4:-"data_wpf/llama3_data/my_llama3_text_document"}     # Data prefix, or "MOCK"

# # Create directories if they don't exist
# mkdir -p "$(dirname "$CHECKPOINT_PATH")"
# mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Distributed training setup
GPUS_PER_NODE=8
NUM_NODES=${SLURM_NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Script Path
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters
# Parallelism Layout: TP * EP * CP * DP = WORLD_SIZE
TP_SIZE=1     
EP_SIZE=8     
CP_SIZE=1     
PP_SIZE=1
MICRO_BATCH_SIZE=4 #2
GLOBAL_BATCH_SIZE=1024  # 256
DTYPE="fp8"  # Changed default to bf16, can switch to fp8 via logic below
# DTYPE="bf16"  # Changed default to bf16, can switch to fp8 via logic below

# Mixtral specific dimensions
NUM_LAYERS=11          # Standard Mixtral 8x7B has 32 layers
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=8128
NUM_ATTN_HEADS=32
KV_CHANNELS=128          # 4096 / 32
SEQ_LENGTH=2048         # Mixtral supports long context
MAX_POSITION_EMBEDDINGS=32768
NUM_EXPERTS=8     #32
ACT_EXPERTS=2      #4

# Data cache path
DATA_CACHE_PATH="${PWD}/benchmark_cache_mixtral_8x7b"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTN_HEADS
    --kv-channels $KV_CHANNELS
    --group-query-attention
    --num-query-groups 8
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000 
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --no-masked-softmax-fusion
    --init-method-std 0.01
    --timing-log-level 2
)

# MoE Specific Arguments
MOE_ARGS=(
    --num-experts $NUM_EXPERTS
    --moe-router-topk $ACT_EXPERTS
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.1  # 0.0
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-fusion
    --moe-router-force-load-balancing
    
    # --moe-token-dispatcher-type alltoall # 'alltoall' is generally faster for Mcore MoE
    --moe-token-dispatcher-type flex
    --moe-enable-deepep

    --moe-router-dtype fp32
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters 200  #200  65  31         # Or use --train-samples
    --lr-decay-iters 80000
    --lr-warmup-iters 100
    --lr 1.0e-4
    --min-lr 1.0e-5
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion

    --cross-entropy-fusion-impl te

    #--calculate-per-token-loss
    --manual-gc
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    #--use-flash-attn
)

# Conditional arguments for FP8 (If needed in future)
if [[ "$DTYPE" == "fp8" ]]; then
    TRAINING_ARGS+=(
        # "--fp8-recipe delayed"
        # "--fp8-format hybrid"
        "--fp8-recipe blockwise"
        "--fp8-format e4m3"

        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
	    "--fp8-param-gather"
        # "--moe-router-padding-for-fp8"
    )
fi

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --expert-model-parallel-size $EP_SIZE
    --context-parallel-size $CP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --sequence-parallel
)

# Distributed Optimizer arguments
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")

# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"
        "--vocab-size 128256" 
        # "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
    )
else
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--data-path $DATA_ARG"
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"
        # "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 10
    --eval-interval 1000
    # --save-interval 2000
    --log-throughput
    # --ckpt-format torch_dist 
    # --save "$CHECKPOINT_PATH"
    # --load "$CHECKPOINT_PATH" 
    # --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    # --no-load-optim 
    # --no-load-rng
    --log-flops
    --log-peak-mem
    --log-metrics-start-iter 159  #39 24
    --log-metrics-end-iter 190  #59 30
)

# Optional: WandB Support
if [ -n "${WANDB_API_KEY}" ]; then
    EVAL_AND_LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral-Mcore"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B_Train"}
    )
fi

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    exit 1
fi

# Run the training command
torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
