#!/bin/bash
# launch_training.sh - Launch script for single node training

# ============================================
# SINGLE NODE LAUNCH SCRIPT (run on each node)
# ============================================

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true

# Flash Attention environment variables
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
export MAX_JOBS=4  # Parallel compilation jobs for flash-attn

# Node-specific configuration
NODE_ID=${1:-0}  # Pass node ID as argument (0, 1, 2, or 3)
DATASET_PATH="./data/training_data_node_${NODE_ID}.csv"
OUTPUT_DIR="./outputs/node_${NODE_ID}_model"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Launch training with accelerate
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 4 \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port 29500 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    fine_tune_qwen3.py \
    --model_name_or_path "Qwen/Qwen3-32B" \
    --dataset_path ${DATASET_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_seq_length 8192 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    --gradient_checkpointing \
    --bf16 \
    --tf32 \
    --use_flash_attn \
    --attn_implementation flash_attention_2 \
    --deepspeed ds_config.json \
    --report_to tensorboard

# ============================================
# ALTERNATIVE: Using torchrun directly
# ============================================
# torchrun \
#     --nproc_per_node=4 \
#     --master_port=29500 \
#     fine_tune_qwen3.py \
#     --model_name_or_path "Qwen/Qwen3-32B" \
#     --dataset_path ${DATASET_PATH} \
#     --output_dir ${OUTPUT_DIR} \
#     --deepspeed ds_config.json \
#     --use_flash_attn \
#     --attn_implementation flash_attention_2 \
#     --learning_rate 5e-5 \
#     --lr_scheduler_type cosine \
#     [other arguments...]

# ============================================
# SLURM SCRIPT FOR HPC CLUSTERS
# ============================================
# Save as: submit_training.slurm

: '
#!/bin/bash
#SBATCH --job-name=qwen3-finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --time=48:00:00
#SBATCH --mem=500G
#SBATCH --output=logs/node_%A_%a.out
#SBATCH --error=logs/node_%A_%a.err
#SBATCH --array=0-3

# Load required modules (adjust based on your cluster)
module load cuda/12.1
module load python/3.10
module load gcc/11.3

# Activate virtual environment
source /path/to/venv/bin/activate

# Set node-specific variables
NODE_ID=${SLURM_ARRAY_TASK_ID}
DATASET_PATH="./data/training_data_node_${NODE_ID}.csv"
OUTPUT_DIR="./outputs/node_${NODE_ID}_model"

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE

# Launch training
srun accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 4 \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port $((29500 + NODE_ID)) \
    --mixed_precision bf16 \
    fine_tune_qwen3.py \
    --model_name_or_path "Qwen/Qwen3-32B" \
    --dataset_path ${DATASET_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --deepspeed ds_config.json \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --use_flash_attn \
    --attn_implementation flash_attention_2
'

# ============================================
# PARALLEL LAUNCH SCRIPT (run from master node)
# ============================================
# Save as: launch_all_nodes.sh

: '
#!/bin/bash

# SSH to each node and launch training
for NODE_ID in 0 1 2 3; do
    NODE_HOST="node${NODE_ID}.cluster.local"  # Adjust hostname
    
    echo "Launching training on ${NODE_HOST}..."
    
    ssh ${NODE_HOST} "cd /path/to/training/dir && \
        nohup bash launch_training.sh ${NODE_ID} > logs/node_${NODE_ID}.log 2>&1 &" &
    
    sleep 5  # Small delay between launches
done

echo "All nodes launched. Monitor logs in ./logs/"
'

# ============================================
# ENVIRONMENT SETUP SCRIPT
# ============================================
# Save as: setup_environment.sh

: '
#!/bin/bash

# Create virtual environment
python3 -m venv qwen3_venv
source qwen3_venv/bin/activate

# Upgrade pip and install build tools
pip install --upgrade pip
pip install packaging ninja

# Install PyTorch with CUDA support (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention 2 (requires CUDA toolkit)
# Option 1: Pre-built wheel (faster, if available for your CUDA/torch version)
pip install flash-attn --no-build-isolation

# Option 2: Build from source (if pre-built not available)
# git clone https://github.com/Dao-AILab/flash-attention.git
# cd flash-attention
# pip install . --no-build-isolation
# cd ..

# Install required packages
pip install transformers>=4.51.0
pip install accelerate
pip install deepspeed
pip install trl
pip install datasets
pip install pandas
pip install tensorboard
pip install sentencepiece
pip install protobuf
pip install peft  # Optional, for future LoRA experiments
pip install bitsandbytes  # Optional, for quantization
pip install triton  # Required for some Flash Attention features

# Verify Flash Attention installation
python -c "from flash_attn import flash_attn_func; print(\"Flash Attention 2 installed successfully\")" || echo "Flash Attention 2 installation failed - will fallback to SDPA"

# Create necessary directories
mkdir -p data
mkdir -p outputs
mkdir -p logs
mkdir -p checkpoints

echo "Environment setup complete!"
echo "Flash Attention 2 status:"
python -c "
try:
    from flash_attn import flash_attn_func
    print(\"  ✓ Flash Attention 2 is available\")
except ImportError:
    print(\"  ✗ Flash Attention 2 not available, will use SDPA fallback\")
"
'
