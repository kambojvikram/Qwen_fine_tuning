# Qwen3-32B Multi-Node Fine-Tuning Guide

## Overview
This setup enables full parameter fine-tuning of Qwen3-32B across 4 nodes, each with 4x A100 80GB GPUs. Each node trains independently on a portion of your dataset.

## System Requirements
- **Per Node**: 4x NVIDIA A100 80GB GPUs
- **Memory**: 500GB+ RAM recommended
- **Storage**: 200GB+ for model and checkpoints
- **CUDA**: 11.8+ (12.1 recommended)
- **CUDA Toolkit**: Required for Flash Attention compilation
- **Python**: 3.8+

## Key Technologies

### Flash Attention 2
Flash Attention 2 is enabled for optimal performance:
- **2-4x faster** training compared to standard attention
- **50% memory reduction** for attention computation
- **Longer sequences** possible with same GPU memory
- **Better scaling** with sequence length

Performance improvements with Flash Attention 2:
- Training speed: ~30-40% faster overall
- Memory efficiency: Enables larger batch sizes
- Supports up to 128K context (with proper configuration)

## Setup Instructions

### 1. Environment Setup
Run on each node:
```bash
# Clone the repository and navigate to it
cd /path/to/training

# Setup Python environment
bash setup_environment.sh

# Activate environment
source qwen3_venv/bin/activate

# Verify Flash Attention installation
python verify_flash_attention.py
```

### Flash Attention Installation
If Flash Attention 2 is not installed:
```bash
# Ensure CUDA toolkit is installed
nvcc --version  # Should show CUDA version

# Install Flash Attention (takes 10-30 minutes)
pip install flash-attn --no-build-isolation

# Alternative: Install specific version
pip install flash-attn==2.5.8 --no-build-isolation

# Verify installation
python -c "from flash_attn import flash_attn_func; print('Flash Attention 2 ready!')"
```

### 2. Data Preparation
On the master node, prepare and split your dataset:
```bash
# Quick benchmark to verify setup
python quick_benchmark.py

# Your dataset should have columns: prompt, question, response
python prepare_data.py \
    --input your_dataset.csv \
    --num_nodes 4 \
    --output_dir ./data
```

This creates:
- `data/training_data_node_0.csv`
- `data/training_data_node_1.csv`
- `data/training_data_node_2.csv`
- `data/training_data_node_3.csv`

### 3. Configuration
The setup includes several configuration options:

#### DeepSpeed Configuration (auto-generated)
- **ZeRO-3**: Full model sharding across GPUs
- **CPU Offloading**: For optimizer and parameters
- **Mixed Precision**: BF16 for optimal A100 performance

#### Training Parameters
Key parameters to adjust:
- `per_device_train_batch_size`: 1 (increase if memory allows)
- `gradient_accumulation_steps`: 8 (effective batch = 32)
- `learning_rate`: 5e-5
- `lr_scheduler_type`: cosine (with warmup)
- `warmup_ratio`: 0.1 (10% of total steps)
- `max_seq_length`: 8192 tokens

## Running Training

### Learning Rate Schedule
The training uses a **cosine scheduler** with warmup:
- Initial LR: 0 → 5e-5 (warmup phase, 10% of steps)
- Main training: 5e-5 → ~1e-7 (cosine decay)
- Benefits: Smoother convergence, better final performance
- Prevents sudden drops in learning rate that can destabilize training

### Option 1: Individual Node Launch
SSH into each node and run:
```bash
# On Node 0
bash launch_training.sh 0

# On Node 1
bash launch_training.sh 1

# On Node 2
bash launch_training.sh 2

# On Node 3
bash launch_training.sh 3
```

### Option 2: SLURM Cluster
If using SLURM:
```bash
sbatch submit_training.slurm
```

### Option 3: Parallel Launch from Master
From a master node with SSH access:
```bash
bash launch_all_nodes.sh
```

### Option 4: Direct Python Execution
For testing or custom configurations:
```bash
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 4 \
    fine_tune_qwen3.py \
    --dataset_path ./data/training_data_node_0.csv \
    --output_dir ./outputs/node_0_model
```

## Memory Optimization

### Estimated Memory Usage
- **Model Weights**: ~65GB (BF16)
- **Optimizer States**: ~130GB (AdamW)
- **Gradients**: ~65GB
- **Activations**: Variable (reduced with gradient checkpointing)

### If OOM Occurs
1. **Reduce batch size**: Set `per_device_train_batch_size=1`
2. **Increase gradient accumulation**: Set to 16 or 32
3. **Reduce sequence length**: Try 4096 or 2048
4. **Enable more aggressive CPU offloading**
5. **Use gradient checkpointing** (already enabled)

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./outputs --port 6006
```

### GPU Monitoring
```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Or use the provided monitoring script
python monitoring.py
```

### Check Training Logs
```bash
# View logs for specific node
tail -f logs/node_0.log

# Check for errors
grep ERROR logs/*.log
```

## Advanced Configuration

### Custom Training Arguments
Create a JSON config file:
```json
{
  "model_args": {
    "model_name_or_path": "Qwen/Qwen3-32B"
  },
  "data_args": {
    "dataset_path": "./data/training_data.csv",
    "max_seq_length": 8192
  },
  "training_args": {
    "num_train_epochs": 5,
    "learning_rate": 5e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1
  }
}
```

Run with:
```bash
python fine_tune_qwen3.py config.json
```

### Adjusting for Different GPU Configurations
For different GPU counts per node, modify:
1. `CUDA_VISIBLE_DEVICES` in launch script
2. `--num_processes` in accelerate launch
3. `per_device_train_batch_size` based on available memory

## Post-Training

### Merge Checkpoints
If training was interrupted:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the best checkpoint
model = AutoModelForCausalLM.from_pretrained(
    "./outputs/node_0_model/checkpoint-best",
    torch_dtype="auto",
    device_map="auto"
)

# Save final model
model.save_pretrained("./final_model")
```

### Evaluate Models
Each node produces an independent model. To evaluate:
```python
from transformers import pipeline

# Load model from specific node
pipe = pipeline(
    "text-generation",
    model="./outputs/node_0_model",
    device_map="auto"
)

# Test generation
result = pipe("Your test prompt here")
```

## Troubleshooting

### Common Issues

1. **CUDA OOM**
   - Reduce batch size or sequence length
   - Check for memory leaks with `torch.cuda.empty_cache()`

2. **Flash Attention Issues**
   - Verify CUDA toolkit: `nvcc --version`
   - Check compatibility: Flash Attn requires CUDA 11.6+
   - If compilation fails: Use pre-built wheels or fallback to SDPA
   - Memory errors: Ensure sequence length is multiple of 8

3. **Slow Training**
   - Verify Flash Attention 2 is active: Check logs for "Flash Attention 2 is available"
   - If not available, training uses SDPA (still optimized but slower)
   - Check network bandwidth between GPUs
   - Verify CPU isn't bottlenecking data loading

4. **Connection Errors**
   - Check firewall rules for ports 29500-29503
   - Ensure all nodes can communicate

5. **Model Loading Errors**
   - Verify transformers>=4.51.0
   - Check internet connection for model download

## Performance Tips

1. **Flash Attention Optimization**
   - Ensure sequence lengths are multiples of 8 for best performance
   - Use BF16 dtype (better than FP16 for stability)
   - Keep batch dimensions consistent
   - Monitor with `python verify_flash_attention.py`

2. **Data Loading**
   - Pre-tokenize dataset if possible
   - Use fast tokenizers
   - Adjust `dataloader_num_workers`

3. **Training Speed**
   - Flash Attention 2 provides 30-40% speedup
   - Use TF32 for A100s (enabled by default)
   - Optimize gradient accumulation steps
   - Consider sequence packing for shorter sequences

4. **Model Quality**
   - Don't use greedy decoding in thinking mode
   - Monitor validation loss
   - Save checkpoints frequently

## Expected Training Time

### With Flash Attention 2 Enabled:
- **Per Epoch**: ~6-8 hours (depends on dataset size)
- **Total (3 epochs)**: ~18-24 hours
- **Throughput**: ~1.5-2.5 samples/second per node
- **Speedup**: 30-40% faster than standard attention

### Without Flash Attention 2 (SDPA fallback):
- **Per Epoch**: ~8-12 hours
- **Total (3 epochs)**: ~24-36 hours
- **Throughput**: ~1-2 samples/second per node

## Support
For issues or questions:
1. Check logs in `./logs/`
2. Monitor GPU usage with `nvidia-smi`
3. Review DeepSpeed documentation
4. Check Qwen3 model card on HuggingFace
