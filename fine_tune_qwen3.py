#!/usr/bin/env python
# fine_tune_qwen3.py
"""
Full parameter fine-tuning script for Qwen3-32B using TRL
Designed for single-node multi-GPU training (4x A100 80GB)
"""

import os
import sys
import json
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from accelerate import Accelerator
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    model_name_or_path: str = field(
        default="Qwen/Qwen3-32B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Enable trust_remote_code for loading model"}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Torch dtype for model weights (float16, bfloat16, float32)"}
    )
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation to use (flash_attention_2, sdpa, eager)"}
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Use Flash Attention 2 if available"}
    )

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    dataset_path: str = field(
        metadata={"help": "Path to the training dataset (CSV, JSON, or HF dataset)"}
    )
    validation_split: float = field(
        default=0.05,
        metadata={"help": "Proportion of dataset to use for validation"}
    )
    max_seq_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length for training"}
    )
    preprocessing_num_workers: int = field(
        default=8,
        metadata={"help": "Number of processes to use for data preprocessing"}
    )

@dataclass
class TrainingConfig(TrainingArguments):
    """Extended training arguments."""
    output_dir: str = field(
        default="./qwen3-32b-finetuned",
        metadata={"help": "Output directory for model checkpoints"}
    )
    num_train_epochs: float = field(default=3.0)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    gradient_checkpointing: bool = field(default=True)
    
    learning_rate: float = field(default=5e-5)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=0.01)
    
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    evaluation_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DeepSpeed config file"}
    )
    optim: str = field(default="adamw_torch")
    seed: int = field(default=42)
    
    # Additional settings for distributed training
    ddp_timeout: int = field(default=7200)
    ddp_find_unused_parameters: bool = field(default=False)
    
    report_to: List[str] = field(
        default_factory=lambda: ["tensorboard"],
        metadata={"help": "Integration(s) to report to"}
    )

def create_deepspeed_config(output_path: str = "ds_config.json"):
    """Create DeepSpeed configuration for ZeRO-3."""
    config = {
        "bf16": {
            "enabled": "auto"
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "total_num_steps": "auto",
                "warmup_type": "linear",
                "cos_min_lr": 1e-7
            }
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }
    
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return output_path

def load_and_prepare_dataset(data_args: DataArguments):
    """Load dataset and prepare for training."""
    dataset_path = data_args.dataset_path
    
    # Load dataset based on file extension
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
        dataset = Dataset.from_pandas(df)
    elif dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=dataset_path)['train']
    else:
        # Assume it's a HuggingFace dataset
        dataset = load_dataset(dataset_path)['train']
    
    # Split into train and validation
    if data_args.validation_split > 0:
        split_dataset = dataset.train_test_split(
            test_size=data_args.validation_split,
            seed=42
        )
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
    else:
        train_dataset = dataset
        eval_dataset = None
    
    return train_dataset, eval_dataset

def format_chat_template(example: Dict, tokenizer) -> Dict:
    """Format dataset for Qwen3 chat template."""
    messages = []
    
    # Add system prompt if exists
    if example.get('prompt') and example['prompt'].strip():
        messages.append({
            "role": "system",
            "content": example['prompt']
        })
    
    # Add user question
    messages.append({
        "role": "user",
        "content": example['question']
    })
    
    # Add assistant response
    messages.append({
        "role": "assistant",
        "content": example['response']
    })
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False  # Use non-thinking mode for training
    )
    
    return {"text": text}

def setup_model_and_tokenizer(model_args: ModelArguments):
    """Initialize model and tokenizer."""
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    
    # Determine torch dtype
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(model_args.torch_dtype, torch.bfloat16)
    
    # Check if Flash Attention 2 is available
    attn_implementation = model_args.attn_implementation
    if model_args.use_flash_attn:
        try:
            from flash_attn import flash_attn_func
            logger.info("Flash Attention 2 is available and will be used")
            attn_implementation = "flash_attention_2"
        except ImportError:
            logger.warning("Flash Attention 2 not found. Falling back to SDPA")
            attn_implementation = "sdpa"  # Scaled Dot Product Attention
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=attn_implementation,
        device_map=None,  # Let DeepSpeed handle device placement
    )
    
    # Enable gradient checkpointing if specified
    model.config.use_cache = False
    
    logger.info(f"Model loaded with attention: {attn_implementation}")
    logger.info(f"Model dtype: {torch_dtype}")
    
    return model, tokenizer

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingConfig))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from JSON config file
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    
    # Create DeepSpeed config if not provided
    if training_args.deepspeed is None:
        training_args.deepspeed = create_deepspeed_config()
        logger.info(f"Created DeepSpeed config at {training_args.deepspeed}")
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # Load and prepare datasets
    logger.info("Loading and preparing datasets...")
    train_dataset, eval_dataset = load_and_prepare_dataset(data_args)
    
    # Format datasets with chat template
    logger.info("Formatting datasets with chat template...")
    train_dataset = train_dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
        desc="Formatting train dataset"
    )
    
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            lambda x: format_chat_template(x, tokenizer),
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            desc="Formatting eval dataset"
        )
    
    # Setup data collator for completion-only training
    response_template = "<|im_start|>assistant\n"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_seq_length=data_args.max_seq_length,
        dataset_text_field="text",
        packing=False,  # Set to True if you want to pack multiple examples
    )
    
    # Start training
    logger.info("Starting training...")
    if accelerator.is_main_process:
        logger.info(f"Training on {len(train_dataset)} examples")
        if eval_dataset:
            logger.info(f"Evaluating on {len(eval_dataset)} examples")
        logger.info(f"Total optimization steps: {trainer.args.max_steps}")
    
    train_result = trainer.train()
    
    # Save final model
    if accelerator.is_main_process:
        logger.info("Saving final model...")
        trainer.save_model()
        trainer.save_state()
        
        # Save training results
        with open(os.path.join(training_args.output_dir, "train_results.json"), "w") as f:
            json.dump(train_result.metrics, f, indent=2)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
