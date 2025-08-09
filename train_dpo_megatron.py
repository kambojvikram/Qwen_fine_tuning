import os, math
import torch
import torch.distributed as dist
from megatron.training import get_args, print_rank_0, get_timers
from megatron.core import mpu
from megatron.training import pretrain
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import build_module_spec
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import forward_backward_generic
from megatron.core.models.gpt.gpt_spec import GPTModelConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from transformers import AutoTokenizer
from dpo_dataset import PairwisePreferenceDataset, megatron_collate

# ----- DPO loss -----
# L = - E[ log σ( β(Δπ - Δref) ) ],  Δ = logp(chosen) - logp(rejected)
def dpo_loss(beta, logp_c, logp_r, logp_ref_c, logp_ref_r):
    import torch.nn.functional as F
    return -torch.log(torch.sigmoid(beta * ((logp_c - logp_r) - (logp_ref_c - logp_ref_r)))).mean()

# Sum of token logprobs over response labels (ignoring -100) with TP/sequence-parallel safe gather
def sequence_nll(logits, labels):
    # logits: [B, T, V] sharded over TP; labels: [B, T]
    # Use standard cross-entropy over valid positions and sum per sample
    import torch.nn.functional as F
    B, T, V = logits.shape
    logits = logits.view(B*T, V)
    labels = labels.view(B*T)
    mask = labels.ne(-100)
    if mask.any():
        ll = -F.cross_entropy(logits[mask], labels[mask], reduction='none')
        # sum per sample
        idx = torch.arange(B, device=labels.device).repeat_interleave(T)[mask]
        per_sample = torch.zeros(B, device=labels.device).index_add_(0, idx, ll)
    else:
        per_sample = torch.zeros(B, device=labels.device)
    # Reduce across tensor-parallel shards (sum log-probs split by vocab partition if using vocab parallel)
    dist.all_reduce(per_sample, group=mpu.get_tensor_model_parallel_group())
    return per_sample  # [B]

def get_dataloader(args, tokenizer):
    from torch.utils.data import DataLoader
    train = PairwisePreferenceDataset(args.train_data, tokenizer, args.max_prompt_len, args.max_resp_len)
    collate = lambda b: megatron_collate(b, tokenizer.pad_token_id)
    return DataLoader(train, batch_size=args.micro_batch_size, shuffle=True, drop_last=True, collate_fn=collate)

def build_model(args):
    # Megatron config
    cfg = GPTModelConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        seq_length=args.seq_length,
        max_position_embeddings=args.seq_length,
        vocab_size=args.vocab_size,
        init_method_std=0.02,
        use_gpu_initialization=True,
        fp16=False,
        bf16=args.bf16,
        apply_layernorm_1p=False,
        attention_dropout=args.attn_dropout,
        hidden_dropout=args.hidden_dropout,
        add_position_embedding=True,
        share_embeddings_and_output_weights=True,
    )
    layer_spec = get_gpt_layer_local_spec()
    return GPTModel(cfg, transformer_layer_spec=layer_spec)

def forward_step(data_iter, model, ref_model, args, timers):
    data = next(data_iter)
    for k in data:
        data[k] = data[k].cuda(non_blocking=True)

    # Forward policy
    ch_logits = model(data["chosen_input_ids"])
    rj_logits = model(data["rejected_input_ids"])

    # Forward ref (frozen)
    with torch.no_grad():
        ch_ref_logits = ref_model(data["chosen_input_ids"])
        rj_ref_logits = ref_model(data["rejected_input_ids"])

    # Per-sample NLL sums (policy and ref)
    logp_c = sequence_nll(ch_logits, data["chosen_labels"])
    logp_r = sequence_nll(rj_logits, data["rejected_labels"])
    logp_ref_c = sequence_nll(ch_ref_logits, data["chosen_labels"])
    logp_ref_r = sequence_nll(rj_ref_logits, data["rejected_labels"])

    loss = dpo_loss(args.beta, logp_c, logp_r, logp_ref_c, logp_ref_r)
    return loss, {'loss': loss.detach()}

def train_valid_test_datasets_provider(args):
    # Megatron expects this factory; we only train here
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return None, None, None  # not used; we feed our own iterator in training loop

def model_provider_func(pre_process=True, post_process=True):
    args = get_args()
    model = build_model(args)
    # Build a separate ref model that shares init weights but stays frozen
    ref_model = build_model(args)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    return model, ref_model

def main():
    args = get_args()
    model_parallel_cuda_manual_seed(args.seed)
    model, ref_model = model_provider_func()

    optimizer = get_megatron_optimizer(model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_loader = get_dataloader(args, tokenizer)
    data_iter = iter(train_loader)

    # Simple loop with pipeline=1; if you add PP>1, replace with Megatron's pipeline schedule
    model.train()
    for step in range(args.train_steps):
        loss, _ = forward_step(data_iter, model, ref_model, args, get_timers())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if mpu.get_data_parallel_rank() == 0 and step % args.log_every == 0:
            print_rank_0(f"step {step} | loss {loss.item():.4f}")

    if mpu.get_data_parallel_rank() == 0:
        print_rank_0("Training done.")

if __name__ == "__main__":
    main()
