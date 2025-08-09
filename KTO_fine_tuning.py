# train_qwen3_32b_kto_qlora.py
import os, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import KTOTrainer

MODEL      = os.getenv("MODEL", "Qwen/Qwen3-32B-Instruct")
TRAIN_CSV  = os.getenv("TRAIN_CSV", "train.csv")
EVAL_CSV   = os.getenv("EVAL_CSV",  "")
MAX_PROMPT = 8192
MAX_TARGET = 64

LR       = float(os.getenv("LR", "2e-4"))
EPOCHS   = int(os.getenv("EPOCHS", "2"))
BSZ      = 1
GRAD_ACC = int(os.getenv("GRAD_ACC", "8"))
WARMUP   = int(os.getenv("WARMUP_STEPS", "200"))
LOG_EVERY= int(os.getenv("LOG_EVERY", "25"))
KTO_GAIN = float(os.getenv("KTO_GAIN", "1.5"))
KTO_LOSS = float(os.getenv("KTO_LOSS", "1.0"))

# Tokenizer (Qwen often needs trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def build_prompt_text(system_prompt: str, user_question: str) -> str:
    msgs = []
    if system_prompt and system_prompt.strip():
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_question})
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return (f"<<SYS>>\n{system_prompt}\n<</SYS>>\n" if system_prompt else "") + f"<|user|>\n{user_question}\n<|assistant|>\n"

def encode_row(ex):
    prompt_text = build_prompt_text(ex["prompt"], ex["question"])
    p = tok(prompt_text, add_special_tokens=False, truncation=True, max_length=MAX_PROMPT)
    r = tok(ex["response"] + tok.eos_token, add_special_tokens=False, truncation=True, max_length=MAX_TARGET)
    input_ids = p["input_ids"] + r["input_ids"]
    labels    = [-100]*len(p["input_ids"]) + r["input_ids"]
    return {"input_ids": input_ids, "labels": labels}

def collate(batch):
    pad = tok.pad_token_id
    maxlen = max(len(b["input_ids"]) for b in batch)
    def pad_to(x, val): return x + [val]*(maxlen - len(x))
    input_ids = torch.tensor([pad_to(b["input_ids"], pad) for b in batch], dtype=torch.long)
    labels    = torch.tensor([pad_to(b["labels"],   -100) for b in batch], dtype=torch.long)
    attn      = (input_ids != pad).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
    attn_implementation=os.getenv("ATTN_IMPL", "flash_attention_2"),  # set ATTN_IMPL=sdpa if FA2 not installed
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

peft_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)

files = {"train": TRAIN_CSV}
if EVAL_CSV and os.path.exists(EVAL_CSV):
    files["eval"] = EVAL_CSV
ds = load_dataset("csv", data_files=files)
for split, d in ds.items():
    missing = {"prompt","question","response"} - set(d.column_names)
    if missing:
        raise ValueError(f"{split} CSV missing: {missing}")

ds = ds.map(encode_row, remove_columns=ds["train"].column_names, num_proc=1)

args = TrainingArguments(
    output_dir="out-kto-qwen3-32b",
    per_device_train_batch_size=BSZ,
    per_device_eval_batch_size=BSZ,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP,
    logging_steps=LOG_EVERY,
    evaluation_strategy="steps" if "eval" in ds else "no",
    eval_steps=500,
    save_steps=1000,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,          # <-- important with custom collate
    ddp_find_unused_parameters=False,     # <-- PEFT/DDP stability
    optim="paged_adamw_8bit",             # <-- QLoRA-friendly optimizer
)

trainer = KTOTrainer(
    model=model,
    args=args,
    beta=1.0,
    train_dataset=ds["train"],
    eval_dataset=ds.get("eval"),
    tokenizer=tok,
    max_prompt_length=MAX_PROMPT,
    max_target_length=MAX_TARGET,
    max_length=MAX_PROMPT + MAX_TARGET,
    data_collator=collate,
    kto_loss_kwargs={"gain_weight": KTO_GAIN, "loss_weight": KTO_LOSS},
)

trainer.train()
trainer.save_model("out-kto-qwen3-32b/adapters")
tok.save_pretrained("out-kto-qwen3-32b/adapters")
print("Done.")
