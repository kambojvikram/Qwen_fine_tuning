import os, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import ORPOTrainer, KTOTrainer

# ---------- Config (override with env) ----------
MODEL       = os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct")
TRAIN_CSV   = os.getenv("TRAIN_CSV", "train.csv")   # columns: prompt,question,response
EVAL_CSV    = os.getenv("EVAL_CSV",  "")           # optional
LOSS_MODE   = os.getenv("LOSS_MODE", "orpo")       # "orpo" or "kto"

MAX_PROMPT  = int(os.getenv("MAX_PROMPT_LEN", "8192"))
MAX_TARGET  = int(os.getenv("MAX_TARGET_LEN", "64"))
LR          = float(os.getenv("LR", "2e-4"))
EPOCHS      = int(os.getenv("EPOCHS", "2"))
BSZ         = int(os.getenv("PER_DEVICE_BATCH", "1"))
GRAD_ACC    = int(os.getenv("GRAD_ACC", "8"))
WARMUP      = int(os.getenv("WARMUP_STEPS", "200"))
LOG_STEPS   = int(os.getenv("LOG_EVERY", "25"))
BETA_ORPO   = float(os.getenv("ORPO_BETA", "0.2"))     # used only if LOSS_MODE=orpo
KTO_GAIN    = float(os.getenv("KTO_GAIN", "1.0"))      # used only if LOSS_MODE=kto
KTO_LOSS    = float(os.getenv("KTO_LOSS", "1.0"))      # used only if LOSS_MODE=kto

# ---------- Tokenizer ----------
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def build_prompt_text(system_prompt: str, user_question: str) -> str:
    """Use the model's chat template if available."""
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_question})
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback manual format (works but prefer template)
    sys = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n" if system_prompt else ""
    return sys + f"<|user|>\n{user_question}\n<|assistant|>\n"

def encode_row(example):
    prompt_text = build_prompt_text(example["prompt"], example["question"])
    # tokenize prompt and response separately to control masking & lengths
    p = tok(prompt_text, add_special_tokens=False, truncation=True, max_length=MAX_PROMPT)
    r = tok(example["response"] + tok.eos_token, add_special_tokens=False, truncation=True, max_length=MAX_TARGET)
    input_ids = p["input_ids"] + r["input_ids"]
    labels = [-100] * len(p["input_ids"]) + r["input_ids"]  # loss on response only
    return {"input_ids": input_ids, "labels": labels}

def collate(batch):
    pad_id = tok.pad_token_id
    max_len = max(len(x["input_ids"]) for x in batch)
    def pad(seq, pad_with):
        return seq + [pad_with] * (max_len - len(seq))
    input_ids = torch.tensor([pad(x["input_ids"], pad_id) for x in batch], dtype=torch.long)
    labels    = torch.tensor([pad(x["labels"],   -100)  for x in batch], dtype=torch.long)
    attn_mask = (input_ids != pad_id).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

# ---------- QLoRA (4-bit) ----------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_cfg,
    torch_dtype=torch.bfloat16,
    attn_implementation=os.getenv("ATTN_IMPL", "flash_attention_2"),  # fallback "sdpa"
    device_map="auto",
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# Attention-only adapters first; add MLP later if needed
peft_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)

# ---------- Load CSVs ----------
files = {"train": TRAIN_CSV}
if EVAL_CSV and os.path.exists(EVAL_CSV):
    files["eval"] = EVAL_CSV

ds = load_dataset("csv", data_files=files)
# ensure required columns exist
for split, d in ds.items():
    missing = {"prompt","question","response"} - set(d.column_names)
    if missing:
        raise ValueError(f"{split} CSV is missing columns: {missing}")

# map -> tokenized fields; keep memory low by not caching huge columns
ds = ds.map(encode_row, remove_columns=ds["train"].column_names, num_proc=1)

# ---------- Training args ----------
args = TrainingArguments(
    output_dir=f"out-{LOSS_MODE}-qlora",
    per_device_train_batch_size=BSZ,
    per_device_eval_batch_size=BSZ,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP,
    logging_steps=LOG_STEPS,
    evaluation_strategy="steps" if "eval" in ds else "no",
    eval_steps=500,
    save_steps=1000,
    save_total_limit=2,
    report_to="none",
)

# ---------- Pick trainer ----------
if LOSS_MODE.lower() == "orpo":
    trainer = ORPOTrainer(
        model=model,
        args=args,
        beta=BETA_ORPO,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        tokenizer=tok,
        max_prompt_length=MAX_PROMPT,
        max_target_length=MAX_TARGET,
        max_length=MAX_PROMPT + MAX_TARGET,
        data_collator=collate,
    )
elif LOSS_MODE.lower() == "kto":
    trainer = KTOTrainer(
        model=model,
        args=args,
        beta=1.0,  # KTO uses gain/loss weights below; beta not like ORPO's KL
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        tokenizer=tok,
        max_prompt_length=MAX_PROMPT,
        max_target_length=MAX_TARGET,
        max_length=MAX_PROMPT + MAX_TARGET,
        data_collator=collate,
        kto_loss_kwargs={"gain_weight": float(os.getenv("KTO_GAIN", KTO_GAIN)),
                         "loss_weight": float(os.getenv("KTO_LOSS", KTO_LOSS))},
    )
else:
    raise ValueError("LOSS_MODE must be 'orpo' or 'kto'")

trainer.train()
trainer.save_model(f"out-{LOSS_MODE}-qlora/adapters")
tok.save_pretrained(f"out-{LOSS_MODE}-qlora/adapters")
print("Done.")
