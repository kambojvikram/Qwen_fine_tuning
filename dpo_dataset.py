scale = 1.4  # try 1.1–1.6; validate before committing

from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, lora_path)

model.set_adapter("default")
# Temporarily exaggerate ΔW at runtime…
model.base_model.model.set_adapter_scaling({"default": scale})

# …then bake that scaled ΔW in ONCE.
merged = model.merge_and_unload()
merged.save_pretrained(f"model-merged-{scale:.1f}x")
