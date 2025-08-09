from torch.utils.data import Dataset
import json

class PairwisePreferenceDataset(Dataset):
    # JSONL with keys: prompt, chosen, rejected
    def __init__(self, path, tokenizer, max_prompt=512, max_target=512):
        self.tok = tokenizer
        self.rows = [json.loads(l) for l in open(path)]
        self.max_prompt = max_prompt
        self.max_target = max_target

    def _encode(self, prompt, response):
        # teacher-forcing format: [prompt] -> [response]
        prompt_ids  = self.tok(prompt, truncation=True, max_length=self.max_prompt, add_special_tokens=False)
        resp_ids    = self.tok(response, truncation=True, max_length=self.max_target, add_special_tokens=False)
        # input_ids = [ ...prompt..., ...response... ]
        input_ids = prompt_ids["input_ids"] + resp_ids["input_ids"] + [self.tok.eos_token_id]
        # labels: ignore prompt tokens; train only on response tokens
        labels = [-100]*len(prompt_ids["input_ids"]) + resp_ids["input_ids"] + [self.tok.eos_token_id]
        return input_ids, labels

    def __getitem__(self, i):
        r = self.rows[i]
        pc_ids, pc_lbl = self._encode(r["prompt"], r["chosen"])
        pr_ids, pr_lbl = self._encode(r["prompt"], r["rejected"])
        return {
            "chosen_input_ids": pc_ids,
            "chosen_labels": pc_lbl,
            "rejected_input_ids": pr_ids,
            "rejected_labels": pr_lbl,
        }

    def __len__(self):
        return len(self.rows)

def megatron_collate(batch, pad_id):
    # Pad to max length per side
    import torch
    def pad_side(key):
        mx = max(len(x[key]) for x in batch)
        out = []
        for x in batch:
            seq = x[key]
            if "labels" in key:
                pad_val = -100
            else:
                pad_val = pad_id
            out.append(seq + [pad_val]*(mx-len(seq)))
        return torch.tensor(out, dtype=torch.long)
    return {
        "chosen_input_ids":   pad_side("chosen_input_ids"),
        "chosen_labels":      pad_side("chosen_labels"),
        "rejected_input_ids": pad_side("rejected_input_ids"),
        "rejected_labels":    pad_side("rejected_labels"),
    }
