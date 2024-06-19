import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import pandas as pd
import torch

with open("D_KV_SUBS.json", "r") as f:
    dataset = json.load(f)

one_row = dataset[232]
print(one_row)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token	

print(tokenizer.encode("My experiments are going strong!"))

# with padding - so it'll not exceed 10 tokens then
# print(tokenizer.encode(outputs[0], padding='max_length', max_length=10))
# print(tokenizer.encode("My experiments are going strong!", padding='max_length', max_length=10))

# pytorch tensors
# print(tokenizer.encode(outputs[0], 
#                  padding='max_length', 
#                  max_length=10,
#                  return_tensors="pt"))
# print(tokenizer.encode("My experiments are going strong!", 
#                  padding='max_length', 
#                  max_length=10,
#                  return_tensors="pt"))

# dividing train and eval datasets
train_dataset = dataset[:-1000]
eval_dataset = dataset[-1000:]

train_table = pd.DataFrame(train_dataset)
eval_table  = pd.DataFrame(eval_dataset)

def pad_eos(ds):
    EOS_TOKEN = "</s>"
    return [f"{row['value']}{EOS_TOKEN}" for row in ds]

train_prompts = [row["key"] for row in train_dataset]
eval_prompts = [row["key"] for row in eval_dataset]

train_outputs = pad_eos(train_dataset)
eval_outputs = pad_eos(eval_dataset)

train_dataset = [{"prompt":s, "output":t, "example": s + t} for s, t in zip(train_prompts, train_outputs)]
eval_dataset = [{"prompt":s, "output":t, "example": s + t} for s, t in zip(eval_prompts, eval_outputs)]

# packing examples with padding
max_seq_len = 1024

# print(tokenizer([s["example"] for s in train_dataset]))

def pack(dataset, max_seq_len=1024):
    tkds_ids = tokenizer([s["example"] for s in dataset])["input_ids"]
    
    all_token_ids = []
    for tokenized_input in tkds_ids:
        all_token_ids.extend(tokenized_input + [tokenizer.eos_token_id])
    
    packed_ds = []
    for i in range(0, len(all_token_ids), max_seq_len+1):
        input_ids = all_token_ids[i : i + max_seq_len+1]
        if len(input_ids) == (max_seq_len+1):
            packed_ds.append({"input_ids": input_ids[:-1], "labels": input_ids[1:]})  # < --- ‼️ ⛔️
	    # if you use the model.output.loss you don't need to shift, it is done for you!
    return packed_ds

train_ds_packed = pack(train_dataset)
eval_ds_packed = pack(eval_dataset)

# length of sequences we get after packing them together
total_sequences = len(train_ds_packed)
print(total_sequences)

# dataloader
from torch.utils.data import DataLoader
from transformers import default_data_collator

torch.manual_seed(seed)
batch_size = 64 # good starter number

train_dataloader = DataLoader(
    train_ds_packed,
    batch_size=batch_size,
    collate_fn=default_data_collator, # we don't need any special collator 😎
)

eval_dataloader = DataLoader(
    eval_ds_packed,
    batch_size=batch_size,
    collate_fn=default_data_collator,
    shuffle=False,
)

# print one batch
b = next(iter(train_dataloader))
print(b)





