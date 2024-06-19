import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    pipeline,
    logging,
)
import pandas as pd
import torch

seed = 42

with open("D_KV_SUBS.json", "r") as f:
    dataset = json.load(f)

one_row = dataset[232]
print(one_row)

model_id = 'meta-llama/Llama-2-7b-hf'
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
train_dataset = dataset[:-4000]
eval_dataset = dataset[-4000:]

print("lengths of datasets: ", len(train_dataset), len(eval_dataset))

train_table = pd.DataFrame(train_dataset)
eval_table  = pd.DataFrame(eval_dataset)

def pad_eos(ds):
    EOS_TOKEN = "</s>"
    return [f"{row['value']}{EOS_TOKEN}" for row in ds]

# adding create_prompt to use as formatting_func argument during training
def prompt_input(row):
    return ("Learn the key value pairings provided in the format of corresponding arithmetic expressions.\n\n"
            "### Key:\n{key}\n\n### Value:\n{value}").format_map(row)

def create_prompt(row):
    return prompt_input(row)

# checking row in dataset
print(train_dataset[0])

train_prompts = [create_prompt(row) for row in train_dataset]
eval_prompts = [create_prompt(row) for row in eval_dataset]

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
batch_size = 8 # good starter number

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

# decoding the batch
print(tokenizer.decode(b["input_ids"][0])[:250])
print(tokenizer.decode(b["labels"][0])[:250])

print(b["input_ids"][0])
print(b["labels"][0])

# adding lora config
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=64,  # the rank of the LoRA matrices
    lora_alpha=16, # the weight
    lora_dropout=0.1, # dropout to add to the LoRA layers
    bias="none", # add bias to the nn.Linear layers?
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
)

# training the model and training arguments
from transformers import TrainingArguments
from trl import SFTTrainer

batch_size = 64
gradient_accumulation_steps = 2
num_train_epochs = 3

total_num_steps = num_train_epochs * 11_210 // (batch_size * gradient_accumulation_steps)

print(total_num_steps)

output_dir = "./output/"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size//2,
    bf16=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio = 0.1,
    max_steps=total_num_steps,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs=dict(use_reentrant=False),
    evaluation_strategy="steps",
    eval_steps=total_num_steps // num_train_epochs,
    # eval_steps=10,
    # logging strategies
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=total_num_steps // num_train_epochs,
)

model_kwargs = dict(
    device_map="auto",
    trust_remote_code=True,
    # low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    # use_flash_attention_2=True,
    use_cache=False,
)

model = AutoModelForCausalLM.from_pretrained(model_id)
model.add_adapter(peft_config, adapter_name="adapter_key_value_pairs")

trainer = Trainer(
    model=model,
    train_dataset=train_dataloader,
    args=training_args,
    # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

# trainer = SFTTrainer(
#     model=model_id,
#     # model_init_kwargs=model_kwargs,
#     train_dataset=train_dataloader,
#     eval_dataset=eval_dataloader,
#     # packing=True,
#     max_seq_length=1024,
#     args=training_args,
#     # formatting_func=create_prompt,
#     peft_config=peft_config,
# )


