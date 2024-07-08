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
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

seed = 42

## Print CUDA Summary

import sys
from subprocess import call
print('_____Python, Pytorch, Cuda info____')
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA RUNTIME API VERSION')
#os.system('nvcc --version')
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('_____nvidia-smi GPU details____')
call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('_____Device assignments____')
print('Number CUDA Devices:', torch.cuda.device_count())
print ('Current cuda device: ', torch.cuda.current_device(), ' **May not correspond to nvidia-smi ID above, check visibility parameter')
print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))


with open("datasets/D_KV_SUBS.json", "r") as f:
    dataset = json.load(f)

one_row = dataset[232]
print("one row of the dataset: ", one_row)

model_id = 'meta-llama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token	

# dividing train and eval datasets
train_dataset = dataset[:-4000]
eval_dataset = dataset[-4000:]

print("lengths of datasets: ", len(train_dataset), len(eval_dataset))

def pad_eos(ds):
    EOS_TOKEN = "</s>"
    return [f"{row['values']}{EOS_TOKEN}" for row in ds]

# adding create_prompt to use as formatting_func argument during training
def prompt_input(row):
    return ("{pairs}; {query} = ").format_map(row)

def create_prompt(row):
    return prompt_input(row)

# checking row in dataset
print("row in dataset: ", train_dataset[0])

train_prompts = [create_prompt(row) for row in train_dataset]
eval_prompts = [create_prompt(row) for row in eval_dataset]

# printing prompt
print("single prompt: ", train_prompts[0])

train_outputs = pad_eos(train_dataset)
eval_outputs = pad_eos(eval_dataset)

# printing output
print("single output: ", train_outputs[0])

train_dataset = [{"prompt":s, "output":t, "example": s + t} for s, t in zip(train_prompts, train_outputs)]
eval_dataset = [{"prompt":s, "output":t, "example": s + t} for s, t in zip(eval_prompts, eval_outputs)]

# checking row in dataset
print("row in formatted dataset: ", train_dataset[0])

# packing examples with padding
max_seq_len = 1024

# breakpoint() 

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
import transformers
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer

batch_size = 8
gradient_accumulation_steps = 4
num_train_epochs = 3

total_num_steps = num_train_epochs * total_sequences // (batch_size * gradient_accumulation_steps)

print(total_num_steps)

output_dir = "./output/"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size//2,
    bf16=True,
    learning_rate=2e-3,
    lr_scheduler_type="cosine",
    warmup_ratio = 0.1,
    max_steps = 10,
    # max_steps=total_num_steps,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs=dict(use_reentrant=False),
    evaluation_strategy="steps",
    eval_steps=total_num_steps // num_train_epochs,
    # logging strategies
    logging_strategy="steps",
    logging_steps=5,
    save_strategy="steps",
    save_steps=total_num_steps // num_train_epochs,
    use_cpu=False,
    do_predict=True,
)

model = AutoModelForCausalLM.from_pretrained(model_id)
model.add_adapter(peft_config, adapter_name="llama2-7b-key-value-pairings-adapter")
model.set_adapter("llama2-7b-key-value-pairings-adapter")

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

torch.cuda.empty_cache()

device = torch.device("cuda")
model.to(device)

#defining callback
class MyCallBack(TrainerCallback):
   def on_evaluate(self, args, state, model, tokenizer):
         tokens = tokenizer("text")
         generated_text  = model.generate(tokens["input_ids"], tokens["prompt"])

trainer = Trainer(
    model=model,
    train_dataset=train_ds_packed,
    eval_dataset=eval_ds_packed,
    # get_train_dataloader=train_ds_packed,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[MyCallBack],
    # use_cache=False,
)

print("active adapter before training: ", model.active_adapters())

trainer.train()

# save model
model.push_to_hub("schaturv/llama2-7b-key-value-adapter")

# testing on one prompt
with open("inference_inputs/inference_for_dataset_1.txt") as file:
    prompts = [line.rstrip() for line in file]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    inputs = inputs.to('cuda')
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    tokenized_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(tokenized_output)