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
from peft import PeftConfig,PeftModel

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

seed = 42

# model
model_id = 'meta-llama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token	

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

# adding create_prompt to use as formatting_func argument during training
def prompt_input(row):
    return ("### Arithmetic Expression:{key} ### Answer:").format_map(row)

def create_prompt(row):
    return prompt_input(row)

# packing examples with padding
max_seq_len = 1024

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
    
def pad_eos(ds):
    EOS_TOKEN = "</s>"
    return [f"{row['value']}{EOS_TOKEN}" for row in ds]

#### DATASET SPECIFIC PRE-PROCESSING

# load dataset
with open("D_SUBS_VAL.json", "r") as f:
    dataset_arithmetic = json.load(f)

with open("D_KV_SUBS.json", "r") as f:
    dataset_pairings = json.load(f)

def dataset_preprocessing_till_packing(dataset):
    # dividing train and eval datasets
    train_dataset = dataset[:-4000]
    eval_dataset = dataset[-4000:]

    train_prompts = [create_prompt(row) for row in train_dataset]
    eval_prompts = [create_prompt(row) for row in eval_dataset]

    # padded outputs
    train_outputs = pad_eos(train_dataset)
    eval_outputs = pad_eos(eval_dataset)

    train_dataset = [{"prompt":s, "output":t, "example": s + t} for s, t in zip(train_prompts, train_outputs)]
    eval_dataset = [{"prompt":s, "output":t, "example": s + t} for s, t in zip(eval_prompts, eval_outputs)]

    train_ds_packed = pack(train_dataset)
    eval_ds_packed = pack(eval_dataset)

    return train_ds_packed, eval_ds_packed

train_ds_packed_pairings, eval_ds_packed_pairings = dataset_preprocessing_till_packing(dataset_pairings)
train_ds_packed_arithmetic, eval_ds_packed_arithmetic = dataset_preprocessing_till_packing(dataset_arithmetic)

# length of sequences we get after packing them together
total_sequences_pairings = len(train_ds_packed_pairings)
total_sequences_arithmetic = len(train_ds_packed_arithmetic)

torch.manual_seed(seed)


#### TRAINING 

# training the model and training arguments
import transformers
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer

batch_size = 8
gradient_accumulation_steps = 4
num_train_epochs = 3

total_num_steps_pairings = num_train_epochs * total_sequences_pairings // (batch_size * gradient_accumulation_steps)
total_num_steps_arithmetic = num_train_epochs * total_sequences_arithmetic // (batch_size * gradient_accumulation_steps)

model = AutoModelForCausalLM.from_pretrained(model_id)
model.add_adapter(peft_config, adapter_name="llama2-7b-key-value-pairings-adapter")
model.add_adapter(peft_config, adapter_name="llama2-7b-arithmetic-calculations-adapter")

print(model)

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


#### TRAINING PAIRINGS ADAPTER

output_dir = "./output/pairings"
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
    eval_steps=total_num_steps_pairings // num_train_epochs,
    # logging strategies
    logging_strategy="steps",
    logging_steps=5,
    save_strategy="steps",
    save_steps=total_num_steps_pairings // num_train_epochs,
    use_cpu=False,
    do_predict=True,
)

model.set_adapter("llama2-7b-key-value-pairings-adapter")

trainer = Trainer(
    model=model,
    train_dataset=train_ds_packed_pairings,
    eval_dataset=eval_ds_packed_pairings,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[MyCallBack],
    # use_cache=False,
)

print("active adapter before training: ", model.active_adapters())

# trainer.train()

# save model
model.push_to_hub("schaturv/llama2-7b-key-value-pairings-adapter")

# testing on one prompt
f = open("test_samples_key_value_pairings.txt", 'r')
content = f.readlines()
expression = content[0]
expected_ans = content[1]
prompt = f"### Arithmetic Expression: {str(expression)} ### Answer: \n"
inputs = tokenizer(prompt, return_tensors="pt").input_ids
inputs = inputs.to('cuda')
outputs = model.generate(inputs, max_new_tokens=200, do_sample=True, top_k=50, top_p=0.95)
tokenized_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("expected_output: ", expected_ans)
print(tokenized_output)


#### TRAINING ARITHMETIC ADAPTER

output_dir = "./output/arithmetic"
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
    eval_steps=total_num_steps_arithmetic // num_train_epochs,
    # logging strategies
    logging_strategy="steps",
    logging_steps=5,
    save_strategy="steps",
    save_steps=total_num_steps_arithmetic // num_train_epochs,
    use_cpu=False,
    do_predict=True,
)

model.set_adapter("llama2-7b-arithmetic-calculations-adapter")

trainer = Trainer(
    model=model,
    train_dataset=train_ds_packed_pairings,
    eval_dataset=eval_ds_packed_pairings,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[MyCallBack],
    # use_cache=False,
)

print("active adapter before training: ", model.active_adapters())

# trainer.train()

# save model
model.push_to_hub("schaturv/llama2-7b-arithmetic-calculations-adapter")

# testing on one prompt
prompt = "### Arithmetic Expression: '24 - 61 + 40' ### Answer: \n"
inputs = tokenizer(prompt, return_tensors="pt").input_ids
inputs = inputs.to('cuda')
outputs = model.generate(inputs, max_new_tokens=200, do_sample=True, top_k=50, top_p=0.95)
tokenized_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(tokenized_output)


#### LOADING THE ADAPTERS

# base_model = "meta-llama/Llama-2-7b-hf"
# compute_dtype = getattr(torch, "float16")

# model = AutoModelForCausalLM.from_pretrained(
#         base_model, device_map={"": 0})

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)

peft_model = PeftModel.from_pretrained(model, "schaturv/llama2-7b-arithmetic-calculations-adapter", adapter_name="arithmetic", is_trainable=True)

peft_model.load_adapter("schaturv/llama2-7b-key-value-pairings-adapter", adapter_name="pairings")

print(peft_model)

# # combining adapters using cat
peft_model.add_weighted_adapter(["arithmetic", "pairings"], [1.0,1.0], combination_type="linear", adapter_name="pairings_arithmetic")

peft_model.set_adapter("pairings_arithmetic")

print("Active adapters: ", peft_model.active_adapters())

# testing on one prompt
f = open("test_samples_key_solution_pairings.txt", 'r')
content = f.readlines()
expression = content[0]
expected_ans = content[1]
prompt = f"### Arithmetic Expression: {str(expression)} ### Answer: \n"
inputs = tokenizer(prompt, return_tensors="pt").input_ids
inputs = inputs.to('cuda')
outputs = model.generate(inputs, max_new_tokens=200, do_sample=True, top_k=50, top_p=0.95)
tokenized_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("expected_output: ", expected_ans)
print(tokenized_output)