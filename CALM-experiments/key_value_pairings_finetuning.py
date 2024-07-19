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

with open("datasets/D_KV_SUBS.json", "r") as f:
    dataset = json.load(f)

one_row = dataset[232]
print("one row of the dataset: ", one_row)

model_id = 'meta-llama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# dividing train and eval datasets
train_dataset = dataset[:-300]
eval_dataset = dataset[-300:]

print("lengths of datasets: ", len(train_dataset), len(eval_dataset))

def pad_eos(ds):
    EOS_TOKEN = "" # "</s>"
    return [f"{row['value']}{EOS_TOKEN}" for row in ds]

# adding create_prompt to use as formatting_func argument during training
def prompt_input(row):
    return ("{key} = ").format_map(row)

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

def pack(dataset, max_seq_len=32):
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
print("total packed sequences: ", total_sequences)
print("total packed sequences eval dataset: ", len(eval_ds_packed))
print("single seq in packed sequences training dataset: ", train_ds_packed[0])

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

batch_size = 4
gradient_accumulation_steps = 16
num_train_epochs = 30

total_num_steps = num_train_epochs * total_sequences // (batch_size * gradient_accumulation_steps)

print("total num of steps ", total_num_steps)

output_dir = "./output/"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    # bf16=True,
    num_train_epochs = num_train_epochs,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio = 0.1,
    # max_steps = 10,
    # max_steps=total_num_steps,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs=dict(use_reentrant=False),
    evaluation_strategy="steps",
    eval_steps=(len(eval_ds_packed) / batch_size),
    # logging strategies
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=total_num_steps,
    use_cpu=False,
    do_predict=True,
)

model = AutoModelForCausalLM.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
device = torch.device("cuda")
model.to(device)

# checking prompt performance on base model
# with open("inference_inputs/context_only_adapter1_prompts.txt") as file:
#     prompts = [line.rstrip() for line in file]

# for prompt in prompts:
#     inputs = tokenizer(prompt, return_tensors="pt").input_ids
#     inputs = inputs.to('cuda')
#     outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=50, top_p=0.95)
#     tokenized_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     print(tokenized_output[0])


# model.add_adapter(peft_config, adapter_name="key-value-context-adapter")
# model.set_adapter("key-value-context-adapter")

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

torch.cuda.empty_cache()

#defining callback
class MyCallBack(TrainerCallback):
   def on_evaluate(self, args, state, model, tokenizer):
         tokens = tokenizer("text")
         generated_text  = model.generate(tokens["input_ids"], tokens["prompt"])

# model= torch.nn.DataParallel(model)
device = torch.device("cuda")
model.to(device)

peft_model = get_peft_model(model, peft_config)

trainer = Trainer(
    model=peft_model,
    train_dataset=train_ds_packed,
    eval_dataset=eval_ds_packed,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    # callbacks=[MyCallBack],
    # use_cache=False,
)

# print("active adapter before training: ", model.active_adapters())

trainer.train()

# save model
model.push_to_hub("schaturv/llama2-7b-key-value-context-adapter")

# testing on one prompt
with open("inference_inputs/context_only_adapter1_prompts.txt") as file:
    prompts = [line.rstrip() for line in file]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    inputs = inputs.to('cuda')
    outputs = model.generate(inputs, max_new_tokens=3)
    tokenized_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(tokenized_output[0])
    
with open("inference_inputs/context_only_adapter1_answers.txt") as file:
    answers = [line.rstrip() for line in file]
    print(answers)