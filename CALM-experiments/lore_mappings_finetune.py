import torch
print(torch.cuda.is_available())

# """#### Load Base Model"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from IPython.display import display, Markdown
from trl import SFTTrainer

from generate_mapping_dataset import D_KV_SUBS, D_KV_VAL, D_SUBS_VAL, create_arithmetic_expressions

import pandas as pd
df = pd.DataFrame(D_KV_SUBS)

from datasets import Dataset
dataset = Dataset.from_pandas(df.rename(columns={0: "train"}), split="train")


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# using tokenizer.use_special_token adds extra padding tokens and increases size of vocabulary, causing an index error during model training. 
tokenizer.pad_token = tokenizer.eos_token

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


"""##### View Model Summary"""

print(model)

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

"""#### Helper Function"""

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

"""#### Obtain LoRA Model"""
# HuggingFace's Inbuilt Lora implementation
 
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

"""#### Train LoRA"""

import transformers

training_arguments = transformers.TrainingArguments( 
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=100,
    learning_rate=1e-3,
    fp16=True,
    logging_steps=1,
    output_dir='outputs',
    # use_cpu=False,
    # push_to_hub=True,
    ),

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=config,
    # dataset_text_field="train",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
    # packing=False,
)

trainer.train()

prompt, expected_ans, _ = create_arithmetic_expressions()
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] Given an arithmetic expression between strings: {prompt}. Find the corresponding arithmetic expression in numeric terms between the numeric mappings of those strings.[/INST]")
print(prompt, expected_ans)
print(result[0]['generated_text'])

access_token = "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"
model.push_to_hub("schaturv/llama-2-7b-lora-key-value-mappings", token = access_token)

# Setting use_cpu = True as a training argument helped me debug and check the memory requirements for training without running 
# into CUDA: Out of Memory error. Once debugged successfully, setting it to False allowed me to train the model using GPU itself.

# push_to_hub added in the Training Arguments requires that you manually access a 'write' token through the Huggingface-CLI login, 
# but since loading the pretrained Llama model required me to login via a 'read' token, I manually identified the access_token in
# the file itself and then used model.push_to_hub command. This action is strictly prohibited in the case of scripts where exposing
# the secret can be detrimental. (Check GitGuardian's guidelines.)
