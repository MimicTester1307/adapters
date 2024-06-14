import torch
import torch.nn as nn
import bitsandbytes as bnb
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from IPython.display import display, Markdown

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

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
(* 
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

f = open("dummy_llama_inference_outputs.txt", "w")

def make_inference(context, question):
  batch = tokenizer(f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n", return_tensors='pt')
  
  with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=200)

  print(tokenizer.decode(output_tokens[0], skip_special_tokens=True))
  f.write((tokenizer.decode(output_tokens[0], skip_special_tokens=True)))
  return

cheese_context = "Cheese is the best food."
cheese_question = "What is the best food?"

make_inference(cheese_context, cheese_question)

cheese_context = "Cheese is the best food."
moon_question = "How far away is the Moon from the Earth?"

make_inference(cheese_context, moon_question)

moon_context = "The Moon orbits Earth at an average distance of 384,400 km (238,900 mi), or about 30 times Earth's diameter. Its gravitational influence is the main driver of Earth's tides and very slowly lengthens Earth's day. The Moon's orbit around Earth has a sidereal period of 27.3 days. During each synodic period of 29.5 days, the amount of visible surface illuminated by the Sun varies from none up to 100%, resulting in lunar phases that form the basis for the months of a lunar calendar. The Moon is tidally locked to Earth, which means that the length of a full rotation of the Moon on its own axis causes its same side (the near side) to always face Earth, and the somewhat longer lunar day is the same as the synodic period. However, 59% of the total lunar surface can be seen from Earth through cyclical shifts in perspective known as libration."
moon_question = "At what distance does the Moon orbit the Earth?"

make_inference(moon_context, moon_question)

f.close()
