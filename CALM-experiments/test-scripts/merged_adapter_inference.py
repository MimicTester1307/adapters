import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

base_model = "meta-llama/Llama-2-7b-hf"
compute_dtype = getattr(torch, "float16")

model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map={"": 0})

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)

peft_model = PeftModel.from_pretrained(model, "schaturv/llama2-7b-arithmetic-calculations-adapter", adapter_name="arithmetic", is_trainable=True)

peft_model.load_adapter("schaturv/llama2-7b-key-value-adapter", adapter_name="pairings")

print(peft_model)

# # combining adapters using cat
peft_model.add_weighted_adapter(["arithmetic", "pairings"], [1.0,1.0], combination_type="linear", adapter_name="pairings_arithmetic")

peft_model.set_adapter("pairings_arithmetic")

print(peft_model)

print("Active adapters: ", peft_model.active_adapters)

# breakpoint()

# testing on prompts
outfile = open("merged_adapter_inference_outputs_smaller.txt", 'w')
with open("inference_for_merged_adapter.txt") as file:
    prompts = [line.rstrip() for line in file]

outfile.write("Smaller prompts with maximum length of expressions having 3 operands.\n\n")

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    inputs = inputs.to('cuda')
    outputs = model.generate(inputs, max_new_tokens=20)
    tokenized_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outfile.write(tokenized_output[0]\n)
    print(tokenized_output[0]\n)

outfile.close()