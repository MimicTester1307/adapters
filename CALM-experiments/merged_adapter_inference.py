import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

base_model = "meta-llama/Llama-2-7b-hf"
compute_dtype = getattr(torch, "float16")

model = AutoModelForCausalLM.from_pretrained(
        base_model)

# device = torch.device("cuda")
# model.to(device)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)

# print(model)

# testing on prompts for base model
outfile = open("inference_outputs/merged_prompt_inference_on_individual_adapters.txt", 'w')
# with open("inference_inputs/inference_for_base_model_merged_adapter.txt") as file:
#     prompts = [line.rstrip() for line in file]

# outfile.write("##INFERENCE ON BASE MODEL\n\n")

# for prompt in prompts:
#     inputs = tokenizer(prompt, return_tensors="pt").input_ids
#     # inputs = inputs.to('cuda')
#     outputs = model.generate(inputs, max_new_tokens=40)
#     tokenized_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     outfile.write(tokenized_output[0])
#     outfile.write("\n")
#     print(tokenized_output[0])

# outfile.write("\n\n\n")

peft_model = PeftModel.from_pretrained(model, "schaturv/llama2-7b-arithmetic-calculations-adapter", adapter_name="arithmetic", is_trainable=True)

peft_model.load_adapter("schaturv/llama2-7b-key-value-adapter", adapter_name="pairings")

# print(peft_model)

# # combining adapters using cat
peft_model.add_weighted_adapter(["arithmetic", "pairings"], [1.0,1.0], combination_type="linear", adapter_name="pairings_arithmetic")

# peft_model.set_adapter("pairings_arithmetic")
peft_model.set_adapter("pairings")

print(peft_model)

print("Active adapters: ", peft_model.active_adapters)

# testing on prompts
with open("inference_inputs/inference_for_merged_adapter.txt") as file:
    prompts = [line.rstrip() for line in file]

outfile.write("##INFERENCE ON PEFT MODEL WITH ACTIVATED ADAPTER\n\n")

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    # inputs = inputs.to('cuda')
    outputs = peft_model.generate(inputs, max_new_tokens=40)
    tokenized_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outfile.write(tokenized_output[0])
    outfile.write("\n")
    print(tokenized_output[0])

outfile.close()

peft_model.set_adapter("arithmetic")

print(peft_model)

print("Active adapters: ", peft_model.active_adapters)

# testing on prompts
with open("inference_inputs/inference_for_merged_adapter.txt") as file:
    prompts = [line.rstrip() for line in file]

outfile.write("##INFERENCE ON PEFT MODEL WITH ACTIVATED ADAPTER\n\n")

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    # inputs = inputs.to('cuda')
    outputs = peft_model.generate(inputs, max_new_tokens=40)
    tokenized_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outfile.write(tokenized_output[0])
    outfile.write("\n")
    print(tokenized_output[0])

outfile.close()