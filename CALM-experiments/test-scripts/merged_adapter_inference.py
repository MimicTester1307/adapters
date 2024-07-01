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

print("Active adapters: ", peft_model.active_adapters)

# breakpoint()

# config = PeftConfig.from_pretrained("schaturv/pairings_arithmetic")
# model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
# lora_model = PeftModel.from_pretrained(model, "schaturv/pairings_arithmetic")

# model.config.to_json_file("adapter_config.json")
# model.push_to_hub("schaturv/pairings_arithmetic")

# model = PeftModel.from_pretrained(model)
# print("active adapters: ", model.active_adapters)
# print("active adapters: ", model.active_adapter)

# prompt generating function
def generate(prompt):
  tokenized_input = tokenizer(prompt, return_tensors="pt")
  print(tokenized_input)
  input_ids = tokenized_input["input_ids"].cuda()

  generation_output = peft_model.generate(
          input_ids=input_ids,
          num_beams=1,
          return_dict_in_generate=True,
          output_scores=True,
          max_new_tokens=200

  )
  for seq in generation_output.sequences:
      output = tokenizer.decode(seq, skip_special_tokens=True)
      print(output.strip())

# peft model not suitable for 
# pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
# result = pipe("Find the computation of this arithmetic expression: 22032 * 17024 - 19049 + 5049 + 21763")
# print(result[0]['generated_text'])

print(generate("Perform the arithmetic calculations to get the desired solution.\n\n"
            "### Key:22032 * 17024 - 19049 + 5049 + 21763\n\n### Value:\n</s>"))
