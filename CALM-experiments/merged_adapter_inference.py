import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "meta-llama/Llama-2-7b-hf"
compute_dtype = getattr(torch, "float16")

model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map={"": 0})

model = PeftModel.from_pretrained(model, "schaturv/llama2-7b-arithmetic-calculations-adapter", adapter_name="arithmetic")

model.load_adapter("schaturv/llama2-7b-key-value-adapter", adapter_name="pairings")

print(model)

