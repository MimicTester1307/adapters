import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

with open("D_KV_SUBS.json", "r") as f:
    dataset = json.load(f)

one_row = dataset[232]
print(one_row)

EOS_TOKEN = "</s>"
outputs = [row['value'] + EOS_TOKEN for row in dataset]

print(outputs[0])

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token	

print(tokenizer.encode(outputs[0]))
print(tokenizer.encode("My experiments are going strong!"))