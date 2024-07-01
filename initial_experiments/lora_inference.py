import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = f"schaturv/llama2-7b-adapter-trained"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=False, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
qa_model = PeftModel.from_pretrained(model, peft_model_id)

f = open("dummy_llama_inference_outputs.txt", "w")

def make_inference(context, question):
  batch = tokenizer(f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n", return_tensors='pt')
  
  with torch.cuda.amp.autocast():
    output_tokens = qa_model.generate(**batch, max_new_tokens=200)

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

llama_lora_model = PeftModel.from_pretrained(model, "haoranxu/ALMA-7B-Pretrain-LoRA")

def peft_make_inference(context, question):
  batch = tokenizer(f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n", return_tensors='pt')
  
  with torch.cuda.amp.autocast():
    output_tokens = llama_lora_model.generate(**batch, max_new_tokens=200)

  print(tokenizer.decode(output_tokens[0], skip_special_tokens=True))
  f.write((tokenizer.decode(output_tokens[0], skip_special_tokens=True)))
  return

f.write("\nPretrained Llama Model\n")

peft_make_inference(cheese_context, cheese_question)
peft_make_inference(cheese_context, moon_question)
peft_make_inference(moon_context, moon_question)

f.close()
