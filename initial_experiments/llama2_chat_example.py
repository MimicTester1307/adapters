from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", use_auth_token=True, device_map = 'auto')

prompt = "Hi, How are You?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=20)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
