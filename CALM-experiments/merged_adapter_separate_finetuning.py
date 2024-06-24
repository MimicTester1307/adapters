

# !pip install transformers peft
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from peft import LoraConfig, get_peft_model

# base_model = 'meta-llama/Llama-2-7b-hf'
base_model = 'roberta-base'

##
model=AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=base_model)

model

model_total_parameters=sum(p.numel() for p in model.parameters())
model_parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'total parameters: {model_total_parameters} | total_trainable_parameters={model_parameters_trainable}')

##loading the peft model
peft_config=LoraConfig(inference_mode=False)

peft_model=get_peft_model(model=model,
                          peft_config=peft_config)

peft_model.print_trainable_parameters() ## only trains 23% of the model

print(peft_model)

"""Adding a adapter works same as adding adapter to peft_model from the hub. Basically we will register numbers of adapters tp the config. This does not change the model itself. While training and inference we can switch to any of them using different methods as explained below. More detail on previous colab note 09_PEFT.ipynb"""

peft_model.add_adapter('adapter1',peft_config)

print(peft_model)

peft_model.print_trainable_parameters()

peft_model.add_adapter('adapter2',peft_config)
# peft_model.add_adapter('adapter3',peft_config)

peft_model

peft_model.set_adapter('adapter1')  #We can use this method to see the active peft adapter

peft_model.active_adapters

"""### peft_model.set_adapter
This method would allow to set different adapter before training or
"""

peft_model.set_adapter('adapter1')

peft_model

"""#### Testing different adapters config during training and testing"""

# !pip install transformers datasets
import transformers
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import numpy as np

import datasets

tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model)
# tokenizer.pad_token = tokenizer.eos_token	

dataset=datasets.load_dataset('ag_news')

num_labels=np.unique(dataset['train']['label']).shape[0]
dataset['train'].features
def tokenize_example(example_dataset):
    text=example_dataset['text']
    return tokenizer(text,padding=True,truncation=True)

tokenize_dataset=dataset.map(tokenize_example,
                             batched=True,
                            remove_columns=['text'])
num_labels = dataset['train'].features['label'].num_classes
classnames=tokenize_dataset['train'].features['label'].names
print(f"number of labels: {num_labels}")
print(f"the labels: {classnames}")

id2label={i:label for i,label in enumerate(classnames)}
print(f'id2label: {id2label}')


data_collator=DataCollatorWithPadding(tokenizer=tokenizer,padding=True,return_tensors='pt') ## we are again padding even though there is already padding in above tokenizer.map(batched=True); This is because
# in map padding is done in a fixed batched size; however; during training using Trainer().train() different batch_size would have been present so we wanna make sure it is agiain padded when we are generating batches
##during training

seed=101
train_dataset=tokenize_dataset['train'].shuffle(seed=seed).select(range(2000))
eval_dataset=tokenize_dataset['test'].shuffle(seed=seed).select(range(2000))

### creating different lora_adapter_config
peft_config=peft_config_A=LoraConfig(inference_mode=False)
peft_config_1=LoraConfig(r=4,lora_alpha=16,inference_mode=False,lora_dropout=0.1)
peft_config_2=LoraConfig(inference_mode=False,r=8,lora_alpha=32,lora_dropout=0.3)

model=AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=base_model,id2label=id2label)
peft_model=get_peft_model(model=model,
                          peft_config=peft_config

                    )

## add adapter
peft_model.add_adapter('adapter1',peft_config=peft_config_1)
peft_model.add_adapter('adapter2',peft_config=peft_config_2)

peft_model

# !pip install evaluate
import evaluate

metrics=evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits,label=eval_pred
    pred_label=np.argmax(logits,axis=-1)
    return metrics.compute(predictions=pred_label,references=label)

"""#### Training with LoRAConfigA"""

from transformers import TrainingArguments,Trainer

peft_model.set_adapter('adapter1')
print("Training Adapter 1. Active Adapter: ", peft_model.active_adapters)

saved_dire='../saved_adapters/adapter_1_config'
args_1=TrainingArguments(output_dir=saved_dire)
trainer_1=Trainer(model=peft_model,
                  args=args_1,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  compute_metrics=compute_metrics,
                  data_collator=data_collator
                  )

trainer_1.train()

trainer_1.evaluate()

# breakpoint()

"""#### Training with LoRAConfig2"""

peft_model.set_adapter('adapter2')
print("Training Adapter 2. Active Adapter: ", peft_model.active_adapters)

saved_dire='../saved_adapters/adapter_2_config'
args_2=TrainingArguments(output_dir=saved_dire)
trainer_2=Trainer(model=peft_model,
                  args=args_2,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  compute_metrics=compute_metrics,
                  data_collator=data_collator
                  )

trainer_2.train()

trainer_2.evaluate()

saved_dire='../saved_adapters/adapter_config'
peft_model.save_pretrained(saved_dire)
tokenizer.save_pretrained(saved_dire)

"""## loading both model_adapter"""

from peft import PeftConfig,PeftModel
from transformers import AutoModelForSequenceClassification,AutoTokenizer

## Path of the save_model dire
saved_dire='../saved_adapters/adapter_config'

## loading the "Pretrained" base model and "Pretrained" tokenizer
id2label={0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
base_model=AutoModelForSequenceClassification.from_pretrained(saved_dire,id2label= id2label)
tokenizer=AutoTokenizer.from_pretrained(saved_dire)

##loading the adapter1_config and adapter2_config
adapter1_config=PeftConfig.from_pretrained(saved_dire+'/adapter1')
adapter2_config=PeftConfig.from_pretrained(saved_dire+'/adapter2')

print(f'adapter1_config: {adapter1_config}')
print(f'adapter2_config: {adapter2_config}')

# Load the entire model with adapters
peft_model = PeftModel.from_pretrained(base_model, saved_dire)

# Load adapter1 and adapter2
peft_model.load_adapter(saved_dire + '/adapter1', adapter_name='adapter1')
peft_model.load_adapter(saved_dire + '/adapter2', adapter_name='adapter2')

import torch.nn.functional as F

def classify(peft_model,text, adapter_name: str):
    # Set the adapter
    peft_model.set_adapter(adapter_name)
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    # Get the model's output
    output = peft_model(**inputs)
    # Get the predicted class and confidence
    probabilities = F.softmax(output.logits, dim=-1)
    prediction = probabilities.argmax(dim=-1).item()
    confidence = probabilities[0, prediction].item()
    print(f'Adapter: {adapter_name} | Text: {text} | Class: {prediction} | Label: {id2label[prediction]} | Confidence: {confidence:.2%}')

text1="Kederis proclaims innocence Olympic champion Kostas Kederis today left hospital ahead of his date with IOC inquisitors claiming his ..."
text2="Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again."

classify(peft_model,text1, 'adapter1')
classify(peft_model,text1, 'adapter2')

classify(peft_model, text2, 'adapter1') ## both correction are wrong 'trained on small dataset so
classify(peft_model, text2, 'adapter2')

# trying to merge adapters
# peft_model.merge_and_unload() - still gave only adapter2 as active
# print(peft_model.active_adapters)


"""### What if saved_pretrained only saved adapter weight?

Load the base_model and tokenizer from the hub and keep everything the same. **Make sure when you are training you also save the classifier_head**



```python
base_model = AutoModelForSequenceClassification.from_pretrained(base_model, id2label=id2label)

tokenizer = AutoTokenizer.from_pretrained(base_model)
```

"""

# Loading adapters pretrained
base_model = 'roberta-base'
saved_dire='../saved_weight/12_config_lora'
## loading the "Pretrained" base model and "Pretrained" tokenizer
id2label={0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
# ## we will load base_model from hub and only use adapter
base_model=AutoModelForSequenceClassification.from_pretrained(base_model,id2label= id2label)
tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model)
##loading the adapter1_config and adapter2_config
adapter1_config=PeftConfig.from_pretrained(saved_dire+'/adapter1')
adapter2_config=PeftConfig.from_pretrained(saved_dire+'/adapter2')
print(f'adapter1_config: {adapter1_config}')
print(f'adapter2_config: {adapter2_config}')

# Load the entire model with adapters
peft_model_ = PeftModel.from_pretrained(base_model, saved_dire)

# Load adapter1 and adapter2
peft_model_.load_adapter(saved_dire + '/adapter1', adapter_name='adapter1')
peft_model_.load_adapter(saved_dire + '/adapter2', adapter_name='adapter2')
model.add_weighted_adapter(
    adapters=["adapter1", "adapter2"],
    weights=[1.0, 1.0],
    adapter_name="combined_adapter",
    combination_type="linear"
)
model.set_adapter("combined_adapter")

print(peft_model.active_adapters)
