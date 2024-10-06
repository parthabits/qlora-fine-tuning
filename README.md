
# LLaMA Fine-Tuning with Amazon Polarity Dataset

This repository contains a script for fine-tuning the LLaMA model using the Amazon Polarity dataset. The purpose of this project is to showcase the steps for loading a dataset, tokenizing the data, and preparing the model for fine-tuning using LoRA (Low-Rank Adaptation).

## Requirements

To run this code, we need the following dependencies:
- `transformers`
- `datasets`
- `huggingface_hub`
- `peft`
- `accelerate`

Install the required packages with:

```bash
pip install transformers datasets huggingface_hub accelerate git+https://github.com/huggingface/peft.git
```

## Authentication

We need to authenticate using a Hugging Face token before accessing the datasets or model hub.

```python
from huggingface_hub import login

# Authentication using Hugging Face token
login(token="")
```

## Dataset

The dataset used is the **Amazon Polarity** dataset, a collection of Amazon reviews classified into positive and negative sentiment categories. Each class contains 1.8 million training samples and 200,000 testing samples.

More information about the dataset can be found [here](https://paperswithcode.com/dataset/amazon-polarity-1).

```python
from datasets import load_dataset

# Loading the Amazon Polarity dataset
dataset = load_dataset('amazon_polarity')

# Checking the dataset structure
print(dataset)
```

## Tokenization

We use the LLaMA tokenizer for preprocessing the data.

```python
from transformers import LlamaTokenizer

# Loading the tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/llama-2-7b")

def tokenize_function(examples):
    return tokenizer(examples['content'], padding='max_length', truncation=True)

# Tokenizing the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Renaming the label column to 'labels'
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')

# Setting the format to PyTorch tensors
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
```

## Fine-Tuning

We are using LoRA to fine-tune the LLaMA model with reduced memory usage. The training process may encounter memory issues, and alternative strategies such as gradient accumulation may help mitigate those.

### Model and LoRA Configuration

```python
from transformers import LlamaForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

# Loading the pre-trained LLaMA model
model = LlamaForSequenceClassification.from_pretrained("meta-Llama/llama-2-7b-hf", num_labels=2)

# Preparing the model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

# Integrating LoRA into the model
model = get_peft_model(model, lora_config)
```

### Training Configuration

```python
# Defining training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Reduce batch size to save memory
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # Enable mixed precision
    gradient_checkpointing=True,  # Enable gradient checkpointing
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10
)

# Initializing the accelerator
accelerator = Accelerator()

# Creating the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    accelerator=accelerator
)

# Starting the training process
trainer.train()
```

### Issues Faced

During training, we encountered GPU memory over-utilization issues, even with reduced batch sizes and gradient accumulation. This may require running the script on hardware with more available memory.

