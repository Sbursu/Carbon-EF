import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Title and Introduction
intro_md = """# Mistral-7B Fine-tuning for Emission Factor Recommendations

This notebook implements the fine-tuning of the Mistral-7B-Instruct-v0.2 model for emission factor recommendations using LoRA (Low-Rank Adaptation).

## Setup Requirements

- GPU Runtime (T4 or A100 recommended)
- Google Drive mounted for saving checkpoints
- Required packages installed
- Training data from GitHub repository

## Notebook Structure

1. Environment Setup
2. Model and Tokenizer Configuration
3. Data Preparation
4. Training Setup
5. Model Training
6. Model Evaluation
7. Model Saving and Export"""

nb.cells.append(nbf.v4.new_markdown_cell(intro_md))

# Environment Setup
env_setup_md = """## 1. Environment Setup

First, we'll mount Google Drive and install the required packages."""

nb.cells.append(nbf.v4.new_markdown_cell(env_setup_md))

env_setup_code = """# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install -q torch>=2.0.0 transformers>=4.34.0 peft>=0.5.0 accelerate>=0.21.0 \\
    bitsandbytes>=0.40.0 trl>=0.7.1 tensorboard>=2.14.0 datasets>=2.14.0 \\
    evaluate>=0.4.0 tqdm>=4.66.1 pandas>=2.1.0 matplotlib>=3.7.2 \\
    seaborn>=0.12.2 sentencepiece>=0.1.99 scipy>=1.11.2 \\
    scikit-learn>=1.3.0 einops>=0.6.1 wandb>=0.15.10"""

nb.cells.append(nbf.v4.new_code_cell(env_setup_code))

# Model Configuration
model_config_md = """## 2. Model and Tokenizer Configuration

Set up the Mistral-7B model with 4-bit quantization and LoRA configuration."""

nb.cells.append(nbf.v4.new_markdown_cell(model_config_md))

model_config_code = """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Model configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Configure LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
print("Model and tokenizer configured successfully!")"""

nb.cells.append(nbf.v4.new_code_cell(model_config_code))

# Data Preparation
data_prep_md = """## 3. Data Preparation

Load and prepare the training data from the GitHub repository."""

nb.cells.append(nbf.v4.new_markdown_cell(data_prep_md))

data_prep_code = """# Clone the repository if not already cloned
!git clone https://github.com/yourusername/Carbon-EF.git
%cd Carbon-EF

# Load training and validation data from GitHub
from datasets import load_dataset

# Load data from the repository
train_data = load_dataset('json', data_files='training/data/instructions_train.json')
val_data = load_dataset('json', data_files='training/data/instructions_val.json')

# Format instruction template
def format_instruction(example):
    instruction = example['instruction']
    input_text = example.get('input', '')
    output = example['output']
    
    if input_text:
        formatted = f"<s>[INST] {instruction}\\n\\n{input_text} [/INST] {output} </s>"
    else:
        formatted = f"<s>[INST] {instruction} [/INST] {output} </s>"
    
    return {'text': formatted}

# Apply formatting
train_data = train_data.map(format_instruction)
val_data = val_data.map(format_instruction)

print(f"Training samples: {len(train_data['train'])}")
print(f"Validation samples: {len(val_data['train'])}")"""

nb.cells.append(nbf.v4.new_code_cell(data_prep_code))

# Training Setup
training_setup_md = """## 4. Training Setup

Configure the training arguments and initialize the trainer."""

nb.cells.append(nbf.v4.new_markdown_cell(training_setup_md))

training_setup_code = """from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    evaluation_strategy='steps',
    eval_steps=50,
    save_strategy='steps',
    save_steps=50,
    warmup_steps=100,
    report_to='tensorboard',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss'
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data['train'],
    eval_dataset=val_data['train'],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

print("Training setup completed!")"""

nb.cells.append(nbf.v4.new_code_cell(training_setup_code))

# Model Training
training_md = """## 5. Model Training

Start the fine-tuning process. This will take several hours depending on your GPU."""

nb.cells.append(nbf.v4.new_markdown_cell(training_md))

training_code = """# Start training
trainer.train()

# Save the final model
trainer.save_model('./final_model')
print('Training completed and model saved!')"""

nb.cells.append(nbf.v4.new_code_cell(training_code))

# Model Evaluation
evaluation_md = """## 6. Model Evaluation

Evaluate the model on the test set to measure its performance."""

nb.cells.append(nbf.v4.new_markdown_cell(evaluation_md))

evaluation_code = """# Load test data from GitHub
test_data = load_dataset('json', data_files='training/data/instructions_test.json')
test_data = test_data.map(format_instruction)

# Evaluate model
eval_results = trainer.evaluate(test_data['train'])

# Print evaluation results
print('Evaluation Results:')
print(eval_results)"""

nb.cells.append(nbf.v4.new_code_cell(evaluation_code))

# Model Saving
saving_md = """## 7. Model Saving and Export

Save the fine-tuned model and configurations to Google Drive."""

nb.cells.append(nbf.v4.new_markdown_cell(saving_md))

saving_code = """import os
import json

# Save model to Google Drive
DRIVE_PATH = '/content/drive/MyDrive/carbon_ef_model'
os.makedirs(DRIVE_PATH, exist_ok=True)

# Save model and tokenizer
model.save_pretrained(f'{DRIVE_PATH}/model')
tokenizer.save_pretrained(f'{DRIVE_PATH}/tokenizer')

# Save training configuration
with open(f'{DRIVE_PATH}/training_config.json', 'w') as f:
    json.dump({
        'model_name': MODEL_NAME,
        'lora_config': lora_config.to_dict(),
        'training_args': training_args.to_dict(),
        'eval_results': eval_results
    }, f, indent=2)

print(f'Model and configurations saved to {DRIVE_PATH}')"""

nb.cells.append(nbf.v4.new_code_cell(saving_code))

# Write the notebook to a file
with open("training/notebooks/mistral_finetuning.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook created successfully!")
