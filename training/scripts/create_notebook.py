import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Title and Prerequisites
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """# Mistral-7B Fine-tuning for Emission Factor Recommendations

This notebook implements the fine-tuning process for the Mistral-7B model to generate emission factor recommendations based on the PRD specifications.

## Prerequisites

1. GPU Runtime in Google Colab
2. Google Drive mounted for saving checkpoints
3. Hugging Face account with access to Mistral-7B-Instruct-v0.2
4. HF_TOKEN in Colab secrets"""
    )
)

# Environment Setup
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 1. Environment Setup

Install required packages and mount Google Drive."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install -q torch transformers peft datasets accelerate scipy wandb trl"""
    )
)

# Hugging Face Authentication
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 2. Hugging Face Authentication

Authenticate with Hugging Face and verify access to the model."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """import os
from huggingface_hub import login, HfApi

# Login to Hugging Face
login(token=os.environ.get('HF_TOKEN'))

# Verify access
api = HfApi()
try:
    api.model_info("mistralai/Mistral-7B-Instruct-v0.2")
    print("Successfully authenticated and have access to the model!")
except Exception as e:
    print(f"Error: {e}")"""
    )
)

# Model Configuration - UPDATED to use 16-bit precision
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 3. Model and Tokenizer Configuration

Set up the Mistral-7B model with 16-bit precision and LoRA configuration."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Model configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load model with 16-bit precision instead of 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Reduced rank for better memory usage
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
print("Model and tokenizer configured successfully!")"""
    )
)

# Data Preparation
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 4. Data Preparation

Load and prepare the training data from the GitHub repository."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Clone the repository
!git clone https://github.com/Sbursu/Carbon-EF.git
%cd Carbon-EF

# Load datasets
from datasets import load_dataset

train_data = load_dataset('json', data_files='training/data/instructions_train.json')
val_data = load_dataset('json', data_files='training/data/instructions_val.json')
test_data = load_dataset('json', data_files='training/data/instructions_test.json')

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
test_data = test_data.map(format_instruction)

print(f"Training samples: {len(train_data['train'])}")
print(f"Validation samples: {len(val_data['train'])}")
print(f"Test samples: {len(test_data['train'])}")"""
    )
)

# Training Configuration
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 5. Training Configuration

Set up training arguments and initialize the trainer."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Training arguments
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/mistral-ef-checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Reduced batch size for better memory usage
    gradient_accumulation_steps=8,  # Increased gradient accumulation steps
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    warmup_ratio=0.1
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data['train'],
    eval_dataset=val_data['train'],
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("Training configuration completed!")"""
    )
)

# Training Process
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 6. Training Process

Start the fine-tuning process."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Start training
trainer.train()

# Save the final model
trainer.save_model("/content/drive/MyDrive/mistral-ef-final")"""
    )
)

# Model Evaluation
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 7. Model Evaluation

Evaluate the fine-tuned model on the test set."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Evaluate on test set
test_results = trainer.evaluate(test_data['train'])
print(f"Test results: {test_results}")"""
    )
)

# Save and Export
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 8. Save and Export

Save the model and tokenizer to Google Drive."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Save model and tokenizer
model.save_pretrained("/content/drive/MyDrive/mistral-ef-final")
tokenizer.save_pretrained("/content/drive/MyDrive/mistral-ef-final")

# Save training configuration
import json
with open("/content/drive/MyDrive/mistral-ef-final/training_config.json", "w") as f:
    json.dump({
        "model_name": MODEL_NAME,
        "lora_config": lora_config.to_dict(),
        "training_args": training_args.to_dict(),
        "test_results": test_results
    }, f, indent=2)

print("Model and configuration saved successfully!")"""
    )
)

# Write the notebook to a file
with open("training/notebooks/mistral_finetuning.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook created successfully!")
