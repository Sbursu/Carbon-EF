#!/usr/bin/env python3
import json
import os
import shutil


def create_notebook():
    # Create a minimal notebook with essential cells
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# Mistral-7B Fine-Tuning for Emission Factor Recommendations\n\nThis notebook implements fine-tuning of the Mistral-7B language model to provide accurate, region-specific emission factor recommendations.\n\n## Setup Instructions\n1. Select Runtime > Change runtime type and choose GPU (T4 or better)\n2. Run cells in sequence",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "source": "# Check GPU availability\n!nvidia-smi",
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "## Install Dependencies",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "source": "# Install core dependencies with specific versions\n!pip install -q transformers==4.36.2 datasets==2.16.1 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.3 trl==0.7.11 wandb==0.16.3\n!pip install -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118\n!pip install -q neo4j==5.10.0",
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "## Setup Clean Environment",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "source": """# Aggressively remove any existing Carbon-EF directories
import os
import shutil
import glob

# Clean up any nested Carbon-EF directories
for path in glob.glob('/content/Carbon-EF*/**/', recursive=True):
    print(f"Removing directory: {path}")
    shutil.rmtree(path, ignore_errors=True)

# Clone fresh copy of repository
!git clone https://github.com/Sbursu/Carbon-EF.git /content/fresh-carbon-ef
os.chdir('/content/fresh-carbon-ef')

# Create required directories
os.makedirs('training/data', exist_ok=True)
os.makedirs('training/scripts', exist_ok=True)
os.makedirs('training/models', exist_ok=True)
os.makedirs('training/logs', exist_ok=True)

print(f"Working directory: {os.getcwd()}")
print("\\nDirectory structure:")
!ls -R training/""",
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "## Create Sample Data",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "source": """# Create sample instruction data directly
import json

sample_instructions = [
    {
        "instruction": "What is the emission factor for cement production in USA?",
        "input": "",
        "output": "The emission factor for cement production in the USA is 0.92 kg CO2e/kg. This data is sourced from USEEIO_v2.1.",
        "metadata": {
            "regions": ["USA"],
            "entity_types": ["product"],
            "difficulty": "basic",
            "sources": ["USEEIO_v2.1"]
        }
    },
    {
        "instruction": "Compare the emission factor for wheat production between France and the USA.",
        "input": "",
        "output": "The emission factor for wheat production in France is 0.38 kg CO2e/kg, while in the USA it is 0.41 kg CO2e/kg. The USA has a slightly higher emission factor (by 7.9%). This data is sourced from Agribalyse_3.1 for France and USEEIO_v2.1 for the USA.",
        "metadata": {
            "regions": ["FR", "USA"],
            "entity_types": ["product"],
            "difficulty": "moderate",
            "sources": ["Agribalyse_3.1", "USEEIO_v2.1"]
        }
    }
]

# Write sample data files directly
with open('training/data/instructions_train.json', 'w') as f:
    json.dump(sample_instructions, f, indent=2)
with open('training/data/instructions_val.json', 'w') as f:
    json.dump([sample_instructions[0]], f, indent=2)
with open('training/data/instructions_test.json', 'w') as f:
    json.dump([sample_instructions[1]], f, indent=2)

print("Sample data files created successfully!")
!ls -l training/data/""",
            },
            {"cell_type": "markdown", "metadata": {}, "source": "## Import and Train"},
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "source": """import sys
sys.path.append('training/scripts')

from data_preparation import load_and_prepare_data, format_instruction
from model_config import setup_model_and_tokenizer, get_training_config
from training import setup_trainer, evaluate_model, save_model

print("Successfully imported all modules!")""",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "source": """# Load and prepare data
train_data, val_data = load_and_prepare_data(
    train_path='training/data/instructions_train.json',
    val_path='training/data/instructions_val.json',
    test_path='training/data/instructions_test.json'
)

# Format data
train_data = train_data.map(format_instruction)
val_data = val_data.map(format_instruction)

print(f"Training examples: {len(train_data['train'])}")
print(f"Validation examples: {len(val_data['train'])}")""",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "source": """# Initialize model and trainer
model, tokenizer = setup_model_and_tokenizer()
config = get_training_config()
trainer = setup_trainer(model, tokenizer, train_data, val_data, config)

print("Model and trainer initialized successfully!")""",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "source": """# Start training
trainer.train()
save_model(model, tokenizer, 'training/models')
print("Training completed and model saved!")""",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "source": """# Evaluate model
results = evaluate_model(model, tokenizer)
for result in results:
    print(f"\\nQuery: {result['query']}")
    print(f"Response: {result['response']}")""",
            },
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    # Ensure the training/notebooks directory exists
    os.makedirs("training/notebooks", exist_ok=True)

    # Write the notebook
    with open("training/notebooks/mistral_finetuning.ipynb", "w") as f:
        json.dump(notebook, f, indent=2)


if __name__ == "__main__":
    create_notebook()
    print("Notebook created successfully!")
