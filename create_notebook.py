#!/usr/bin/env python3
import json

# Create a minimal notebook with essential cells
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Mistral-7B Fine-Tuning\n\nThis notebook implements fine-tuning of Mistral-7B for emission factor recommendations.\n\n## Setup\n1. Select Runtime > Change runtime type and choose GPU\n2. Run cells in sequence",
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": "# Check GPU availability\n!nvidia-smi",
        },
        {"cell_type": "markdown", "metadata": {}, "source": "## Install Dependencies"},
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": "!pip install -q transformers==4.36.2 datasets==2.16.1 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.3 trl==0.7.11 wandb==0.16.3\n!pip install -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118",
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Clone Repository and Import Scripts",
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": "!git clone https://github.com/Sbursu/Carbon-EF.git\n%cd Carbon-EF\n\n# Add repository root to Python path\nimport os\nimport sys\nsys.path.append(os.getcwd())\n\nfrom training.scripts.data_preparation import load_and_prepare_data, format_instruction\nfrom training.scripts.model_config import setup_model_and_tokenizer, get_training_config\nfrom training.scripts.training import setup_trainer, evaluate_model, save_model",
        },
        {"cell_type": "markdown", "metadata": {}, "source": "## Prepare Training Data"},
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": '# Load and prepare data\ntrain_data, val_data = load_and_prepare_data()\n\n# Format data for training\ntrain_data = train_data.map(format_instruction)\nval_data = val_data.map(format_instruction)\n\n# Print summary\nprint("Training examples:", len(train_data["train"]))',
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

# Write the notebook to a file
with open("training/notebooks/mistral_finetuning.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook created successfully!")
