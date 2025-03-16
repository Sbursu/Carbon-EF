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
            "source": "# Install core dependencies\n!pip install -q transformers==4.36.2 datasets==2.16.1 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.3 trl==0.7.11 wandb==0.16.3\n!pip install -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118\n\n# Install neo4j for database access (optional, used only if Neo4j data source is enabled)\n!pip install -q neo4j==5.10.0",
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
            "source": '# First, make sure we\'re in the content directory\nimport os\n\n# Start with a clean slate - remove any existing Carbon-EF directory\n!rm -rf Carbon-EF\n\n# Clone the repository fresh\n!git clone https://github.com/Sbursu/Carbon-EF.git\n\n# Navigate to the repository directory\n%cd Carbon-EF\n\n# Create logs directory to prevent errors\n!mkdir -p training/logs\n\n# Verify directory structure\n!pwd\n!ls -la training/scripts\n\n# Add repository root to Python path\nimport sys\nsys.path.append(os.getcwd())\n\n# Modify logging in data_preparation at runtime to avoid errors\ndata_prep_path = \'training/scripts/data_preparation.py\'\nif os.path.exists(data_prep_path):\n    with open(data_prep_path, \'r\') as f:\n        code = f.read()\n    \n    if "__name__ == \\"__main__\\"" in code:\n        # Add code that makes the logging path dynamic\n        with open(data_prep_path, \'a\') as f:\n            f.write("""\n# Add runtime logging patch to avoid errors\ndef configure_logging(base_dir=None):\n    \'\'\'Configure logging with dynamic base directory\'\'\'\n    if base_dir is None:\n        base_dir = os.getcwd()\n    \n    log_dir = os.path.join(base_dir, \'training/logs\')\n    os.makedirs(log_dir, exist_ok=True)\n    \n    logging.basicConfig(\n        level=logging.INFO,\n        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",\n        handlers=[\n            logging.FileHandler(os.path.join(log_dir, "data_preparation.log")),\n            logging.StreamHandler(),\n        ],\n    )\n    return log_dir\n""")\n\n# Import necessary modules with error handling\ntry:\n    from training.scripts.data_preparation import load_and_prepare_data, format_instruction\n    from training.scripts.model_config import setup_model_and_tokenizer, get_training_config\n    from training.scripts.training import setup_trainer, evaluate_model, save_model\n    print("Successfully imported all required modules")\nexcept ImportError as e:\n    print(f"Import error: {e}")\n    print("Please check that all required packages are installed")',
        },
        {"cell_type": "markdown", "metadata": {}, "source": "## Prepare Training Data"},
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": '# Check if data files exist\ndata_files = {\n    "train": "training/data/instructions_train.json",\n    "val": "training/data/instructions_val.json",\n    "test": "training/data/instructions_test.json"\n}\n\nfor split, file_path in data_files.items():\n    if os.path.exists(file_path):\n        print(f"Found {split} data: {file_path}")\n    else:\n        print(f"Warning: {file_path} not found")\n\n# Load and prepare data\ntry:\n    # Use file-based loading (don\'t use Neo4j in Colab)\n    train_data, val_data = load_and_prepare_data(use_neo4j=False)\n    \n    # Format data for training\n    train_data = train_data.map(format_instruction)\n    val_data = val_data.map(format_instruction)\n    \n    # Print summary\n    print(f"Training examples: {len(train_data[\'train\'])}")\n    print(f"Validation examples: {len(val_data[\'train\'])}")\n    \n    # Show sample\n    print("\\nSample training example:")\n    print(train_data["train"][0]["text"][:300] + "...")\nexcept Exception as e:\n    print(f"Error preparing data: {e}")\n    print("Please check that the data files exist and are properly formatted")',
        },
        {"cell_type": "markdown", "metadata": {}, "source": "## Initialize Model"},
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": '# Set up model and tokenizer\ntry:\n    model, tokenizer = setup_model_and_tokenizer()\n    print("Model and tokenizer successfully initialized")\n    \n    # Get training configuration\n    config = get_training_config()\n    print("\\nTraining configuration:")\n    for key, value in config.items():\n        print(f"  {key}: {value}")\n    \n    # Set up trainer\n    trainer = setup_trainer(model, tokenizer, train_data, val_data, config)\n    print("\\nTrainer set up successfully")\nexcept Exception as e:\n    print(f"Error setting up model: {e}")\n    print("Please check your GPU availability and memory")',
        },
        {"cell_type": "markdown", "metadata": {}, "source": "## Start Training"},
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": '# Start training\ntry:\n    print("Starting training...")\n    trainer.train()\n    print("Training completed successfully!")\n    \n    # Save model\n    save_model(model, tokenizer, config[\'output_dir\'])\n    print(f"Model saved to {config[\'output_dir\']}/final_model")\nexcept Exception as e:\n    print(f"Error during training: {e}")\n    print("\\nTroubleshooting tips:")\n    print("1. Check if you have enough VRAM (T4 or better GPU recommended)")\n    print("2. Try reducing batch size or gradient accumulation steps")',
        },
        {"cell_type": "markdown", "metadata": {}, "source": "## Evaluate Model"},
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": '# Run evaluation\ntry:\n    print("Running evaluation...")\n    results = evaluate_model(model, tokenizer)\n    \n    # Display results\n    print("\\nEvaluation results:")\n    for result in results:\n        print(f"\\nQuery: {result[\'query\']}")\n        print(f"Response: {result[\'response\']}")\n        print()\nexcept Exception as e:\n    print(f"Error during evaluation: {e}")',
        },
        {"cell_type": "markdown", "metadata": {}, "source": "## Test Your Own Queries"},
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": 'from training.scripts.training import generate_recommendation\n\nquery = "What is the emission factor for cement production in India?"\ntry:\n    response = generate_recommendation(model, tokenizer, query)\n    print(f"Query: {query}")\n    print(f"Response: {response}")\nexcept Exception as e:\n    print(f"Error generating recommendation: {e}")',
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
