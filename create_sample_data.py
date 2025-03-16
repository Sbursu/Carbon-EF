#!/usr/bin/env python3
import json
import os

# Create data directory if it doesn't exist
os.makedirs("training/data", exist_ok=True)

# Sample instruction data
sample_instructions = [
    {
        "instruction": "What is the emission factor for cement production in USA?",
        "input": "",
        "output": "The emission factor for cement production in the USA is 0.92 kg CO2e/kg. This data is sourced from USEEIO_v2.1.",
        "metadata": {
            "regions": ["USA"],
            "entity_types": ["product"],
            "difficulty": "basic",
            "sources": ["USEEIO_v2.1"],
        },
    },
    {
        "instruction": "Compare the emission factor for wheat production between France and the USA.",
        "input": "",
        "output": "The emission factor for wheat production in France is 0.38 kg CO2e/kg, while in the USA it is 0.41 kg CO2e/kg. The USA has a slightly higher emission factor (by 7.9%). This data is sourced from Agribalyse_3.1 for France and USEEIO_v2.1 for the USA.",
        "metadata": {
            "regions": ["FR", "USA"],
            "entity_types": ["product"],
            "difficulty": "moderate",
            "sources": ["Agribalyse_3.1", "USEEIO_v2.1"],
        },
    },
    {
        "instruction": "What are the emission factors for the top 3 contributors to agricultural emissions in Europe?",
        "input": "",
        "output": "The top 3 contributors to agricultural emissions in Europe are: 1) Beef production at 22.9 kg CO2e/kg, 2) Dairy farming at 1.4 kg CO2e/kg of milk, and 3) Rice cultivation at 3.7 kg CO2e/kg. These figures are from EXIOBASE_3.8 and represent EU average values.",
        "metadata": {
            "regions": ["EU"],
            "entity_types": ["product", "sector"],
            "difficulty": "complex",
            "sources": ["EXIOBASE_3.8"],
        },
    },
    {
        "instruction": "How should I adjust the global emission factor for rice production when using it in Thailand?",
        "input": "",
        "output": "The global emission factor for rice production is 2.8 kg CO2e/kg, but for Thailand, you should apply a regional multiplier of 1.32, resulting in an adjusted factor of 3.7 kg CO2e/kg. This is due to specific cultivation practices and methane emissions from flooded rice fields in the region. Source: IPCC AR6 regional multipliers.",
        "metadata": {
            "regions": ["GLB", "THA"],
            "entity_types": ["product"],
            "difficulty": "methodological",
            "sources": ["IPCC_AR6"],
        },
    },
    {
        "instruction": "What emission factor should I use for a new plant-based leather alternative?",
        "input": "",
        "output": "For a new plant-based leather alternative, you can use 2.1 kg CO2e/kg as a proxy based on similar plant-based materials. This is significantly lower than animal leather (17.0 kg CO2e/kg). However, this is an estimate and should be used with a confidence factor of 0.6. Consider conducting a specific LCA study for your exact production process. Source: OpenLCA database.",
        "metadata": {
            "regions": ["GLB"],
            "entity_types": ["product"],
            "difficulty": "edge_case",
            "sources": ["OpenLCA"],
        },
    },
]

# Write sample data files if they don't exist
data_files = {
    "training/data/instructions_train.json": sample_instructions[:3],
    "training/data/instructions_val.json": sample_instructions[3:4],
    "training/data/instructions_test.json": sample_instructions[4:],
}

for file_path, data in data_files.items():
    if not os.path.exists(file_path):
        print(f"Creating sample data file: {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        print(f"File already exists: {file_path}")

print("Sample data files created/checked.")
