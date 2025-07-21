# Adaptive Global LCA Advisor

## Overview

The Adaptive Global LCA Advisor is an AI-driven framework for delivering precise, region-specific emission factor (EF) recommendations to support accurate carbon accounting across global supply chains. This system addresses critical limitations in existing solutions, such as manual EF selection errors (15-30%), static datasets, and limited regional coverage.

Our system integrates:

- A fine-tuned Mistral-7B large language model (LLM) distilled into Phi-2
- A Neo4j knowledge graph for structured data management
- A Qdrant-based retrieval-augmented generation (RAG) pipeline

Key performance metrics:

- 87.2% Precision@3
- 4.8% Mean Absolute Percentage Error (MAPE)
- 148ms latency
- Coverage of 44+ regions globally

## System Architecture

```
┌────────────┐      ┌────────────┐      ┌────────────┐
│  Streamlit │      │   Mistral  │      │ Climate    │
│  Interface │◄────►│   Model    │◄────►│ TRACE      │
└────────────┘      └────────────┘      └────────────┘
      ▲                    ▲                   ▲
      │                    │                   │
      ▼                    ▼                   ▼
┌────────────┐      ┌────────────┐      ┌────────────--┐
│    Phi-2   │      │   Qdrant   │      │   Neo4j      │
│  Embeddings│◄───► │  Vector DB │◄────►│  Knowledge   │
└────────────┘      └────────────┘      └────────────--┘
```

Our system architecture illustrates the interactions between:

- **Streamlit frontend** for user interaction and query input
- **Phi-2 model** for efficient query embedding (267MB quantized model)
- **Qdrant vector database** for fast similarity search (23,520 embeddings)
- **Neo4j knowledge graph** for structured data validation
- **Mistral-7B model** for natural language response generation
- **Climate TRACE** integration for weekly dynamic data updates

_Note: Consider adding a proper system architecture diagram to the repository at images/system_architecture.png_

## Key Features

- **Global Coverage**: Supports 44+ regions with region-specific emission factors
- **Dynamic Data Integration**: Weekly updates from Climate TRACE for real-time accuracy
- **Edge Deployment**: Quantized 267MB Phi-2 model for resource-constrained environments
- **Comprehensive Dataset**: Harmonized data from multiple sources:
  - Agribalyse 3.1 (2,793 food EFs)
  - USEEIO v2.1 (13,561 industrial EFs)
  - EXIOBASE 3.8 (1,030 multi-regional EFs)
  - OpenLCA (961 process-based EFs)
  - IPCC AR6 (10,769 climate metrics)
  - IPCC EFDB (191 specific EFs)
  - GREET Model (234 transport EFs)
  - Climate TRACE (4,681 real-time EFs)
- **User-Friendly Interface**: Streamlit-based UI with natural language queries
- **High Performance**: 150ms average latency, handling 50 concurrent queries

## Data Pipeline

```
┌────────────────────────────────────────────────────┐
│           Data Pipeline Overview                   │
│                                                    │
│  ┌─────────────┐      ┌─────────────┐              │
│  │  Data       │      │             │              │
│  │  Extraction │─────►│  Cleaning   │              │
│  │             │      │             │              │
│  └─────────────┘      └─────────────┘              │
│         ▲                    │                     │
│         │                    ▼                     │
│  ┌─────────────┐      ┌─────────────┐              │
│  │             │      │             │              │
│  │  Sources    │      │Harmonization│              │
│  │             │      │             │              │
│  └─────────────┘      └─────────────┘              │
│                              │                     │
│                              ▼                     │
│  ┌─────────────────────────────────────────────┐   │
│  │                                             │   │
│  │            Neo4j Knowledge Graph            │   │
│  │                                             │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│           Data Sources (23,520 Total Records)      │
├────────────────┬─────────────┬────────────────────-┤
│ Source         │ Records     │ Focus Area          │
├────────────────┼─────────────┼────────────────────-┤
│ USEEIO v2.1    │ 13,561      │ Industrial          │
│ Climate TRACE  │ 4,681       │ Real-time emissions │
│ Agribalyse 3.1 │ 2,793       │ Food products       │
│ IPCC AR6       │ 10,769      │ Climate metrics     │
│ EXIOBASE 3.8   │ 1,030       │ Multi-regional      │
│ OpenLCA        │ 961         │ Process-based       │
│ GREET Model    │ 234         │ Transportation      │
│ IPCC EFDB      │ 191         │ Specific sectors    │
└────────────────┴─────────────┴────────────────────-┘
```

The data pipeline constructs a Neo4j knowledge graph by aggregating and harmonizing EF data from diverse sources. Key processing steps include:

1. **Unit Normalization**: All EFs standardized to kg CO2e per activity
2. **Deduplication**: Removal of 10,700+ redundant records
3. **Regional Adjustment**: IPCC AR6 multipliers adjust EFs for regional variations
4. **Outlier Detection**: Z-score analysis to correct or exclude 474 outliers (2.0% of records)
5. **Imputation**: Missing EFs for niche activities interpolated from similar regional data

The resulting 23,520 records form a Neo4j graph with nodes (activities, regions, EFs) and relationships, enabling Cypher queries in <50ms.

_Note: Consider adding data pipeline diagrams to the repository at images/data_pipeline.png_

## Model Architecture

```
┌────────────────────────────────────────────────────┐
│                                                    │
│                    Fine-Tuning                     │
│                                                    │
│  ┌──────────────┐           ┌──────────────────┐   │
│  │              │           │                  │   │
│  │  Mistral-7B  │──────────►│  LoRA (rank=16)  │   │
│  │              │           │                  │   │
│  └──────────────┘           └──────────────────┘   │
│                                       │            │
│                                       ▼            │
│                          ┌────────────────────┐    │
│                          │   Fine-tuned       │    │
│                          │    Mistral-7B      │    │
│                          └────────────────────┘    │
│                                       │            │
│                                       ▼            │
│                             Distillation           │
│                                       │            │
│                                       ▼            │
│                          ┌────────────────────┐    │
│                          │                    │    │
│                          │      Phi-2         │    │
│                          │  (2.7B parameters) │    │
│                          │                    │    │
│                          └────────────────────┘    │
│                                       │            │
│                                       ▼            │
│                             4-bit NF4 Quantization │
│                                       │            │
│                                       ▼            │
│                          ┌────────────────────┐    │
│                          │   Optimized Phi-2  │    │
│                          │     (267MB size)   │    │
│                          └────────────────────┘    │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Fine-Tuning

Our system employs two large language models:

- **Mistral-7B** for context-aware response generation
- **Phi-2** for efficient query embedding

Mistral-7B is fine-tuned using Low-Rank Adaptation (LoRA) with a rank of 16 on 12,000 instruction-based question-answer pairs derived from the knowledge graph. Fine-tuning is performed on an NVIDIA A100 GPU with 4-bit quantization, achieving a 4.8% MAPE on the validation set.

### Model Distillation

To enhance efficiency for edge deployment, Mistral-7B is distilled into Phi-2, a compact model with 2.7 billion parameters and a size of 267MB. The distillation process minimizes the Kullback-Leibler (KL) divergence between the teacher (Mistral-7B) and student (Phi-2) outputs, retaining 98% of Mistral-7B's performance with a MAPE of 4.9%.

### Quantization

Phi-2 is further quantized to 4-bit precision using the NormalFloat4 (NF4) quantization method, reducing its size while maintaining high performance. The quantized model achieves a MAPE of 5.1% and an inference latency of 120ms on a 4GB RAM device.

_Note: Consider adding model architecture diagrams to the repository at images/model_architecture.png_

## RAG Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  User Query: "What is the emission factor for wheat in France?"    │
│                                                                    │
│  ┌─────────────────┐                                               │
│  │                 │                                               │
│  │  Query Embedding│                                               │
│  │  (Phi-2 Model)  │                                               │
│  │                 │                                               │
│  └─────────────────┘                                               │
│           │                                                        │
│           ▼                                                        │
│  ┌─────────────────┐     ┌─────────────────────────────────┐       │
│  │                 │     │                                 │       │
│  │  Vector Search  │────►│  Qdrant Database                │       │
│  │                 │     │  (23,520 EF Embeddings)         │       │
│  └─────────────────┘     │                                 │       │
│           │              └─────────────────────────────────┘       │
│           ▼                                                        │
│  ┌─────────────────┐     ┌─────────────────────────────────┐       │
│  │                 │     │                                 │       │
│  │  KG Validation  │────►│  Neo4j Knowledge Graph          │       │
│  │                 │     │                                 │       │
│  └─────────────────┘     └─────────────────────────────────┘       │
│           │                                                        │
│           ▼                                                        │
│  ┌─────────────────┐                                               │
│  │                 │                                               │
│  │  Response Gen   │                                               │
│  │  (Mistral-7B)   │                                               │
│  │                 │                                               │
│  └─────────────────┘                                               │
│           │                                                        │
│           ▼                                                        │
│  "The emission factor for wheat in France is 0.31 kg CO2e/kg       │
│   with a confidence score of 0.95. This data comes from            │
│   Agribalyse 3.1 and was last updated in 2023."                    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

The Retrieval-Augmented Generation (RAG) pipeline ensures accurate and low-latency EF recommendations:

1. **Query Embedding**: User queries embedded into dense vectors using Phi-2
2. **Vector Search**: Embeddings searched against Qdrant database with 23,520 EF vectors
3. **Knowledge Graph Validation**: Retrieved candidates validated against Neo4j graph
4. **Response Generation**: Validated candidates passed to Mistral-7B for natural language response

The pipeline achieves an end-to-end latency of ~150ms, making it suitable for real-time applications.

_Note: Consider adding a RAG pipeline diagram to the repository at images/rag_pipeline.png_

## User Interface

The Streamlit-based user interface enhances accessibility for stakeholders including supply chain managers, auditors, and policymakers.

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ┌───────────────┐   Adaptive Global LCA Advisor    │
│  │               │                                  │
│  │    Query      │   [ What is the EF for cement in │
│  │    Input      │     Germany?                    ]│
│  │               │                                  │
│  └───────────────┘   [Search]                       │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │ Results                                       │  │
│  │                                               │  │
│  │ Emission Factor: 0.58 kg CO2e/kg              │  │
│  │ Confidence: 92%                               │  │
│  │ Source: IPCC AR6                              │  │
│  │ Region: Germany                               │  │
│  │                                               │  │
│  │ [View Graph] [Export Results] [Compare]       │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │                                               │  │
│  │              [Chart Visualization]            │  │
│  │                                               │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

Key features include:

- Natural language queries and dropdown-based inputs
- Comprehensive result display with metadata (EF, confidence, source, uncertainty)
- Interactive Plotly charts for visualization
- Knowledge graph excerpts for transparency
- Export functionality (CSV, PDF reports)
- User authentication and role-based access control

_Note: Consider adding actual UI screenshots to the repository at images/streamlit_interface.png and images/admin_dashboard.png_

## Performance Evaluation

```
┌─────────────────────────────────────────────────────┐
│ Key Performance Metrics                             │
├───────────────────┬─────────────────────────────────┤
│ Precision@3       │ 87.2%                           │
│ MAPE              │ 4.8%                            │
│ Latency           │ ~150ms                          │
│ Regions Covered   │ 44+                             │
│ Concurrent Queries│ 50 (235ms), 100 (300ms)         │
│ SUS Score         │ 82/100                          │
└───────────────────┴─────────────────────────────────┘
```

The system demonstrates robust performance across diverse metrics:

- **Precision@3**: 87.2% (matching ground truth from EXIOBASE 3.8 and IPCC AR6)
- **MAPE**: 4.8% (compared to manual methods' 15-30% error rates)
- **Latency**: ~150ms end-to-end
- **Scalability**: Handles 50 concurrent queries, degrades to 300ms at 100 queries
- **User Experience**: System Usability Scale (SUS) score of 82 (above the 68 benchmark)

### Case Studies

```
┌────────────────────────────────────────────────────┐
│ Region-Specific Emission Factor Examples           │
├────────────────┬──────────────┬────────────────────┤
│ Product/Process│ Region       │ EF (kg CO2e/unit)  │
├────────────────┼──────────────┼────────────────────┤
│ Wheat          │ France       │ 0.31 kg/kg         │
│ Wheat          │ India        │ 0.45 kg/kg         │
│ Cement         │ Germany      │ 0.58 kg/kg         │
│ Steel          │ China        │ 1.92 kg/kg         │
│ Rice           │ Thailand     │ 2.80 kg/kg         │
│ Diesel Fuel    │ United States│ 2.68 kg/liter      │
│ Electricity    │ Brazil       │ 0.12 kg/kWh        │
└────────────────┴──────────────┴────────────────────┘
```

These case studies validate the system's adaptability across regions and sectors, with results consistently matching reference databases like IPCC AR6, EXIOBASE 3.8, and the GREET Model.

_Note: Consider adding performance visualization charts to the repository at images/performance_metrics.png_

## Installation and Usage

### Prerequisites

- Python 3.9+
- NVIDIA GPU (recommended for model fine-tuning)
- Neo4j Database
- Qdrant Vector Database

### Installation

```bash
# Clone the repository
git clone https://github.com/Sbursu/Carbon-EF.git
cd Carbon-EF

# Install dependencies
pip install -r requirements.txt

# Additional dependencies for enhanced PDF extraction
pip install "camelot-py[cv]" PyMuPDF
```

### Running the System

```bash
# Start the Streamlit interface
streamlit run app.py
```

### Accessing the Model

The quantized Phi-2 model is available on Hugging Face:
[https://huggingface.co/Surendra-Aitest/phi2-env-factors-merged](https://huggingface.co/Surendra-Aitest/phi2-env-factors-merged)

## Data Sources

The system harmonizes data from multiple sources:

- **Agribalyse 3.1**: French agricultural product emission factors
- **USEEIO v2.1**: US environmentally-extended input-output model
- **EXIOBASE 3.8**: Multi-regional input-output database
- **OpenLCA**: Process-based LCA data
- **IPCC AR6**: Enhanced regional multipliers
- **IPCC EFDB**: Emission factors for various sectors
- **GREET Model**: Transportation fuel lifecycle emissions
- **Climate TRACE**: Real-time emissions data (updated weekly)

## Project Structure

```
data/
├── raw/                # Raw data files from sources
├── interim/            # Intermediate processed data
├── processed/          # Final cleaned datasets
├── scripts/            # Python scripts for processing
│   ├── extractors/     # Dataset-specific extractors
│   ├── harmonization/  # Data harmonization modules
│   ├── main.py         # Pipeline execution script
│   └── utils.py        # Utility functions
├── logs/               # Log files
└── documentation/      # Documentation

models/
├── mistral/            # Mistral-7B fine-tuned model
├── phi2/               # Distilled and quantized Phi-2
└── training/           # Training scripts and logs

neo4j/
├── scripts/            # Neo4j database setup scripts
└── cypher/             # Cypher queries

qdrant/
├── config/             # Qdrant configuration
└── scripts/            # Vector database setup

streamlit/
├── app.py              # Main Streamlit application
├── pages/              # Additional UI pages
└── components/         # UI components
```

## Future Work

Planned enhancements include:

- Expanding regional coverage to 100+ countries
- Integrating real-time APIs for dynamic updates
- Optimizing computational efficiency for mobile devices
- Enhancing explainability with SHAP and other methods
- Incorporating multi-modal data (e.g., satellite imagery)

## Citation

If you use this system in your research, please cite:

```
@inproceedings{burusu2025adaptive,
  title={Adaptive Global LCA Advisor: A Region-Specific Emission Factor Recommendation System with Dynamic Retrieval for Accurate Carbon Accounting},
  author={Burusu, Surendra and Chaduvu, Vinith and Bondre, Ankita},
  booktitle={Proceedings of the International Conference on Sustainable Computing},
  year={2025},
  organization={ACM}
}
```

## Contributors

- Surendra Burusu - Yeshiva University
- Vinith Chaduvu - Yeshiva University
- Ankita Bondre - Yeshiva University

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the following organizations for providing datasets and support:

- ADEME for Agribalyse 3.1
- EPA for USEEIO v2.1
- Climate TRACE Consortium
- IPCC Data Distribution Centre
- Argonne National Laboratory for GREET Model

```

```

```

```
