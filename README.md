# Carbon Emission Factor System

A hybrid Retrieval-Augmented Generation (RAG) system to provide accurate information about carbon emission factors using Neo4j and Qdrant for GraphRAG.

## System Overview

This system implements a hybrid GraphRAG architecture combining:

1. **Graph Database (Neo4j)**: Stores emission factors as a knowledge graph with relationships
2. **Vector Database (Qdrant)**: Stores embeddings for semantic search capabilities
3. **Phi-2 Agent**: Processes queries and reasons over the retrieved contexts
4. **API Service**: Exposes endpoints for querying the system
5. **Streamlit Interface**: Provides a user-friendly web interface

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Git

### Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd Carbon\ EF
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

```

```

```

```

3. Start the Docker containers:

```

docker-compose up -d neo4j qdrant

```

### Running the Application

#### Option 1: Using Docker (Recommended)

Start all services using Docker Compose:

```

docker-compose up -d

```

This will start:

- Neo4j database on port 7687 (Bolt) and 7474 (Browser)
- Qdrant vector database on port 6333
- API service on port 8000
- Streamlit interface on port 8501

Access the services:

- Streamlit interface: http://localhost:8501
- API documentation: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474

#### Option 2: Local Development

1. Start the API server:

```

# Run the optimized slim API for improved reliability

python slim_api.py

# Or run the full API (may require more resources)

python -m src.api.main

```

2. Start the Streamlit interface:

```

streamlit run src/streamlit/app.py

```

## Troubleshooting

### Common Issues

#### API Fails to Start

1. **Check all dependencies are installed:**

```

python troubleshoot.py

```

2. **Verify Neo4j connection:**

- Ensure Neo4j is running: `docker ps | grep neo4j`
- Check Neo4j logs: `docker logs carbon_ef_neo4j`
- Test connection directly: `python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'carbon_ef_secure')); with driver.session() as session: result = session.run('RETURN 1'); print(result.single()); driver.close()"`

3. **Memory issues:**

- Run the model optimization tool to check resource usage:

```

python optimize_models.py

```

- Try the slim API: `python slim_api.py`

#### Model Loading Issues

1. **Run the verification script:**

```

python verify_imports.py

```

2. **Check model cache:**

- Ensure the HuggingFace cache directory exists and has sufficient permissions
- Try downloading models manually: `python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/phi-2')"`

3. **Reduce model size:**

- Edit `configs/config.yaml` to use a smaller model or enable quantization

### Optimization

To optimize the application for your environment:

1. Run the optimization script:

```

python optimize_models.py

```

2. Consider using the slimmer API implementation (`slim_api.py`) which:

- Lazy loads models to reduce memory usage
- Provides fallbacks for component failures
- Handles errors gracefully

## Architecture

### Components

- `src/api/`: API server implementation
- `src/graph_rag/`: GraphRAG implementation (Neo4j + Qdrant)
- `src/agent/`: Phi-2 based agent for reasoning
- `src/streamlit/`: Streamlit web interface
- `src/vector_store/`: Vector database integration
- `src/neo4j/`: Neo4j database integration
- `src/cache/`: Semantic caching implementation
- `configs/`: Configuration files
- `docker/`: Docker configuration files
- `neo4j/`: Neo4j plugins and data

### Key Files

- `docker-compose.yml`: Docker services configuration
- `slim_api.py`: Optimized API server implementation
- `optimize_models.py`: Tool for optimizing model loading
- `troubleshoot.py`: Troubleshooting utility
- `verify_imports.py`: Import verification script

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft Phi-2 model
- Neo4j Graph Database
- Qdrant Vector Database
- FastAPI
- Streamlit

```

```

```

```
