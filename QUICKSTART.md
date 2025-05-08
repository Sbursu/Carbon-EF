# Quick Start Guide: Carbon EF

This guide provides the fastest way to get the Carbon EF system up and running with minimal issues.

## Option 1: Run with Docker (Recommended)

The simplest way to run the entire system is with Docker Compose:

```bash
# Start all services
docker-compose up -d

# Check if services are running
docker-compose ps
```

Access the application:

- Web interface: http://localhost:8501
- API: http://localhost:8000
- Neo4j Browser: http://localhost:7474 (username: neo4j, password: carbon_ef_secure)

## Option 2: Run Locally (Development Mode)

### 1. Start Neo4j with Docker

```bash
# Start only Neo4j database
docker-compose up -d neo4j

# Wait for Neo4j to be ready
docker-compose logs -f neo4j
```

### 2. Run the Slim API

```bash
# Make sure you have dependencies installed
pip install -r requirements.txt

# Run the optimized API (better for reliability)
python slim_api.py
```

The slim API provides a more reliable starting point with:

- Lazy loading of models
- Graceful handling of component failures
- Minimal memory usage

### 3. Run the Streamlit Interface

```bash
# In a new terminal window
streamlit run src/streamlit/app.py
```

## Troubleshooting

If you encounter issues:

1. **Run the verification script**:

   ```bash
   python verify_imports.py
   ```

   This will check for missing dependencies and test Neo4j connectivity.

2. **Run the optimization tool**:

   ```bash
   python optimize_models.py
   ```

   This will help diagnose model loading issues and memory usage.

3. **Common issues and fixes**:

   - **Neo4j connection errors**:

     ```bash
     # Check if Neo4j is running
     docker ps | grep neo4j

     # Restart Neo4j if needed
     docker-compose restart neo4j
     ```

   - **Memory errors**:

     ```bash
     # Edit configs/config.yaml to reduce model size
     # Or use the slim API
     python slim_api.py
     ```

   - **Import errors**:
     ```bash
     # Run the troubleshooting tool
     python troubleshoot.py
     ```

## Key API Endpoints

- `GET /health`: Check API health
- `POST /query`: Process a natural language query

Example query:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the emission factor for electricity in Germany?"}'
```

## Minimal Configuration

To customize the basic settings, edit `configs/config.yaml`:

```yaml
# Minimal configuration
neo4j:
  uri: bolt://localhost:7687
  user: neo4j
  password: carbon_ef_secure

qdrant:
  host: localhost
  port: 6333
  collection_name: emission_factors

api:
  port: 8000
```
