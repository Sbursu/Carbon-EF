#!/bin/bash

# Define colors for output
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# Define default ports
API_PORT=8000
STREAMLIT_PORT=8501
API_HOST="localhost"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --api-port)
      API_PORT="$2"
      shift 2
      ;;
    --streamlit-port)
      STREAMLIT_PORT="$2"
      shift 2
      ;;
    --api-host)
      API_HOST="$2"
      shift 2
      ;;
    --help)
      echo -e "${BLUE}Carbon EF Demo Runner${NC}"
      echo "Usage: ./run_demo.sh [options]"
      echo ""
      echo "Options:"
      echo "  --api-port PORT       Port for the API server (default: 8000)"
      echo "  --streamlit-port PORT Port for the Streamlit interface (default: 8501)"
      echo "  --api-host HOST       Host for the API server (default: localhost)"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run './run_demo.sh --help' for usage information."
      exit 1
      ;;
  esac
done

echo -e "${GREEN}=== Carbon EF Demo Runner ===${NC}"
echo -e "${BLUE}Starting API server on ${API_HOST}:${API_PORT}${NC}"
echo -e "${BLUE}Starting Streamlit interface on port ${STREAMLIT_PORT}${NC}"

# Create necessary directories if they don't exist
mkdir -p logs
mkdir -p src/cache src/vector_store
touch src/cache/__init__.py
touch src/vector_store/__init__.py

# Create a virtual environment if it doesn't exist
if [ ! -d "carbon_ef_env" ]; then
  echo -e "${YELLOW}Setting up virtual environment...${NC}"
  python -m venv carbon_ef_env
  source carbon_ef_env/bin/activate
  echo -e "${YELLOW}Installing dependencies...${NC}"
  pip install -r requirements.txt
  
  # Fix known dependency issues
  echo -e "${YELLOW}Fixing dependencies...${NC}"
  pip install huggingface_hub==0.12.0
  pip install --upgrade sentence-transformers
else
  echo -e "${YELLOW}Activating existing environment...${NC}"
  source carbon_ef_env/bin/activate
fi

# Load environment variables if .env file exists
if [ -f .env ]; then
  echo -e "${YELLOW}Loading environment variables from .env${NC}"
  set -a
  source .env
  set +a
fi

# Set Neo4j environment variables if not already set
export NEO4J_URI=${NEO4J_URI:-"bolt://localhost:7687"}
export NEO4J_USERNAME=${NEO4J_USERNAME:-"neo4j"}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-"password"}

# Function to check if a port is available
check_port() {
  local port=$1
  if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
    return 1
  else
    return 0
  fi
}

# Check if ports are available
if ! check_port $API_PORT; then
  echo -e "${RED}Error: Port $API_PORT is already in use. Please choose a different port.${NC}"
  exit 1
fi

if ! check_port $STREAMLIT_PORT; then
  echo -e "${RED}Error: Port $STREAMLIT_PORT is already in use. Please choose a different port.${NC}"
  exit 1
fi

# Start the API server
echo -e "${YELLOW}Starting API server...${NC}"
python -m src.api.server --host $API_HOST --port $API_PORT --reload > logs/api.log 2>&1 &
API_PID=$!

# Wait for the API server to start
echo -e "${YELLOW}Waiting for API server to start...${NC}"
API_STARTED=false
MAX_ATTEMPTS=30
ATTEMPTS=0

while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
  if curl -s http://${API_HOST}:${API_PORT}/health > /dev/null; then
    API_STARTED=true
    break
  fi
  sleep 1
  ATTEMPTS=$((ATTEMPTS+1))
  echo -e "${YELLOW}Waiting for API server (attempt $ATTEMPTS/$MAX_ATTEMPTS)...${NC}"
done

if [ "$API_STARTED" = false ]; then
  echo -e "${RED}Error: API server failed to start. Check logs/api.log for details.${NC}"
  kill $API_PID 2>/dev/null
  exit 1
fi

echo -e "${GREEN}API server started successfully! (PID: $API_PID)${NC}"

# Start the Streamlit interface
echo -e "${YELLOW}Starting Streamlit interface...${NC}"
python -m src.streamlit.run --port $STREAMLIT_PORT --api-host $API_HOST --api-port $API_PORT > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!

echo -e "${GREEN}Streamlit interface starting! (PID: $STREAMLIT_PID)${NC}"
echo -e "${GREEN}To access the interface, open your browser and navigate to:${NC}"
echo -e "${BLUE}http://localhost:${STREAMLIT_PORT}${NC}"

# Function to handle script termination
cleanup() {
  echo -e "\n${YELLOW}Shutting down services...${NC}"
  kill $API_PID 2>/dev/null
  kill $STREAMLIT_PID 2>/dev/null
  echo -e "${GREEN}Services stopped.${NC}"
  exit 0
}

# Set up trap for clean shutdown
trap cleanup SIGINT SIGTERM

# Keep the script running
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
tail -f logs/api.log logs/streamlit.log
