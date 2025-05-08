import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import requests

import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Page configuration
st.set_page_config(
    page_title="Carbon Emission Factor Lookup",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2196F3;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        background-color: #f9f9f9;
    }
    .ef-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #FF5722;
    }
    .ef-unit {
        font-size: 0.9rem;
        color: #616161;
    }
    .source-tag {
        font-size: 0.8rem;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        color: white;
        background-color: #9E9E9E;
    }
    .source-tag.graph {
        background-color: #2196F3;
    }
    .source-tag.vector {
        background-color: #FF9800;
    }
    .query-box {
        background-color: #f0f8ff;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #f1f8e9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
    }
    .explanation-box {
        background-color: #fff8e1;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .graph-context-box {
        background-color: #e8eaf6;
        border-left: 5px solid #3F51B5;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        white-space: pre-wrap;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Helper functions
def call_api(
    endpoint: str, data: Dict = None, method: str = "post", retry_count: int = 0
) -> Dict:
    """Call API endpoint with retry logic."""
    url = f"{API_BASE_URL}/{endpoint}"

    try:
        if method.lower() == "get":
            response = requests.get(url, params=data, timeout=30)
        else:
            response = requests.post(url, json=data, timeout=30)

        response.raise_for_status()
        return response.json()

    except ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        if retry_count < MAX_RETRIES:
            logger.info(
                f"Retrying in {RETRY_DELAY} seconds... (Attempt {retry_count + 1}/{MAX_RETRIES})"
            )
            time.sleep(RETRY_DELAY)
            return call_api(endpoint, data, method, retry_count + 1)
        else:
            logger.error("Max retries reached. API server may be down.")
            return {
                "error": True,
                "message": "Failed to connect to API server. Please check if the server is running.",
            }

    except Timeout as e:
        logger.error(f"Timeout error: {str(e)}")
        return {
            "error": True,
            "message": "API request timed out. The server might be overloaded or experiencing issues.",
        }

    except RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return {"error": True, "message": f"API request failed: {str(e)}"}

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": True, "message": f"An unexpected error occurred: {str(e)}"}


def create_emission_factor_card(ef: Dict):
    """
    Create a card to display emission factor information
    """
    source_class = "graph" if ef.get("source") == "graph" else "vector"

    st.markdown(
        f"""
    <div class="card">
        <h3>{ef.get('entity_name', 'Unknown entity')}</h3>
        <p>Type: {ef.get('entity_type', 'Unknown type')}, Region: {ef.get('region', 'Global')}</p>
        <p class="ef-value">{ef.get('ef_value', 'N/A')} <span class="ef-unit">{ef.get('ef_unit', '')}</span></p>
        <p>Year: {ef.get('year', 'N/A')}</p>
        <span class="source-tag {source_class}">{ef.get('source', 'unknown')}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


# Sidebar
st.sidebar.markdown('<h1 class="main-header">Carbon EF</h1>', unsafe_allow_html=True)
st.sidebar.markdown("### Emission Factor Explorer")

# Page selection
page = st.sidebar.radio(
    "Select Page",
    [
        "Query Interface",
        "GraphRAG Query",
        "Emission Factors Lookup",
        "Regional Comparison",
        "About",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
use_cache = st.sidebar.checkbox("Use semantic cache", value=True)

if st.sidebar.button("Clear Cache"):
    cache_result = call_api("/cache/clear")
    if "error" not in cache_result:
        st.sidebar.success("Cache cleared successfully")
    else:
        st.sidebar.error(cache_result["error"])

# Check API connection
# api_status = check_api_health() # Commented out due to NameError
# if not api_status:
#     st.error(
#         f"Cannot connect to API server at {API_BASE_URL}. Please ensure the server is running."
#     )
#     st.stop()

# Page content
if page == "Query Interface":
    st.markdown(
        '<h1 class="main-header">Carbon Emission Factor Query Interface</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    Ask natural language questions about carbon emission factors, entity relationships, 
    and make comparisons between regions or entities.
    """
    )

    # Query input
    query = st.text_area(
        "Enter your query",
        height=100,
        placeholder="What is the emission factor for electricity in the US?",
    )

    if st.button("Submit Query"):
        if query:
            with st.spinner("Processing query..."):
                result = call_api(
                    "/query",
                    method="POST",
                    params={"use_cache": use_cache},
                    data={"query": query},
                )

                if "error" in result:
                    st.error(result["error"])
                else:
                    # Display query
                    st.markdown(
                        f"""
                    <div class="query-box">
                        <h4>Query:</h4>
                        <p>{result.get('query', query)}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Display answer
                    st.markdown(
                        f"""
                    <div class="answer-box">
                        <h4>Answer:</h4>
                        <p>{result.get('answer', 'No answer provided')}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Display explanation if available
                    if "explanation" in result and result["explanation"]:
                        with st.expander("See explanation"):
                            st.markdown(
                                f"""
                            <div class="explanation-box">
                                {result.get('explanation', '')}
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                    # Show cache info
                    if result.get("cache_used"):
                        st.info("⚡ Result retrieved from cache")
        else:
            st.warning("Please enter a query")

elif page == "GraphRAG Query":
    st.markdown(
        '<h1 class="main-header">GraphRAG Query Interface</h1>', unsafe_allow_html=True
    )
    st.markdown(
        """
    Ask complex questions that require both semantic search and graph relationships.
    GraphRAG combines Qdrant vector search with Neo4j graph database to provide more contextual answers.
    """
    )

    # Query input
    query = st.text_area(
        "Enter your query",
        height=100,
        placeholder="How are emission factors related between different regions?",
    )

    col1, col2 = st.columns(2)
    with col1:
        entity_type = st.text_input(
            "Entity Type (optional)", placeholder="e.g., electricity"
        )
    with col2:
        region = st.text_input("Region (optional)", placeholder="e.g., US")

    if st.button("Submit GraphRAG Query"):
        if query:
            with st.spinner("Processing with GraphRAG..."):
                data = {
                    "query": query,
                    "entity_type": entity_type if entity_type else None,
                    "region": region if region else None,
                }

                result = call_api("/graphrag", method="POST", data=data)

                if "error" in result:
                    st.error(result["error"])
                else:
                    # Display query
                    st.markdown(
                        f"""
                    <div class="query-box">
                        <h4>Query:</h4>
                        <p>{query}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Display answer
                    st.markdown(
                        f"""
                    <div class="answer-box">
                        <h4>Answer:</h4>
                        <p>{result.get('answer', 'No answer generated')}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Display graph context
                    with st.expander("View Graph Context"):
                        st.markdown(
                            f"""
                        <div class="graph-context-box">
                        {result.get('graph_context', 'No graph context available')}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    # Display vector results
                    with st.expander("View Vector Search Results"):
                        st.json(result.get("vector_results", []))

                    # Display subgraph data if available
                    if result.get("subgraph") and result["subgraph"].get("nodes"):
                        with st.expander("View Graph Data"):
                            tab1, tab2 = st.tabs(["Nodes", "Relationships"])

                            with tab1:
                                nodes_df = pd.json_normalize(
                                    [
                                        n["properties"]
                                        for n in result["subgraph"]["nodes"]
                                    ]
                                )
                                st.dataframe(nodes_df)

                            with tab2:
                                if result["subgraph"].get("relationships"):
                                    rels = []
                                    for rel in result["subgraph"]["relationships"]:
                                        start = (
                                            rel.get("start_node", {})
                                            .get("properties", {})
                                            .get("entity_name", "Unknown")
                                        )
                                        end = (
                                            rel.get("end_node", {})
                                            .get("properties", {})
                                            .get("entity_name", "Unknown")
                                        )
                                        rel_type = rel.get("type", "RELATED_TO")
                                        rels.append(
                                            {
                                                "Source": start,
                                                "Relationship": rel_type,
                                                "Target": end,
                                            }
                                        )

                                    if rels:
                                        st.dataframe(pd.DataFrame(rels))
                                    else:
                                        st.info("No relationships found")
                                else:
                                    st.info("No relationships found")
        else:
            st.warning("Please enter a query")

elif page == "Emission Factors Lookup":
    st.markdown(
        '<h1 class="main-header">Emission Factors Lookup</h1>', unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        entity_type = st.text_input(
            "Entity Type", placeholder="e.g., electricity, fuel"
        )

    with col2:
        entity_name = st.text_input(
            "Entity Name", placeholder="e.g., grid_mix, natural_gas"
        )

    with col3:
        region = st.text_input("Region (optional)", placeholder="e.g., US, EU, global")

    if st.button("Look Up Emission Factors"):
        if entity_type and entity_name:
            with st.spinner("Looking up emission factors..."):
                params = {"entity_type": entity_type, "entity_name": entity_name}

                if region:
                    params["region"] = region

                result = call_api("/emission-factors", params=params)

                if "error" in result:
                    st.error(result["error"])
                else:
                    # Display emission factors
                    emission_factors = result.get("emission_factors", [])
                    metadata = result.get("metadata", {})

                    if metadata.get("warning"):
                        st.warning(metadata["warning"])

                    if not emission_factors:
                        st.info("No emission factors found for the specified criteria")
                    else:
                        st.success(f"Found {len(emission_factors)} emission factor(s)")

                        # Convert to DataFrame for easier display
                        df = pd.DataFrame(emission_factors)
                        st.dataframe(df)

                        # Display cards for each emission factor
                        st.markdown(
                            '<h3 class="sub-header">Emission Factor Details</h3>',
                            unsafe_allow_html=True,
                        )

                        for ef in emission_factors:
                            create_emission_factor_card(ef)
        else:
            st.warning("Please enter entity type and name")

elif page == "Regional Comparison":
    st.markdown(
        '<h1 class="main-header">Regional Comparison</h1>', unsafe_allow_html=True
    )
    st.markdown("Compare emission factors across different regions for the same entity")

    col1, col2 = st.columns(2)

    with col1:
        entity_type = st.text_input(
            "Entity Type", placeholder="e.g., electricity, fuel"
        )

    with col2:
        entity_name = st.text_input(
            "Entity Name", placeholder="e.g., grid_mix, natural_gas"
        )

    regions_input = st.text_input(
        "Regions (comma-separated)", placeholder="e.g., US,EU,CN,IN,GB"
    )

    if st.button("Compare Regions"):
        if entity_type and entity_name and regions_input:
            with st.spinner("Comparing regions..."):
                regions = [r.strip() for r in regions_input.split(",")]

                params = {
                    "entity_type": entity_type,
                    "entity_name": entity_name,
                    "regions": regions,
                }

                result = call_api("/regions/compare", params=params)

                if "error" in result:
                    st.error(result["error"])
                else:
                    # Display comparison results
                    emission_factors = result.get("emission_factors", [])
                    statistics = result.get("statistics", {})

                    if not emission_factors:
                        st.info("No emission factors found for the specified regions")
                    else:
                        st.success(f"Comparing {len(emission_factors)} regions")

                        # Create DataFrame
                        comparison_data = []
                        for ef in emission_factors:
                            comparison_data.append(
                                {
                                    "Region": ef.get("region", "Unknown"),
                                    "Emission Factor": float(ef.get("ef_value", 0)),
                                    "Unit": ef.get("ef_unit", ""),
                                }
                            )

                        df = pd.DataFrame(comparison_data)

                        # Create bar chart
                        if not df.empty:
                            fig = px.bar(
                                df,
                                x="Region",
                                y="Emission Factor",
                                title=f"Emission Factors for {entity_name} ({entity_type}) by Region",
                                labels={
                                    "Emission Factor": f"Emission Factor ({df['Unit'].iloc[0] if not df['Unit'].empty else 'Unknown'})"
                                },
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Display statistics
                            if statistics:
                                st.markdown(
                                    '<h3 class="sub-header">Statistics</h3>',
                                    unsafe_allow_html=True,
                                )

                                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(
                                    4
                                )

                                with stat_col1:
                                    st.metric(
                                        "Minimum", f"{statistics.get('min', 'N/A'):.4f}"
                                    )

                                with stat_col2:
                                    st.metric(
                                        "Maximum", f"{statistics.get('max', 'N/A'):.4f}"
                                    )

                                with stat_col3:
                                    st.metric(
                                        "Average", f"{statistics.get('avg', 'N/A'):.4f}"
                                    )

                                with stat_col4:
                                    st.metric(
                                        "Range", f"{statistics.get('range', 'N/A'):.4f}"
                                    )

                            # Show raw data
                            with st.expander("View raw data"):
                                st.dataframe(df)
        else:
            st.warning("Please enter entity type, name, and at least one region")

elif page == "About":
    st.markdown('<h1 class="main-header">About Carbon EF</h1>', unsafe_allow_html=True)

    st.markdown(
        """
    **Carbon EF** is a hybrid Retrieval-Augmented Generation (RAG) system designed to provide accurate and up-to-date 
    carbon emission factor information.
    
    ### Key Features
    
    - **Natural Language Interface**: Ask questions about emission factors in plain English
    - **GraphRAG**: Combines Qdrant vector search and Neo4j graph database for better reasoning
    - **Hybrid Retrieval**: Combines graph database and vector search for accurate answers
    - **Semantic Caching**: Reduces latency by caching similar queries
    - **Regional Comparison**: Compare emission factors across different regions
    
    ### Technology Stack
    
    - **Agent Framework**: Phi-2 based reasoning agent
    - **Graph Database**: Neo4j for structured emission factor data
    - **Vector Store**: Qdrant for semantic search capabilities
    - **API**: FastAPI for serving emission factor data
    - **UI**: Streamlit for interactive exploration
    
    ### Usage Examples
    
    - "What is the emission factor for electricity in California?"
    - "Compare emission factors for natural gas across EU, US, and Canada"
    - "How do emission factors for coal compare to natural gas?"
    - "What is the lowest emission factor for grid electricity worldwide?"
    - "What entities are related to transportation in the US region?"
    """
    )

# Footer
st.markdown("---")
st.markdown("Carbon EF - Emission Factor Explorer | v0.1.0")


# Function to check API health
def check_api_health() -> bool:
    """Check if the API server is healthy."""
    try:
        result = call_api("health", method="get")
        return not result.get("error", False)
    except Exception:
        return False
