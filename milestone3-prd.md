# Product Requirements Document
# Milestone 3: Hybrid RAG Pipeline with Graph RAG + Agentic RAG

## 1. Overview

### 1.1 Purpose
This document outlines the requirements and specifications for Milestone 3 of the Regional Emission Factor (EF) project. This milestone focuses on implementing a hybrid Retrieval-Augmented Generation (RAG) pipeline that combines Graph RAG and Agentic RAG approaches to deliver accurate, region-specific emission factor data with low latency.

### 1.2 Project Background
Building upon the successful completion of Milestones 1 (Regional EF Knowledge Graph) and 2 (Phi-2 Fine-Tuning and Merging), this phase aims to create an efficient retrieval system that leverages both the structured relationships in our Neo4j graph and the environmental domain knowledge of our fine-tuned Phi-2 model.

### 1.3 Scope
This milestone encompasses the development of:
- A vector embedding pipeline using Qdrant
- A Graph RAG component leveraging Neo4j
- An Agentic RAG system for reasoning and query decomposition using our fine-tuned Phi-2 model
- A semantic caching mechanism
- A unified API for accessing the system

## 2. Key Objectives

### 2.1 Performance Targets
- **Latency**: < 200ms average response time for EF retrieval
- **Accuracy**: > 90% accuracy for region-specific EF queries
- **Availability**: 99.9% uptime for the retrieval pipeline
- **Scalability**: Support for at least 100 concurrent queries

### 2.2 Functional Goals
- Enable precise retrieval of region-specific emission factors
- Provide transparent reasoning for retrieved results
- Support complex multi-part queries through decomposition
- Optimize performance through intelligent caching

## 3. System Architecture

### 3.1 Key Components

#### 3.1.1 Vector Storage (Qdrant)
- **Purpose**: Store and retrieve vector embeddings for semantic similarity search
- **Features**:
  - Support for 768-dimensional embeddings (compatible with Phi-2 architecture)
  - HNSW indexing for fast approximate nearest neighbor search
  - Filtering capabilities based on metadata (region, industry)
  - Collection schema with appropriate payload fields

#### 3.1.2 Graph Database (Neo4j)
- **Purpose**: Store structured relationships between entities (EFs, regions, industries)
- **Features**:
  - Optimized Cypher query templates
  - Fast traversal capabilities
  - Rich metadata on relationships

#### 3.1.3 Agent Framework (Phi-2 Based)
- **Purpose**: Provide reasoning and coordination capabilities
- **Features**:
  - Query decomposition using phi2-env-factors-merged model
  - Retrieval strategy selection optimized for environmental context
  - Result synthesis and explanation with domain specificity
  - Confidence scoring

#### 3.1.4 Semantic Cache
- **Purpose**: Reduce latency for common queries
- **Features**:
  - Storage for both vector and graph-based results
  - Reasoning chain caching
  - Semantic similarity matching for cache hits
  - Time-to-live (TTL) policies

#### 3.1.5 API Layer
- **Purpose**: Provide unified access to the system
- **Features**:
  - REST or GraphQL endpoints
  - Query configuration options
  - Debugging and tracing capabilities

#### 3.1.6 Streamlit Web Interface
- **Purpose**: Provide user-friendly demonstration of the system capabilities
- **Features**:
  - Interactive query input with region selection
  - Visualization of retrieval results 
  - Explanation of reasoning process
  - Performance metrics display
  - Graph visualization of knowledge connections

### 3.2 Data Flow
1. User query received through API
2. Query analyzed and decomposed by Phi-2 Agent
3. Parallel retrieval from:
   - Qdrant for semantic similarity
   - Neo4j for structured relationships
4. Results combined and synthesized
5. Response returned to user with explanation if requested

## 4. Functional Requirements

### 4.1 Vector Embedding Pipeline

#### 4.1.1 Embedding Generation
- **REQ-V1**: System shall generate 768-dimensional embeddings compatible with Phi-2 architecture
- **REQ-V2**: All EF nodes from Neo4j must have corresponding vector embeddings
- **REQ-V3**: Embedding quality shall be validated through similarity tests

#### 4.1.2 Qdrant Integration
- **REQ-V4**: System shall store embeddings in Qdrant with appropriate metadata
- **REQ-V5**: Qdrant collection schema must include fields for filtering by region and industry
- **REQ-V6**: Vector similarity search must complete in < 50ms

### 4.2 Graph RAG Component

#### 4.2.1 Query Templates
- **REQ-G1**: System shall maintain a library of parameterized Cypher queries
- **REQ-G2**: Queries must be optimized for < 50ms execution time
- **REQ-G3**: Templates must cover common EF retrieval patterns

#### 4.2.2 Context Extraction
- **REQ-G4**: System shall extract relevant subgraphs around entities of interest
- **REQ-G5**: Graph structures must be convertible to text context for Phi-2 model consumption
- **REQ-G6**: Graph paths shall be ranked by relevance to original query

#### 4.2.3 Hybrid Retrieval
- **REQ-G7**: System shall combine graph traversal with vector similarity results
- **REQ-G8**: Combined results must use weighted scoring between structural and semantic relevance
- **REQ-G9**: Retrieval interface must support both approaches seamlessly

### 4.3 Agentic RAG Implementation

#### 4.3.1 Phi-2 Agent Framework
- **REQ-A1**: System shall implement a reasoning layer using the fine-tuned phi2-env-factors-merged model
- **REQ-A2**: Distinct agent roles (analyzer, retriever, synthesizer) must be defined
- **REQ-A3**: Agent communication protocol shall enable task delegation

#### 4.3.2 Query Planning
- **REQ-A4**: System shall leverage Phi-2's environmental domain knowledge to decompose complex queries
- **REQ-A5**: Decision trees must route queries to appropriate retrieval methods
- **REQ-A6**: Query reformulation templates shall enhance precision for emission factor contexts

#### 4.3.3 Response Synthesis
- **REQ-A7**: System shall combine information from multiple sources with attribution
- **REQ-A8**: Confidence scoring must be provided for different response components
- **REQ-A9**: Reasoning steps shall be traceable and explainable

### 4.4 Semantic Cache

#### 4.4.1 Cache Structure
- **REQ-C1**: System shall cache both vector and graph traversal results
- **REQ-C2**: Complete reasoning chains must be stored when beneficial
- **REQ-C3**: Metadata system shall track contributing components

#### 4.4.2 Cache Strategy
- **REQ-C4**: TTL policies must be configurable per query type
- **REQ-C5**: Semantic similarity matching shall enable fuzzy cache hits
- **REQ-C6**: Priority system must optimize cache refreshing

#### 4.4.3 Agent Integration
- **REQ-C7**: Phi-2 agents shall make decisions about cache utilization
- **REQ-C8**: Confidence thresholds must govern cache use
- **REQ-C9**: Cache effectiveness shall be tracked and logged

### 4.5 API and Integration

#### 4.5.1 API Design
- **REQ-API1**: System shall expose REST or GraphQL endpoints
- **REQ-API2**: Query parameters must allow control of retrieval strategies
- **REQ-API3**: Response format shall be consistent JSON

#### 4.5.2 Debugging Tools
- **REQ-API4**: System shall provide endpoints for viewing agent reasoning traces
- **REQ-API5**: Visualization tools must help understand retrieved contexts
- **REQ-API6**: Admin interface shall monitor system performance

### 4.6 Streamlit Web Interface

#### 4.6.1 User Interface Design
- **REQ-UI1**: System shall provide a Streamlit-based web interface for demonstration
- **REQ-UI2**: Interface must support natural language query input
- **REQ-UI3**: Region and industry selection must be available via dropdown menus

#### 4.6.2 Result Visualization
- **REQ-UI4**: System shall display retrieved emission factors in a tabular format
- **REQ-UI5**: Interactive visualization must show relevant knowledge graph connections
- **REQ-UI6**: Confidence scores shall be displayed for each result component

#### 4.6.3 Explanation Components
- **REQ-UI7**: System shall provide a step-by-step explanation of the reasoning process
- **REQ-UI8**: Contribution breakdown must show vector vs. graph influence on results
- **REQ-UI9**: Performance metrics shall display retrieval and processing times

## 5. Non-Functional Requirements

### 5.1 Performance
- **REQ-NF1**: End-to-end query response time must be < 200ms (P95)
- **REQ-NF2**: System shall support at least 100 concurrent requests
- **REQ-NF3**: Cache hit rate must exceed 80% for common queries

### 5.2 Scalability
- **REQ-NF4**: Architecture must allow horizontal scaling of components
- **REQ-NF5**: Database connections shall use connection pooling
- **REQ-NF6**: System performance must degrade gracefully under load

### 5.3 Reliability
- **REQ-NF7**: System shall implement fallback strategies for component failures
- **REQ-NF8**: Error handling must provide informative feedback
- **REQ-NF9**: System shall maintain 99.9% uptime

### 5.4 Security
- **REQ-NF10**: API access must be authenticated and authorized
- **REQ-NF11**: Sensitive data shall be protected appropriately
- **REQ-NF12**: All external connections must use TLS

## 6. Testing Requirements

### 6.1 Component Testing
- **REQ-T1**: Each component must have unit tests with > 80% coverage
- **REQ-T2**: Performance benchmarks shall validate latency requirements
- **REQ-T3**: Integration tests must verify cross-component functionality

### 6.2 End-to-End Testing
- **REQ-T4**: Comprehensive test suite shall cover diverse query patterns
- **REQ-T5**: Load tests must simulate concurrent requests
- **REQ-T6**: Edge case tests shall verify graceful failure handling

## 7. Acceptance Criteria

For Milestone 3 to be considered complete, the system must:

1. Successfully retrieve region-specific EF data with > 90% accuracy using the phi2-env-factors-merged model
2. Maintain average response times < 200ms
3. Demonstrate integration of Graph RAG and Agentic RAG approaches
4. Show improved results compared to vector-only or graph-only approaches
5. Provide comprehensive API documentation
6. Pass all specified performance and reliability tests

## 8. Deliverables

1. Vector database (Qdrant) populated with EF embeddings
2. Graph RAG component with optimized query templates
3. Phi-2 based Agentic RAG framework for query processing
4. Semantic cache implementation
5. API endpoints with documentation
6. Streamlit web interface for demonstration
7. Performance testing reports
8. User and technical documentation

## 9. Timeline and Milestones

| Week | Focus Area | Key Deliverables |
|------|------------|------------------|
| 1    | Vector Pipeline & Qdrant Setup | Embedding generation, Qdrant configuration |
| 2    | Graph RAG Implementation | Query templates, context extraction |
| 3    | Phi-2 Agentic RAG Development | Agent framework, query planning |
| 4    | Semantic Cache | Cache structure, strategy implementation |
| 5    | API & Integration | Endpoint development, documentation |
| 6    | Streamlit Web Interface | UI components, visualizations, demo scenarios |
| 7    | Testing & Optimization | Performance tuning, comprehensive testing |
| 8    | Documentation & Handoff | Final documentation, knowledge transfer |

## 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance targets not met | Medium | High | Early performance testing, component-level optimization |
| Integration complexity between Graph and Agentic RAG | High | Medium | Modular design, clear interfaces, incremental integration |
| Data inconsistency across stores | Medium | High | Transaction safety, automated verification |
| Cache efficiency below target | Medium | Medium | Adaptive caching strategies, continuous monitoring |
| Phi-2 model latency exceeds targets | Medium | High | Model quantization, batching strategies, response caching |

## 11. Dependencies

1. Successful completion of Milestone 2 (phi2-env-factors-merged model)
2. Functioning Neo4j knowledge graph from Milestone 1
3. Computing resources for Qdrant and agent operations
4. Hugging Face integration for model hosting and serving

## 12. Implementation Details

### 12.1 phi2-env-factors-merged Model Integration
- **Hosting**: Utilize the Hugging Face model repository (https://huggingface.co/Surendra-Aitest/phi2-env-factors-merged)
- **Deployment**: Implement model serving via:
  - Direct Hugging Face Inference API integration
  - Self-hosted Inference endpoints
  - Containerized deployment with optimized runtime

### 12.2 Model Configuration
- **Tokenization**: Use the model's specialized tokenizer with environmental tokens
- **Generation Parameters**:
  - Temperature: 0.7 for balanced creativity/precision
  - Top-p: 0.9
  - Max new tokens: Configurable based on query complexity

### 12.3 Model Performance Optimization
- Implement quantization (INT8) for production deployment
- Utilize batching for concurrent requests
- Evaluate KV cache strategies for repeated similar queries

### 12.4 Streamlit Web Interface Implementation
- **Framework**: Implement using Streamlit 1.26+ for interactive data apps
- **Architecture**: 
  - Frontend UI elements using Streamlit components
  - Backend connection to API endpoints
  - Caching of query results for demo performance
- **Key Features**:
  - Query building with auto-suggestions
  - Interactive region selection via map interface
  - Step-by-step explanation visualization
  - Graph visualization using Pyvis or NetworkX
  - Performance metrics dashboard
- **Deployment**:
  - Containerized deployment for consistent environment
  - Integration with authentication if required
  - Support for demo recording and sharing

## 13. Stakeholders

| Role | Responsibilities |
|------|------------------|
| Product Manager | Requirements prioritization, feature approval |
| Data Engineer | Neo4j and Qdrant integration, data pipeline |
| ML Engineer | Phi-2 integration, agent implementation |
| Backend Developer | API development, system integration |
| QA Engineer | Testing, performance validation |
| DevOps | Deployment, monitoring, scaling |

## 14. Glossary

- **EF**: Emission Factor
- **RAG**: Retrieval-Augmented Generation
- **Graph RAG**: RAG approach utilizing graph database relationships
- **Agentic RAG**: RAG approach with reasoning agents for query handling
- **HNSW**: Hierarchical Navigable Small World (vector indexing algorithm)
- **TTL**: Time-to-Live (cache expiration policy)
- **Phi-2**: Microsoft's 2.7B parameter language model
- **phi2-env-factors-merged**: Our custom fine-tuned and merged model specialized for environmental factors
