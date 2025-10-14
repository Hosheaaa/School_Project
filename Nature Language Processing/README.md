# Natural Language Processing Projects

## Overview
This folder contains NLP course assignments and a group project focused on Retrieval-Augmented Generation (RAG).

## Individual Assignments

### Assignment 1
- Text processing and analysis
- NLP pipeline implementation
- Jupyter notebook exercises

### Assignment 2
- Text classification tasks
- Implementation using RNN and DeBERTa models
- Performance evaluation and comparison

## Group Project (Group 6)
### RAG-based Marketing Content Generation

**Objective**: Build a RAG system for generating personalized marketing content for XYZ Bank using AlloyDB and Vertex AI.

**Key Components**:
- **Backend**: Google Cloud AlloyDB with pgvector
- **Embeddings**: Vertex AI text-embedding-gecko model
- **LLM**: Gemini-1.5-pro and text-bison models
- **Prompting Strategies**: Zero-shot, few-shot, and direct prompting

**Features**:
- Semantic search with vector embeddings (768 dimensions)
- IVFFlat indexing for efficient similarity search
- Marketing content templates for various banking services
- Context-aware content generation

**Evaluation Metrics**:
- Coherence (using GEval)
- Answer Relevancy
- Contextual Precision
- Comparison of ChatGPT vs Vertex AI responses

## Technologies
- Python
- TensorFlow/PyTorch
- Transformers (DeBERTa, RNN)
- Google Cloud Platform (AlloyDB, Vertex AI)
- LangChain
- DeepEval for evaluation

## Project Deliverables
- Implementation notebooks
- Evaluation results and analysis
- Project report and presentation slides
