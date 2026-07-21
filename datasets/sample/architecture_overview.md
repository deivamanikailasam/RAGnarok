# Architecture Overview

This document gives a visual overview of the retrieval-augmented generation pipeline.

The diagram below shows how a user query flows through embedding, vector search, and the LLM to
produce a grounded answer.

![RAG pipeline architecture diagram showing embedding, vector search, reranking, and LLM answer generation](assets/rag_architecture.png)

The pipeline embeds the query, retrieves relevant chunks from the vector database, reranks them, and
passes the top results to the language model for answer generation.
