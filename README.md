<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin-right: 20px; height: 55px" height="55px">

# Agentic Ops & Retrieval-Augmented Generation (RAG) in Practice

Agentic Ops & Retrieval-Augmented Generation (RAG) has become one of the most powerful patterns in applied AI—but building a high-performing, production-ready RAG pipeline requires more than plugging documents into a vector store.

Participants will code along with instructors to experiment with embeddings, chunking strategies, routing, hybrid search, retrieval fusion, and evaluation frameworks. You’ll leave with working notebooks, templates you can adapt for real systems, and a deep mental model of what actually moves performance.

## Getting Started

### Step 1: Check Prerequisites

Before opening any notebooks, make sure your environment is set up correctly. See the prerequisites guide here:

[Prerequisites Guide](prerequisites/prerequisites.md)

This covers Python version requirements, package installation, and how to verify your setup.

### Step 2: Download the course repository

To get all course files on your local machine, clone this repository using Git.

```
git clone
```

---

## Intended outcome

In this training, we’ll cover:

1. Architecting Robust RAG Pipelines
   - Dense vs. sparse retrieval, hybrid ranking stacks, and retrieval routing
   - Designing multi-stage pipelines (pre-filter → retrieve → rerank → generate)
   - Latency budgets and system-level performance consideration
2. Advanced Chunking, Embeddings & Indexing
   - Adaptive chunking heuristics for structured vs. unstructured data
   - Embedding model selection criteria (domain specificity, multilingual, cost/perf)
   - Indexing patterns: HNSW, IVF-Flat, PQ, hybrid indices, memory optimization
3. Retrieval & Reranking Techniques (Hands-On)
   - Implementing cross-encoder rerankers
   - Experimenting with ColBERT-style late interaction
   - Vector store configuration tuning (Faiss, Milvus, Weaviate, Elastic)
4. Guardrails, Observability & Evaluation
   - Hallucination mitigation strategies using retrieval-level signals
   - Automated RAG evaluation: answer correctness, grounding, context utilization
   - Live instrumentation: tracing, latency tracking, prompt/response logging
5. Putting It Together: Build & Benchmark an Advanced RAG Stack
   - Small-group coding challenge building an end-to-end RAG system
   - Benchmark across multiple retrieval configurations
   - Discuss tradeoffs, scalability, and deployment pathways
6. Large-Document Hierarchical RAG
   - Index page/section-aware evidence without loading the full document per query
   - Route lookup, numeric, global, cross-reference, and amendment questions
   - Retrieve small blocks, expand selected parents/references, and pack to a token budget
7. Evidence-First Evals & Experiments
   - Diagnose retrieval, packing, answer, citation, numeric, and abstention failures separately
   - Run controlled ablations, stress tests, paired uncertainty estimates, and release gates
   - Slice results by financial/legal task type and evidence position

---

## Assumed background

### Who This Is For

- **Machine Learning Engineers** who own retrieval quality
- **Applied AI / LLM Engineers** shipping agentic and RAG features
- **Data Scientists** wrangling unstructured, messy, real-world documents
- **Backend Engineers** bolting AI features onto existing production services

### Prerequisites

You should walk in comfortable with:

- Python (you'll write real code, not just run cells)
- Jupyter / Colab notebooks
- Basic vector database usage
- Embeddings and "simple RAG" concepts (chunk → embed → search → generate)

You do **not** need prior experience with hybrid search, rerankers, HNSW/IVF tuning, or RAG evaluation frameworks — that's the whole point of this course.

**No API key or network access is required for any lab.**
Every embedding, reranker, and "LLM judge" used in this course is a small, transparent, deterministic Python function rather than a live model call.

> **Why no live LLM or embedding API calls?**
> Every embedding, cross-encoder, ColBERT scorer, and "LLM judge" in this course is a small, transparent, deterministic function rather than a call to OpenAI, Cohere, or a hosted model.
>
> This makes every result in every notebook exactly reproducible offline, with no API key, network access, or model download required, and lets you read the full arithmetic behind every score.
>
> The architectural lessons — why hybrid search beats single-mode retrieval, why reranking matters, why guardrails are a separate concern from retrieval quality — transfer directly to real embedding models and LLMs; only the specific scoring functions are simplified, inspectable stand-ins.

---

## Course Content

| #   | Module                                                           | Notebook                                                         | Solutions                                                             | Time       |
| --- | ---------------------------------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------- | ---------- |
| 1   | **Architecting Robust RAG Pipelines**                            | [Notebook](notebooks/01_architecting_pipelines_lab.ipynb)        | [Solution](solutions/01_architecting_pipelines_solution.ipynb)        | 25 minutes |
| 2   | **Advanced Chunking, Embeddings & Indexing**                     | [Notebook](notebooks/02_chunking_embeddings_indexing_lab.ipynb)  | [Solution](solutions/02_chunking_embeddings_indexing_solution.ipynb)  | 50 minutes |
| -   | **Break**                                                        | -                                                                | -                                                                     | 15 minutes |
| 3   | **Retrieval & Reranking Techniques**                             | [Notebook](notebooks/03_retrieval_reranking_lab.ipynb)           | [Solution](solutions/03_retrieval_reranking_solution.ipynb)           | 40 minutes |
| 4   | **Guardrails, Observability & Evaluation**                       | [Notebook](notebooks/04_guardrails_observability_eval_lab.ipynb) | [Solution](solutions/04_guardrails_observability_eval_solution.ipynb) | 50 minutes |
| 5   | **Putting It Together: Build & Benchmark an Advanced RAG Stack** | [Lab](notebooks/05_capstone_lab.ipynb)                           | [Solution](solutions/05_capstone_solution.ipynb)                      | 50 minutes |
| 6   | **Large-Document Hierarchical RAG**                             | [Lab](notebooks/06_large_document_hierarchical_rag_lab.ipynb)    | [Solution](solutions/06_large_document_hierarchical_rag_solution.ipynb) | 70-90 minutes |
| 7   | **Large-Document RAG Evals & Experiments**                       | [Lab](notebooks/07_large_document_rag_evals_lab.ipynb)           | [Solution](solutions/07_large_document_rag_evals_solution.ipynb)      | 75-100 minutes |

---

## Capstone Lab

- [Capstone Project: Build & Benchmark an Advanced RAG Stack](notebooks/05_capstone_lab.ipynb)

---

## Advanced Large-Document Track

Modules 6 and 7 are a self-contained advanced track for financial filings, contracts, and
other documents that exceed a model's practical context window. The core labs remain fully
offline and deterministic, using a fictional 90-block annual-report/MSA benchmark with page,
section, version, table, footnote, amendment, and cross-reference provenance.

- [Research and experiment design](docs/large-document-rag-research.md)
- [Synthetic benchmark](data/large_document_benchmark.json)
- [Hierarchical retrieval lab](notebooks/06_large_document_hierarchical_rag_lab.ipynb)
- [Evidence-first evals lab](notebooks/07_large_document_rag_evals_lab.ipynb)

---

**Find a 👾 bug 👾 or have suggestions? [Let us know]()!**
