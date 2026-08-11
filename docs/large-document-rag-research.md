# Large-Document RAG for Financial and Legal Work

Research and experiment design current to **August 10, 2026**.

## Executive recommendation

Do not treat a 300–2,000 page PDF as one prompt or as hundreds of anonymous chunks.
Treat it as a **navigable, versioned evidence store**.

The strongest practical baseline is:

```text
layout-aware parse
  → immutable page/section/table source map
  → leaf + parent-section indexes
  → query-type router
  → hybrid retrieve and rerank
  → expand only winning parents, neighbors, footnotes, and references
  → pack original evidence under a token budget
  → calculate/answer with claim-level citations
  → verify support or abstain
```

This is the “retrieve small, read big” pattern: retrieve short units for precision, then
read only the relevant contiguous section or linked evidence for context. The entire
document can be indexed once without being loaded into each model call.

The first experiment should compare this baseline against fixed-size flat RAG and a
full-document long-context route. Add agentic decomposition, vision, recursive summaries,
or GraphRAG only for question classes where diagnostic metrics show a gap.

## 1. Preserve the document before optimizing retrieval

The ingestion record should retain:

- `doc_id`, title, document type, immutable version/hash, issue/effective/filing date;
- page, character span and—where available—PDF bounding box;
- ordered heading path, clause/table/figure/footnote identifier, parent and neighbors;
- modality, OCR confidence, reading order and links to the original page image;
- finance metadata: issuer, filing type, reporting period, currency, unit/scale;
- legal metadata: parties, jurisdiction, authority date, defined terms, amendment and
  supersession status.

The cited payload should remain the exact normalized source block. Enriched summaries,
contextual prefixes, synthetic questions, and embeddings are index representations, not
the source of record.

This matters because PDF failure is often upstream of RAG. [Docling](https://arxiv.org/abs/2408.09869)
uses dedicated layout and table-structure models, while [ColPali](https://arxiv.org/abs/2407.01449)
shows why page layout, tables, figures and other visual cues can justify a parallel page-image
retriever. A text-only retrieval score cannot reveal a table header that the parser dropped.

## 2. Use multiple retrieval resolutions

Maintain at least two searchable resolutions:

1. **Leaf blocks**: paragraphs, clauses, lists, table rows/cells, figures and footnotes.
   These produce precise, minimal citations.
2. **Parents**: pages, sections, clauses with children, and concise section summaries.
   These preserve local context and support broader questions.

An optional third level can recursively cluster/summarize parents. [RAPTOR](https://arxiv.org/abs/2401.18059)
retrieves across a recursive summary tree and is especially relevant to multi-step or
holistic questions. In standardized filings, [HiREC](https://aclanthology.org/2025.findings-acl.855/)
uses document-then-passage retrieval plus evidence curation to reduce confusion among
near-duplicate filing structures. [FinGEAR](https://aclanthology.org/2025.findings-emnlp.382/)
likewise combines filing-item guidance, dual hierarchies and two-stage reranking.

At query time:

1. shortlist documents/sections using metadata and parent summaries;
2. retrieve/rerank precise leaves within those parents;
3. expand only relevant parent text, neighboring blocks, table headers/footnotes and
   explicit links;
4. deduplicate and pack to a fixed evidence budget.

The parent-to-leaf path should be logged so every retrieval decision is explainable.

## 3. Index multiple representations, return one source of truth

For each source block, compare these index representations:

- raw text;
- `document + version/period + heading path + raw text`;
- LLM-generated contextual prefix + raw text;
- synthetic questions or concept labels;
- late-chunked embedding;
- text embedding plus a separate page-image embedding.

[Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) prepends
chunk-specific context before both dense and BM25 indexing. Its reported improvements are
vendor results and should be reproduced on the target corpus. For a document larger than
the preprocessing model's window, generate context from the document summary plus section
parent—not by repeatedly passing the entire PDF.

[Late Chunking](https://arxiv.org/abs/2409.04701) embeds a long token sequence first and
pools chunk spans afterward, allowing a chunk representation to retain surrounding context.
Test it against the cheaper explicit prefix; do not assume either wins for legal citations,
tables or exact financial line items.

## 4. Use a hybrid, multi-stage ranking stack

The default ranking stack should be:

```text
metadata filter
  → BM25 candidate set + dense candidate set
  → reciprocal-rank fusion
  → cross-encoder/LLM rerank of a fixed pool
  → redundancy removal
  → parent/reference expansion
  → evidence-budget packing
```

BM25 is valuable for clause numbers, defined terms, dates, entities, filing items and exact
account names. Dense retrieval catches paraphrase. Reranking improves precision after a
wide candidate stage. Freeze the candidate pool when comparing rerankers; otherwise the
experiment cannot attribute the gain.

For legal corpora with many structurally similar documents, explicitly track the
**wrong-document/entity/period/version rate**. Work on reliable retrieval over large legal
datasets identifies document-level mismatch as a distinct failure mode
([Reuter et al., 2025](https://aclanthology.org/2025.nllp-1.3/)).

## 5. Route by the shape of evidence

| Question class | Retrieval path | Reader/tool path |
|---|---|---|
| Direct fact or locator | metadata → hybrid leaf → rerank | small cited packet |
| Exact clause/definition | BM25-heavy leaf + typed legal links | clause plus definition/exception |
| Amendment/version conflict | retrieve original and controlling amendment + priority rule | compare; state controlling date |
| Comparison/multi-hop | decompose; retrieve subquestions; gap check | compose only after evidence suffices |
| Numeric/table | table/page route; header + row/cell retrieval | execute formula in Python/SQL; cite operands |
| Global synthesis | several parent summaries; map/reduce or recursive tree | cited key-point synthesis |
| Entity/network exploration | graph local/global/DRIFT route | graph plus raw-text citations |
| Unanswerable/ambiguous | evidence-sufficiency check | abstain, partial answer, or clarify |

[Adaptive-RAG](https://arxiv.org/abs/2403.14403) is a useful complexity-routing reference.
Microsoft's current [GraphRAG query documentation](https://microsoft.github.io/graphrag/query/overview/)
separates local, global and DRIFT-style search. GraphRAG's global map/reduce path is useful
for corpus-wide themes, but is more expensive and should not be the route for a local fact.
For legal documents, deterministic edges such as `defined_term → definition`,
`clause → cross-reference`, and `amendment → superseded clause` are often safer than relying
only on unconstrained entity extraction.

PageIndex-style reasoning over a semantic table-of-contents is an emerging alternative to
vector-first retrieval. Its [open-source project](https://github.com/VectifyAI/PageIndex)
is useful design inspiration, but benchmark claims are vendor-reported; compare tree-only,
vector-only and hybrid variants on a private test set before adopting it.

## 6. Financial documents need table and calculation tools

Flattening every table into prose destroys headers, row/column relationships, units and
footnotes. Preserve the full table block and also index:

- caption/title and surrounding section;
- schema/header path;
- row and column keys;
- normalized cells with value, currency, scale, period and sign;
- footnotes and continuation-page links.

For a calculation, retrieve the table, select operands, execute a symbolic program and
emit the formula, values, units and citations. [TAT-QA](https://arxiv.org/abs/2105.07624)
explicitly evaluates reasoning over financial tables plus text. [TableRAG](https://aclanthology.org/2025.emnlp-main.710/)
uses decomposition, text retrieval and SQL execution for heterogeneous documents.

Current benchmarks increasingly emphasize realistic multi-page/multimodal evidence.
[FinMRAGBench](https://aclanthology.org/2026.findings-acl.187/) contains expert-verified
questions spanning multiple pages and documents; [FinRAGBench-V](https://arxiv.org/abs/2505.17471)
adds visual citations over a large bilingual page corpus.

## 7. Legal documents need authority and precedence, not “latest wins” alone

Preserve and retrieve:

- parties, defined terms and the exact scope of each definition;
- obligation, exception, negation, survival and remedy clauses;
- clause-to-clause references;
- amendment effective date and the precise text replaced;
- order-of-precedence language;
- governing jurisdiction and source authority.

A query about an amended liability cap normally needs the original clause, amendment,
priority rule, definition of Fees and any relevant carve-outs. Filtering to the newest
document can erase necessary explanation; retrieving only the original produces a dangerous
stale answer.

[LegalBench-RAG](https://arxiv.org/abs/2408.10343) focuses on minimal, precise retrieval for
legal questions and provides 6,858 expert-annotated query/evidence pairs. The 2026
[Legal RAG Bench preprint](https://arxiv.org/abs/2603.01710) uses factorial error decomposition
and reports retrieval as the main driver in its tested setups; treat this as promising recent
evidence, not a settled universal result.

## 8. Long context is a baseline and a route

Advertised context length is not the same as reliable use of every token. [Lost in the
Middle](https://aclanthology.org/2024.tacl-1.9/) found substantial position sensitivity even
in long-context models. Therefore test:

- evidence at the beginning, middle and end;
- document length and distractor count;
- distance between jointly required facts;
- exact lexical match versus paraphrase;
- clean versus OCR-damaged/table-heavy pages.

Always include these baselines:

1. no-context model (memorization/contamination check);
2. oracle evidence → reader (generation/reasoning ceiling);
3. full-document long context where technically feasible;
4. selected contiguous section in long context;
5. flat RAG and the proposed hierarchical RAG.

Route a full document only when it fits the model's measured *effective* context, the task
is genuinely global, and its quality/cost/latency wins the controlled comparison.

## 9. Evaluate each boundary, not just the final prose

```text
PDF → parse → index → retrieve → rerank → pack → answer → cite → abstain
```

Assign every failed case to the earliest failed boundary:

```text
parse/source-map failed
retrieval failed
packing truncated evidence
reader ignored available evidence
reasoning/calculation failed
citation failed
answerability policy failed
```

[RAGChecker](https://arxiv.org/abs/2408.08067) is a strong reference for modular RAG
diagnostics. Citation quality requires separate support and completeness metrics; [ALCE](https://arxiv.org/abs/2305.14627)
provides an influential evaluation formulation. [GaRAGe](https://arxiv.org/abs/2506.07671)
adds grounding, attribution and deflection annotations and reports that current systems
remain weak at strict grounding and abstention.

### Gold schema

Do not annotate only chunk IDs: a new chunker would invalidate the test. Store immutable
source anchors and allow alternative valid evidence sets.

```json
{
  "question_id": "fin_001",
  "question_type": "cross_table_numeric",
  "answerability": "answer",
  "required_claims": ["operating margin changed by -1.7 percentage points"],
  "acceptable_evidence_sets": [["page72.cellA", "page119.cellB"]],
  "expected_answer": {
    "value": -1.7,
    "unit": "percentage_point",
    "program": "subtract(margin_2024, margin_2023)"
  },
  "cohorts": {
    "hops": 2,
    "modality": "table",
    "position_decile": 6,
    "evidence_distance_pages": 47
  }
}
```

### Primary metrics

| Layer | Primary metrics | Diagnostics |
|---|---|---|
| Parsing | reading-order accuracy, table structure/header score | OCR error, footnote retention, page/bbox accuracy |
| Retrieval | Recall@k, strict all-evidence recall, MRR, nDCG | wrong entity/period/version, span/cell recall |
| Packing | packed evidence recall | evidence density, duplicates, truncation, evidence position |
| Answer | typed value/F1 or expert correctness | required-claim recall, unsupported/contradicted claims |
| Citation | support precision and completeness | locator, entity, period, version, authority |
| Numeric | normalized value accuracy | operand, unit/scale, sign, period, program execution |
| Legal | entail/contradict/not-mentioned | exception scope, precedence, governing authority |
| Abstention | unsafe false-answer rate, selective accuracy | risk–coverage curve, ECE/Brier, over-abstention |
| Operations | p50/p95 latency, prompt tokens, cost | index size, ingestion time, Pareto frontier |

## 10. Controlled experiment sequence

### First 2×2×2 factorial

- chunk/index text: raw fixed chunks vs structure-aware contextual blocks;
- retrieval: lexical vs hybrid;
- reranker/reference expansion: off vs on.

Hold the parser, corpus, questions, top-k, token budget, reader, prompt and seed fixed. Then
sweep candidate pool size, packed-token budget and parent expansion around the winner.

### Hypotheses worth testing

1. Layout/table-aware parsing improves numeric evidence recall more than swapping embedding
   models.
2. Hybrid retrieval improves exact identifiers, clauses and financial line items.
3. Parent/neighbor/reference expansion improves strict multi-evidence recall but may reduce
   evidence density.
4. Increasing `k` initially improves recall and then harms reader accuracy through noise.
5. Query decomposition helps comparison/global questions at higher latency and cost.
6. Explicit amendment/definition graphs reduce stale legal answers.
7. Visual page retrieval helps scans, figures and complex tables but is unnecessary for
   clean text clauses.
8. Citation-constrained answers improve attribution but may increase abstention.

### Statistical protocol

- split by company/document/contract and add a time-based holdout;
- maintain a private final test set and prevent prompt/judge tuning on it;
- use paired per-query comparisons and paired bootstrap confidence intervals;
- use McNemar's test for paired binary critical errors and multiplicity correction when
  testing many variants;
- repeat stochastic generation with fixed paired seeds;
- report macro, production-weighted and worst-cohort outcomes;
- calibrate LLM judges against a double-annotated expert sample by injected failure type.

## 11. Notebook artifacts in this repository

- `notebooks/06_large_document_hierarchical_rag_lab.ipynb` implements provenance blocks,
  contextual/parent indexes, routing, decomposition, reference expansion and token packing.
- `solutions/06_large_document_hierarchical_rag_solution.ipynb` is the worked version.
- `notebooks/07_large_document_rag_evals_lab.ipynb` implements evidence metrics, a factorial
  ablation, first-failure attribution, numeric/citation checks, OCR and position slices,
  paired bootstrap uncertainty and a release gate.
- `solutions/07_large_document_rag_evals_solution.ipynb` is the worked version.
- `data/large_document_benchmark.json` is a deterministic fictional 90-block benchmark
  covering a 124-page annual report, MSA, amendment, tables, footnotes, multi-hop questions,
  conflicts and unanswerables.

The notebooks use transparent offline stand-ins so the arithmetic and experimental design
are reproducible. Their production-extension tables identify where to substitute layout
parsers, embeddings, rerankers, visual retrieval, LLM readers and calibrated judges.
