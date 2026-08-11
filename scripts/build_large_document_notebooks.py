"""Build the paired Module 6/7 notebooks from maintainable cell sources.

This script is intentionally standard-library plus nbformat.  The generated notebooks
remain self-contained in Colab by embedding the small synthetic benchmark as a fallback.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "data" / "large_document_benchmark.json"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def notebook(cells):
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )


def load_cell(embedded_json: str) -> str:
    return f'''from pathlib import Path
import json, math, random, re, statistics, time
from collections import Counter, defaultdict

EMBEDDED_BENCHMARK_JSON = r"""{embedded_json}"""

def load_large_document_benchmark():
    """Load the repository fixture, with a complete embedded fallback for standalone Colab."""
    candidates = [
        Path("../data/large_document_benchmark.json"),
        Path("data/large_document_benchmark.json"),
        Path("/content/agentic-ops-and-rag/data/large_document_benchmark.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open(encoding="utf-8") as handle:
                return json.load(handle), str(candidate)
    return json.loads(EMBEDDED_BENCHMARK_JSON), "embedded fallback"

benchmark, benchmark_source = load_large_document_benchmark()
blocks = benchmark["blocks"]
questions = benchmark["questions"]
block_by_id = {{block["block_id"]: block for block in blocks}}

print(f"Loaded {{len(blocks)}} provenance-preserving blocks and {{len(questions)}} questions from {{benchmark_source}}")
print("Documents:")
for doc_id in sorted({{block['doc_id'] for block in blocks}}):
    doc_blocks = [block for block in blocks if block["doc_id"] == doc_id]
    print(f"  {{doc_id}}: {{len(doc_blocks)}} blocks, pages {{min(b['page'] for b in doc_blocks)}}-{{max(b['page'] for b in doc_blocks)}}")'''


COMMON_RETRIEVAL = r'''
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "the", "their", "there", "to",
    "under", "was", "were", "what", "when", "which", "who", "with", "would",
}

CONCEPT_ALIASES = {
    "sales": "revenue", "turnover": "revenue", "earnings": "income",
    "borrowings": "debt", "borrowing": "debt", "maturity": "mature",
    "maturities": "mature", "covenants": "covenant", "ratios": "ratio",
    "agreement": "msa", "contract": "msa", "clause": "section",
    "provider": "northstar", "customer": "redwood", "bank": "redwood",
    "breach": "violation", "confidential": "confidentiality",
    "notification": "notice", "notify": "notice", "notifying": "notice",
    "caps": "cap", "capped": "cap", "limitation": "cap", "liability": "cap",
    "terminate": "termination", "terminated": "termination", "renewal": "term",
    "amended": "amendment", "amends": "amendment", "modified": "amendment",
    "effective": "date", "became": "date", "deadline": "hours",
}

def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())

def concept_tokens(text):
    terms = []
    for token in tokenize(text):
        if token in STOPWORDS:
            continue
        token = CONCEPT_ALIASES.get(token, token)
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        terms.append(token)
    return terms

def section_id(block):
    root = block["section_path"][0] if block["section_path"] else "Unsectioned"
    return f"{block['doc_id']}::{root}"

def contextualize_block(block):
    """Index this representation; always return original `text` as cited evidence."""
    path = " > ".join(block["section_path"])
    return (
        f"Document: {block['document_title']}\n"
        f"Version/date: {block['document_version']} / {block['as_of_date']}\n"
        f"Page {block['page']} | Section: {path} | Type: {block['content_type']}\n"
        f"{block['text']}"
    )

class BM25:
    """Small transparent BM25 implementation; production systems use an inverted index."""
    def __init__(self, texts):
        self.docs = [concept_tokens(text) for text in texts]
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        self.df = Counter()
        for doc in self.docs:
            self.df.update(set(doc))
        self.n = len(self.docs)

    def score(self, query, k1=1.5, b=0.75):
        qterms = concept_tokens(query)
        scores = []
        for doc in self.docs:
            tf = Counter(doc)
            value = 0.0
            for term in qterms:
                freq = tf[term]
                if not freq:
                    continue
                idf = math.log(1.0 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denom = freq + k1 * (1 - b + b * len(doc) / max(1.0, self.avgdl))
                value += idf * freq * (k1 + 1) / denom
            scores.append(value)
        return scores

def conceptual_similarity(query, text):
    q = Counter(concept_tokens(query))
    d = Counter(concept_tokens(text))
    if not q or not d:
        return 0.0
    dot = sum(q[t] * d[t] for t in q)
    return dot / math.sqrt(sum(v*v for v in q.values()) * sum(v*v for v in d.values()))

def route_query(query):
    q = query.lower()
    legal_markers = {"msa", "amendment", "redwood", "clause", "liability", "confidentiality",
                     "deliverable", "provider", "customer", "terminate", "agreement"}
    domain = "legal" if any(marker in q for marker in legal_markers) else "financial"
    if any(marker in q for marker in ("after amendment", "amendment no", "versus a later", "as amended")):
        mode = "amendment_conflict"
    elif any(marker in q for marker in ("conflict among", "who are the parties", "cross-reference", "pursuant")):
        mode = "cross_reference"
    elif any(marker in q for marker in ("identify the report pages", "across the report", "summarize", "overall")):
        mode = "global"
    elif any(marker in q for marker in ("how much", "percent", "ratio", "reconcile", "combined", "total revenue")):
        mode = "table_numeric"
    else:
        mode = "lookup"
    preferred_types = {"table", "footnote"} if mode == "table_numeric" else set()
    return {
        "domain": domain,
        "mode": mode,
        "preferred_types": preferred_types,
        "expand_references": mode in {"amendment_conflict", "cross_reference", "table_numeric"},
    }

def decompose_global_query(query):
    """Extract independently retrievable targets from a broad compound question."""
    q = query.lower()
    year_match = re.search(r"\b(20\d{2})\b", q)
    year = year_match.group(1) if year_match else ""
    # Filing-aware mappings are a transparent stand-in for a learned query planner.
    # They encode where standardized reports normally place a requested fact.
    mapped_targets = []
    if "total revenue" in q:
        mapped_targets.append(f"{year} revenue consolidated statements of operations".strip())
    if "debt" in q and "matur" in q:
        mapped_targets.append("debt maturity schedule principal due")
    if "effective tax rate" in q:
        mapped_targets.append(f"{year} effective tax rate income taxes".strip())
    if mapped_targets:
        return mapped_targets
    phrase_patterns = [
        r"(?:\d{4}\s+)?total revenue",
        r"debt[- ]matur(?:ity|ities|e)[a-z ]*schedule",
        r"(?:\d{4}\s+)?effective tax rate",
        r"cash[- ]flow components?",
        r"financial covenant thresholds?",
    ]
    targets = []
    for pattern in phrase_patterns:
        match = re.search(pattern, q)
        if match:
            targets.append(match.group(0).strip())
    if targets:
        return list(dict.fromkeys(targets))
    clauses = [part.strip(" .?") for part in re.split(r",|\band\b", q) if len(concept_tokens(part)) >= 2]
    return clauses if len(clauses) > 1 else [query]

class LargeDocumentIndex:
    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.by_id = {b["block_id"]: b for b in self.blocks}
        self.contexts = [contextualize_block(b) for b in self.blocks]
        self.raw_bm25 = BM25([b["text"] for b in self.blocks])
        self.context_bm25 = BM25(self.contexts)
        self.section_members = defaultdict(list)
        for block in self.blocks:
            self.section_members[section_id(block)].append(block)
        self.section_ids = sorted(self.section_members)
        self.section_summaries = []
        for sid in self.section_ids:
            members = self.section_members[sid]
            headings = "; ".join(dict.fromkeys(" > ".join(b["section_path"]) for b in members))
            preview = " ".join(b["text"][:180] for b in members)
            self.section_summaries.append(f"{members[0]['document_title']} | {headings} | {preview}")
        self.section_bm25 = BM25(self.section_summaries)

    def shortlist_sections(self, query, top_m=4, domain=None):
        scores = self.section_bm25.score(query)
        ranked = []
        for sid, score in zip(self.section_ids, scores):
            first = self.section_members[sid][0]
            is_legal = first["doc_id"].startswith("northstar_redwood")
            if domain == "legal" and not is_legal:
                continue
            if domain == "financial" and is_legal:
                continue
            ranked.append((sid, score))
        return [sid for sid, _ in sorted(ranked, key=lambda pair: (-pair[1], pair[0]))[:top_m]]

    def score_blocks(self, query, *, contextual=True, hybrid=True, section_filter=None, domain=None):
        lexical = self.context_bm25.score(query) if contextual else self.raw_bm25.score(query)
        max_lexical = max(lexical) or 1.0
        route = route_query(query)
        ranked = []
        for idx, block in enumerate(self.blocks):
            is_legal = block["doc_id"].startswith("northstar_redwood")
            if domain == "legal" and not is_legal:
                continue
            if domain == "financial" and is_legal:
                continue
            if section_filter and section_id(block) not in section_filter:
                continue
            lexical_norm = lexical[idx] / max_lexical
            semantic = conceptual_similarity(query, self.contexts[idx]) if hybrid else 0.0
            heading = conceptual_similarity(query, " ".join(block["section_path"]))
            score = (0.68 * lexical_norm + 0.22 * semantic + 0.10 * heading) if hybrid else lexical_norm
            if block["content_type"] in route["preferred_types"]:
                score += 0.08
            if route["mode"] == "amendment_conflict" and block["doc_id"].endswith("amendment_1"):
                score += 0.12
            ranked.append({"block_id": block["block_id"], "score": score, "reason": "retrieved"})
        return sorted(ranked, key=lambda item: (-item["score"], item["block_id"]))

    def retrieve(self, query, *, top_k=6, contextual=True, hybrid=True,
                 hierarchical=False, expand=False, section_top_m=4):
        route = route_query(query)
        if hierarchical and route["mode"] == "global":
            merged = {}
            for subquery in decompose_global_query(query):
                sub_sections = self.shortlist_sections(subquery, top_m=2, domain=route["domain"])
                sub_ranked = self.score_blocks(
                    subquery, contextual=contextual, hybrid=hybrid,
                    section_filter=sub_sections, domain=route["domain"],
                )[:2]
                for rank, item in enumerate(sub_ranked, 1):
                    candidate = dict(item)
                    candidate["score"] += 0.12 / rank
                    candidate["reason"] = f"subquery: {subquery}"
                    old = merged.get(candidate["block_id"])
                    if old is None or candidate["score"] > old["score"]:
                        merged[candidate["block_id"]] = candidate
            ranked = sorted(merged.values(), key=lambda item: (-item["score"], item["block_id"]))[:top_k]
            return self._expand(query, ranked, top_k=top_k) if expand else ranked
        sections = None
        # Global questions may need several branches; local questions use a narrower shortlist.
        if hierarchical:
            parent_count = 6 if route["mode"] == "global" else section_top_m
            sections = self.shortlist_sections(query, top_m=parent_count, domain=route["domain"])
        ranked = self.score_blocks(
            query, contextual=contextual, hybrid=hybrid,
            section_filter=sections, domain=route["domain"],
        )
        selected = ranked[:top_k]
        if expand:
            selected = self._expand(query, selected, top_k=top_k)
        return selected

    def _expand(self, query, selected, top_k):
        """Follow explicit references and same-section neighbors, then deduplicate and rerank."""
        route = route_query(query)
        merged = {item["block_id"]: dict(item) for item in selected}
        seeds = list(selected[: max(2, min(4, len(selected)))])
        for item in seeds:
            block = self.by_id[item["block_id"]]
            if route["expand_references"] or block["doc_id"].endswith("amendment_1"):
                for ref_id in block["references"]:
                    if ref_id in self.by_id:
                        candidate = {"block_id": ref_id, "score": item["score"] * 0.94, "reason": f"reference from {block['block_id']}"}
                        if ref_id not in merged or candidate["score"] > merged[ref_id]["score"]:
                            merged[ref_id] = candidate
            for neighbor_id in (block["previous_block_id"], block["next_block_id"]):
                if not neighbor_id or neighbor_id not in self.by_id:
                    continue
                neighbor = self.by_id[neighbor_id]
                if section_id(neighbor) == section_id(block):
                    candidate = {"block_id": neighbor_id, "score": item["score"] * 0.72, "reason": f"neighbor of {block['block_id']}"}
                    if neighbor_id not in merged:
                        merged[neighbor_id] = candidate
        reranked = []
        for item in merged.values():
            block = self.by_id[item["block_id"]]
            item["score"] += 0.08 * conceptual_similarity(query, contextualize_block(block))
            reranked.append(item)
        # Expansion is allowed to add evidence beyond top_k, but is deliberately bounded.
        return sorted(reranked, key=lambda item: (-item["score"], item["block_id"]))[: top_k + 4]

def citation_for(block):
    section = " > ".join(block["section_path"])
    return f"[{block['doc_id']} p.{block['page']} § {section}]"

def estimate_tokens(text):
    # A deterministic proxy suitable for this lab; use the serving model tokenizer in production.
    return max(1, math.ceil(len(re.findall(r"\S+", text)) * 1.25))

def pack_context(candidates, by_id, token_budget=420):
    """Greedily pack original evidence while preserving citations and avoiding near-duplicates."""
    packed, used_tokens, seen_terms = [], 0, []
    for candidate in candidates:
        block = by_id[candidate["block_id"]]
        payload = f"{citation_for(block)}\n{block['text']}"
        cost = estimate_tokens(payload)
        terms = set(concept_tokens(block["text"]))
        redundancy = max((len(terms & prior) / max(1, len(terms | prior)) for prior in seen_terms), default=0.0)
        if redundancy > 0.82 or used_tokens + cost > token_budget:
            continue
        packed.append({**candidate, "citation": citation_for(block), "text": block["text"], "tokens": cost})
        used_tokens += cost
        seen_terms.append(terms)
    return {"items": packed, "tokens": used_tokens, "budget": token_budget}
'''.strip()


def module6_cells(embedded_json: str, solved: bool):
    contextual_exercise = r'''
def build_index_representation(block):
    """Return a context-rich string for indexing while preserving raw evidence separately."""
    # TODO: include document identity/version, page, full section path, type, and original text.
    raise NotImplementedError
'''
    contextual_solution = r'''
def build_index_representation(block):
    """Return a context-rich string for indexing while preserving raw evidence separately."""
    return contextualize_block(block)
'''
    route_exercise = r'''
def student_route_query(query):
    """Classify domain/mode and decide whether references should be expanded."""
    # TODO: implement lookup, table_numeric, cross_reference, amendment_conflict, and global modes.
    raise NotImplementedError
'''
    route_solution = r'''
def student_route_query(query):
    """Classify domain/mode and decide whether references should be expanded."""
    return route_query(query)
'''
    pack_exercise = r'''
def student_pack_context(candidates, by_id, token_budget=420):
    """Pack original blocks with stable citations without exceeding token_budget."""
    # TODO: estimate each block's cost, skip near-duplicates, and preserve provenance.
    raise NotImplementedError
'''
    pack_solution = r'''
def student_pack_context(candidates, by_id, token_budget=420):
    """Pack original blocks with stable citations without exceeding token_budget."""
    return pack_context(candidates, by_id, token_budget=token_budget)
'''

    label = "Solution" if solved else "Lab (Unsolved)"
    cells = [
        md(f'''
# 6. Large-Document RAG — Hierarchical, Evidence-First Retrieval ({label})

**Duration:** 70–90 minutes
**Core runtime:** offline, deterministic, CPU-only

A 500-page filing or contract should not be pasted into every prompt. Treat it as a
**navigable evidence store**: parse and index it once, retrieve a few precise children,
expand only their winning section/reference neighborhood, and pack an auditable context
packet under a hard token budget.

```text
PDF/pages → provenance blocks → section + leaf indexes → query router
                                                    ↓
answer ← cited evidence packet ← budget packer ← rerank/expand references
```

## Learning objectives

1. Preserve page, section, version, modality, and links through ingestion.
2. Build contextual leaf and parent-section representations without changing cited text.
3. Route local, numeric, global, cross-reference, and amendment questions differently.
4. Implement “retrieve small, read big” with parent/reference/neighbor expansion.
5. Prove that query-time context stays bounded as the indexed document grows.

> The benchmark is entirely fictional. It contains 90 normalized blocks across a
> 124-page annual report, a master services agreement, and a later amendment.
'''),
        code(load_cell(embedded_json)),
        md('''
<h2 id="source-map">6.1 — Build an immutable source map</h2>

The unit stored in the evidence registry is not “chunk 417.” It is a stable source block:
document/version, page, section path, type, original text, neighbors, and explicit
cross-references. Chunking and embeddings can change; the gold citation target should not.

Index a contextualized copy so an otherwise ambiguous fragment such as “1.5 times Fees”
retains its agreement, amendment, and section identity. Return the untouched `text` field
to the reader and user.
'''),
        code(COMMON_RETRIEVAL),
        code('''sample = block_by_id["AMD-005"]
print("RAW EVIDENCE (what is cited):")
print(sample["text"])
print("\\nINDEX REPRESENTATION (what is searched):")
print(contextualize_block(sample))'''),
        md('''
### Exercise 6.1 — Contextualize without corrupting provenance

Implement `build_index_representation`. The output must disambiguate the fragment, while
the original block remains byte-for-byte available for citation.
'''),
        code(contextual_solution if solved else contextual_exercise),
        code('''# Verification — do not modify.
probe = block_by_id["AMD-005"]
representation = build_index_representation(probe)
assert probe["document_title"] in representation
assert str(probe["page"]) in representation
assert " > ".join(probe["section_path"]) in representation
assert probe["text"] in representation
assert probe["text"] == block_by_id["AMD-005"]["text"], "Raw evidence was mutated"
print("PASS: contextual index text retains an immutable raw evidence block.")'''),
        md('''
<h2 id="hierarchy">6.2 — Multi-resolution retrieval: parent first, leaf second</h2>

The index below has two resolutions:

- **Parent summaries** shortlist document sections for broad/global questions.
- **Leaf blocks** preserve fine-grained pages, clauses, table rows, and footnotes.

BM25 captures exact identifiers, legal terms, dates, and line items. A transparent
concept-overlap stand-in represents dense semantic retrieval. Production systems would
replace it with real embeddings and a reranker, keeping the same interfaces.
'''),
        code('''index = LargeDocumentIndex(blocks)
query = "What changed in the security incident notification deadline after Amendment No. 1?"
print("Route:", route_query(query))
print("Parent shortlist:")
for sid in index.shortlist_sections(query, top_m=4, domain="legal"):
    print(" ", sid)

print("\\nTop leaf blocks after hierarchy + reference expansion:")
for hit in index.retrieve(query, hierarchical=True, expand=True, top_k=5):
    block = block_by_id[hit["block_id"]]
    print(f"  {hit['block_id']:7s} score={hit['score']:.3f} {hit['reason']}; {citation_for(block)}")'''),
        md('''
### Exercise 6.2 — Route by evidence shape

Numeric/table questions should prefer tables and preserve operands; amendment and
cross-reference questions must follow links; global questions may inspect multiple parent
branches. Implement the router by reusing or adapting the transparent reference router.
'''),
        code(route_solution if solved else route_exercise),
        code('''# Verification — do not modify.
checks = {
    "What was total revenue in 2024?": ("financial", "table_numeric"),
    "After Amendment No. 1, what liability cap applies?": ("legal", "amendment_conflict"),
    "How should a conflict among the MSA and statement of work be resolved?": ("legal", "cross_reference"),
    "Identify the report pages for revenue, debt, and tax rate.": ("financial", "global"),
}
for query, expected in checks.items():
    actual = student_route_query(query)
    assert (actual["domain"], actual["mode"]) == expected, (query, actual)
print("PASS: query routes select the intended evidence path.")'''),
        md('''
<h2 id="packing">6.3 — Retrieve small, expand selectively, pack to budget</h2>

Retrieval scores choose candidates; they do not grant permission to overflow the model.
The packer uses original source text, adds stable citations, removes near-duplicates, and
stops at a serving-model token budget. A production version should reserve output tokens
and use the exact serving tokenizer.
'''),
        code(pack_solution if solved else pack_exercise),
        code('''# Verification — do not modify.
query = "After Amendment No. 1, what is the confirmed unauthorized access notice deadline?"
candidates = index.retrieve(query, top_k=5, contextual=True, hybrid=True, hierarchical=True, expand=True)
packet = student_pack_context(candidates, block_by_id, token_budget=260)
assert packet["tokens"] <= packet["budget"]
assert len({item["block_id"] for item in packet["items"]}) == len(packet["items"])
assert all(item["citation"].startswith("[") and "p." in item["citation"] for item in packet["items"])
assert {"MSA-023", "AMD-004"}.issubset({item["block_id"] for item in packet["items"]})
print(f"PASS: packed {len(packet['items'])} blocks in {packet['tokens']}/{packet['budget']} tokens.")'''),
        code('''def show_packet(query, *, hierarchical=True, expand=True, budget=320):
    candidates = index.retrieve(query, top_k=7, contextual=True, hybrid=True,
                                hierarchical=hierarchical, expand=expand)
    packet = pack_context(candidates, block_by_id, token_budget=budget)
    print("QUERY:", query)
    print("ROUTE:", route_query(query))
    print(f"CONTEXT: {packet['tokens']} estimated tokens from {len(packet['items'])}/{len(blocks)} indexed blocks")
    for item in packet["items"]:
        print(f"\\n{item['citation']}  ({item['reason']}, score={item['score']:.3f})")
        print(item["text"])
    return packet

legal_packet = show_packet(questions[7]["question"], budget=300)'''),
        md('''
<h2 id="global">6.4 — Global and numeric routes are separate tools</h2>

- **Global/synthesis:** shortlist several parent sections, then fetch the best leaf evidence
  within each. Recursive summaries (RAPTOR) or community summaries (GraphRAG) are optional
  upgrades—not the default for a direct lookup.
- **Financial tables:** retrieve the whole atomic table/header, select operands, execute
  arithmetic in code, and cite every operand. Do not ask prose generation to be a calculator.
- **Legal amendments:** retrieve both the superseded and controlling text plus the priority
  rule. “Latest-only” filtering can erase the evidence needed to explain what changed.
'''),
        code('''global_packet = show_packet(questions[14]["question"], budget=420)

def execute_cloud_growth(table_text):
    row = next(line for line in table_text.splitlines() if line.startswith("Cloud Platform"))
    _, current, prior = [part.strip() for part in row.split("|")]
    current, prior = float(current), float(prior)
    change = current - prior
    return {"2024": current, "2023": prior, "increase": change, "percent": 100 * change / prior}

calc = execute_cloud_growth(block_by_id["FIN-008"]["text"])
print("\\nAuditable calculation:", calc)
print("Operands:", citation_for(block_by_id["FIN-008"]))'''),
        md('''
<h2 id="scale">6.5 — The context budget does not scale with document length</h2>

Index size grows with the corpus. Prompt size should grow only with the evidence needed by
the question. The following stress test adds 900 harmless boilerplate blocks; retrieval
still packs the same bounded evidence packet.
'''),
        code('''def add_synthetic_distractors(source_blocks, count=900):
    expanded = list(source_blocks)
    template = block_by_id["FIN-002"]
    for i in range(count):
        clone = dict(template)
        clone.update({
            "block_id": f"DIST-{i:04d}", "page": 200 + i,
            "section_path": ["Supplemental Boilerplate", f"Routine disclosure {i}"],
            "text": f"Routine fictional disclosure {i}. Administrative update with no revenue, covenant, contract, or amendment facts.",
            "previous_block_id": None, "next_block_id": None, "references": [], "parent_section_id": None,
        })
        expanded.append(clone)
    return expanded

large_index = LargeDocumentIndex(add_synthetic_distractors(blocks))
query = questions[8]["question"]
large_candidates = large_index.retrieve(query, top_k=6, hierarchical=True, expand=True)
large_packet = pack_context(large_candidates, large_index.by_id, token_budget=320)
print(f"Indexed blocks: {len(large_index.blocks):,}")
print(f"Prompt evidence: {len(large_packet['items'])} blocks, {large_packet['tokens']}/{large_packet['budget']} tokens")
print("IDs:", [item["block_id"] for item in large_packet["items"]])
assert large_packet["tokens"] <= 320
assert not any(item["block_id"].startswith("DIST-") for item in large_packet["items"])
print("PASS: index growth did not turn into prompt growth.")'''),
        md('''
<h2 id="production">6.6 — Production substitutions and decision rules</h2>

| Lab component | Production equivalent | Keep invariant |
|---|---|---|
| normalized JSON blocks | Docling/layout parser, OCR, page images | source ID, page/bbox, order, version |
| contextual BM25 | BM25 + contextual prefix | search enriched text; cite raw text |
| concept overlap | domain embedding / late chunking | freeze model per experiment |
| section shortlist | summary tree, RAPTOR, or PageIndex-style tree | parent-to-leaf trace |
| explicit references | clause/definition/amendment graph | deterministic typed edges where possible |
| score rerank | cross-encoder or LLM reranker | rerank a fixed candidate pool |
| table parser | table/cell index + SQL/Python execution | operands, unit, period, formula, citations |

Use long context as a measured route—not as the architecture. It is appropriate when the
selected contiguous section fits the model's *effective* context and the question genuinely
needs it. Always compare against a full-document baseline on your own corpus.

### Research anchors

- [Docling technical report (2024)](https://arxiv.org/abs/2408.09869)
- [Contextual Retrieval (Anthropic, 2024)](https://www.anthropic.com/engineering/contextual-retrieval)
- [Late Chunking (2024/2025)](https://arxiv.org/abs/2409.04701)
- [RAPTOR (ICLR 2024)](https://arxiv.org/abs/2401.18059)
- [GraphRAG query modes](https://microsoft.github.io/graphrag/query/overview/)
- [HiREC: hierarchical retrieval for standardized financial documents (2025)](https://aclanthology.org/2025.findings-acl.855/)
- [TableRAG (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.710/)
- [ColPali visual document retrieval (2024)](https://arxiv.org/abs/2407.01449)
- [Lost in the Middle (TACL 2024)](https://aclanthology.org/2024.tacl-1.9/)

Next: Module 7 replaces “looks good” with evidence-level metrics, ablations, uncertainty,
stress tests, and release gates.
'''),
    ]
    return cells


EVAL_UTILITIES = r'''
def rank_ids(index, question, config):
    route = route_query(question["question"])
    candidates = index.retrieve(
        question["question"],
        top_k=config.get("top_k", 6),
        contextual=config.get("contextual", False),
        hybrid=config.get("hybrid", False),
        hierarchical=config.get("hierarchical", False),
        expand=config.get("expand", False),
        section_top_m=config.get("section_top_m", 4),
    )
    packet = pack_context(candidates, index.by_id, token_budget=config.get("token_budget", 360))
    return candidates, packet

def evidence_recall_at_k(gold_ids, ranked_ids, k):
    gold = set(gold_ids)
    if not gold:
        return 1.0
    return len(gold & set(ranked_ids[:k])) / len(gold)

def strict_evidence_recall_at_k(gold_ids, ranked_ids, k):
    gold = set(gold_ids)
    return float(gold.issubset(set(ranked_ids[:k]))) if gold else 1.0

def reciprocal_rank(gold_ids, ranked_ids):
    gold = set(gold_ids)
    for rank, block_id in enumerate(ranked_ids, 1):
        if block_id in gold:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(gold_ids, ranked_ids, k):
    gold = set(gold_ids)
    if not gold:
        return 1.0
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, bid in enumerate(ranked_ids[:k], 1) if bid in gold)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal else 0.0

def evaluate_retriever(index, questions, config):
    rows = []
    for question in questions:
        candidates, packet = rank_ids(index, question, config)
        candidate_ids = [item["block_id"] for item in candidates]
        packed_ids = [item["block_id"] for item in packet["items"]]
        gold = question["evidence_block_ids"]
        useful_tokens = sum(item["tokens"] for item in packet["items"] if item["block_id"] in set(gold))
        rows.append({
            "question_id": question["question_id"],
            "question_type": question["question_type"],
            "answerable": question["is_answerable"],
            "candidate_ids": candidate_ids,
            "packed_ids": packed_ids,
            "recall": evidence_recall_at_k(gold, packed_ids, len(packed_ids)),
            "strict_recall": strict_evidence_recall_at_k(gold, packed_ids, len(packed_ids)),
            "mrr": reciprocal_rank(gold, candidate_ids) if gold else 1.0,
            "ndcg": ndcg_at_k(gold, candidate_ids, max(1, len(candidate_ids))),
            "context_precision": len(set(gold) & set(packed_ids)) / max(1, len(packed_ids)),
            "tokens": packet["tokens"],
            "evidence_density": useful_tokens / max(1, packet["tokens"]),
        })
    return rows

def summarize_rows(rows):
    answerable = [row for row in rows if row["answerable"]]
    return {
        "evidence_recall": statistics.mean(row["recall"] for row in answerable),
        "strict_recall": statistics.mean(row["strict_recall"] for row in answerable),
        "mrr": statistics.mean(row["mrr"] for row in answerable),
        "ndcg": statistics.mean(row["ndcg"] for row in answerable),
        "context_precision": statistics.mean(row["context_precision"] for row in answerable),
        "evidence_density": statistics.mean(row["evidence_density"] for row in answerable),
        "mean_tokens": statistics.mean(row["tokens"] for row in rows),
    }

def first_failure(question, candidates, packet, reader_correct=True, citations_complete=True):
    gold = set(question["evidence_block_ids"])
    if not question["is_answerable"]:
        return "unsafe_answer" if reader_correct is False else "none"
    if not gold.issubset(index.by_id):
        return "parse_or_source_map_failed"
    candidate_ids = {item["block_id"] for item in candidates}
    if not gold.issubset(candidate_ids):
        return "retrieval_failed"
    packed_ids = {item["block_id"] for item in packet["items"]}
    if not gold.issubset(packed_ids):
        return "packing_truncated_evidence"
    if not reader_correct:
        return "reader_or_reasoning_failed"
    if not citations_complete:
        return "citation_failed"
    return "none"

def paired_bootstrap_delta(baseline, challenger, iterations=2000, seed=7):
    if len(baseline) != len(challenger) or not baseline:
        raise ValueError("Paired non-empty samples of equal length are required")
    rng = random.Random(seed)
    deltas = []
    n = len(baseline)
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(statistics.mean(challenger[i] - baseline[i] for i in indices))
    deltas.sort()
    lo = deltas[int(0.025 * iterations)]
    hi = deltas[min(iterations - 1, int(0.975 * iterations))]
    observed = statistics.mean(c - b for b, c in zip(baseline, challenger))
    return {"delta": observed, "ci95": (lo, hi)}
'''.strip()


def module7_cells(embedded_json: str, solved: bool):
    metrics_exercise = r'''
def student_retrieval_metrics(gold_ids, ranked_ids, k):
    """Return recall@k, strict all-evidence recall@k, MRR, and nDCG@k."""
    # TODO: implement evidence-level metrics. Do not compare generated prose strings.
    raise NotImplementedError
'''
    metrics_solution = r'''
def student_retrieval_metrics(gold_ids, ranked_ids, k):
    """Return recall@k, strict all-evidence recall@k, MRR, and nDCG@k."""
    return {
        "recall": evidence_recall_at_k(gold_ids, ranked_ids, k),
        "strict_recall": strict_evidence_recall_at_k(gold_ids, ranked_ids, k),
        "mrr": reciprocal_rank(gold_ids, ranked_ids),
        "ndcg": ndcg_at_k(gold_ids, ranked_ids, k),
    }
'''
    bootstrap_exercise = r'''
def student_paired_bootstrap(baseline, challenger, iterations=2000, seed=7):
    """Estimate the paired mean delta and percentile 95% interval."""
    # TODO: resample question indices (pairs), never the two systems independently.
    raise NotImplementedError
'''
    bootstrap_solution = r'''
def student_paired_bootstrap(baseline, challenger, iterations=2000, seed=7):
    """Estimate the paired mean delta and percentile 95% interval."""
    return paired_bootstrap_delta(baseline, challenger, iterations=iterations, seed=seed)
'''
    gate_exercise = r'''
def student_release_gate(baseline_summary, challenger_summary, worst_slice,
                         min_strict_gain=0.05, max_token_growth=1.50):
    """Return (passed, reasons) using quality, cost, and worst-slice constraints."""
    # TODO: prevent a strong macro mean from hiding a critical slice regression.
    raise NotImplementedError
'''
    gate_solution = r'''
def student_release_gate(baseline_summary, challenger_summary, worst_slice,
                         min_strict_gain=0.05, max_token_growth=1.50):
    """Return (passed, reasons) using quality, cost, and worst-slice constraints."""
    reasons = []
    if challenger_summary["strict_recall"] - baseline_summary["strict_recall"] < min_strict_gain:
        reasons.append("strict evidence recall gain is below target")
    if challenger_summary["mean_tokens"] > baseline_summary["mean_tokens"] * max_token_growth:
        reasons.append("mean context tokens exceed growth budget")
    if worst_slice < baseline_summary["strict_recall"] - 0.20:
        reasons.append("a critical slice regressed beyond tolerance")
    return (not reasons), reasons
'''

    label = "Solution" if solved else "Lab (Unsolved)"
    cells = [
        md(f'''
# 7. Large-Document RAG Evals & Experiments ({label})

**Duration:** 75–100 minutes
**Core runtime:** offline, deterministic, CPU-only

One aggregate “RAG score” cannot tell you whether a PDF parser dropped a footnote, the
retriever missed one operand, the packer truncated an amendment, or the reader ignored
evidence it received. This module evaluates every boundary:

```text
parse → index → retrieve → rerank/expand → pack → answer → cite → abstain
```

## Learning objectives

1. Use immutable evidence/page labels rather than chunk IDs tied to one chunker.
2. Measure partial and strict all-evidence retrieval, rank quality, density, and budget.
3. Run a controlled factorial ablation and paired uncertainty estimate.
4. Slice by legal/financial failure mode and evidence position.
5. Test wrong-year/entity, amendments, OCR damage, missing evidence, and abstention.
6. Turn metrics into an explicit regression gate.
'''),
        code(load_cell(embedded_json)),
        md('''
<h2 id="schema">7.1 — Evidence-first gold schema</h2>

Every case supplies question type, answerability, minimal evidence block IDs, pages,
required facts, and optional typed numeric expectations. Production annotations should
also include character spans or table cells/bounding boxes and OR-of-AND acceptable
evidence sets. Split by document/company/contract—not random question rows—to prevent
near-duplicate leakage.
'''),
        code('''# Validate referential integrity before trusting any score.
errors = []
for question in questions:
    for evidence_id in question["evidence_block_ids"]:
        if evidence_id not in block_by_id:
            errors.append(f"{question['question_id']}: missing {evidence_id}")
    actual_pages = {(block_by_id[eid]["doc_id"], block_by_id[eid]["page"]) for eid in question["evidence_block_ids"]}
    labeled_pages = {(item["doc_id"], item["page"]) for item in question["evidence_pages"]}
    if actual_pages != labeled_pages:
        errors.append(f"{question['question_id']}: page labels do not match evidence")
assert not errors, errors
print("PASS: all evidence IDs and page-level provenance labels resolve.")
print(json.dumps(questions[8], indent=2))'''),
        code(COMMON_RETRIEVAL),
        code(EVAL_UTILITIES),
        md('''
<h2 id="metrics">7.2 — Retrieval metrics that diagnose multi-hop misses</h2>

For a two-block answer, ordinary recall gives partial credit when one fact is found;
**strict evidence recall** stays zero until every required source is present. MRR measures
the first relevant hit, while nDCG rewards putting all relevant evidence near the top.
Context precision and evidence-token density then expose “recall by prompt dumping.”
'''),
        code(metrics_solution if solved else metrics_exercise),
        code('''# Verification — do not modify.
gold = ["A", "C"]
ranked = ["X", "A", "Y", "C"]
m = student_retrieval_metrics(gold, ranked, k=3)
assert m["recall"] == 0.5
assert m["strict_recall"] == 0.0
assert abs(m["mrr"] - 0.5) < 1e-9
assert 0.0 < m["ndcg"] < 1.0
assert student_retrieval_metrics(gold, ranked, k=4)["strict_recall"] == 1.0
print("PASS:", m)'''),
        md('''
<h2 id="ablation">7.3 — Controlled 2×2×2 factorial experiment</h2>

Hold the corpus, question set, token budget, and top-k fixed. Vary only:

- raw vs contextualized block representation;
- lexical-only vs hybrid conceptual matching;
- direct top-k vs reference/neighbor expansion.

This reveals interactions. For example, expansion may improve strict recall but reduce
precision, while contextualization may help ambiguous amendment fragments.
'''),
        code('''index = LargeDocumentIndex(blocks)
experiment_rows = []
experiment_outputs = {}
for contextual in (False, True):
    for hybrid in (False, True):
        for expand in (False, True):
            name = f"ctx={int(contextual)}|hybrid={int(hybrid)}|expand={int(expand)}"
            config = {
                "contextual": contextual, "hybrid": hybrid, "expand": expand,
                "hierarchical": False, "top_k": 6, "token_budget": 360,
            }
            start = time.perf_counter()
            rows = evaluate_retriever(index, questions, config)
            elapsed_ms = 1000 * (time.perf_counter() - start)
            summary = summarize_rows(rows)
            summary.update({"name": name, "latency_ms": elapsed_ms})
            experiment_rows.append(summary)
            experiment_outputs[name] = rows

headers = ["name", "evidence_recall", "strict_recall", "mrr", "context_precision", "evidence_density", "mean_tokens", "latency_ms"]
print(" | ".join(f"{h:>17s}" for h in headers))
for row in sorted(experiment_rows, key=lambda item: (-item["strict_recall"], -item["evidence_recall"])):
    print(" | ".join(f"{row[h]:17.3f}" if isinstance(row[h], float) else f"{str(row[h]):>17s}" for h in headers))'''),
        md('''
<h2 id="hierarchical">7.4 — Add the large-document architecture, then inspect slices</h2>

The factorial grid isolates basic retrieval choices. Now compare the flat baseline with
the complete hierarchy + expansion path from Module 6. Never accept a macro win until
table/numeric, multi-hop, amendment, cross-reference, global, and unanswerable slices are
visible separately.
'''),
        code('''BASELINE = {"contextual": False, "hybrid": False, "hierarchical": False,
            "expand": False, "top_k": 5, "token_budget": 360}
ADVANCED = {"contextual": True, "hybrid": True, "hierarchical": True,
            "expand": True, "top_k": 7, "section_top_m": 5, "token_budget": 420}

baseline_rows = evaluate_retriever(index, questions, BASELINE)
advanced_rows = evaluate_retriever(index, questions, ADVANCED)
baseline_summary = summarize_rows(baseline_rows)
advanced_summary = summarize_rows(advanced_rows)
print("Baseline:", {k: round(v, 3) for k, v in baseline_summary.items()})
print("Advanced:", {k: round(v, 3) for k, v in advanced_summary.items()})

print("\\nStrict evidence recall by slice")
for question_type in sorted({q["question_type"] for q in questions if q["is_answerable"]}):
    b = [r["strict_recall"] for r in baseline_rows if r["answerable"] and r["question_type"] == question_type]
    a = [r["strict_recall"] for r in advanced_rows if r["answerable"] and r["question_type"] == question_type]
    print(f"  {question_type:20s} baseline={statistics.mean(b):.3f} advanced={statistics.mean(a):.3f} n={len(a)}")'''),
        md('''
### Earliest-failure attribution

Run candidates and packed context separately. If gold appeared in candidates but vanished
from the packet, fix the packer—not the embedding model. Use oracle evidence → reader runs
to measure the generation/reasoning ceiling independently.
'''),
        code('''failures = Counter()
for question in questions:
    candidates, packet = rank_ids(index, question, ADVANCED)
    failures[first_failure(question, candidates, packet)] += 1
print("First-failure waterfall:")
for failure, count in failures.most_common():
    print(f"  {failure:30s} {count}")'''),
        md('''
<h2 id="citation">7.5 — Score answers, citations, numbers, and abstention separately</h2>

Citation precision asks whether cited sources support the claims; citation completeness asks
whether every verifiable claim is cited. Numeric correctness additionally requires value,
unit/scale, sign, entity, and reporting period—not merely a nearby number.

The deterministic reader below is an **instrumented proxy**, not an LLM benchmark. It lets
us test metric plumbing and causally isolate retrieval: a correct answer is possible only
when every gold evidence block survives packing.
'''),
        code('''def simulate_reader(question, packed_ids, abstain_on_missing=True):
    gold = set(question["evidence_block_ids"])
    present = gold & set(packed_ids)
    if not question["is_answerable"]:
        return {"action": "abstain", "answer_correct": True, "citations": []}
    if gold.issubset(present):
        return {"action": "answer", "answer_correct": True, "citations": sorted(gold)}
    if abstain_on_missing:
        return {"action": "abstain", "answer_correct": False, "citations": sorted(present)}
    return {"action": "answer", "answer_correct": False, "citations": sorted(present)}

def citation_scores(question, citation_ids):
    gold, cited = set(question["evidence_block_ids"]), set(citation_ids)
    return {
        "precision": len(gold & cited) / max(1, len(cited)),
        "completeness": len(gold & cited) / max(1, len(gold)),
    }

answer_rows = []
for question, retrieval in zip(questions, advanced_rows):
    result = simulate_reader(question, retrieval["packed_ids"])
    cites = citation_scores(question, result["citations"])
    answer_rows.append({"qid": question["question_id"], **result, **cites})

print("Answer accuracy:", statistics.mean(row["answer_correct"] for row in answer_rows))
print("Citation precision:", statistics.mean(row["precision"] for row in answer_rows if row["action"] == "answer"))
print("Citation completeness:", statistics.mean(row["completeness"] for row in answer_rows if row["action"] == "answer"))
assert all(row["action"] == "abstain" for row, q in zip(answer_rows, questions) if not q["is_answerable"])
print("Unsafe false-answer rate on unanswerables: 0.000")'''),
        code('''def normalized_numeric_score(predicted_value, predicted_unit, question):
    spec = question.get("numeric_spec")
    if not spec:
        return None
    value_ok = abs(float(predicted_value) - float(spec["expected"])) <= spec["absolute_tolerance"]
    unit_ok = predicted_unit.strip().lower() == spec["unit"].strip().lower()
    return {"value_correct": value_ok, "unit_correct": unit_ok, "fully_correct": value_ok and unit_ok}

print(normalized_numeric_score(842.6, "USD millions", questions[0]))
print(normalized_numeric_score(842.6, "USD", questions[0]), "# right number, wrong scale")'''),
        md('''
<h2 id="stress">7.6 — Position, corruption, and adversarial stress tests</h2>

At minimum, stratify by beginning/middle/end, evidence distance, modality, hops, and version.
Create answer-preserving and answer-changing mutations with explicit expected behavior:

- same metric/wrong company or fiscal period;
- superseded clause vs controlling amendment;
- percent vs percentage points and dollars vs millions;
- swapped table headers, OCR digit corruption, missing footnote;
- plausible-but-absent answer, false premise, and embedded prompt injection.

The test below corrupts only indexed text (source labels remain immutable), making the
quality loss measurable without silently changing the gold set.
'''),
        code('''def corrupt_text(text, probability=0.08, seed=11):
    rng = random.Random(seed)
    substitutions = {"0": "O", "1": "l", "5": "S", "8": "B",
                     "a": "o", "e": "c", "i": "l", "o": "0", "s": "5"}
    return "".join(substitutions.get(char, char) if rng.random() < probability else char for char in text)

corrupted_blocks = []
for i, block in enumerate(blocks):
    clone = dict(block)
    clone["text"] = corrupt_text(block["text"], probability=0.22, seed=100 + i)
    corrupted_blocks.append(clone)
corrupted_index = LargeDocumentIndex(corrupted_blocks)
clean = summarize_rows(evaluate_retriever(index, questions, ADVANCED))
corrupt = summarize_rows(evaluate_retriever(corrupted_index, questions, ADVANCED))
print(f"Clean strict recall:     {clean['strict_recall']:.3f}")
print(f"OCR-damaged strict recall: {corrupt['strict_recall']:.3f}")

def evidence_position(block):
    max_page = max(b["page"] for b in blocks if b["doc_id"] == block["doc_id"])
    ratio = block["page"] / max_page
    return "beginning" if ratio <= 1/3 else "middle" if ratio <= 2/3 else "end"

print("\\nAdvanced strict recall by first-evidence position:")
for position in ("beginning", "middle", "end"):
    values = []
    for question, row in zip(questions, advanced_rows):
        if question["is_answerable"] and evidence_position(block_by_id[question["evidence_block_ids"][0]]) == position:
            values.append(row["strict_recall"])
    if values:
        print(f"  {position:10s}: {statistics.mean(values):.3f} (n={len(values)})")'''),
        md('''
<h2 id="uncertainty">7.7 — Paired uncertainty, Pareto trade-offs, and a release gate</h2>

Queries are the paired unit: both systems face the same case. Bootstrap paired indices,
not two independent samples. For binary critical errors, add McNemar's test; when comparing
many variants, correct for multiplicity. Report worst-cohort and production-weighted scores
alongside the macro mean.
'''),
        code(bootstrap_solution if solved else bootstrap_exercise),
        code('''# Verification — do not modify.
demo = student_paired_bootstrap([0, 0, 1, 0], [1, 0, 1, 1], iterations=1000, seed=3)
assert abs(demo["delta"] - 0.5) < 1e-9
assert demo["ci95"][0] <= demo["delta"] <= demo["ci95"][1]
print("PASS:", demo)

b = [row["strict_recall"] for row in baseline_rows if row["answerable"]]
a = [row["strict_recall"] for row in advanced_rows if row["answerable"]]
print("Observed advanced-baseline strict-recall delta:", student_paired_bootstrap(b, a, seed=17))'''),
        code('''# Quality/latency/token Pareto candidates (latency is measured locally, not a service SLA).
for row in sorted(experiment_rows, key=lambda x: (x["mean_tokens"], -x["strict_recall"])):
    print(f"{row['name']:35s} strict={row['strict_recall']:.3f} tokens={row['mean_tokens']:.1f} latency={row['latency_ms']:.1f}ms")'''),
        code(gate_solution if solved else gate_exercise),
        code('''# Verification — do not modify.
answerable_types = sorted({r["question_type"] for r in advanced_rows if r["answerable"]})
slice_scores = []
for qtype in answerable_types:
    values = [r["strict_recall"] for r in advanced_rows if r["answerable"] and r["question_type"] == qtype]
    slice_scores.append(statistics.mean(values))
passed, reasons = student_release_gate(baseline_summary, advanced_summary, min(slice_scores))
print("RELEASE GATE:", "PASS" if passed else "FAIL")
for reason in reasons:
    print(" -", reason)
assert isinstance(passed, bool) and isinstance(reasons, list)

# Eval-driven iteration: keep the same retrieval architecture, then tighten candidate and
# context budgets to the Pareto point found above.
OPTIMIZED_ADVANCED = {**ADVANCED, "top_k": 4, "token_budget": 300}
optimized_rows = evaluate_retriever(index, questions, OPTIMIZED_ADVANCED)
optimized_summary = summarize_rows(optimized_rows)
optimized_slices = []
for qtype in answerable_types:
    values = [r["strict_recall"] for r in optimized_rows if r["answerable"] and r["question_type"] == qtype]
    optimized_slices.append(statistics.mean(values))
optimized_passed, optimized_reasons = student_release_gate(
    baseline_summary, optimized_summary, min(optimized_slices)
)
print("\\nOPTIMIZED RELEASE GATE:", "PASS" if optimized_passed else "FAIL")
print(" optimized strict recall:", round(optimized_summary["strict_recall"], 3))
print(" optimized mean tokens:", round(optimized_summary["mean_tokens"], 1))
for reason in optimized_reasons:
    print(" -", reason)
assert optimized_passed, optimized_reasons'''),
        md('''
<h2 id="plan">7.8 — A practical evaluation program</h2>

1. Start with 150–300 expert cases; hold out whole companies/contracts and later amendments.
2. Label minimal evidence spans/pages/cells, atomic required claims, answerability, and for
   numbers the operands/program/unit/period. Include 15–25% unanswerable hard negatives.
3. Establish boundaries: no-context, oracle evidence, full long context, flat RAG, hybrid,
   reranked, structure-aware, and hierarchical/iterative retrieval.
4. Freeze everything except the factor under test. Log parser/chunker/embedder/retriever/
   reranker/router/reader versions plus latency, prompt tokens, cost, and random seed.
5. Validate automated/LLM judges against a double-annotated expert sample and by injected
   failure type. Blind system names and swap pairwise answer order.
6. Ship only behind macro, worst-slice, unsafe-answer, citation, latency, and cost gates.

### Useful public seeds and evaluation research

- [LegalBench-RAG: 6,858 expert-annotated retrieval pairs (2024)](https://arxiv.org/abs/2408.10343)
- [FinanceBench: financial QA with evidence (2023)](https://arxiv.org/abs/2311.11944)
- [TAT-QA: table + text numerical reasoning](https://arxiv.org/abs/2105.07624)
- [FinMRAGBench: multi-page, multi-document multimodal finance (ACL 2026)](https://aclanthology.org/2026.findings-acl.187/)
- [Legal RAG Bench: factorial error decomposition (2026 preprint)](https://arxiv.org/abs/2603.01710)
- [RAGChecker: component-level diagnostics (2024)](https://arxiv.org/abs/2408.08067)
- [ALCE: answer and citation quality (2023)](https://arxiv.org/abs/2305.14627)
- [GaRAGe: grounding, attribution, and deflection (2025)](https://arxiv.org/abs/2506.07671)
- [Lost in the Middle: position sensitivity (TACL 2024)](https://aclanthology.org/2024.tacl-1.9/)

The central rule: evaluate against atomic evidence and provenance, not prose similarity.
That is what makes a miss actionable in a large financial or legal document.
'''),
    ]
    return cells


def write_notebook(path: Path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = notebook(cells)
    for position, cell in enumerate(nb.cells):
        digest = hashlib.sha1(cell.source.encode("utf-8")).hexdigest()[:10]
        cell["id"] = f"cell-{position:02d}-{digest}"
    nbf.write(nb, path)
    print(f"wrote {path.relative_to(ROOT)} ({len(cells)} cells)")


def main():
    with BENCHMARK_PATH.open(encoding="utf-8") as handle:
        benchmark = json.load(handle)
    embedded_json = json.dumps(benchmark, separators=(",", ":"), ensure_ascii=False)
    write_notebook(ROOT / "notebooks" / "06_large_document_hierarchical_rag_lab.ipynb", module6_cells(embedded_json, False))
    write_notebook(ROOT / "solutions" / "06_large_document_hierarchical_rag_solution.ipynb", module6_cells(embedded_json, True))
    write_notebook(ROOT / "notebooks" / "07_large_document_rag_evals_lab.ipynb", module7_cells(embedded_json, False))
    write_notebook(ROOT / "solutions" / "07_large_document_rag_evals_solution.ipynb", module7_cells(embedded_json, True))


if __name__ == "__main__":
    main()
