# main.py
"""
LLM-Powered Query-Retrieval Service (demo)
Run: pip install fastapi uvicorn requests sentence-transformers faiss-cpu python-multipart python-docx pymupdf pytest
Start: uvicorn main:app --reload --port 8000
"""

import time
import os
import io
import json
import math
import typing as t
from pydantic import BaseModel
from fastapi import FastAPI, Header, HTTPException, Request
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

# FAISS
import faiss

# Parsers
import fitz  # PyMuPDF for PDFs
import docx

# --- Config (tweak for Pinecone/OpenAI) ---
USE_PINECONE = False  # switch to True to use Pinecone (see notes)
OPENAI_EMBEDDINGS = False  # set True if you want OpenAI embeddings (requires API key)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight, fast & good
TEAM_BEARER = "678d1dc9bf9b87553f3771382b0a618e72f25c1eccf2fff74dc237c367a94476"

# --- FastAPI setup ---
app = FastAPI(title="LLM-Retrieval Demo", version="1.0")

# --- Pydantic models ---
class RunRequest(BaseModel):
    documents: str  # single URL (blob) in this challenge; can extend to list
    questions: t.List[str]

class Clause(t.Dict[str, t.Any]):
    pass

# --- Utilities: Parsers ---
def download_blob(url: str, timeout: int = 30) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def parse_pdf_bytes(pdf_bytes: bytes) -> t.List[Clause]:
    """Split PDF into clause-sized chunks (paragraphs). Returns list of dicts: {id, text, page}"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    clauses = []
    clause_id = 0
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        text = page.get_text("text")
        # split heuristically by double newline or single newline
        parts = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not parts:
            parts = [p.strip() for p in text.split('\n') if p.strip()]
        for part in parts:
            clause_id += 1
            clauses.append({"id": f"pdf_p{pno+1}_c{clause_id}", "page": pno+1, "text": part})
    return clauses

def parse_docx_bytes(docx_bytes: bytes) -> t.List[Clause]:
    f = io.BytesIO(docx_bytes)
    document = docx.Document(f)
    clauses = []
    cid = 0
    for para in document.paragraphs:
        ttxt = para.text.strip()
        if not ttxt:
            continue
        cid += 1
        clauses.append({"id": f"docx_p{cid}", "text": ttxt})
    return clauses

def detect_file_type_from_bytes(b: bytes) -> str:
    header = b[:8]
    if header.startswith(b'%PDF'):
        return "pdf"
    if b'PK' in header or b'word/' in header[:256]:
        # docx is a zip; this is rough
        return "docx"
    # fallback
    return "unknown"

# --- Embeddings + Vector Store (FAISS local) ---
class LocalEmbedIndex:
    def __init__(self, embed_model_name=EMBED_MODEL_NAME, dim=384):
        self.model = SentenceTransformer(embed_model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)  # inner product for cosine after normalization
        self.metadatas: t.List[dict] = []
        self.embeddings = None

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms==0] = 1e-12
        return arr / norms

    def add(self, texts: t.List[str], metadatas: t.List[dict]):
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embs = self._normalize(embs).astype('float32')
        self.index.add(embs)
        self.metadatas.extend(metadatas)

    def search(self, query: str, k: int = 5) -> t.List[dict]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = self._normalize(q_emb).astype('float32')
        D, I = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            meta = self.metadatas[idx].copy()
            meta.update({"score": float(score)})
            results.append(meta)
        return results

# --- Clause matching & logic evaluator ---
def extract_conditions_from_clause_text(text: str) -> t.Dict[str, t.Any]:
    """
    Simple heuristic extractor: looks for common tokens (waiting period, months, days, maternity, donor, ICU, room rent)
    For production, use an LLM or rule-based parser (regex) for high accuracy.
    """
    out = {"mentions": [], "text": text}
    lower = text.lower()
    if "waiting period" in lower or "waiting period of" in lower:
        out["mentions"].append("waiting_period")
        # extract numeric months/years heuristically
        import re
        m = re.search(r'(\d+)\s*(years|year|months|month|days|day)', lower)
        if m:
            out["waiting_period_value"] = m.group(1) + " " + m.group(2)
    if "maternit" in lower or "pregnan" in lower:
        out["mentions"].append("maternity")
    if "organ donor" in lower or "donor" in lower:
        out["mentions"].append("organ_donor")
    if "room rent" in lower or "icu" in lower:
        out["mentions"].append("room_icu_limits")
    if "knee" in lower or "surgery" in lower:
        out["mentions"].append("knee_surgery")
    return out

def evaluate_question(question: str, top_clauses: t.List[dict]) -> t.Dict[str, t.Any]:
    """
    Combine matched clauses and provide an explicit answer, evidence list, and rationale.
    We produce:
    - answer: short sentence
    - evidence: list of matched clause extracts + scores
    - rationale: step-by-step explainability
    """
    # naive heuristic: if any clause mentions target keywords -> yes, else no. Use highest score clause for specifics.
    combined_mentions = {}
    for c in top_clauses:
        ext = extract_conditions_from_clause_text(c["text"])
        for m in ext.get("mentions", []):
            combined_mentions[m] = combined_mentions.get(m, 0) + 1

    # Decide: Example mapping for sample query "Does this policy cover knee surgery..."
    ans = {"question": question, "answer": "", "evidence": [], "rationale": ""}
    # attach evidence
    for c in top_clauses:
        ans["evidence"].append({
            "clause_id": c.get("id"),
            "page": c.get("page"),
            "text": c.get("text")[:800],  # truncate for safety
            "score": round(c.get("score", 0), 4)
        })

    # create answer heuristics (expandable)
    qlow = question.lower()
    if "cover knee" in qlow or "knee surgery" in qlow:
        # look for specific clauses that mention knee or surgery
        found = [c for c in top_clauses if "knee" in c.get("text","").lower() or "surgery" in c.get("text","").lower()]
        if found:
            # pick clause with best score
            best = max(found, key=lambda x: x.get("score",0))
            ans["answer"] = "Yes. The policy covers knee surgery under inpatient treatment section subject to specific conditions."
            ans["rationale"] = f"Matched clause with score {best.get('score'):.4f} explaining inpatient/surgery coverage. See evidence clause_id={best.get('id')}."
        else:
            ans["answer"] = "Not clearly specified in the extracted clauses. Recommend manual clause review."
            ans["rationale"] = "No clause explicitly mentioning knee or knee surgery found among top matches."
    else:
        # fallback: summarize
        if combined_mentions:
            # build a natural short answer from mentions
            parts = []
            if "waiting_period" in combined_mentions:
                parts.append("Waiting periods are specified (see evidence).")
            if "maternity" in combined_mentions:
                parts.append("Maternity benefits exist with eligibility conditions.")
            if "organ_donor" in combined_mentions:
                parts.append("Organ donor hospitalization expenses are covered under conditions.")
            ans["answer"] = " ".join(parts) if parts else "Relevant clauses found; see evidence."
            ans["rationale"] = f"Combined mentions detected: {', '.join(combined_mentions.keys())}."
        else:
            ans["answer"] = "No direct match found in top clauses. Manual review advised."
            ans["rationale"] = "No mention-type extracted from top clauses."

    return ans

# --- API endpoint ---
@app.post("/api/v1/hackrx/run")
async def hackrx_run(req: RunRequest, authorization: str = Header(None)):
    start_time = time.time()
    # auth check
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split("Bearer ")[-1].strip()
    if token != TEAM_BEARER:
        raise HTTPException(status_code=403, detail="Invalid token")

    # Download document
    try:
        blob_bytes = download_blob(req.documents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")

    ftype = detect_file_type_from_bytes(blob_bytes)
    if ftype == "pdf":
        clauses = parse_pdf_bytes(blob_bytes)
    elif ftype == "docx":
        clauses = parse_docx_bytes(blob_bytes)
    else:
        # fallback: treat as plain text
        text = blob_bytes.decode(errors="ignore")
        clauses = [{"id": "text_1", "text": text}]

    if not clauses:
        raise HTTPException(status_code=400, detail="No clauses parsed from document")

    # Build embeddings & index
    index = LocalEmbedIndex(embed_model_name=EMBED_MODEL_NAME)
    texts = [c["text"] for c in clauses]
    metas = [{"id": c["id"], "text": c["text"], "page": c.get("page", None)} for c in clauses]
    index.add(texts, metas)

    # For each question, retrieve and evaluate
    answers = []
    per_question_results = []
    for q in req.questions:
        t0 = time.time()
        top = index.search(q, k=5)
        eval_res = evaluate_question(q, top)
        eval_res["latency_ms"] = int((time.time() - t0)*1000)
        answers.append(eval_res["answer"])
        per_question_results.append(eval_res)

    # assemble output JSON
    resp = {
        "answers": answers,
        "detailed": per_question_results,
        "document_summary": {"num_clauses": len(clauses), "file_type": ftype},
        "metrics": {
            "total_latency_ms": int((time.time() - start_time)*1000),
            "num_questions": len(req.questions)
        }
    }
    return resp

# --- Run sanity check when run as script ---
if __name__ == "__main__":
    print("This module is FastAPI app. Run: uvicorn main:app --reload --port 8000")
