# app.py
import os, tempfile, shutil, uuid, requests
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import docx
from pdfminer.high_level import extract_text as pdf_extract_text
import email
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Retrieval System - Demo")

TEAM_TOKEN = os.getenv("TEAM_TOKEN", "678d1dc9bf9b87553f3771382b0a618e72f25c1eccf2fff74dc237c367a94476")

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

def download_url_to_file(url: str, dest_folder: str):
    local_filename = os.path.join(dest_folder, url.split("/")[-1].split("?")[0] or str(uuid.uuid4()))
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return local_filename

def parse_pdf(path: str) -> List[str]:
    text = pdf_extract_text(path)
    return [text] if text else []

def parse_docx(path: str) -> List[str]:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return paragraphs

def parse_eml(path: str) -> List[str]:
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f)
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                parts.append(part.get_payload(decode=True).decode(errors="ignore"))
            elif ctype == "text/html":
                html = part.get_payload(decode=True).decode(errors="ignore")
                parts.append(BeautifulSoup(html, "html.parser").get_text())
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            parts.append(payload.decode(errors="ignore"))
    return parts

def chunk_texts(texts, source_label):
    chunks = []
    for i, t in enumerate(texts):
        parts = [p.strip() for p in t.split("\n\n") if len(p.strip()) > 40]
        if not parts:
            parts = [s.strip() for s in t.split(". ") if len(s.strip()) > 40]
        for j, p in enumerate(parts):
            cid = f"{source_label}::{i}::{j}"
            chunks.append({"id": cid, "text": p, "source": source_label})
    return chunks

def build_index(chunks):
    texts = [c["text"] for c in chunks]
    if not texts:
        return None, None
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vec.fit_transform(texts)
    return vec, matrix

def retrieve_top_k(query, vec, matrix, chunks, k=5):
    if matrix is None:
        return []
    qv = vec.transform([query])
    sims = cosine_similarity(qv, matrix)[0]
    idxs = sims.argsort()[::-1][:k]
    results = []
    for i in idxs:
        results.append({
            "clause_id": chunks[i]["id"],
            "text": chunks[i]["text"],
            "source": chunks[i]["source"],
            "score": float(sims[i])
        })
    return results

@app.post("/api/v1/hackrx/run")
def run(payload: RunRequest, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer ") or authorization.split()[-1] != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    tmp = tempfile.mkdtemp()
    try:
        file_path = download_url_to_file(payload.documents, tmp)
        ext = Path(file_path).suffix.lower()
        texts = []
        if ext == ".pdf":
            texts = parse_pdf(file_path)
        elif ext in [".docx", ".doc"]:
            texts = parse_docx(file_path)
        elif ext in [".eml", ".msg"]:
            texts = parse_eml(file_path)
        else:
            with open(file_path, "r", errors="ignore") as f:
                texts = [f.read()]

        chunks = chunk_texts(texts, source_label=Path(file_path).name)
        vec, matrix = build_index(chunks)

        answers = []
        for q in payload.questions:
            retrieved = retrieve_top_k(q, vec, matrix, chunks, k=5) if vec is not None else []
            # Simple "synthesis": return top clause text as the answer summary
            if retrieved:
                top = retrieved[0]
                answer_text = top["text"][:1000]
                rationale = f"Top matched clause {top['clause_id']} (score={top['score']:.3f})"
                confidence = float(top["score"])
            else:
                answer_text = "No relevant clause found in the document."
                rationale = "No matches"
                confidence = 0.0
            answers.append({
                "answer": answer_text,
                "supporting_clauses": retrieved,
                "rationale": rationale,
                "confidence": confidence
            })

        return {"answers": answers}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
