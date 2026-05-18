import re
import os
import math
import time
import pickle
import random
import warnings
import requests
from collections import Counter

import numpy as np
import pandas as pd
import faiss
import PyPDF2
import torch
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rcParams
from sentence_transformers import SentenceTransformer, CrossEncoder
from langdetect import detect, DetectorFactory
from gtts import gTTS
from transformers import pipeline as hfpipeline
from transformers import pipeline
from groq import Groq
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

warnings.filterwarnings("ignore")
DetectorFactory.seed = 0
random.seed(42)
np.random.seed(42)
rcParams["figure.facecolor"] = "#FFFDF8"
rcParams["axes.facecolor"] = "#FFF9F0"
rcParams["savefig.facecolor"] = "#FFFDF8"

APP_ACCENT = "#6C63FF"
APP_ACCENT_2 = "#FF8A65"
APP_ACCENT_3 = "#26A69A"
APP_ACCENT_4 = "#AB47BC"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

KB_TEXTS, KB_META, FAISS_INDEX, KB_EMB = [], [], None, None
DOC_TYPE_INFO = {"type": "📄 General", "is_economic": False, "score": 0}
PER_FILE_INFO = {}
CHAT_STATS = {"questions": 0, "found": 0, "notfound": 0, "general_used": 0}
MIN_SIMILARITY = 0.18
STRICT_RAG_MODE = False
LAST_BUILD_SECONDS = 0.0

PERSIST_DIR = "/tmp"
KB_TEXTS_PATH = f"{PERSIST_DIR}/kbtexts.pkl"
KB_META_PATH = f"{PERSIST_DIR}/kbmeta.pkl"
FAISS_PATH = f"{PERSIST_DIR}/faiss.index"
os.makedirs(PERSIST_DIR, exist_ok=True)

ECONOMIC_KEYWORDS = ["gdp", "inflation", "monetary", "fiscal", "forecast", "exchange rate", "interest rate", "unemployment", "recession", "growth rate", "trade balance", "budget deficit", "central bank", "economic outlook", "imf", "world bank", "cpi", "macro", "revenue", "expenditure", "deficit", "surplus", "debt", "croissance", "taux", "banque centrale", "prévision", "economique", "économique", "pib", "التضخم", "الناتج المحلي", "النمو", "البنك المركزي", "سعر الصرف", "السياسة النقدية", "المالية العامة", "البطالة", "العجز", "الفائض", "الدين", "الركود"]
MEDICAL_KEYWORDS = ["patient", "diagnosis", "treatment", "clinical", "hospital", "symptom", "disease"]
LEGAL_KEYWORDS = ["article", "law", "contract", "clause", "jurisdiction", "court", "legal"]
ACADEMIC_KEYWORDS = ["abstract", "methodology", "hypothesis", "conclusion", "references", "doi", "journal"]

ECON_POSITIVE = ["growth", "recovery", "surplus", "improvement", "stability", "increase", "expansion", "acceleration", "resilience", "upturn", "robust", "favorable", "strengthened", "progress", "rebound", "optimistic", "confidence", "boom", "prosper", "thrive", "advance", "gain", "rise", "positive", "upward", "exceed", "outperform", "strong", "healthy", "dynamic", "sustainable", "croissance", "reprise", "amélioration", "stabilité", "excédent", "hausse", "dynamique", "favorable", "progrès", "rebond", "solide", "تعافي", "نمو", "استقرار", "تحسن", "ارتفاع", "توسع", "إيجابي", "انتعاش", "قوي"]
ECON_NEGATIVE = ["deficit", "recession", "inflation", "decline", "contraction", "debt", "crisis", "deterioration", "slowdown", "downturn", "unemployment", "pressure", "risk", "vulnerability", "shock", "uncertainty", "war", "sanctions", "drought", "collapse", "default", "volatile", "instability", "weak", "fragile", "pessimistic", "loss", "shrink", "fall", "negative", "downward", "slump", "stagnation", "turbulence", "disruption", "imbalance", "burden", "déficit", "récession", "crise", "ralentissement", "chômage", "incertitude", "guerre", "effondrement", "instabilité", "baisse", "fragilité", "pression", "عجز", "تضخم", "ركود", "انكماش", "أزمة", "تدهور", "بطالة", "انخفاض", "ضغط", "مخاطر", "صدمة"]
ECON_TRIGGER = list(dict.fromkeys(ECONOMIC_KEYWORDS + ["risk", "crisis", "slowdown", "policy", "reform", "budget", "revenue", "trade", "forecast", "outlook", "growth", "deficit", "croissance", "prévision", "سياسة", "ميزانية", "اقتصاد"]))


def save_index():
    if FAISS_INDEX is None or not KB_TEXTS:
        return "⚠️ No index to save."
    try:
        with open(KB_TEXTS_PATH, "wb") as f:
            pickle.dump(KB_TEXTS, f)
        with open(KB_META_PATH, "wb") as f:
            pickle.dump(KB_META, f)
        faiss.write_index(FAISS_INDEX, FAISS_PATH)
        return f"💾 Saved! {len(KB_TEXTS):,} chunks"
    except Exception as e:
        return f"❌ Save error: {e}"


def load_saved_index():
    global KB_TEXTS, KB_META, FAISS_INDEX, DOC_TYPE_INFO
    try:
        if not os.path.exists(FAISS_PATH):
            return "_No saved index found._"
        with open(KB_TEXTS_PATH, "rb") as f:
            KB_TEXTS = pickle.load(f)
        with open(KB_META_PATH, "rb") as f:
            KB_META = pickle.load(f)
        FAISS_INDEX = faiss.read_index(FAISS_PATH)
        DOC_TYPE_INFO = detect_document_type(KB_TEXTS)
        return f"✅ Index loaded! {len(KB_TEXTS):,} chunks | Type: {DOC_TYPE_INFO['type']}"
    except Exception as e:
        return f"❌ Load error: {e}"


def economic_lexicon_score(text: str) -> float:
    text_lower = str(text).lower()
    pos = sum(1 for w in ECON_POSITIVE if w in text_lower)
    neg = sum(1 for w in ECON_NEGATIVE if w in text_lower)
    total = max(pos + neg, 1)
    return round((pos - neg) / total, 4)


def detect_document_type(texts: list) -> dict:
    if not texts:
        return {"type": "📄 General", "is_economic": False, "score": 0, "confidence": 0.0}
    full_text = " ".join(map(str, texts[:30])).lower()
    scores = {
        "economic": sum(1 for kw in ECONOMIC_KEYWORDS if kw in full_text),
        "medical": sum(1 for kw in MEDICAL_KEYWORDS if kw in full_text),
        "legal": sum(1 for kw in LEGAL_KEYWORDS if kw in full_text),
        "academic": sum(1 for kw in ACADEMIC_KEYWORDS if kw in full_text),
        "general": 1,
    }
    doc_type = max(scores, key=scores.get)
    confidence = round(scores[doc_type] / max(sum(scores.values()), 1), 2)
    icons = {"economic": "📊 Economic", "medical": "🏥 Medical", "legal": "⚖️ Legal", "academic": "🎓 Academic", "general": "📄 General"}
    return {"type": icons.get(doc_type, "📄 General"), "rawtype": doc_type, "is_economic": doc_type == "economic" and scores["economic"] >= 3, "score": scores[doc_type], "confidence": confidence}


def is_economic_text(text: str) -> bool:
    t = str(text).lower()
    return sum(1 for kw in ECON_TRIGGER if kw in t) >= 2


BASE_WEIGHTS = {"finbert": 0.40, "xlm": 0.20, "lexicon": 0.15, "econ_ft": 0.25}
ECON_WEIGHTS = {"finbert": 0.15, "econ_ft": 0.60, "lexicon": 0.25}

try:
    finbert_pipe = pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert", return_all_scores=True, device=0 if torch.cuda.is_available() else -1)
    FINBERT_OK = True
except Exception:
    finbert_pipe = None
    FINBERT_OK = False

try:
    xlm_pipe = pipeline("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment", return_all_scores=True, device=0 if torch.cuda.is_available() else -1)
    XLM_OK = True
except Exception:
    xlm_pipe = None
    XLM_OK = False

try:
    econ_ft_pipe = pipeline("text-classification", model="sitayeb/economic_sentiment_finetunedto", tokenizer="sitayeb/economic_sentiment_finetunedto", return_all_scores=True, device=0 if torch.cuda.is_available() else -1)
    ECON_FT_OK = True
except Exception:
    econ_ft_pipe = None
    ECON_FT_OK = False


def normalize_clf(raw):
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    return raw if isinstance(raw, list) else [raw]


def clf_finbert(text: str) -> float:
    if not FINBERT_OK or finbert_pipe is None:
        return 0.0
    try:
        items = normalize_clf(finbert_pipe(str(text)[:512]))
        d = {r["label"].lower(): float(r["score"]) for r in items}
        return round(d.get("positive", 0.0) - d.get("negative", 0.0), 4)
    except Exception:
        return 0.0


def clf_xlm(text: str) -> float:
    if not XLM_OK or xlm_pipe is None:
        return 0.0
    try:
        items = normalize_clf(xlm_pipe(str(text)[:512]))
        d = {r["label"]: float(r["score"]) for r in items}
        pos = d.get("LABEL_2", d.get("positive", d.get("Positive", 0.0)))
        neg = d.get("LABEL_0", d.get("negative", d.get("Negative", 0.0)))
        return round(pos - neg, 4)
    except Exception:
        return 0.0


def clf_econ_ft(text: str) -> float:
    if not ECON_FT_OK or econ_ft_pipe is None:
        return 0.0
    try:
        items = normalize_clf(econ_ft_pipe(str(text)[:512]))
        mapping = {str(r["label"]).lower().strip(): float(r["score"]) for r in items}
        positive_keys = ["positive", "pos", "label_2", "2", "bullish", "optimistic", "إيجابي"]
        negative_keys = ["negative", "neg", "label_0", "0", "bearish", "pessimistic", "سلبي"]
        pos = max([mapping.get(k, 0.0) for k in positive_keys] + [0.0])
        neg = max([mapping.get(k, 0.0) for k in negative_keys] + [0.0])
        if pos == 0.0 and neg == 0.0 and len(mapping) == 3:
            vals = list(mapping.values())
            neg, _, pos = vals[0], vals[1], vals[2]
        return round(pos - neg, 4)
    except Exception:
        return 0.0


def sentiment_score_numeric(text: str) -> float:
    text = str(text)
    lex = economic_lexicon_score(text)
    econ_mode = is_economic_text(text) or DOC_TYPE_INFO.get("is_economic", False)
    if econ_mode and ECON_FT_OK:
        fb = clf_finbert(text)
        ft = clf_econ_ft(text)
        return round(ECON_WEIGHTS["finbert"] * fb + ECON_WEIGHTS["econ_ft"] * ft + ECON_WEIGHTS["lexicon"] * lex, 4)
    fb = clf_finbert(text)
    xlm = clf_xlm(text)
    ft = clf_econ_ft(text) if ECON_FT_OK else 0.0
    return round(BASE_WEIGHTS["finbert"] * fb + BASE_WEIGHTS["xlm"] * xlm + BASE_WEIGHTS["lexicon"] * lex + BASE_WEIGHTS["econ_ft"] * ft, 4)


def run_sentiment(text: str):
    score = sentiment_score_numeric(text)
    if score > 0.05:
        sent = "Positive"
    elif score < -0.05:
        sent = "Negative"
    else:
        sent = "Neutral"
    return sent, round(min(abs(score), 1.0), 4)


def run_sentiment_detailed(text: str) -> str:
    fb = clf_finbert(text)
    xlm = clf_xlm(text)
    lex = economic_lexicon_score(text)
    ft = clf_econ_ft(text) if ECON_FT_OK else 0.0
    final = sentiment_score_numeric(text)
    def bars(s):
        filled = max(0, min(10, round((s + 1) / 2 * 10)))
        icon = "🟩" if s > 0.05 else ("🟥" if s < -0.05 else "🟨")
        return icon * filled + "⬜" * (10 - filled)
    label = "Positive" if final > 0.05 else "Negative" if final < -0.05 else "Neutral"
    return "\n".join([
        "## Ensemble Sentiment Breakdown", "", "| Model | Score | Visual |", "|---|---:|---|",
        f"| FinBERT | {fb:.4f} | {bars(fb)} |", f"| XLM-RoBERTa | {xlm:.4f} | {bars(xlm)} |",
        f"| Fine-Tuned Econ | {ft:.4f} | {bars(ft)} |", f"| Lexicon | {lex:.4f} | {bars(lex)} |",
        f"| **Final** | **{final:.4f}** | {bars(final)} |", "", f"**Label:** {label}"])


embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
asr = hfpipeline("automatic-speech-recognition", model="openai/whisper-small", device=0 if torch.cuda.is_available() else -1)
embedder.encode(["warmup"], convert_to_numpy=True)
load_saved_index()


def clean_filename(path: str) -> str:
    return os.path.basename(str(path))


def detect_lang(text: str) -> str:
    try:
        return "ar" if str(detect(str(text)[:300])).startswith("ar") else "en"
    except Exception:
        return "en"


def extract_year_from_filename(filename: str):
    full_path = str(filename).replace("\\", "/")
    for part in reversed(full_path.split("/")):
        m = re.findall(r"\b(20\d{2}|19\d{2})\b", part)
        if m:
            return int(m[0])
    all_years = re.findall(r"(19\d{2}|20\d{2})", full_path)
    return int(all_years[0]) if all_years else None


def chunk_text(text, chunk_size=320, overlap=90):
    text = re.sub(r"\s+", " ", str(text).strip())
    sentences = re.split(r"(?<=[?.!؟])\s+", text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += " " + sent
        else:
            if current.strip():
                chunks.append(current.strip())
            words = current.split()
            current = " ".join(words[-overlap // 5:]) + " " + sent if words else sent
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 35]


def load_file(path):
    path = str(path)
    if path.endswith(".pdf"):
        pages = []
        try:
            import pypdf
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for i, pg in enumerate(reader.pages[:80]):
                    t = pg.extract_text()
                    if t and t.strip():
                        pages.append({"text": t, "page": i + 1})
        except Exception:
            pass
        if not pages:
            try:
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for i, pg in enumerate(reader.pages[:80]):
                        t = pg.extract_text()
                        if t and t.strip():
                            pages.append({"text": t, "page": i + 1})
            except Exception:
                pass
        return pages or [{"text": "Could not extract text.", "page": 1}]
    if path.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(path)
            pars = [p.text for p in doc.paragraphs if p.text.strip()]
            return [{"text": "\n".join(pars[i:i+50]), "page": i // 50 + 1} for i in range(0, len(pars), 50)] or [{"text": "Empty DOCX.", "page": 1}]
        except Exception as e:
            return [{"text": f"DOCX error: {e}", "page": 1}]
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        col = "text" if "text" in df.columns else df.columns[0]
        return [{"text": t, "page": i + 1} for i, t in enumerate(df[col].dropna().astype(str))]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [{"text": f.read(), "page": 1}]


def build_index(files):
    global KB_TEXTS, KB_META, FAISS_INDEX, KB_EMB, DOC_TYPE_INFO, PER_FILE_INFO, LAST_BUILD_SECONDS
    t0 = time.time()
    KB_TEXTS, KB_META, PER_FILE_INFO = [], [], {}
    if not files:
        raise gr.Error("⚠️ Upload at least one file.")
    if not isinstance(files, list):
        files = [files]
    file_paths = []
    for f in files:
        if isinstance(f, str):
            file_paths.append(f)
        elif isinstance(f, dict):
            file_paths.append(f.get("path") or f.get("name") or str(f))
        elif hasattr(f, "name"):
            file_paths.append(f.name)
        else:
            file_paths.append(str(f))
    for p in file_paths:
        fname = clean_filename(p)
        year = extract_year_from_filename(p)
        pages = load_file(p)
        file_texts = []
        for pg in pages:
            for ch in chunk_text(pg["text"]):
                KB_TEXTS.append(ch)
                KB_META.append({"name": fname, "lang": detect_lang(ch), "page": pg["page"], "year": year})
                file_texts.append(ch)
        ti = detect_document_type(file_texts)
        ti["year"] = year
        PER_FILE_INFO[fname] = ti
    if not KB_TEXTS:
        raise gr.Error("⚠️ No text extracted.")
    KB_EMB = embedder.encode(KB_TEXTS, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    FAISS_INDEX = faiss.IndexFlatIP(KB_EMB.shape[1])
    FAISS_INDEX.add(KB_EMB)
    DOC_TYPE_INFO = detect_document_type(KB_TEXTS)
    lang_count = Counter(m["lang"] for m in KB_META)
    save_index()
    elapsed = time.time() - t0
    LAST_BUILD_SECONDS = elapsed
    tbl = "\n\n### Per-File Analysis\n\n| File | Year | Type | Conf | Chunks |\n|---|---:|---|---:|---:|\n"
    for fname, info in PER_FILE_INFO.items():
        n = sum(1 for m in KB_META if m["name"] == fname)
        tbl += f"| {fname} | {str(info.get('year', 'NA'))} | {info['type']} | {info['confidence']:.2f} | {n} |\n"
    status_md = f"## ✅ Index built successfully\n\n- ⏱️ Build time: **{elapsed:.2f} seconds**\n- 📚 Total chunks: **{len(KB_TEXTS):,}**\n- 📄 Files: **{len(file_paths)}**\n- 🇸🇦 Arabic: **{lang_count.get('ar', 0)}**\n- 🇬🇧 English: **{lang_count.get('en', 0)}**\n- 🧠 Detected type: **{DOC_TYPE_INFO['type']}**" + tbl
    timer_md = f"### ⏳ Build Timer\n\n**Elapsed:** `{elapsed:.2f} sec`\n\n**Status:** Index is ready."
    return status_md, timer_md


def bm25_score(query_terms, doc, k1=1.5, b=0.75, avg_dl=200):
    try:
        if not KB_TEXTS or not isinstance(doc, str):
            return 0.0
        dl, score = len(doc.split()), 0.0
        df = Counter(doc.lower().split())
        for term in query_terms:
            if not term:
                continue
            tl = term.lower()
            n_doc = sum(1 for t in KB_TEXTS if tl in t.lower())
            tf = df.get(tl, 0)
            idf = math.log((len(KB_TEXTS) + 1) / (1 + n_doc))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        return score
    except Exception:
        return 0.0


def keyword_overlap(query_terms, text):
    txt = str(text).lower()
    terms = [t for t in query_terms if len(t) > 2]
    if not terms:
        return 0.0
    hits = sum(1 for t in terms if re.search(rf"\b{re.escape(t)}\b", txt))
    return hits / len(terms)


def dedupe_candidates(candidates, max_per_file=2):
    final = []
    seen = set()
    per_file = Counter()
    for c in candidates:
        norm = re.sub(r"\s+", " ", c["text"][:220].lower()).strip()
        key = (c["file"], norm)
        if key in seen or per_file[c["file"]] >= max_per_file:
            continue
        seen.add(key)
        per_file[c["file"]] += 1
        final.append(c)
    return final


def rag_retrieve(query, k=8, top_n=4):
    if FAISS_INDEX is None or not KB_TEXTS:
        return []
    try:
        q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idx = FAISS_INDEX.search(q_emb, min(k * 6, len(KB_TEXTS)))
        candidates, q_terms = [], [t for t in re.findall(r"\w+", str(query).lower()) if t]
        for rank, i in enumerate(idx[0]):
            if i == -1:
                continue
            sem = float(scores[0][rank])
            if sem < MIN_SIMILARITY:
                continue
            text = KB_TEXTS[i]
            kw = bm25_score(q_terms, text)
            overlap = keyword_overlap(q_terms, text)
            low_terms = [t for t in q_terms if len(t) > 2]
            exact = all(re.search(rf"\b{re.escape(t)}\b", text.lower()) for t in low_terms) if low_terms else False
            hybrid = sem * 0.45 + min(kw / 8, 0.25) + overlap * 0.20 + (0.10 if exact else 0.0)
            candidates.append({"idx": i, "sem": sem, "kw": kw, "overlap": overlap, "exact": exact, "hybrid": hybrid, "lang": KB_META[i]["lang"], "file": KB_META[i]["name"], "page": KB_META[i]["page"], "year": KB_META[i].get("year"), "text": text})
        if not candidates:
            return []
        ce_scores = reranker.predict([[query, c["text"]] for c in candidates])
        for c, ce in zip(candidates, ce_scores):
            c["ce_score"] = float(ce)
            ce_norm = 1 / (1 + math.exp(-float(ce) / 3))
            c["final"] = c["hybrid"] * 0.45 + ce_norm * 0.55
        candidates.sort(key=lambda x: x["final"], reverse=True)
        candidates = dedupe_candidates(candidates, max_per_file=2)
        chosen = candidates[:top_n]
        for i, c in enumerate(chosen):
            c["rank"] = i + 1
        return chosen
    except Exception:
        return []


def detect_query_intent(question: str) -> str:
    q = str(question).strip().lower()
    summarize_patterns = ["explain the pdf", "explain pdf", "summarize the pdf", "summarize pdf", "what is this pdf about", "what is this document about", "summarize this file", "explain this file", "summarize this document", "اشرح الملف", "لخص الملف", "اشرح الوثيقة", "ما مضمون الملف", "ملخص الملف", "summary of the pdf", "overview of the document"]
    if any(pat in q for pat in summarize_patterns):
        return "summarize_document"
    if len(q.split()) <= 3 and any(k in q for k in ["pdf", "file", "document", "doc", "ملف", "وثيقة"]):
        return "summarize_document"
    return "qa"


def format_context(results):
    return "\n\n".join([f"[DOC {r['rank']}] file={r['file']} | page={r['page']} | score={r.get('final', 0.9):.3f}\n{r['text']}" for r in results])


def build_file_summary_context(max_files=2, max_chunks_per_file=5):
    if not KB_TEXTS or not KB_META:
        return []
    by_file = {}
    for text, meta in zip(KB_TEXTS, KB_META):
        by_file.setdefault(meta['name'], []).append({'text': text, 'page': meta.get('page', 1), 'year': meta.get('year')})
    ranked_files = sorted(by_file.items(), key=lambda kv: len(kv[1]), reverse=True)[:max_files]
    results = []
    rank = 1
    for fname, chunks in ranked_files:
        seen = set()
        chosen = []
        for ch in chunks:
            txt = ch['text'].strip()
            if len(txt) < 120:
                continue
            norm = re.sub(r"\s+", " ", txt[:180].lower())
            if norm in seen:
                continue
            seen.add(norm)
            chosen.append(ch)
            if len(chosen) >= max_chunks_per_file:
                break
        for ch in chosen:
            results.append({'rank': rank, 'file': fname, 'page': ch['page'], 'text': ch['text'], 'final': 0.90, 'sem': 0.90})
            rank += 1
    return results


def llm_groq(question, rag_context, history, lang, grounded=True):
    if groq_client is None:
        return "⚠️ Groq API key missing."
    if grounded:
        system_prompt = (
            "You are a strict multilingual grounded AI assistant. "
            "Always reply in the same language as the user. Use only the provided context. "
            "If the context is only partially relevant, explicitly say it is partial and avoid overclaiming."
        )
        user_content = f"DOCUMENT CONTEXT:\n{rag_context}\n\nQUESTION:\n{question}\n\nAnswer only from the document context."
        temperature = 0.1
    else:
        system_prompt = "You are a smart multilingual AI assistant. Always reply in the same language as the user. Be concise and helpful."
        user_content = question
        temperature = 0.3
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_content})
    try:
        r = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, temperature=temperature, max_tokens=512)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Groq error: {e}"


def llm_summarize_document(question, results, history, lang):
    if groq_client is None:
        return "⚠️ Groq API key missing."
    context = format_context(results)[:5000]
    if lang == 'ar':
        system_prompt = "أنت مساعد ذكي. المطلوب هنا شرح الوثيقة أو تلخيصها بشكل منظم: الموضوع، الهدف، المنهجية، وأهم النتائج أو المحاور، اعتمادًا فقط على المقاطع المتاحة من الوثيقة."
        user_prompt = f"سؤال المستخدم: {question}\n\nمقاطع من الوثيقة:\n{context}\n\nأعطني شرحًا أو ملخصًا عامًا منظمًا للوثيقة اعتمادًا على هذا السياق فقط."
    else:
        system_prompt = "You are a grounded assistant. The user wants an overall explanation of the document. Provide a structured summary of the document topic, objective, methodology, and key themes using only the provided excerpts."
        user_prompt = f"User request: {question}\n\nDocument excerpts:\n{context}\n\nGive a structured overall explanation of the document using this context only."
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_prompt})
    try:
        r = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, temperature=0.15, max_tokens=700)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Groq error: {e}"


def smart_answer(question, history):
    lang = detect_lang(question)
    intent = detect_query_intent(question)

    if intent == "summarize_document":
        summary_results = build_file_summary_context(max_files=2, max_chunks_per_file=5)
        if summary_results:
            answer_text = llm_summarize_document(question, summary_results, history, lang)
            if lang == "ar":
                src = "، ".join(f"`{r['file']}` ص.{r['page']}" for r in summary_results[:6])
                prefix = "✅ **وجدت محتوى كافيًا في الوثائق لشرح الملف / تلخيصه:**\n\n"
                badge = f"\n\n📄 **المصادر المستخدمة في التلخيص:** {src}"
            else:
                src = ", ".join(f"`{r['file']}` p.{r['page']}" for r in summary_results[:6])
                prefix = "✅ **I found enough content in the uploaded documents to explain / summarize the file:**\n\n"
                badge = f"\n\n📄 **Sources used for the summary:** {src}"
            CHAT_STATS["found"] += 1
            return prefix + answer_text + badge, "rag_summary"

    results = rag_retrieve(question, k=8, top_n=4)
    top = results[0] if results else None
    strong_evidence = False
    if top:
        strong_evidence = top["final"] >= 0.52 or (top["sem"] >= 0.46 and top.get("overlap", 0) >= 0.20) or (len(results) >= 2 and sum(1 for r in results if r.get("overlap", 0) >= 0.20) >= 2)

    if strong_evidence:
        rag_context = format_context(results)
        answer_text = llm_groq(question, rag_context[:3200], history, lang, grounded=True)
        if lang == "ar":
            src = "، ".join(f"`{r['file']}` ص.{r['page']}" for r in results)
            prefix = "✅ **تم العثور على جواب مباشر في الوثائق المرفوعة / ملفات PDF:**\n\n"
            badge = f"\n\n📄 **المصادر:** {src}"
        else:
            src = ", ".join(f"`{r['file']}` p.{r['page']}" for r in results)
            prefix = "✅ **I found a direct answer in the uploaded documents / PDFs:**\n\n"
            badge = f"\n\n📄 **Sources:** {src}"
        CHAT_STATS["found"] += 1
        return prefix + answer_text + badge, "rag"

    if results:
        rag_context = format_context(results)
        partial_answer = llm_groq(question, rag_context[:3200], history, lang, grounded=True)
        if lang == "ar":
            src = "، ".join(f"`{r['file']}` ص.{r['page']}" for r in results)
            preface = "⚠️ **وجدت مقاطع مرتبطة في الوثائق، لكن لم أجد جوابًا مباشرًا وواضحًا تمامًا على سؤالك.**\n\n### أفضل ما وجدته في الوثائق:\n\n"
            suffix = f"\n\n📄 **المقاطع المرتبطة:** {src}"
        else:
            src = ", ".join(f"`{r['file']}` p.{r['page']}" for r in results)
            preface = "⚠️ **I found related passages in the uploaded documents, but not a fully direct clear answer to your question.**\n\n### Best grounded answer from the documents:\n\n"
            suffix = f"\n\n📄 **Related passages:** {src}"
        CHAT_STATS["notfound"] += 1
        return preface + partial_answer + suffix, "partial_rag"

    general_answer = llm_groq(question, "", history, lang, grounded=False)
    CHAT_STATS["notfound"] += 1
    CHAT_STATS["general_used"] += 1
    if lang == "ar":
        msg = "⚠️ **لم أجد جوابًا واضحًا ومباشرًا في الوثائق المرفوعة / ملفات PDF.**\n\n### إجابة عامة بالاعتماد على المعرفة العامة:\n\n" + general_answer + "\n\n_ملاحظة: الجزء أعلاه ليس مستخرجًا مباشرة من الوثائق._"
    else:
        msg = "⚠️ **I did not find a clear direct answer in the uploaded documents / PDFs.**\n\n### General-knowledge answer:\n\n" + general_answer + "\n\n_Note: the answer above is based on general knowledge, not directly extracted from the documents._"
    return msg, "general"


def chat_text(message, history):
    if not message.strip():
        return "", history
    answer, _ = smart_answer(message, history)
    CHAT_STATS["questions"] += 1
    return "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]


def tts_save(text, lang="en"):
    path = "/tmp/ans.mp3"
    gTTS(text=re.sub(r"[#*_`>]", "", str(text))[:600], lang="ar" if lang == "ar" else "en").save(path)
    return path


def chat_voice(audio, history):
    if audio is None:
        raise gr.Error("No audio received.")
    sr, y = audio
    y = np.array(y) if isinstance(y, list) else y
    if y.ndim > 1:
        y = y.mean(axis=1)
    transcript = asr({"array": y.astype(np.float32), "sampling_rate": sr})["text"]
    lang = detect_lang(transcript)
    answer, _ = smart_answer(transcript, history)
    new_history = history + [{"role": "user", "content": f"🎤 {transcript}"}, {"role": "assistant", "content": answer}]
    audio_path = tts_save(answer, lang)
    CHAT_STATS["questions"] += 1
    return new_history, audio_path, transcript


def export_chat(history):
    path = "/tmp/chat.txt"
    with open(path, "w", encoding="utf-8") as f:
        for turn in history:
            f.write(f"{turn['role']}: {turn['content']}\n\n")
    return path


def predict_with_rag(text):
    text = "" if text is None else str(text).strip()
    if not text:
        raise gr.Error("⚠️ Enter text first.")
    q_terms = [t for t in re.findall(r"\w+", text.lower()) if len(t) > 2]
    exact_hits = []
    for i, chunk in enumerate(KB_TEXTS):
        cl = chunk.lower()
        for term in q_terms:
            if re.search(rf"\b{re.escape(term)}\b", cl):
                for s in re.split(r"(?<=[?.!؟])\s+", chunk):
                    if re.search(rf"\b{re.escape(term)}\b", s.lower()):
                        exact_hits.append({"word": term, "file": KB_META[i]["name"], "sentence": s.strip(), "lang": KB_META[i]["lang"], "chunkid": i, "page": KB_META[i]["page"]})
    sem_results, md = rag_retrieve(text, k=8, top_n=4), ""
    if exact_hits:
        seen, unique = set(), []
        for h in exact_hits:
            key = (h["word"], h["file"], h["sentence"][:80])
            if key not in seen:
                seen.add(key)
                unique.append(h)
        md += "## Word Found\n\n"
        for h in unique[:12]:
            flag = "🇸🇦" if h["lang"] == "ar" else "🇬🇧"
            md += f"- **{h['word']}** → `{h['file']}` p.{h['page']} {flag}\n  - {h['sentence']}\n"
        detail = run_sentiment_detailed(text)
        sent, conf = run_sentiment(text)
        md += f"\n---\n\n{detail}\n"
    else:
        sent, conf = ("Not found", 0.0)
        md += f"## Word Not Found\n\n`{text}` not found literally in the indexed documents.\n"
    if sem_results:
        md += "\n---\n\n## Semantic Results\n"
        for r in sem_results:
            snippet = r["text"][:320].strip()
            md += f"\n### Result {r['rank']}\n- File: `{r['file']}`\n- Page: {r['page']}\n- Final score: `{r['final']:.3f}`\n\n> {snippet}...\n"
    return sent, round(conf, 4), md


def generate_report(text, sent, conf, md):
    path = "/tmp/report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Report\n\n**Query:** {text}\n\n**Sentiment:** {sent}\n\n**Score:** {conf}\n\n{md}")
    return path


def simple_token_f1(a: str, b: str) -> float:
    a_tokens = re.findall(r"\w+", str(a).lower())
    b_tokens = re.findall(r"\w+", str(b).lower())
    if not a_tokens or not b_tokens:
        return 0.0
    a_count = Counter(a_tokens)
    b_count = Counter(b_tokens)
    common = sum((a_count & b_count).values())
    if common == 0:
        return 0.0
    precision = common / max(len(a_tokens), 1)
    recall = common / max(len(b_tokens), 1)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def build_auto_eval_set(max_samples=10):
    if not KB_TEXTS:
        return []
    sample_idx = np.linspace(0, len(KB_TEXTS) - 1, min(max_samples, len(KB_TEXTS)), dtype=int)
    dataset = []
    for idx in sample_idx:
        chunk = KB_TEXTS[int(idx)]
        meta = KB_META[int(idx)]
        lang = meta.get("lang", "en")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?؟])\s+", chunk) if len(s.strip()) > 30]
        if not sentences:
            continue
        anchor = max(sentences, key=len)[:240]
        q = f"ما المعلومة الأساسية الواردة في الملف {meta['name']} حول: {anchor[:120]}؟" if lang == "ar" else f"What is the main information stated in {meta['name']} about: {anchor[:120]}?"
        dataset.append({"question": q, "ground_truth": anchor, "lang": lang})
    return dataset


def evaluate_rag_pipeline(max_samples=10):
    if not KB_TEXTS or FAISS_INDEX is None:
        return "⚠️ Build the index first.", None
    eval_set = build_auto_eval_set(max_samples=max_samples)
    if not eval_set:
        return "⚠️ Could not build evaluation dataset.", None
    rows = []
    for item in eval_set:
        results = rag_retrieve(item["question"], k=8, top_n=4)
        contexts = [r["text"] for r in results]
        best_context = contexts[0] if contexts else ""
        answer = llm_groq(item["question"], format_context(results)[:3200], [], item["lang"], grounded=True) if contexts else ""
        gt = item["ground_truth"]
        rows.append({
            "context_recall": round(float(max([simple_token_f1(gt, c) for c in contexts], default=0.0)), 4),
            "faithfulness": round(float(simple_token_f1(answer, best_context) if best_context else 0.0), 4),
            "answer_relevancy": round(float(simple_token_f1(answer, gt)), 4),
            "context_precision": round(float(np.mean([simple_token_f1(c, gt) for c in contexts]) if contexts else 0.0), 4)
        })
    df = pd.DataFrame(rows)
    avg = df[["context_recall", "faithfulness", "answer_relevancy", "context_precision"]].mean().round(4)
    fig, ax = plt.subplots(figsize=(8, 5))
    avg.plot(kind="bar", ax=ax, color=[APP_ACCENT, APP_ACCENT_3, APP_ACCENT_2, APP_ACCENT_4])
    ax.set_ylim(0, 1)
    ax.set_title("Improved Automatic RAG Evaluation", color="#4A3F7A", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    img_path = "/tmp/rag_eval.png"
    plt.savefig(img_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    md = f"## 📊 Automatic RAG Evaluation\n\nSamples: **{len(df)}**\n\n- Context Recall: **{avg['context_recall']:.4f}**\n- Faithfulness: **{avg['faithfulness']:.4f}**\n- Answer Relevancy: **{avg['answer_relevancy']:.4f}**\n- Context Precision: **{avg['context_precision']:.4f}**"
    return md, img_path


def build_word_cloud_image(max_words=80):
    if not KB_TEXTS:
        return None
    text = " ".join(KB_TEXTS).lower()
    stop = {"this", "that", "with", "from", "have", "been", "were", "they", "their", "there", "what", "when", "which", "will", "also", "than", "into", "more", "about", "your", "them", "dans", "avec", "pour", "mais", "على", "من", "إلى", "في", "عن", "هذا", "هذه", "ذلك", "التي", "الذي"}
    words = [w for w in re.findall(r"\b[\w\u0600-\u06FF]{3,}\b", text) if w not in stop and not w.isdigit()]
    counts = Counter(words).most_common(max_words)
    if not counts:
        return None
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("#FFF9F0")
    fig.patch.set_facecolor("#FFFDF8")
    ax.axis("off")
    colors = [APP_ACCENT, APP_ACCENT_2, APP_ACCENT_3, APP_ACCENT_4, "#5C6BC0", "#EF5350", "#29B6F6"]
    xs = np.random.uniform(0.05, 0.95, len(counts))
    ys = np.random.uniform(0.08, 0.92, len(counts))
    mx = max(c for _, c in counts)
    for i, ((word, cnt), x, y) in enumerate(zip(counts, xs, ys)):
        size = 12 + (cnt / mx) * 28
        ax.text(x, y, word, fontsize=size, color=colors[i % len(colors)], ha="center", va="center", alpha=0.90, transform=ax.transAxes, fontweight="bold")
    ax.set_title("Word Cloud", fontsize=20, color="#4A3F7A", fontweight="bold", pad=20)
    path = "/tmp/wordcloud.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def build_keyword_bar_chart(top_n=15):
    if not KB_TEXTS:
        return None
    stop = {"this", "that", "with", "from", "have", "been", "were", "they", "their", "there", "what", "when", "which", "will", "also", "than", "into", "more"}
    top = Counter(w for w in re.findall(r"\b[a-zA-Z]{4,}\b", " ".join(KB_TEXTS).lower()) if w not in stop).most_common(top_n)
    if not top:
        return None
    words, vals = zip(*top)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(words[::-1], vals[::-1], color=[APP_ACCENT, APP_ACCENT_2, APP_ACCENT_3, APP_ACCENT_4] * 5)
    ax.set_title("Top Keywords", fontsize=18, color="#4A3F7A", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for b in bars:
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height()/2, f"{int(b.get_width())}", va="center", fontsize=10)
    plt.tight_layout()
    path = "/tmp/top_keywords_chart.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def build_language_pie_chart():
    if not KB_META:
        return None
    counts = Counter(m["lang"] for m in KB_META)
    labels = ["Arabic" if k == "ar" else "English" for k in counts.keys()]
    sizes = list(counts.values())
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=[APP_ACCENT, APP_ACCENT_3, APP_ACCENT_2], wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Language Distribution", fontsize=18, color="#4A3F7A", fontweight="bold")
    path = "/tmp/lang_pie.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def refresh_analytics():
    stats_md = f"# Session Stats\n\n- Questions asked: **{CHAT_STATS['questions']}**\n- Answers from PDFs: **{CHAT_STATS['found']}**\n- Questions not fully answered from PDFs: **{CHAT_STATS['notfound']}**\n- General-knowledge fallbacks: **{CHAT_STATS['general_used']}**\n- Chunks indexed: **{len(KB_TEXTS):,}**\n- Similarity threshold: **{MIN_SIMILARITY:.2f}**\n- Last build time: **{LAST_BUILD_SECONDS:.2f} sec**\n"
    return stats_md, build_word_cloud_image(), build_keyword_bar_chart(), build_language_pie_chart()


def update_threshold(val):
    global MIN_SIMILARITY
    MIN_SIMILARITY = val
    return f"✅ Threshold set to **{val:.2f}**"


def update_strict_mode(val):
    global STRICT_RAG_MODE
    STRICT_RAG_MODE = bool(val)
    return f"✅ Strict RAG Mode: **{'ON' if STRICT_RAG_MODE else 'OFF'}**"


def get_world_bank_data(country_code, indicator, start_year, end_year):
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?date={start_year}:{end_year}&per_page=100&format=json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data or len(data) < 2 or not data[1]:
            return pd.DataFrame()
        rows = [{"year": int(e["date"]), "value": float(e["value"])} for e in data[1] if e.get("value") is not None and e.get("date") is not None]
        return pd.DataFrame(rows).dropna().sort_values("year").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_economic_chunks(texts: list, max_chunks: int = 40) -> list:
    econ = [t for t in texts if any(kw in t.lower() for kw in ECON_TRIGGER)]
    if len(econ) < 10:
        n = len(texts)
        econ = list(dict.fromkeys(texts[:min(10, n)] + texts[max(0, n//2-5):n//2+5] + texts[-min(10, n):] + econ))
    if len(econ) > max_chunks:
        step = max(1, len(econ) // max_chunks)
        econ = econ[::step][:max_chunks]
    return econ


def build_doc_sentiment_index():
    if not KB_TEXTS or not KB_META:
        return None, None
    files_texts = {}
    for text, meta in zip(KB_TEXTS, KB_META):
        files_texts.setdefault(meta["name"], []).append(text[:400])
    yearly_sentiment, file_results = {}, []
    for fname, texts in files_texts.items():
        sample = get_economic_chunks(texts, max_chunks=40)
        scores = [sentiment_score_numeric(t) for t in sample]
        avg = round(float(np.mean(scores)), 4) if scores else 0.0
        year = next((m["year"] for m in KB_META if m["name"] == fname and m.get("year")), None)
        file_results.append({"file": fname, "year": year if year else "NA", "sentiment": avg, "nchunks": len(sample), "label": "Optimistic" if avg > 0.05 else "Pessimistic" if avg < -0.05 else "Neutral"})
        if year:
            yearly_sentiment.setdefault(year, []).append(avg)
    df_files = pd.DataFrame(file_results).sort_values("year") if file_results else None
    df_yearly = pd.DataFrame([{"year": y, "sentiment": round(float(np.mean(v)), 4)} for y, v in sorted(yearly_sentiment.items())]) if yearly_sentiment else None
    return df_files, df_yearly


def run_adf_check(series: np.ndarray, name: str):
    from statsmodels.tsa.stattools import adfuller
    def adf_p(s):
        try:
            return adfuller(s, autolag="AIC")[1]
        except Exception:
            return 1.0
    s = series.copy()
    p0 = adf_p(s)
    if p0 < 0.05:
        return s, f"{name}: Stationary at level (p={p0:.4f})", False
    s1 = np.diff(s)
    p1 = adf_p(s1)
    if p1 < 0.05:
        return s1, f"{name}: Non-stationary (p={p0:.4f}) → 1st diff stationary (p={p1:.4f})", True
    s2 = np.diff(s1)
    p2 = adf_p(s2)
    return s2, f"{name}: Non-stationary (p={p0:.4f}) → 1st diff (p={p1:.4f}) → 2nd diff {'stationary' if p2 < 0.05 else 'non-stationary'} (p={p2:.4f})", True


def run_granger_test(series_y, series_exog, maxlag=4):
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        if len(series_y) < 10:
            return "Granger Test skipped: need at least 10 points.", False
        sy, status_y, _ = run_adf_check(series_y.copy(), "Target")
        sexog, status_exog, _ = run_adf_check(series_exog.copy(), "Sentiment")
        minlen = min(len(sy), len(sexog))
        sy, sexog = sy[-minlen:], sexog[-minlen:]
        maxlag = min(maxlag, max(1, len(sy) - 1), 3)
        if len(sy) < 5:
            return "Granger Test skipped: too few observations after differencing.", False
        gc_result = grangercausalitytests(np.column_stack([sy, sexog]), maxlag=maxlag, verbose=False)
        rows, any_pass, best_p = [], False, 1.0
        for lag, res in gc_result.items():
            pval = res[0]["ssr_ftest"][1]
            fval = res[0]["ssr_ftest"][0]
            sig = "Yes" if pval < 0.05 else "Marginal" if pval < 0.10 else "No"
            if pval < 0.05:
                any_pass = True
            best_p = min(best_p, pval)
            rows.append(f"| {lag} | {fval:.4f} | {pval:.4f} | {sig} |")
        table = "## Granger Causality Test\n\n| Lag | F-stat | p-value | Significant? |\n|---:|---:|---:|---|\n" + "\n".join(rows)
        verdict = "\n\n✅ PASS: Sentiment significantly Granger-causes target." if any_pass else (f"\n\n🟡 Marginal: best p={best_p:.4f}" if best_p < 0.10 else "\n\n❌ FAIL: No significant Granger causality.")
        return table + "\n\n### ADF Pre-check\n- " + status_y + "\n- " + status_exog + verdict, any_pass
    except Exception as e:
        return f"Granger test error: {e}", False


def run_dm_test(actual, pred_arima, pred_sarimax):
    try:
        n = len(actual)
        if n < 3:
            return "DM Test skipped: n < 3.", False
        d = (actual - pred_arima) ** 2 - (actual - pred_sarimax) ** 2
        dmean = np.mean(d)
        dstd = np.std(d, ddof=1)
        if dstd < 1e-10:
            return "DM Test skipped: models nearly identical.", False
        dm_stat = dmean / (dstd / np.sqrt(n))
        pval = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
        sig = "Yes" if pval < 0.05 else "Marginal" if pval < 0.10 else "No"
        better = "SARIMAX+Ensemble" if dm_stat > 0 else "ARIMA"
        table = "## Diebold-Mariano Test\n\n| DM Statistic | p-value | n(test) | Significant? | Better Model |\n|---:|---:|---:|---|---|\n" + f"| {dm_stat:.4f} | {pval:.4f} | {n} | {sig} | {better} |"
        passed = pval < 0.05 and dm_stat > 0
        verdict = "\n\n✅ PASS: SARIMAX+Ensemble is significantly better." if passed else (f"\n\n🟡 Marginal evidence (p={pval:.4f})." if pval < 0.10 and dm_stat > 0 else f"\n\n❌ FAIL: Not statistically significant (p={pval:.4f}).")
        return table + verdict, passed
    except Exception as e:
        return f"DM error: {e}", False


def run_economic_forecast(country_code, target_var, start_year, end_year):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from sklearn.metrics import mean_squared_error, mean_absolute_error
    except ImportError:
        return "Install statsmodels and scikit-learn first.", None
    indicator_map = {"Inflation (CPI %)": "FP.CPI.TOTL.ZG", "GDP Growth": "NY.GDP.MKTP.KD.ZG", "Unemployment": "SL.UEM.TOTL.ZS", "Exchange Rate": "PA.NUS.FCRF"}
    econ_df = get_world_bank_data(country_code, indicator_map.get(target_var, "FP.CPI.TOTL.ZG"), int(start_year), int(end_year))
    if econ_df.empty or len(econ_df) < 5:
        return "No enough World Bank data found. Widen the year range.", None
    df_files, df_yearly = build_doc_sentiment_index()
    if df_yearly is not None and len(df_yearly) >= 2:
        merged = econ_df.merge(df_yearly, on="year", how="left")
        merged["sentiment"] = merged["sentiment"].fillna(float(df_yearly["sentiment"].mean()))
        has_yearly = True
        mode_msg = "Yearly Ensemble Sentiment"
    else:
        global_sent = float(pd.to_numeric(df_files["sentiment"], errors="coerce").mean()) if df_files is not None and len(df_files) > 0 else 0.0
        merged = econ_df.copy()
        merged["sentiment"] = global_sent
        has_yearly = False
        mode_msg = "Global Sentiment"
    if merged["sentiment"].std() < 1e-6:
        scaler = MinMaxScaler(feature_range=(-0.3, 0.3))
        merged["sentiment"] = scaler.fit_transform(merged[["sentiment"]]).flatten().round(4)
    series = merged["value"].values.astype(float)
    exog = merged["sentiment"].values.reshape(-1, 1)
    years = merged["year"].values
    n = len(series)
    split = max(n - 3, 5)
    train_y, test_y = series[:split], series[split:]
    train_exog, test_exog = exog[:split], exog[split:]
    test_years = years[split:]
    m1 = ARIMA(train_y, order=(1, 1, 1)).fit()
    pred_arima = m1.forecast(len(test_y))
    rmse_a = float(np.sqrt(mean_squared_error(test_y, pred_arima)))
    mae_a = float(mean_absolute_error(test_y, pred_arima))
    mape_a = float(np.mean(np.abs(test_y - pred_arima) / np.maximum(np.abs(test_y), 1e-8)) * 100)
    m2 = SARIMAX(train_y, exog=train_exog, order=(1, 1, 1)).fit(disp=False)
    pred_s = m2.forecast(len(test_y), exog=test_exog)
    rmse_s = float(np.sqrt(mean_squared_error(test_y, pred_s)))
    mae_s = float(mean_absolute_error(test_y, pred_s))
    mape_s = float(np.mean(np.abs(test_y - pred_s) / np.maximum(np.abs(test_y), 1e-8)) * 100)
    impr_rmse = (rmse_a - rmse_s) / rmse_a * 100 if rmse_a != 0 else 0.0
    if has_yearly and df_yearly is not None and len(df_yearly) >= 5:
        real_merged = econ_df.merge(df_yearly, on="year", how="inner")
        gc_y = real_merged["value"].values.astype(float)
        gc_exog = real_merged["sentiment"].values.astype(float)
    else:
        gc_y = series
        gc_exog = merged["sentiment"].values
    granger_md, granger_pass = run_granger_test(gc_y, gc_exog, maxlag=4)
    dm_md, dm_pass = run_dm_test(test_y, np.array(pred_arima), np.array(pred_s))
    fig, axes = plt.subplots(4, 1, figsize=(11, 18))
    fig.patch.set_facecolor("#FFFDF8")
    axes[0].plot(years, series, "o-", color="#3F51B5", label="Actual", lw=2.8, ms=6)
    axes[0].plot(test_years, pred_arima, "s--", color="#FF7043", label="ARIMA(1,1,1)", lw=2.6)
    axes[0].plot(test_years, pred_s, "^-.", color="#26A69A", label="SARIMAX+Ensemble", lw=2.8)
    axes[0].axvline(x=years[split - 1], color="#8D6E63", linestyle="--", alpha=0.8, label="Train/Test")
    axes[0].set_title(f"{target_var} — {country_code} ({mode_msg})", color="#4A3F7A", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)
    colors = ["#66BB6A" if s > 0.05 else "#FF8A65" if s < -0.05 else "#FFD54F" for s in merged["sentiment"]]
    axes[1].bar(years, merged["sentiment"], color=colors, edgecolor="white", width=0.6)
    axes[1].axhline(y=0, color="#5D4037", lw=0.8)
    axes[1].set_title("Ensemble Sentiment Index", color="#4A3F7A", fontweight="bold")
    bars = axes[2].bar(["ARIMA(1,1,1)", "SARIMAX+Ensemble"], [rmse_a, rmse_s], color=["#FF8A65" if rmse_a <= rmse_s else "#26A69A", "#26A69A" if rmse_s < rmse_a else "#FF8A65"], width=0.45, edgecolor="white")
    for bar, val in zip(bars, [rmse_a, rmse_s]):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11, color="#4A3F7A")
    axes[2].set_title("RMSE Comparison (lower is better)", color="#4A3F7A", fontweight="bold")
    axes[2].grid(True, alpha=0.25, axis="y")
    axes[3].axis("off")
    test_data = [["Granger + ADF", "PASS" if granger_pass else "FAIL", "Sentiment Granger-causes target" if granger_pass else "No causal link detected"], ["Diebold-Mariano", "PASS" if dm_pass else "FAIL", "SARIMAX significantly better" if dm_pass else f"n(test)={len(test_y)} may limit power"]]
    tbl4 = axes[3].table(cellText=test_data, colLabels=["Test", "Result", "Interpretation"], cellLoc="center", loc="center", colWidths=[0.35, 0.2, 0.45])
    tbl4.auto_set_font_size(False)
    tbl4.set_fontsize(11)
    tbl4.scale(1, 2.5)
    for (row, col), cell in tbl4.get_celld().items():
        if row == 0:
            cell.set_facecolor("#6C63FF")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == 1:
            cell.set_facecolor("#E8F5E9" if granger_pass else "#FFEBEE")
        elif row == 2:
            cell.set_facecolor("#E8F5E9" if dm_pass else "#FFEBEE")
    axes[3].set_title("Statistical Tests Summary", color="#4A3F7A", fontweight="bold", pad=20)
    plt.tight_layout(pad=3.0)
    img_path = "/tmp/forecast_plot.png"
    plt.savefig(img_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    result_md = f"# Forecast — {country_code} / {target_var}\n\n- Sentiment Mode: **{mode_msg}**\n- Train samples: **{split}**\n- Test samples: **{len(test_y)}**\n\n| Model | RMSE | MAE | MAPE |\n|---|---:|---:|---:|\n| ARIMA(1,1,1) | {rmse_a:.4f} | {mae_a:.4f} | {mape_a:.1f}% |\n| SARIMAX+Ensemble | {rmse_s:.4f} | {mae_s:.4f} | {mape_s:.1f}% |\n\n- RMSE improvement: **{impr_rmse:.1f}%**\n\n---\n\n{granger_md}\n\n---\n\n{dm_md}"
    return result_md, img_path


CUSTOM_CSS = """
.gradio-container {background: linear-gradient(135deg, #FFFDF8 0%, #FFF8F0 40%, #F8F4FF 100%) !important;}
.block {border-radius: 18px !important; box-shadow: 0 8px 24px rgba(108,99,255,0.10) !important;}
button.primary {background: linear-gradient(90deg, #6C63FF 0%, #26A69A 100%) !important; border: none !important;}
"""

with gr.Blocks(title="RAG Sentiment Forecast", theme=gr.themes.Soft(), css=CUSTOM_CSS) as app:
    gr.Markdown("# Hybrid Multilingual RAG + Fine-Tuned Economic Sentiment + Economic Forecast\n\n**ENSSEA — Masters Thesis** | Si Tayeb Houari | 2025–2026")

    with gr.Tab("1 · Upload files"):
        files = gr.File(label="Upload Files (PDF / TXT / CSV / DOCX)", file_types=[".pdf", ".txt", ".csv", ".docx"], file_count="multiple", type="filepath")
        build_btn = gr.Button("Build Index", variant="primary")
        status = gr.Markdown("No index built yet.")
        timer_box = gr.Markdown("### ⏳ Build Timer\n\nPress **Build Index** to start.")
        with gr.Row():
            save_btn = gr.Button("Save Index")
            load_btn = gr.Button("Load Saved Index")
        persist_status = gr.Markdown()
        sim_slider = gr.Slider(0.05, 0.60, value=0.18, step=0.01, label="Similarity Threshold")
        threshold_status = gr.Markdown()
        strict_toggle = gr.Checkbox(value=False, label="Strict RAG Mode")
        strict_status = gr.Markdown("✅ Strict RAG Mode: **OFF**")
        build_btn.click(build_index, inputs=files, outputs=[status, timer_box])
        save_btn.click(save_index, outputs=persist_status)
        load_btn.click(load_saved_index, outputs=persist_status)
        sim_slider.change(update_threshold, inputs=sim_slider, outputs=threshold_status)
        strict_toggle.change(update_strict_mode, inputs=strict_toggle, outputs=strict_status)

    with gr.Tab("2 · Sentiment / Search"):
        inp = gr.Textbox(lines=3, label="Enter text or keyword")
        run_btn = gr.Button("Analyze / Search", variant="primary")
        with gr.Row():
            out_sent = gr.Textbox(label="Sentiment")
            out_conf = gr.Number(label="Score")
        out_full = gr.Markdown()
        rep_btn = gr.Button("Download Report")
        rep_file = gr.File(label="Report")
        run_btn.click(predict_with_rag, inputs=inp, outputs=[out_sent, out_conf, out_full])
        rep_btn.click(generate_report, inputs=[inp, out_sent, out_conf, out_full], outputs=rep_file)

    with gr.Tab("3 · Smart Chatbot"):
        chatbot = gr.Chatbot(height=430, type="messages", show_label=False)
        msg = gr.Textbox(placeholder="Ask anything about your documents", label="Message")
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")
            exp_btn = gr.Button("Export")
        exp_file = gr.File(label="Chat Export")
        msg.submit(chat_text, inputs=[msg, chatbot], outputs=[msg, chatbot])
        send_btn.click(chat_text, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
        exp_btn.click(export_chat, inputs=chatbot, outputs=exp_file)

    with gr.Tab("4 · Voice"):
        gr.Markdown("Speak your question and get a spoken answer.")
        voice_input = gr.Audio(sources=["microphone"], type="numpy", label="Record")
        voice_btn = gr.Button("Ask by Voice", variant="primary")
        voice_chat = gr.Chatbot(height=300, type="messages")
        audio_output = gr.Audio(label="Answer Voice", autoplay=True)
        transcript_out = gr.Textbox(label="Transcript")
        voice_btn.click(chat_voice, inputs=[voice_input, voice_chat], outputs=[voice_chat, audio_output, transcript_out])

    with gr.Tab("5 · Analytics"):
        analytics_btn = gr.Button("Refresh Interactive Analytics", variant="primary")
        analytics_stats = gr.Markdown()
        with gr.Row():
            wc_img = gr.Image(label="Word Cloud", type="filepath")
            lang_pie = gr.Image(label="Language Distribution", type="filepath")
        kw_chart = gr.Image(label="Top Keywords", type="filepath")
        analytics_btn.click(refresh_analytics, outputs=[analytics_stats, wc_img, kw_chart, lang_pie])

    with gr.Tab("6 · About"):
        gr.Markdown("# Hybrid Multilingual RAG Framework\n\nThis fixed version improves the chatbot so it does not overclaim that it found a direct answer in PDFs when it only found related passages. It also routes 'explain the PDF' and similar requests to a document-summary mode instead of narrow QA.")

    with gr.Tab("7 · Forecast"):
        gr.Markdown("Economic Forecast: ARIMA vs SARIMAX with ensemble sentiment. Forecast logic is preserved; colors were improved.")
        with gr.Row():
            country_input = gr.Textbox(value="DZ", label="Country Code (ISO)")
            target_input = gr.Dropdown(choices=["Inflation (CPI %)", "GDP Growth", "Unemployment", "Exchange Rate"], value="Inflation (CPI %)", label="Target Variable")
        with gr.Row():
            start_year = gr.Slider(minimum=1990, maximum=2022, value=2000, step=1, label="Start Year")
            end_year = gr.Slider(minimum=2000, maximum=2025, value=2023, step=1, label="End Year")
        forecast_btn = gr.Button("Run Forecast", variant="primary", size="lg")
        forecast_result = gr.Markdown()
        forecast_plot = gr.Image(label="Forecast Chart", type="filepath")
        forecast_btn.click(run_economic_forecast, inputs=[country_input, target_input, start_year, end_year], outputs=[forecast_result, forecast_plot])

    with gr.Tab("8 · Auto RAG Eval"):
        eval_samples = gr.Slider(minimum=4, maximum=20, value=10, step=1, label="Number of evaluation samples")
        eval_btn = gr.Button("Run Automatic RAG Evaluation", variant="primary")
        eval_md = gr.Markdown()
        eval_plot = gr.Image(label="Evaluation Chart", type="filepath")
        eval_btn.click(evaluate_rag_pipeline, inputs=eval_samples, outputs=[eval_md, eval_plot])

app.launch(server_name="0.0.0.0", server_port=7860, show_api=False) 