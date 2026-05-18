# 🤖 Multilingual RAG + Sentiment Web Application

> **Live Demo:** 🚀 [huggingface.co/spaces/sitayeb/xlmr](https://huggingface.co/spaces/sitayeb/xlmr)  
> Part of the ENSSEA Master's Thesis — Si Tayeb Houari | 2025–2026

---

## ✨ What This App Does

A full-stack AI-powered web application for **multilingual economic document analysis**, featuring:

| Feature | Description |
|---------|-------------|
| 🔍 Hybrid RAG | FAISS + BM25 + CrossEncoder reranking |
| 🎭 Sentiment Analysis | Ensemble of FinBERT + XLM-RoBERTa + Economic Lexicon |
| 🤖 Smart Chatbot | Llama-3.3-70B via Groq — RAG-grounded answers |
| 🎙️ Voice Interface | Whisper (speech-to-text) + gTTS (text-to-speech) |
| 📈 Economic Forecasting | ARIMA vs SARIMAX + Sentiment Index |
| 📊 Analytics Dashboard | Word cloud, language distribution, keyword charts |
| ✅ RAG Evaluation | Automated evaluation pipeline with metrics |
| 💾 Persistent Index | Save & reload FAISS index across sessions |

---

## 📂 Supported File Types

| Format | Status |
|--------|--------|
| PDF    | ✅ |
| TXT    | ✅ |
| CSV    | ✅ |
| DOCX   | ✅ |

---

## 🏷️ Auto Document Type Detection

The app automatically detects document type from content:

| Type     | Icon | Keyword Triggers |
|----------|------|-----------------|
| Economic | 📊 | GDP, inflation, fiscal, IMF… |
| Medical  | 🏥 | patient, diagnosis, clinical… |
| Legal    | ⚖️ | article, contract, jurisdiction… |
| Academic | 🎓 | abstract, methodology, DOI… |
| General  | 📄 | fallback |

---

## 🧠 Sentiment Ensemble Logic

The app uses a **weighted ensemble** of three signals:

```
Standard mode:
  FinBERT          → weight: 0.40
  XLM-RoBERTa      → weight: 0.20
  Economic Lexicon → weight: 0.15
  Fine-tuned FinBERT (Economic) → weight: 0.25

Economic document mode (auto-detected):
  Fine-tuned FinBERT → weight: 0.60
  Economic Lexicon   → weight: 0.25
  XLM-RoBERTa        → weight: 0.15
```

---

## 🚀 Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rag-sentiment-app.git
cd rag-sentiment-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key
```bash
export GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the app
```bash
python app.py
```

Then open: `http://localhost:7860`

---

## ☁️ HuggingFace Deployment

1. Fork or upload this repo to a HuggingFace Space
2. Set `GROQ_API_KEY` in Space **Secrets** (Settings → Variables and Secrets)
3. The app launches automatically

---

## 🗂️ App Tabs

```
Tab 1 · Upload Files    → Build FAISS + BM25 index from documents
Tab 2 · Sentiment       → Analyze text sentiment + RAG document search
Tab 3 · Smart Chatbot   → LLM-powered Q&A grounded in your documents
Tab 4 · Voice           → Record voice query → spoken answer
Tab 5 · Analytics       → Word cloud, language distribution, keywords
Tab 6 · About           → Project description
Tab 7 · Forecast        → ARIMA vs SARIMAX economic forecasting
Tab 8 · RAG Evaluation  → Automated retrieval quality metrics
```

---

## 📦 Requirements

```
gradio>=5.0.0
faiss-cpu>=1.7.4
sentence-transformers>=3.0.0
transformers>=4.44.0
groq>=0.11.0
langdetect>=1.0.9
gTTS>=2.5.0
torch>=2.1.0
statsmodels>=0.14.0
python-docx>=1.1.0
PyPDF2>=3.0.0
```

---

## 🔗 Related Repositories

| Repo | Description |
|------|-------------|
| [thesis-multilingual-rag-sentiment](https://github.com/YOUR_USERNAME/thesis-multilingual-rag-sentiment) | Main thesis repository |
| [thesis-notebooks](https://github.com/YOUR_USERNAME/thesis-notebooks) | Analysis notebooks |
| [thesis-evaluation-data](https://github.com/YOUR_USERNAME/thesis-evaluation-data) | Evaluation data & gold labels |

---

*© 2026 Si Tayeb Houari — ENSSEA Master's Thesis*
