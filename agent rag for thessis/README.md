---
title: Multilingual RAG Sentiment Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: false
license: mit
---

# 🤖 Multilingual RAG + Sentiment Assistant

A smart multilingual AI assistant combining:
- 🔍 Hybrid RAG (FAISS + BM25 + CrossEncoder)
- 🎭 Multilingual Sentiment Analysis (XLM-RoBERTa)
- 🤖 LLM Chat (Llama-3.3-70B via Groq)
- 🎙️ Voice Interface (Whisper + gTTS)
- 💾 Persistent Index Storage
- 📝 DOCX / PDF / TXT / CSV Support
- 🏷️ Automatic Document Type Detection

## 🚀 Setup

1. Add `GROQ_API_KEY` to HuggingFace Space Secrets
2. Upload your files and click **Build Index**
3. Start chatting!

## 📂 Supported File Types
| Format | Support |
|--------|---------|
| PDF    | ✅ |
| TXT    | ✅ |
| CSV    | ✅ |
| DOCX   | ✅ NEW |

## 🏷️ Document Types Detected
| Type     | Icon |
|----------|------|
| Economic | 📊   |
| Medical  | 🏥   |
| Legal    | ⚖️   |
| Academic | 🎓   |
| General  | 📄   |
