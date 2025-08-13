# DataChat

Upload a CSV → map column types → get quick EDA with **automated insights** → **chat with your data**.  
Answers come back as **tables** and **clean plots** (no code needed).

**Live app:** _add your URL here_  
**Repo:** https://github.com/joseph-sclar/DataChat

---

## ✨ Features

- **CSV upload** with optional sep/encoding
- **Column type mapping** (auto, int, float, bool, category, datetime, text)
- **Quick EDA**: metrics, missingness, describe, top categories, correlations, **auto insights**
- **AI chat**: ask questions in plain English; app returns:
  - `answer_df` (always a table)
  - Optional text note
  - Plot (bar charts formatted with readable labels)
- **Privacy by design**: only a compact schema + a few head rows go to the model
- **No code shown** in the UI; optional console output for debugging

---

## 🖼️ Demo (what it looks like)

1. Upload a CSV  
2. Map columns  
3. Click **Chatbot** and ask:  
   - “Top 10 products by revenue”  
   - “Monthly orders by country”  
   - “Average price and quantity per category”

---

## 🚀 Quickstart

### Prerequisites
- Python 3.9+  
- An OpenAI API key

### 1) Clone & install
```bash
git clone https://github.com/joseph-sclar/DataChat.git
cd DataChat

# (recommended) create a venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
