# 📂 AI File Analyst

**AI-powered document analyzer built with Streamlit**  
Upload a file — get instant summaries, insights, and Q&A using LLMs.

---

## 📦 Features

- ✅ Upload files: `.csv`, `.xlsx`, `.pdf`, `.docx`, `.txt`, `.jpg`, `.png`, and more  
- ✅ Automatically extract and display clean text  
- ✅ Ask questions about the content using LLMs  
- ✅ Summarization, keyword extraction, and entity detection  
- ✅ Secure API integration via Streamlit Secrets  
- ✅ Simple, interactive UI built with Streamlit

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/abhinavyadav11/AI-DataAnalyst-Agent
cdAI-DataAnalyst-Agent
```

### 2. Create virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```

---

## 🔐 API Key Setup

### 1. Create `.streamlit/secrets.toml`

```toml
[api_keys]
openai = "your-openai-key"
together = "your-together-api-key"
```

### 2. Add to `.gitignore`

```bash
# .gitignore
.streamlit/secrets.toml
.env
```

### 3. Access in your code

```python
import streamlit as st

openai_key = st.secrets["api_keys"]["openai"]
```

---

## 🧪 Running the App Locally

```bash
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push your code to a public GitHub repository  
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)  
3. Deploy the repo  
4. Add your secrets via **Manage App → Secrets**

---

## 🛠️ Tech Stack

- Streamlit  
- Python  
- OpenAI / Together AI API  
- PyMuPDF, python-docx, pandas  
- Pillow, pdfplumber  
- dotenv / toml

---

## 📄 Supported File Types

| Format   | Use Case                         |
|----------|----------------------------------|
| `.csv`   | Tabular data parsing             |
| `.pdf`   | Document analysis & summarization|
| `.docx`  | Word documents                   |
| `.txt`   | Raw text processing              |
| `.jpg`   | OCR for image-based text         |
| `.png`   | OCR for image-based text         |

---

## 🤖 Future Plans

- [ ] Multi-file Q&A  
- [ ] Chat history memory  
- [ ] PDF visual layout extraction  
- [ ] Export answers to PDF/CSV  

---

## 👤 Author

**Abhinav Yadav**  
💼 AI/ML | LLMs | Data Science  
[LinkedIn](https://www.linkedin.com/in/abhinav-yadav-70088a252/)

---

## 📄 License

MIT License — feel free to use and modify.
