# Persian Retrieval-Augmented Generation (RAG) System

## Overview
A production-ready system for Persian document understanding using Qwen and GPT models:
- FastAPI backend
- FAISS vector store
- Local and OpenAI LLM support
- Handles Persian PDF, Word, Excel with table extraction

## APIs
| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/health` | GET | Check system health |
| `/documents/upload` | POST | Upload and process documents |
| `/query` | POST | Ask questions about indexed documents |

## Usage Example (Python)
```python
import requests

files = {'file': open('report.pdf','rb')}
r = requests.post('http://localhost:8000/documents/upload', files=files)
print(r.json())

query = {"question": "جدول درآمد شرکت چه اطلاعاتی دارد؟"}
resp = requests.post("http://localhost:8000/query", json=query)
print(resp.json())

## Run the Server
bash
pip install -r requirements.txt
python main.py


---