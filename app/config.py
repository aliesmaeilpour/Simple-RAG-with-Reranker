from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
EXTRACTED_DIR = DATA_DIR / "extracted"

# Models
EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking
CHUNK_SIZE = 550         # 400-600 chars
CHUNK_OVERLAP = 120      # about 20-25 percent
TOP_K = 12               # initial retrieve
FINAL_K = 5              # after rerank

# Scoring weight
SEM_WEIGHT = 0.6
RERANK_WEIGHT = 0.4
