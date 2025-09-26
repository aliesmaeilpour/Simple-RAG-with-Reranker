from typing import List, Dict
import re
import pymupdf
from hazm import Normalizer, SentenceTokenizer

class PDFProcessor:
    def __init__(self):
        self.normalizer = Normalizer()
        self.sent_tokenizer = SentenceTokenizer()

    def extract_text(self, pdf_path: str) -> str:
        doc = pymupdf.open(pdf_path)
        texts = []
        for page in doc:
            texts.append(page.get_text())
        text = "\n".join(texts)
        # Normalize Persian text
        text = self.normalizer.normalize(text)
        # Clean noisy whitespace
        text = re.sub(r"[\u200c\u200f\u202a-\u202e]", "", text)  # remove bidi marks
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def chunk_text(self, text: str, chunk_size: int = 550, overlap: int = 120) -> List[str]:
        # Split to sentences with Hazm, then accumulate into chunks near target size
        sentences = self.sent_tokenizer.tokenize(text)
        chunks = []
        current = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(current) + len(s) + 1 <= chunk_size:
                current = f"{current} {s}".strip()
            else:
                if current:
                    chunks.append(current)
                # start new with overlap from end of previous
                if overlap > 0 and chunks:
                    tail = chunks[-1][-overlap:]
                    current = f"{tail} {s}".strip()
                else:
                    current = s
        if current:
            chunks.append(current)
        return chunks
