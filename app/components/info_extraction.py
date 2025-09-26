from typing import Dict, Any, List
import re
import json
from pathlib import Path
from datetime import datetime

CURRENCY_RE = re.compile(r"(?:\b|\s)(?:(?:USD|EUR|IRR|ریال|تومان)\s*)?([0-9]{1,3}(?:[,\u066C\u066B]?[0-9]{3})*(?:\.[0-9]+)?)", re.UNICODE)
DATE_RE = re.compile(r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
NUMBER_RE = re.compile(r"(?<!\w)(\d{2,}(?:[\.,]\d+)?)")

class InfoExtractionAgent:
    def __init__(self, extracted_dir: Path):
        self.dir = extracted_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def extract(self, doc_id: str, text: str, chunks: List[str]) -> Dict[str, Any]:
        data = {
            "doc_id": doc_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": self._summarize(text),
            "financial_figures": self._find_financials(text),
            "dates": self._find_dates(text),
            "percents": self._find_percents(text),
            "notable_numbers": self._find_numbers(text)[:50],
            "top_sentences": self._top_sentences(chunks)
        }
        path = self.dir / f"{doc_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data

    def _summarize(self, text: str) -> str:
        # Heuristic summary, first 3 long sentences
        sents = re.split(r"(?<=[.!؟\?])\s+", text)
        top = [s for s in sents if len(s) > 80][:3]
        return " ".join(top)[:1200]

    def _find_financials(self, text: str) -> List[str]:
        vals = []
        for m in CURRENCY_RE.finditer(text):
            vals.append(m.group(0).strip())
        return list(dict.fromkeys(vals))[:100]

    def _find_dates(self, text: str) -> List[str]:
        vals = [m.group(0) for m in DATE_RE.finditer(text)]
        return list(dict.fromkeys(vals))[:100]

    def _find_percents(self, text: str) -> List[str]:
        vals = [m.group(0) for m in PERCENT_RE.finditer(text)]
        return list(dict.fromkeys(vals))[:100]

    def _find_numbers(self, text: str) -> List[str]:
        vals = [m.group(0) for m in NUMBER_RE.finditer(text)]
        return list(dict.fromkeys(vals))

    def _top_sentences(self, chunks: List[str]) -> List[str]:
        # naive scoring by presence of numbers or keywords
        KEYWORDS = ["مالی", "درآمد", "هزینه", "بودجه", "قرارداد", "سود", "زیان", "پرداخت", "تاریخ"]
        scored = []
        for c in chunks:
            score = 0
            if any(k in c for k in KEYWORDS):
                score += 2
            if PERCENT_RE.search(c):
                score += 1
            if CURRENCY_RE.search(c):
                score += 2
            if NUMBER_RE.search(c):
                score += 1
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in scored[:10]]
