import hashlib
import re
import unicodedata
from collections import Counter

from qdrant_client import models


BM25_VECTOR_NAME = "bm25"


class BM25Encoder:
    """
    Inference-time BM25 sparse encoder for Persian/Arabic and Latin text.

    Token IDs are 32-bit Blake2s hashes; collision probability is negligible
    for typical document sizes but non-zero — known, accepted tradeoff.

    `average_length` should ideally be the corpus-wide average token count.
    The default (256) is a reasonable prior when corpus statistics are unavailable.
    """

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        average_length: float = 256.0,
    ) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"b must be in [0, 1], got {b}")
        if average_length <= 0:
            raise ValueError(f"average_length must be > 0, got {average_length}")
        self.k1 = k1
        self.b = b
        self.average_length = average_length

    def __repr__(self) -> str:
        return (
            f"BM25Encoder("
            f"k1={self.k1}, b={self.b}, average_length={self.average_length})"
        )

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def tokenize(text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        normalized = normalized.translate(
            str.maketrans({
                # Arabic/Persian letter normalization
                "ي": "ی", "ى": "ی", "ك": "ک",
                # Zero-width non-joiner → space; tatweel → deleted
                "\u200c": " ", "\u0640": None,
                # Extended-Arabic-Indic digits → ASCII
                "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
                "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
                # Arabic-Indic digits → ASCII
                "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
                "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
            })
        )
        # Strip combining marks (diacritics)
        normalized = "".join(
            ch for ch in normalized if unicodedata.category(ch) != "Mn"
        )
        return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)

    @staticmethod
    def _token_id(token: str) -> int:
        # 32-bit hash → compact Qdrant sparse indices; see class docstring re collisions
        return int.from_bytes(
            hashlib.blake2s(token.encode("utf-8"), digest_size=4).digest(), "big"
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_document(self, text: str) -> models.SparseVector:
        tokens = self.tokenize(text)
        if not tokens:
            return models.SparseVector(indices=[], values=[])

        frequencies = Counter(self._token_id(token) for token in tokens)
        normalization = self.k1 * (
            1 - self.b + self.b * len(tokens) / self.average_length
        )
        indices = sorted(frequencies)
        return models.SparseVector(
            indices=indices,
            values=[
                frequencies[idx] * (self.k1 + 1) / (frequencies[idx] + normalization)
                for idx in indices
            ],
        )

    def encode_query(self, text: str) -> models.SparseVector:
        # Unique token IDs, equal weight — standard simplified BM25 query encoding
        indices = sorted({self._token_id(token) for token in self.tokenize(text)})
        return models.SparseVector(indices=indices, values=[1.0] * len(indices))
