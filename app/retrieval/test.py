# 1. Confirm what BM25 produces for Persian
from bm25 import BM25Encoder
enc = BM25Encoder()
result = enc.encode_query("صفحه دو رو میتونی بهم توضیح بدی؟")
print(result.indices, result.values)  # if empty → always falls back to dense