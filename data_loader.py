!pip install sentence-transformers faiss-cpu rank-bm25 fastapi uvicorn httpx rapidfuzz sacrebleu rouge-score sqlalchemy

import pandas as pd
import re
import json

df1 = pd.read_csv('/content/Natural-Questions-Base.csv')
df2 = pd.read_csv('/content/Natural-Questions-Filtered.csv')
df = pd.concat([df1, df2], ignore_index=True).drop_duplicates()

print("Dataset shape:", df.shape)

def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()
