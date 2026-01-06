# rag_api.py
import os
import time
import pickle
import numpy as np
import pandas as pd
import faiss
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai

# ---------- Config ----------
# Lấy API key từ biến môi trường (không hardcode)
API_KEY = "AIzaSyAUTFN98Qy5AbfmVf2G1bsBHyXo0Xxad-Y"

genai.configure(api_key=API_KEY)

EMBED_MODEL = "text-embedding-004"     # expected dim (vd 768)
LLM_MODEL = "models/gemini-2.5-flash"

DATA_CSV = "marketing_recommendations.csv"
EMB_FILE = "embeddings.npy"
DOC_FILE = "docs.pkl"
INDEX_FILE = "faiss.index"

BATCH_SIZE = 16
SLEEP_PER_CALL = 0.12
MAX_RETRIES = 3

# ---------- Utilities from your notebook ----------
def row_to_chunk_with_meta(row):
    product_id = str(row.get("Mã sản phẩm", "")).strip()
    product_name = str(row.get("Tên sản phẩm", "")).strip()

    # Chuẩn hóa định lượng
    quantity_info = ""
    if row.get("Định lượng (g)", ""):
        quantity_info = f"{row.get('Định lượng (g)')} g"
    elif row.get("Định lượng (ml)", ""):
        quantity_info = f"{row.get('Định lượng (ml)')} ml"
    else:
        quantity_info = str(row.get("Định lượng", ""))

    text = f"""
[PRODUCT_PROFILE]
Mã sản phẩm: {product_id}
Tên sản phẩm: {product_name}
Danh mục: {row.get('Danh mục','')}
Thương hiệu: {row.get('Thương hiệu','')}
Xuất xứ: {row.get('Xuất xứ','')}
Loại sản phẩm: {row.get('Loại sản phẩm','')}
Mùa phổ biến: {row.get('Mùa phổ biến','')}

[PRICE]
Giá gốc: {row.get('Giá gốc','')}
Giá bán hiện tại: {row.get('Giá bán hiện tại','')}
Giảm giá: {row.get('Giảm giá','')}

[SIZE]
Định lượng: {quantity_info}

[PERFORMANCE]
Số lượng đã bán: {row.get('Số lượng đã bán','')}
Số lượt xem sản phẩm: {row.get('Số lượt xem sản phẩm','')}
Điểm đánh giá: {row.get('Điểm đánh giá','')}
Số lượng đánh giá: {row.get('Số lượng đánh giá','')}

[INVENTORY]
Số lượng hàng tồn kho: {row.get('Số lượng hàng tồn kho','')}

[INSIGHT_HINT]
- Nếu lượt xem cao nhưng bán thấp → vấn đề giá hoặc chuyển đổi
- Nếu tồn kho cao → ưu tiên đẩy khuyến mãi
- Nếu rating cao → nên khai thác quảng cáo & niềm tin
""".strip()

    return {
        "id": product_id,
        "title": f"{product_name} ({product_id})",
        "text": text
    }


def embed_one(text: str):
    last_err = None
    for attempt in range(1, MAX_RETRIES+1):
        try:
            r = genai.embed_content(model=EMBED_MODEL, content=text)
            emb = np.array(r["embedding"], dtype="float32")
            time.sleep(SLEEP_PER_CALL)
            return emb
        except Exception as e:
            last_err = e
            wait = 0.5 * attempt
            print(f"Warning: embed error (attempt {attempt}/{MAX_RETRIES}): {e}. retry after {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Embedding failed after {MAX_RETRIES} retries. Last error: {last_err}")

# ---------- Load / build resources (on startup) ----------
print("Starting RAG API - loading resources...")

# load docs_meta from CSV (and normalizing)
if not os.path.exists(DATA_CSV):
    raise RuntimeError(f"Missing data file: {DATA_CSV}")
df = pd.read_csv(DATA_CSV).fillna("")
docs_meta = [row_to_chunk_with_meta(row) for _, row in df.iterrows()]

# Embeddings + FAISS index load or build
EMB_DIM = None
# quick test embed to get dim
print("Checking embedding dimension with test call...")
test_emb = embed_one("test")
EMB_DIM = test_emb.shape[0]
print("Embedding dimension:", EMB_DIM)

if os.path.exists(EMB_FILE) and os.path.exists(INDEX_FILE) and os.path.exists(DOC_FILE):
    print("Loading existing embeddings, index, docs...")
    embeddings = np.load(EMB_FILE)
    with open(DOC_FILE, "rb") as f:
        docs_meta = pickle.load(f)
    index = faiss.read_index(INDEX_FILE)
    if embeddings.shape[1] != EMB_DIM:
        raise RuntimeError("Loaded embeddings dimension != current embed dim. Delete files and rebuild.")
else:
    # build embeddings and index (this will take time)
    print("Embeddings/index not found - building from CSV (this may take a while)...")
    texts = [m["text"] for m in docs_meta]
    embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i+BATCH_SIZE]
        for t in chunk:
            embs.append(embed_one(t))
    embeddings = np.vstack(embs).astype("float32")
    np.save(EMB_FILE, embeddings)
    with open(DOC_FILE, "wb") as f:
        pickle.dump(docs_meta, f)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print("Built FAISS index and saved embeddings/docs.")

# LLM init
llm = genai.GenerativeModel(LLM_MODEL)

# ---------- Retrieval & RAG ----------
def find_exact_product(product_name: str):
    name_lower = product_name.lower().strip()

    for doc in docs_meta:
        if name_lower in doc["title"].lower():
            return doc
    return None

def retrieve(query: str, k=5):
    q_emb = embed_one(query).reshape(1, -1).astype("float32")
    D, I = index.search(q_emb, k)

    results = []
    for rank, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(docs_meta):
            continue
        meta = docs_meta[idx]
        results.append({
            "id": meta["id"],
            "title": meta["title"],
            "text": meta["text"],
            "distance": float(D[0][rank])
        })
    return results


def rag_pipeline(product_name: str, k: int = 4):
    # 1️⃣ Tìm đúng sản phẩm trước
    exact_doc = find_exact_product(product_name)

    if exact_doc:
        context = exact_doc["text"]
        retrieved = [exact_doc]
    else:
        # 2️⃣ fallback: dùng semantic search
        retrieved = retrieve(product_name, k=1)
        context = retrieved[0]["text"] if retrieved else "Không có dữ liệu."

    prompt = f"""
Bạn là chuyên gia marketing TMĐT.

Hãy xây dựng kế hoạch marketing CHỈ cho SẢN PHẨM SAU:

{context}

Yêu cầu:
1. Phân tích tình trạng sản phẩm
2. Xác định vấn đề chính
3. Đề xuất chiến lược marketing phù hợp
4. Hành động cụ thể trong 1–2 tháng

Lưu ý:
- KHÔNG so sánh với sản phẩm khác
- KHÔNG tổng hợp nhiều sản phẩm
"""

    try:
        response = llm.generate_content(prompt)
        return response.text.strip(), retrieved
    except Exception as e:
        return f"❌ Lỗi LLM: {e}", retrieved



# ---------- FastAPI endpoints ----------
app = FastAPI(title="RAG Marketing API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    product_name: str
    top_k: Optional[int] = 4

class GenerateResponse(BaseModel):
    product_name: str
    answer: str
    retrieved_docs: List[dict]

@app.post("/generate-marketing-plan", response_model=GenerateResponse)
def generate_marketing_plan(req: GenerateRequest):
    if not req.product_name or req.product_name.strip() == "":
        raise HTTPException(status_code=400, detail="product_name is required")

    # Build query using product_name (you can change phrasing)
    q = f"Hãy viết kế hoạch marketing chi tiết cho sản phẩm {req.product_name.strip()}."
    answer, retrieved = rag_pipeline(q, k=req.top_k or 4)

    return {
        "product_name": req.product_name,
        "answer": answer,
        "retrieved_docs": retrieved
    }

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Run ----------
if __name__ == "__main__":
    # chạy dev server
    uvicorn.run("rag_api:app", host="0.0.0.0", port=4000, reload=False)
