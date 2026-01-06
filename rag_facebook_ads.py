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

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG Facebook Ads API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    global docs_meta, embeddings, index, llm

    print("🚀 Starting RAG Facebook Ads - loading resources...", flush=True)

    if not os.path.exists(DATA_CSV):
        raise RuntimeError(f"Missing data file: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV).fillna("")
    docs_meta = [row_to_chunk_with_meta(row) for _, row in df.iterrows()]

    print("🔍 Checking embedding dimension...", flush=True)
    test_emb = embed_one("test")
    emb_dim = test_emb.shape[0]
    print("Embedding dimension:", emb_dim, flush=True)

    if os.path.exists(EMB_FILE) and os.path.exists(INDEX_FILE) and os.path.exists(DOC_FILE):
        print("Loading existing embeddings, index, docs...", flush=True)
        embeddings = np.load(EMB_FILE)
        with open(DOC_FILE, "rb") as f:
            docs_meta = pickle.load(f)
        index = faiss.read_index(INDEX_FILE)
    else:
        raise RuntimeError("Missing FAISS resources, please build first")

    llm = genai.GenerativeModel(LLM_MODEL)

    print("RAG Facebook Ads READY", flush=True)


# ---------- Google Sheets ----------
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SPREADSHEET_ID = '1NgDk-c5rusOUw8LhJXZWzh9cEn4rnimQ5lS4r_JqoSE'
SHEET_RANGE = 'Sheet1!A:F'

def append_ads_to_sheet(data: dict):
    creds = Credentials.from_service_account_file(
        'credentials.json',
        scopes=SCOPES
    )
    service = build('sheets', 'v4', credentials=creds)

    values = [[
        data['id'],
        data['product'],
        data['content'],
        data['image'],
        "To do",
        ""  # URL facebook
    ]]

    body = {"values": values}

    service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=SHEET_RANGE,
        valueInputOption="RAW",
        body=body
    ).execute()

# ---------- Config ----------
# Lấy API key từ biến môi trường (không hardcode)
API_KEY = "AIzaSyA45DbNgVzDh_4EpPjGHA5ITtlCf3GvnFY"

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
Bạn là copywriter Facebook Ads chuyên viết bài quảng cáo để ĐĂNG / CHẠY ADS TRỰC TIẾP.

Dựa trên thông tin sản phẩm bên dưới, hãy VIẾT 1 BÀI QUẢNG CÁO FACEBOOK HOÀN CHỈNH NHẤT:

{context}

Yêu cầu bắt buộc:
- CHỈ VIẾT 1 BÀI POST FACEBOOK DUY NHẤT
- Viết liền mạch như người bán thật, KHÔNG chia mục, KHÔNG đánh số
- KHÔNG sử dụng hoặc nhắc tới các thuật ngữ:
  Hook, CTA, Headline, Primary text, Body, Insight
- KHÔNG giải thích, KHÔNG phân tích, KHÔNG liệt kê lý do
- Không nói đang quảng cáo, không nói “bài viết này”

Yêu cầu nội dung:
- Văn phong Facebook bán hàng PHỔ THÔNG, dễ chạy ads đại trà
- Ngắn gọn, dễ đọc, có xuống dòng hợp lý
- Có thể dùng emoji vừa phải (1-3 emoji)
- Nội dung cân bằng giữa:
  + Lợi ích sản phẩm    
  + Giá / ưu đãi (nếu có)
  + Độ tin cậy (bán chạy, đánh giá, tiện lợi… nếu dữ liệu cho phép)
- Có lời kêu gọi hành động TỰ NHIÊN như người bán thật
  (ví dụ: inbox, đặt ngay, mua liền hôm nay… nhưng KHÔNG ghi chữ “CTA”)

Giọng văn:
- Trung tính - thân thiện - bán hàng tự nhiên
- Không quá hype, không quá review, không quá cảm xúc
- Phù hợp làm bài ads chính để test hoặc scale

Output:
- Chỉ xuất DUY NHẤT nội dung bài quảng cáo Facebook
- KHÔNG thêm tiêu đề, KHÔNG thêm chú thích, KHÔNG thêm phân cách
"""

    try:
        response = llm.generate_content(prompt)
        return response.text.strip(), retrieved
    except Exception as e:
        return f"Lỗi LLM: {e}", retrieved
# ---------- FastAPI app ----------
class GenerateAdsRequest(BaseModel):
    product_name: str
    product_id: Optional[str] = ""
    image: Optional[str] = ""

@app.post("/api/facebook-ads/generate")
def generate_facebook_ads(req: GenerateAdsRequest):
    # 1️⃣ Sinh ads từ RAG
    print("HIT /api/facebook-ads/generate", flush=True)
    print("Request:", req, flush=True)

    content, retrieved = rag_pipeline(req.product_name)

    print("Generated content length:", len(content), flush=True)

    # 2️⃣ Lưu Google Sheet
    try:
        append_ads_to_sheet({
            "id": req.product_id or "",
            "product": req.product_name,
            "content": content,
            "image": req.image or ""
        })
    except Exception as e:
        print("⚠️ Google Sheet error:", e, flush=True)
        
    # 3️⃣ Trả về cho web
    return {
        "product": req.product_name,
        "content": content,
        "image": req.image,
        "status": "saved_to_sheet"
    }

if __name__ == "__main__":
    uvicorn.run("rag_facebook_ads:app", host="0.0.0.0", port=8000, reload=False)

