import os
import time
import pickle
import numpy as np
import pandas as pd
import faiss
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# --- CẤU HÌNH ---
# Lấy từ Biến môi trường
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("⚠️ WARNING: Chưa có GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

EMBED_MODEL = "text-embedding-004"
LLM_MODEL = "models/gemini-2.5-flash"
DATA_CSV = "marketing_recommendations.csv"
EMB_FILE = "embeddings.npy"
DOC_FILE = "docs.pkl"
INDEX_FILE = "faiss.index"
BATCH_SIZE = 16
SLEEP_PER_CALL = 0.12
MAX_RETRIES = 3

# --- KHỞI TẠO APP ---
app = FastAPI(title="BepSachViet AI Service")

# Cấu hình CORS (Cho phép Frontend gọi)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc điền domain Vercel cụ thể để bảo mật hơn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- BIẾN TOÀN CỤC ---
docs_meta = []
embeddings = None
index = None
llm = None


# --- HÀM TIỆN ÍCH ---
def row_to_chunk_with_meta(row):
    product_id = str(row.get("Mã sản phẩm", "")).strip()
    product_name = str(row.get("Tên sản phẩm", "")).strip()
    quantity_info = str(row.get("Định lượng", ""))

    text = f"""
[PRODUCT_PROFILE]
Mã: {product_id} | Tên: {product_name}
Danh mục: {row.get('Danh mục','')}
Giá: {row.get('Giá bán hiện tại','')}
Tồn kho: {row.get('Số lượng hàng tồn kho','')}
Đánh giá: {row.get('Điểm đánh giá','')} ({row.get('Số lượng đánh giá','')} reviews)
""".strip()
    return {"id": product_id, "title": f"{product_name}", "text": text}


def embed_one(text: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = genai.embed_content(model=EMBED_MODEL, content=text)
            emb = np.array(r["embedding"], dtype="float32")
            time.sleep(SLEEP_PER_CALL)
            return emb
        except Exception:
            time.sleep(0.5 * attempt)
    raise RuntimeError("Embedding failed")


# --- STARTUP EVENT (Load dữ liệu 1 lần duy nhất) ---
@app.on_event("startup")
def startup_event():
    global docs_meta, embeddings, index, llm
    print("🚀 Starting AI Service...", flush=True)

    if not os.path.exists(DATA_CSV):
        print(f"⚠️ Missing data file: {DATA_CSV}")
        return

    # Load dữ liệu thô
    df = pd.read_csv(DATA_CSV).fillna("")
    docs_meta = [row_to_chunk_with_meta(row) for _, row in df.iterrows()]

    # Load FAISS
    if os.path.exists(EMB_FILE) and os.path.exists(INDEX_FILE):
        print("Loading FAISS cache...", flush=True)
        embeddings = np.load(EMB_FILE)
        with open(DOC_FILE, "rb") as f:
            docs_meta = pickle.load(f)
        index = faiss.read_index(INDEX_FILE)
    else:
        print(
            "⚠️ Không tìm thấy file index. Vui lòng build index dưới local và push lên!",
            flush=True,
        )

    llm = genai.GenerativeModel(LLM_MODEL)
    print("✅ AI Service READY", flush=True)


# --- RAG LOGIC ---
def find_exact_product(product_name: str):
    name_lower = product_name.lower().strip()
    for doc in docs_meta:
        if name_lower in doc["title"].lower():
            return doc
    return None


def rag_pipeline(product_name: str, mode="facebook"):
    exact_doc = find_exact_product(product_name)
    context = exact_doc["text"] if exact_doc else "Không có dữ liệu chi tiết."

    if mode == "facebook":
        prompt = f"""
        Bạn là chuyên gia Copywriter. Hãy viết 1 bài quảng cáo Facebook cho sản phẩm:
        {context}
        Yêu cầu: Hấp dẫn, có emoji, không chia mục, giọng văn tự nhiên.
        """
    else:  # marketing plan
        prompt = f"""
        Lập kế hoạch marketing ngắn hạn cho sản phẩm:
        {context}
        Yêu cầu: Phân tích vấn đề, đề xuất chiến lược cụ thể.
        """

    try:
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Lỗi AI: {str(e)}"


# --- GOOGLE SHEETS LOGIC (Dùng biến môi trường) ---
def append_to_sheet(data: dict):
    json_creds = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if not json_creds:
        print("⚠️ Chưa cấu hình GOOGLE_CREDENTIALS_JSON")
        return

    creds_dict = json.loads(json_creds)
    creds = Credentials.from_service_account_info(
        creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    service = build("sheets", "v4", credentials=creds)

    values = [[data["id"], data["product"], data["content"], data["image"], "To do"]]
    service.spreadsheets().values().append(
        spreadsheetId="1NgDk-c5rusOUw8LhJXZWzh9cEn4rnimQ5lS4r_JqoSE",
        range="Sheet1!A:E",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()


# --- API ENDPOINTS ---
class AdsRequest(BaseModel):
    product_name: str
    product_id: Optional[str] = ""
    image: Optional[str] = ""


@app.get("/")
def health_check():
    return {"status": "ok", "service": "BepSachViet AI"}


@app.post("/api/facebook-ads/generate")
def generate_ads(req: AdsRequest):
    content = rag_pipeline(req.product_name, mode="facebook")
    try:
        append_to_sheet(
            {
                "id": req.product_id,
                "product": req.product_name,
                "content": content,
                "image": req.image,
            }
        )
    except Exception as e:
        print(f"Lỗi Sheet: {e}")

    return {"content": content}


@app.post("/generate-marketing-plan")
def generate_plan(req: AdsRequest):
    content = rag_pipeline(req.product_name, mode="plan")
    return {"answer": content}
