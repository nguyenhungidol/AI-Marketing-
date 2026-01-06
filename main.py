# main.py – FINAL (Render Production Ready)

from __future__ import annotations

import os
import json
import time
import pickle
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Any

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


# ======================================================
# LOGGING
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("BepSachViet-AI")


# ======================================================
# ENV CONFIG
# ======================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

DATA_CSV = "marketing_recommendations.csv"
EMB_FILE = "embeddings.npy"
DOC_FILE = "docs.pkl"
INDEX_FILE = "faiss.index"

EMBED_MODEL = "text-embedding-004"
LLM_MODEL = "models/gemini-2.5-flash"

SLEEP_PER_CALL = 0.12
MAX_RETRIES = 3
DEFAULT_TOP_K = 3

# Google Sheets (optional)
SHEET_ID = os.getenv("SHEET_ID")  # nếu thiếu -> skip
SHEET_RANGE = os.getenv("SHEET_RANGE", "Trang tính1!A:E")

# CORS
ALLOW_ORIGINS = os.getenv(
    "ALLOW_ORIGINS",
    "https://bepsachviet-fe.vercel.app"
).split(",")


# ======================================================
# INIT GEMINI
# ======================================================
if not GOOGLE_API_KEY:
    logger.warning("⚠️ GOOGLE_API_KEY is missing")

genai.configure(api_key=GOOGLE_API_KEY)


# ======================================================
# GLOBAL OBJECTS
# ======================================================
docs_meta: List[Dict[str, Any]] = []
index: Optional[faiss.Index] = None
llm = None
sheets_service = None


# ======================================================
# GOOGLE SHEETS SERVICE (OPTIONAL)
# ======================================================
def init_google_sheets():
    global sheets_service
    if not GOOGLE_CREDENTIALS_JSON or not SHEET_ID:
        logger.info("Google Sheets disabled (missing env)")
        sheets_service = None
        return

    try:
        creds_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        sheets_service = build("sheets", "v4", credentials=creds)
        logger.info("Google Sheets enabled")
    except Exception as e:
        logger.error(f"Google Sheets init failed: {e}")
        sheets_service = None


def append_ads_to_sheet(data: dict):
    if not sheets_service:
        return

    try:
        values = [[
            data.get("id", ""),
            data.get("product", ""),
            data.get("content", ""),
            data.get("image", ""),
            "To do",
        ]]

        sheets_service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range=SHEET_RANGE,
            valueInputOption="RAW",
            body={"values": values},
        ).execute()

    except Exception as e:
        logger.warning(f"Google Sheet append error: {e}")


# ======================================================
# RAG UTILITIES
# ======================================================
def row_to_chunk(row: pd.Series) -> Dict[str, str]:
    return {
        "id": str(row.get("Mã sản phẩm", "")).strip(),
        "title": str(row.get("Tên sản phẩm", "")).strip(),
        "text": f"""
Mã sản phẩm: {row.get('Mã sản phẩm','')}
Tên sản phẩm: {row.get('Tên sản phẩm','')}
Danh mục: {row.get('Danh mục','')}
Thương hiệu: {row.get('Thương hiệu','')}
Giá bán: {row.get('Giá bán hiện tại','')}
Giảm giá: {row.get('Giảm giá','')}
Đánh giá: {row.get('Điểm đánh giá','')} ({row.get('Số lượng đánh giá','')} đánh giá)
Tồn kho: {row.get('Số lượng hàng tồn kho','')}
""".strip()
    }


def embed_one(text: str) -> np.ndarray:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = genai.embed_content(model=EMBED_MODEL, content=text)
            time.sleep(SLEEP_PER_CALL)
            return np.array(r["embedding"], dtype="float32")
        except Exception:
            time.sleep(0.5 * attempt)
    raise RuntimeError("Embedding failed")


def find_exact_product(name: str):
    name = name.lower().strip()
    for doc in docs_meta:
        if name in doc["title"].lower():
            return doc
    return None


def retrieve_semantic(query: str, k=3):
    if not index:
        return []
    q_emb = embed_one(query).reshape(1, -1)
    D, I = index.search(q_emb, k)
    return [docs_meta[i] for i in I[0] if i >= 0]


# ======================================================
# RAG PIPELINES
# ======================================================
def build_context(product_name: str, k: int):
    exact = find_exact_product(product_name)
    if exact:
        return exact["text"], "exact"

    retrieved = retrieve_semantic(product_name, k)
    if retrieved:
        return "\n\n".join(d["text"] for d in retrieved), "faiss"

    return "Không có dữ liệu sản phẩm.", "none"


def generate_facebook_ads(product_name: str):
    context, mode = build_context(product_name, DEFAULT_TOP_K)

    prompt = f"""
Bạn là copywriter Facebook Ads.

Hãy viết 1 bài quảng cáo Facebook cho sản phẩm sau:
{context}

Yêu cầu:
- Viết liền mạch, tự nhiên
- Có emoji vừa phải
- Không chia mục, không giải thích
"""

    try:
        return llm.generate_content(prompt).text.strip(), mode
    except Exception as e:
        return f"Lỗi AI: {e}", mode


def generate_marketing_plan(product_name: str):
    context, mode = build_context(product_name, DEFAULT_TOP_K)

    prompt = f"""
Bạn là chuyên gia marketing TMĐT.

Hãy lập kế hoạch marketing ngắn hạn (1–2 tháng) cho sản phẩm:
{context}
"""

    try:
        return llm.generate_content(prompt).text.strip(), mode
    except Exception as e:
        return f"Lỗi AI: {e}", mode


# ======================================================
# FASTAPI APP
# ======================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global docs_meta, index, llm

    logger.info("🚀 Starting AI Service...")

    # Load data
    df = pd.read_csv(DATA_CSV).fillna("")
    docs_meta = [row_to_chunk(r) for _, r in df.iterrows()]

    if os.path.exists(INDEX_FILE) and os.path.exists(DOC_FILE):
        with open(DOC_FILE, "rb") as f:
            docs_meta = pickle.load(f)
        index = faiss.read_index(INDEX_FILE)
        logger.info("FAISS loaded")
    else:
        logger.warning("FAISS not found – semantic search limited")

    llm = genai.GenerativeModel(LLM_MODEL)
    init_google_sheets()

    logger.info("✅ AI Service READY")
    yield
    logger.info("🛑 Shutting down")


app = FastAPI(title="BepSachViet AI Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# API MODELS
# ======================================================
class AdsRequest(BaseModel):
    product_name: str
    product_id: Optional[str] = ""
    image: Optional[str] = ""


# ======================================================
# API ROUTES
# ======================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "docs": len(docs_meta),
        "faiss": bool(index),
    }


@app.post("/api/facebook-ads/generate")
def api_generate_ads(req: AdsRequest):
    content, mode = generate_facebook_ads(req.product_name)

    append_ads_to_sheet({
        "id": req.product_id,
        "product": req.product_name,
        "content": content,
        "image": req.image,
    })

    return {
        "content": content,
        "rag_mode": mode,
    }


@app.post("/generate-marketing-plan")
def api_generate_plan(req: AdsRequest):
    content, mode = generate_marketing_plan(req.product_name)
    return {
        "answer": content,
        "rag_mode": mode,
    }
