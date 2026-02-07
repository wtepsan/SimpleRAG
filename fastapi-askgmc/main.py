import os
import re
import json
import asyncio
import base64
import hmac
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI
import textwrap

import tools.Rag_retrived as rag

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gmc-assistant")

# =========================
# Env & OpenAI client
# =========================
load_dotenv("settings/.env") or load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

PORT = int(os.getenv("PORT", "7778"))
SESSION_TIMEOUT_MIN = int(os.getenv("SESSION_TIMEOUT_MIN", "2"))
ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",") if o.strip()]

# =========================
# RAG env
# =========================
RAG_OUTPUT_DIR = os.getenv("RAG_OUTPUT_DIR", "./RAG/RAG_database")
RAG_PREFIX = os.getenv("RAG_PREFIX", "knowledge")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 10))
RAG_METHOD = os.getenv("RAG_METHOD", "equal")  # equal | weighted | rrf

RAG_TOP_K_DENSE = int(os.getenv("RAG_TOP_K_DENSE", "50"))
RAG_TOP_K_SPARSE = int(os.getenv("RAG_TOP_K_SPARSE", "200"))
RAG_ALPHA = float(os.getenv("RAG_ALPHA", "0.6"))
RAG_RRF_K = int(os.getenv("RAG_RRF_K", "60"))

# =========================
# LINE OA env (DO NOT hardcode)
# =========================
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Set it in .env or environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# App & CORS
# =========================
app = FastAPI(title="GMC Assistant API (RAG)", version="2.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Sessions (in-memory) + persistence
# =========================
conversation_histories: Dict[str, List[Dict[str, str]]] = {}
conversation_timestamps: Dict[str, datetime] = {}

SAVE_DIR = os.getenv("CONVERSATION_SAVE_DIR", "conversations_history")
os.makedirs(SAVE_DIR, exist_ok=True)

# 0 means unlimited (but system message is always retained)
MAX_IN_MEMORY_TURNS = int(os.getenv("MAX_IN_MEMORY_TURNS", "0"))

# =========================
# Static blocks (data & templates)
# =========================
GMC_MAP_URL = "https://maps.app.goo.gl/mL1SnTCBW6MEakq49"
GMC_FACEBOOK = "https://www.facebook.com/gmc.cmu/"
GMC_WEBSITE = "https://excellent.med.cmu.ac.th/website/en/gmc"

APPOINTMENT_PHONES = textwrap.dedent("""\
• ตรวจสิทธิ/สอบถามสิทธิรักษา: 053-934710
• เวชศาสตร์ฟื้นฟู / InBody: 053-920609 (08:00–16:00)
• ห้องยา/สอบถามมียาหรือไม่: 053-934725, 053-934729
• ทันตกรรม (อาคารศูนย์เวชศาสตร์ผู้สูงอายุ, ต้องนัดล่วงหน้า): 053-920638 (08:00–20:00)
• ศูนย์การแพทย์เพื่อการมีบุตร (IUI/IVF): 053-934714
• แพ็กเกจตรวจสุขภาพประจำปี: ขอรับลิงก์ Facebook ทางการของศูนย์ได้จากเจ้าหน้าที่
""")

STATIC_COSTS = textwrap.dedent("""\
• สิทธิกรมบัญชีกลาง (เบิกตรง): เบิกได้ตามสิทธิ ยกเว้นค่าบริการหน่วย 250 บาท/ครั้ง และค่าแพทย์ ~100–500 บาท/ครั้ง (ชำระเอง)
• ประกันสุขภาพเอกชนแบบ OPD: ศูนย์ไม่ได้ทำสัญญาตรง ต้องสำรองจ่ายแล้วนำเอกสารไปเคลมเอง
• วัคซีนไข้หวัดใหญ่: 500 บาท/เข็ม (ไม่รวมค่าบริการและค่าแพทย์)
• ส่องกล้องคัดกรองมะเร็งลำไส้ (เงินสด): ประมาณ 16,000–18,000 บาท (ราคาประเมิน)
""")

STATIC_GMC_INFO = textwrap.dedent(f"""\
# ข้อมูลศูนย์เวชศาสตร์ผู้สูงอายุ (Geriatric Medical Center – GMC)
• สังกัด: ศูนย์ความเป็นเลิศทางการแพทย์ (Center for Medical Excellence), คณะแพทยศาสตร์ มหาวิทยาลัยเชียงใหม่
• อาคารบริการสุขภาพแบบครบวงจร ขนาด 7 ชั้น ให้บริการผู้ป่วยนอกและผู้ป่วยใน โดยแพทย์ผู้เชี่ยวชาญ
• จุดเด่น: เป็นมิตรกับผู้สูงวัย เข้าถึงสะดวก ได้มาตรฐานระดับสากล

# เวลาทำการ
• จันทร์–ศุกร์: 08:00–20:00
• เสาร์–อาทิตย์: 08:00–16:00
(หมายเหตุ: มีแหล่งข้อมูลที่ระบุ 07:00–20:00 ทุกวัน แนะนำให้ยืนยันกับเจ้าหน้าที่ก่อน)

# ช่องทางการติดต่อหลัก
• โทร (ศูนย์ความเป็นเลิศทางการแพทย์): 053-934710
• โทร (GMC โดยตรง – กรุณานัดหมายล่วงหน้า): 053-920666
• อีเมล: cmex.medcmu@gmail.com
• Facebook: Geriatric Medical Center {GMC_FACEBOOK}
• LINE Official: @mca4022m
• เว็บไซต์: {GMC_WEBSITE}

# ที่ตั้ง
• เลขที่ 110 ถนนอินทวโรรส ซอย 2 ตำบลสุเทพ อำเภอเมืองเชียงใหม่ จังหวัดเชียงใหม่ 50200
• แผนที่ (Google Maps): [เปิดแผนที่คลิกที่นี่]({GMC_MAP_URL})
""")

STATIC_GMC_SERVICES = textwrap.dedent("""\
# บริการในอาคารศูนย์เวชศาสตร์ผู้สูงอายุ (GMC)

## 🔹 ชั้น 1: คลินิกอายุรกรรมทั่วไปและเฉพาะทาง
• โรคความดันโลหิตสูง
• โรคไขมันในเลือดสูง
• โรคเบาหวาน
• โรคไทรอยด์
• โรคหัวใจ
• โรคไต
• โรคผิวหนัง
• วิตเธอร์ (เวชศาสตร์ฟื้นฟู)
• โรคระบบประสาทและสมอง
• โรคระบบทางเดินอาหาร
• โรคทางเดินปัสสาวะ
• โรคกระดูกและข้อ
• โรคอายุรกรรมผู้สูงอายุ
• โรคติดเชื้อ

## 🔹 ชั้น 2: คลินิกเฉพาะทางและบริการตรวจสุขภาพ
• คลินิกโรคเฉพาะทาง ได้แก่ หู ตา จมูก
• คลินิกหัวใจผู้สูงวัย
• คลินิกทันตกรรมผู้สูงอายุ
• ห้องเวชศาสตร์ฟื้นฟู
• การตรวจวิเคราะห์ข้อมูลสุขภาพ และการตรวจร่างกายเชิงป้องกัน

## 🔹 ชั้น 3: หน่วยบริการเสริมและกิจกรรมผู้สูงอายุ
• หน่วยกายภาพบำบัด
• ห้องอาหาร Healthy Tasty
• ห้องกิจกรรม
• ห้องออกกำลังกายสำหรับผู้สูงอายุ
""")

# =========================
# Prompt (build once)
# =========================
def gmc_safety_suffix() -> str:
    return textwrap.dedent("""
    หมายเหตุด้านความปลอดภัย: ข้อมูลนี้เป็นคำแนะนำทั่วไป ไม่ใช่การวินิจฉัยทางการแพทย์
    หากมีอาการรุนแรง เช่น เจ็บหน้าอกเฉียบพลัน หายใจลำบาก ซึม/สับสนมาก หรือแขนขาอ่อนแรงครึ่งซีก
    โปรดไปห้องฉุกเฉินใกล้บ้านหรือโทร 1669 ทันที
    """).strip()

def build_system_prompt() -> str:
    return textwrap.dedent(f"""
    คุณคือ **น้องจีจี้ (Gee Jee)**
    คุณใช้คำว่าหนูเป็นสรรพนามแทนตัวเอง
    เป็นผู้ช่วยหญิง สุภาพ อบอุ่น ของศูนย์เวชศาสตร์ผู้สูงอายุ (GMC)
    โหมดการทำงาน: **RAG (ตอบจากคลังความรู้ที่ให้มา)**

    ขอบเขตที่อนุญาต:
    • เวลา/ขั้นตอนทั่วไป/สิทธิการรักษา(ภาพรวม)/ค่าบริการตัวอย่าง/ช่องทางติดต่อ/เบอร์โทร
    • ใช้ข้อมูลคงที่ต่อไปนี้เท่านั้น หากคำถามเกินขอบเขตให้ปฏิเสธอย่างสุภาพและชี้ช่องทางติดต่อ
    • ข้อมูลทั่วไปเกี่ยวกับบริการของศูนย์เวชศาสตร์ผู้สูงอายุ (GMC)

    ขอบเขตที่ไม่อนุญาต:
    • ห้ามรับ **นัดหมาย/เลื่อนนัด/จองคิว** ในแชท
    • ห้ามวินิจฉัยอาการ สั่งยา ตีความผลตรวจ

    นโยบายการนัดหมาย:
    • หากผู้ใช้ต้องการนัดหมาย/เลื่อนนัด/ตรวจสอบคิว ให้ตอบสั้น ๆ ว่า:
      "หากต้องการนัดหมาย/เลื่อนนัด กรุณาโทรติดต่อเจ้าหน้าที่ตามเบอร์ด้านล่างนะคะ"
    • แล้วแสดงเบอร์โทรที่เกี่ยวข้อง

    ------------------------------
    # STATIC DATA — ค่าบริการ/สิทธิ
    {STATIC_COSTS.strip()}

    # เบอร์โทรติดต่อ (สำหรับการนัด/เลื่อนนัด/สอบถาม)
    {APPOINTMENT_PHONES.strip()}

    # ข้อมูลศูนย์ (ที่ตั้ง/เวลา/ช่องทาง + แผนที่)
    {STATIC_GMC_INFO.strip()}

    # บริการของศูนย์ (แต่ละชั้น)
    {STATIC_GMC_SERVICES.strip()}

    ------------------------------
    วิธีตอบ (สั้น กระชับ):
    1) ทักทาย
    2) ตอบเฉพาะข้อมูล static ที่ตรงคำถาม (เป็นข้อ ๆ)
    3) ถ้าเป็นเรื่องนัด/เลื่อนนัด ให้ขึ้นประโยคโทรติดต่อ + ใส่เบอร์
    4) ถ้าเกินขอบเขต ให้ปฏิเสธอย่างสุภาพ + ชี้ช่องทางติดต่อ + ใส่เบอร์

    ความปลอดภัย:
    • ข้อมูลนี้เป็นคำแนะนำทั่วไป ไม่ใช่การวินิจฉัยทางการแพทย์
    • หากมีอาการฉุกเฉิน ให้ไปห้องฉุกเฉินใกล้บ้านหรือโทร 1669 ทันที
    """).strip()

SYSTEM_PROMPT = build_system_prompt()

# =========================
# Guardrails (server-side)
# =========================
APPOINTMENT_REGEX = re.compile(
    r"(นัดหมาย|นัด|เลื่อนนัด|จองคิว|walk[\s-]*in|คิว|ตารางแพทย์|ตรวจวันนี้ได้ไหม|เปลี่ยนวัน|เลื่อนวัน)",
    flags=re.IGNORECASE
)
MEDICAL_ADVICE_REGEX = re.compile(
    r"(อาการ|ป่วย|เจ็บ|ปวด|ไข้|ผื่น|ติดเชื้อ|วินิจฉัย|สั่งยา|ยาอะไร|ผลตรวจ|ค่าเลือด|x-?ray|เอกซเรย์|mri|ct)",
    flags=re.IGNORECASE
)

APPOINTMENT_REPLY = textwrap.dedent(f"""\
หากต้องการนัดหมาย/เลื่อนนัด/ตรวจสอบคิว กรุณาโทรติดต่อเจ้าหน้าที่ตามเบอร์ด้านล่างนะคะ

{APPOINTMENT_PHONES.strip()}
""").strip()

MEDICAL_REPLY = textwrap.dedent(f"""\
ขออภัยค่ะ น้องจีจี้ไม่สามารถประเมินอาการ/วินิจฉัย หรือแนะนำยาได้
เพื่อความปลอดภัย แนะนำให้ติดต่อเจ้าหน้าที่หรือพบแพทย์นะคะ

{gmc_safety_suffix()}
""").strip()

# =========================
# Models
# =========================
class QueryRequest(BaseModel):
    session_id: str
    query: str
    user_info: Optional[Dict[str, Any]] = None  # optional; ignored except logging

class QueryResponse(BaseModel):
    response: str
    type: Literal["text", "html", "markdown"] = "text"

# =========================
# Persistence helpers
# =========================
def _safe_session_id_for_filename(session_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", session_id)[:64]
    return safe or "unknown"

def _jsonl_path(session_id: str) -> str:
    return os.path.join(SAVE_DIR, f"{_safe_session_id_for_filename(session_id)}.jsonl")

def _snapshot_txt_path(session_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(SAVE_DIR, f"{ts}_{_safe_session_id_for_filename(session_id)}.txt")

def _format_history_as_text(session_id: str, history: List[Dict[str, str]]) -> str:
    lines = [
        f"Session: {session_id}",
        f"Saved at: {datetime.now().isoformat(timespec='seconds')}",
        "-" * 60
    ]
    for msg in history:
        role = msg.get("role", "?")
        content = (msg.get("content") or "").strip()
        lines.append(f"{role.upper()}:")
        lines.append(content)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

def _atomic_write(path: str, content: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)

def _persist_append_jsonl(session_id: str, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "content": content,
    }
    if meta:
        payload["meta"] = meta
    with open(_jsonl_path(session_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def _snapshot_session_to_txt(session_id: str, history: List[Dict[str, str]]) -> None:
    try:
        _atomic_write(_snapshot_txt_path(session_id), _format_history_as_text(session_id, history))
    except Exception as e:
        logger.warning("Failed to snapshot session %s: %s", session_id, e)

# =========================
# Session mgmt
# =========================
def _ensure_session(session_id: str) -> None:
    if session_id not in conversation_histories:
        conversation_histories[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        _persist_append_jsonl(session_id, "system", SYSTEM_PROMPT, meta={"event": "session_start"})
    conversation_timestamps[session_id] = datetime.now()

def _append_history(session_id: str, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
    _ensure_session(session_id)

    history = conversation_histories[session_id]
    history.append({"role": role, "content": content})

    # keep system always, then keep last N (if configured)
    if MAX_IN_MEMORY_TURNS > 0:
        system_msg = history[0]
        tail = history[1:][-MAX_IN_MEMORY_TURNS:]
        history = [system_msg] + tail
        conversation_histories[session_id] = history

    conversation_timestamps[session_id] = datetime.now()
    _persist_append_jsonl(session_id, role, content, meta=meta)

def _prune_expired_sessions() -> None:
    now = datetime.now()
    expired = [
        sid for sid, ts in list(conversation_timestamps.items())
        if now - ts > timedelta(minutes=SESSION_TIMEOUT_MIN)
    ]
    for sid in expired:
        hist = conversation_histories.get(sid, [])
        if hist:
            _snapshot_session_to_txt(sid, hist)
        conversation_histories.pop(sid, None)
        conversation_timestamps.pop(sid, None)

async def _prune_loop() -> None:
    while True:
        try:
            _prune_expired_sessions()
        except Exception as e:
            logger.warning("prune loop error: %s", e)
        await asyncio.sleep(60)

@app.on_event("startup")
async def _on_startup():
    asyncio.create_task(_prune_loop())

@app.on_event("shutdown")
async def _on_shutdown():
    try:
        for sid, hist in list(conversation_histories.items()):
            if hist:
                _snapshot_session_to_txt(sid, hist)
    except Exception as e:
        logger.warning("shutdown snapshot error: %s", e)

# =========================
# LLM call
# =========================
def call_llm(messages: List[Dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_tokens=800,
    )
    return (resp.choices[0].message.content or "").strip()

# =========================
# RAG helpers
# =========================
def _retrieve_context(user_query: str) -> Dict[str, Any]:
    try:
        items = rag.hybrid_search(
            query=user_query,
            output_dir=RAG_OUTPUT_DIR,
            prefix=RAG_PREFIX,
            embed_model_name=EMBED_MODEL_NAME,
            top_k=RAG_TOP_K,
            top_k_dense=RAG_TOP_K_DENSE,
            top_k_sparse=RAG_TOP_K_SPARSE,
            alpha=RAG_ALPHA,
            method=RAG_METHOD,
            rrf_k=RAG_RRF_K,
        )
    except Exception as e:
        logger.warning("RAG retrieve failed: %s", e)
        items = []

    blocks: List[str] = []
    for idx, it in enumerate(items, start=1):
        chunk = (it.get("chunk") or "").strip()
        if not chunk:
            continue

        title = (it.get("topic_title") or it.get("title") or "").strip()
        tag = f"{it.get('retrieval', 'rag')}"
        score = it.get("score", None)

        header = f"[{idx}] {title}".strip() if title else f"[{idx}]"
        header += f"  ({tag}, score={score:.4f})" if isinstance(score, (int, float)) else f"  ({tag})"
        blocks.append(header + "\n" + chunk)

    # print(blocks)
    return {"items": items, "context_text": "\n\n".join(blocks).strip()}

def _build_rag_messages(session_id: str, user_query: str, context_text: str) -> List[Dict[str, str]]:
    # base: system prompt once
    _ensure_session(session_id)
    history = conversation_histories[session_id]

    messages: List[Dict[str, str]] = [history[0]]  # system
    messages.extend(history[1:])  # prior turns

    # injected retrieved context (system-like)
    if context_text:
        messages.append({"role": "system", "content": "CONTEXT (จากคลังความรู้):\n" + context_text})
    else:
        messages.append({"role": "system", "content": "CONTEXT: (ไม่พบข้อมูลที่ตรงคำถามในคลังความรู้)"})

    messages.append({"role": "user", "content": user_query})
    return messages

def _geejee_answer(session_id: str, user_query: str) -> str:
    q = (user_query or "").strip()
    if not q:
        return "ขออภัยค่ะ หนูไม่เห็นคำถาม รบกวนพิมพ์ใหม่อีกครั้งนะคะ"

    # Guardrails first
    if APPOINTMENT_REGEX.search(q):
        _append_history(session_id, "user", q)
        _append_history(session_id, "assistant", APPOINTMENT_REPLY)
        return APPOINTMENT_REPLY

    if MEDICAL_ADVICE_REGEX.search(q):
        _append_history(session_id, "user", q)
        _append_history(session_id, "assistant", MEDICAL_REPLY)
        return MEDICAL_REPLY

    # Retrieve
    rag_pack = _retrieve_context(q)
    context_text = rag_pack["context_text"]

    # LLM
    _append_history(session_id, "user", q, meta={"rag_method": RAG_METHOD})
    messages = _build_rag_messages(session_id, q, context_text)
    answer = call_llm(messages)

    # If no RAG context, add contact suggestion
    if not context_text:
        answer = (answer + "\n\n"
                  "ถ้าต้องการความชัดเจนเพิ่มเติม แนะนำให้โทรสอบถามเจ้าหน้าที่นะคะ\n\n"
                  + APPOINTMENT_PHONES.strip()).strip()

    _append_history(
        session_id,
        "assistant",
        answer,
        meta={
            "rag_used": True,
            "rag_k": RAG_TOP_K,
            "rag_method": RAG_METHOD,
            "rag_hits": len(rag_pack["items"]),
        },
    )
    return answer

# =========================
# LINE helpers
# =========================
def _verify_line_signature(raw_body: bytes, x_line_signature: str) -> bool:
    if not LINE_CHANNEL_SECRET or not x_line_signature:
        return False
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), raw_body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, x_line_signature)

def _line_session_id(event: Dict[str, Any]) -> str:
    src = (event.get("source") or {})
    return src.get("userId") or src.get("groupId") or src.get("roomId") or "line_unknown"

async def _line_reply(reply_token: str, text: str) -> None:
    if not LINE_CHANNEL_ACCESS_TOKEN:
        logger.warning("LINE_CHANNEL_ACCESS_TOKEN missing; cannot reply.")
        return

    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text[:2000]}],
    }
    async with httpx.AsyncClient(timeout=10) as client_http:
        r = await client_http.post(LINE_REPLY_URL, headers=headers, json=payload)
        if r.status_code >= 400:
            logger.warning("LINE reply failed %s: %s", r.status_code, r.text)

# =========================
# Routes
# =========================
@app.get("/healthcheck")
def healthcheck():
    _prune_expired_sessions()
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "mode": "rag",
        "timeout_min": SESSION_TIMEOUT_MIN,
        "save_dir": SAVE_DIR,
        "rag": {
            "output_dir": RAG_OUTPUT_DIR,
            "prefix": RAG_PREFIX,
            "embed_model": EMBED_MODEL_NAME,
            "top_k": RAG_TOP_K,
            "method": RAG_METHOD,
        },
        "line_webhook_ready": True,
    }

@app.post("/query", response_model=QueryResponse)
def query_agent(req: QueryRequest):
    _prune_expired_sessions()

    session_id = (req.session_id or "").strip()
    user_query = (req.query or "").strip()
    if not session_id or not user_query:
        return QueryResponse(response="⚠️ session_id หรือ query ว่างเปล่า", type="text")

    answer = _geejee_answer(session_id, user_query)
    return QueryResponse(response=answer, type="text")

# LINE Webhook
@app.get("/line/webhook")
def line_webhook_get():
    return PlainTextResponse("OK", status_code=200)

@app.post("/line/webhook")
async def line_webhook_post(
    request: Request,
    x_line_signature: Optional[str] = Header(default=None, convert_underscores=False),
):
    raw_body = await request.body()

    if not x_line_signature or not _verify_line_signature(raw_body, x_line_signature):
        return PlainTextResponse("OK", status_code=200)

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return PlainTextResponse("OK", status_code=200)

    events = payload.get("events") or []
    for ev in events:
        if ev.get("type") != "message":
            continue
        msg = ev.get("message") or {}
        if msg.get("type") != "text":
            continue

        reply_token = ev.get("replyToken")
        if not reply_token:
            continue

        text_in = msg.get("text", "")
        session_id = _line_session_id(ev)
        answer = _geejee_answer(session_id, text_in)
        await _line_reply(reply_token, answer)

    return PlainTextResponse("OK", status_code=200)

# =========================
# Dev server
# =========================
if __name__ == "__main__":
    import uvicorn
    # If this file is main.py, use "main:app"
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
