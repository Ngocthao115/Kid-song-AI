import os, time, json, csv, re, glob, uuid, shutil
from datetime import datetime
from typing import Optional, List

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from openai import OpenAI

# ========================= 0) PAGE CONFIG & THEME TWEAKS =========================
st.set_page_config(page_title="Kids Song AI", page_icon="🎵", layout="wide")
st.set_option("client.showErrorDetails", False)

st.markdown(
    """
    <style>
      .main .block-container{max-width:1200px;padding-top:1.2rem;padding-bottom:3rem;}
      .card{background:rgba(255,255,255,0.95);border:1px solid rgba(107,124,255,0.18);
            box-shadow:0 8px 24px rgba(107,124,255,0.12);border-radius:18px;padding:18px;}
      .stButton>button,.stDownloadButton>button{border-radius:14px;font-weight:600;padding:.6rem 1rem}
      .stTabs [data-baseweb="tab-list"]{gap:.4rem}
      .stTabs [data-baseweb="tab"]{background:#eef1ff;border-radius:12px;padding:10px 14px;font-weight:600}
      .stTabs [aria-selected="true"]{background:#6b7cff;color:white}
      .stTextInput>div>div>input,.stTextArea textarea,.stSelectbox>div>div{border-radius:12px!important}
      .badge{display:inline-block;padding:.18rem .5rem;border-radius:999px;font-size:.78rem;background:#eef1ff;color:#5860ff;font-weight:700}
      .subtle{color:#64748b}
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================= 1) CONFIG & ENV (Cloud + Local) =========================
load_dotenv()  # Local reads .env; Cloud will use st.secrets

def get_secret(name: str, default: Optional[str] = None):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

OPENAI_API_KEY    = get_secret("OPENAI_API_KEY")
SUNO_API_KEY      = get_secret("SUNO_API_KEY")
SUNO_API_BASE     = get_secret("SUNO_API_BASE", "https://api.sunoapi.org")
SUNO_MODEL        = get_secret("SUNO_MODEL", "V4_5")
SUNO_CALLBACK_URL = get_secret("SUNO_CALLBACK_URL", "https://webhook.site/your-id")
DEFAULT_SUNOSTYLE = get_secret(
    "DEFAULT_SUNOSTYLE",
    "Kids, cheerful, playful, educational",
)

if not OPENAI_API_KEY:
    st.error("Thiếu OPENAI_API_KEY – vào Manage app → Settings → Secrets để thêm.")
    st.stop()
if not SUNO_API_KEY:
    st.warning("Chưa có SUNO_API_KEY – có thể dùng phần tạo lời; tạo nhạc sẽ báo lỗi nếu thiếu.")

# Make available for SDKs reading env
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

HEADERS = {"Authorization": f"Bearer {SUNO_API_KEY}", "Content-Type": "application/json"}

# ========================= 2) PATHS & STORAGE =========================
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
COVERS_DIR = os.path.join(OUTPUT_DIR, "covers")
MP3_DIR    = os.path.join(OUTPUT_DIR, "mp3")
HISTORY_CSV = os.path.join(OUTPUT_DIR, "tracks.csv")
for d in (OUTPUT_DIR, COVERS_DIR, MP3_DIR):
    os.makedirs(d, exist_ok=True)

# History CSV schema (id,title,lyrics,style,lang,verses,bridge,instrumental,created_at,cover_path,mp3_path,cover_url,audio_url)
if not os.path.exists(HISTORY_CSV):
    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id","title","lyrics","style","lang","verses","bridge","instrumental",
            "created_at","cover_path","mp3_path","cover_url","audio_url"
        ])

# ========================= 3) HELPERS =========================

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text)
    return text[:60] or ("song-" + uuid.uuid4().hex[:8])

@st.cache_data(show_spinner=False)
def _download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def download_to_path(url: str, path: str) -> str:
    try:
        data = _download_bytes(url)
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception as e:
        st.warning(f"Không tải được: {e}")
        return ""

def append_history(row: List[str]):
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# ========================= 4) OPENAI – LYRICS =========================

def build_lyrics_prompt(title: str, desc: str, keywords: str, lang: str, verses: int, bridge: bool, style: str) -> List[dict]:
    sys = (
        "Bạn là nhà thơ – nhạc sĩ chuyên nghiệp viết lời bài hát thiếu nhi tiếng Việt cho giáo dục mầm non."
        "Hãy sáng tác lời bài hát hoặc dùng nội dung bài thơ, câu chuyện thiếu nhi thành lời bài hát, phù hợp với lứa tuổi 3 - 6, tươi vui, tích cực hoặc tình cảm, biết ơn."
        "Mỗi câu từ 5 - 10 từ ngữ, vần điệu rõ ràng, từ ngữ đơn giản, có điệp khúc dễ nhớ."
    )
    user = f"""
    Viết lời bài hát cho thiếu nhi.
    • Ngôn ngữ: {lang}
    • Tựa đề: {title or '(tự đặt nếu trống)'}
    • Mô tả: {desc}
    • Từ khóa chính (phân tách dấu phẩy): {keywords}
    • Số verse: {verses}
    • Có bridge: {"có" if bridge else "không"}
    • Phong cách: {style}
    • Yêu cầu: Có điệp khúc; mỗi câu 5–10 từ; tích cực, giáo dục; phù hợp mầm non.
    Trả về phần lời theo định dạng:
    [Verse 1]\n...
    [Chorus]\n...
    [Verse 2]\n...
    {"[Bridge]\n..." if bridge else ''}
    """
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]

@st.chat_input  # optional: enable quick feedback channel

def _ignore_chat(_=None):
    return None

def openai_generate_lyrics(title: str, desc: str, keywords: str, lang: str, verses: int, bridge: bool, style: str) -> str:
    msgs = build_lyrics_prompt(title, desc, keywords, lang, verses, bridge, style)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        temperature=0.8,
    )
    return resp.choices[0].message.content.strip()

def openai_refine_lyrics(lyrics: str, hint: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"Bạn là biên tập viên lời bài hát thiếu nhi; tinh chỉnh theo chỉ dẫn mà vẫn giữ cấu trúc verse/chorus."},
            {"role":"user","content": f"Chỉ dẫn refine: {hint}\n\nLời hiện tại:\n{lyrics}"}
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

# ========================= 5) SUNO – MUSIC (V4_5) =========================

SUNO_CREATE_PATH = "/api/generate"           # may vary per provider
SUNO_STATUS_PATH = "/api/generate/{}"        # format with job_id


def suno_create_music(lyrics: str, title: str, style: str, instrumental: bool = False) -> dict:
    """Kick off music generation. Returns dict with job_id or direct urls if provider supports sync."""
    payload = {
        "model": SUNO_MODEL,
        "prompt": lyrics if not instrumental else (lyrics + "\n[Instrumental only]"),
        "title": title or "Kids Song",
        "style": style,
        "callback_url": SUNO_CALLBACK_URL,
    }
    try:
        r = requests.post(SUNO_API_BASE + SUNO_CREATE_PATH, headers=HEADERS, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        data = r.json()
        return data  # expected: {"job_id": "..."} or {"audio_url": "...", "image_url": "..."}
    except Exception as e:
        st.error(f"Suno create lỗi: {e}")
        return {}


def suno_poll_result(job_id: str, timeout_s: int = 180) -> dict:
    """Poll status endpoint until done or timeout. Expected to return {status, audio_url, image_url}."""
    start = time.time()
    last_err = None
    while time.time() - start < timeout_s:
        try:
            r = requests.get(SUNO_API_BASE + SUNO_STATUS_PATH.format(job_id), headers=HEADERS, timeout=60)
            if r.status_code == 404:
                last_err = "404 – chưa có job hoặc sai endpoint"; time.sleep(2); continue
            r.raise_for_status()
            data = r.json()
            status = str(data.get("status", "")).lower()
            if status in {"succeeded", "completed", "done"} or (data.get("audio_url") and data.get("image_url")):
                return data
            if status in {"failed", "error"}:
                return {"error": data}
        except Exception as e:
            last_err = e
        time.sleep(2)
    return {"error": f"Timeout khi đợi Suno. Chi tiết: {last_err}"}

# ========================= 6) HERO HEADER =========================

st.markdown(
    """
<div class="card" style="padding:20px; margin-bottom:14px;">
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="font-size:2rem">🎶</div>
    <div>
      <div style="font-size:1.9rem;font-weight:800;background: linear-gradient(90deg,#2b2f77,#6b7cff);-webkit-background-clip:text;background-clip:text;color:transparent;">Kids Song AI</div>
      <div class="subtle">OpenAI Lyrics · Suno Music · Dành cho giáo viên mầm non</div>
    </div>
  </div>
  <div style="margin-top:10px">
    <span class="badge">{model}</span>
    <span class="badge">Kids • cheerful • gentle</span>
  </div>
</div>
    """.format(model=SUNO_MODEL),
    unsafe_allow_html=True,
)

# ========================= 7) TABS =========================

tab1, tab2, tab3, tab4 = st.tabs(["✨ Tạo bài hát", "📚 Thư viện", "🗂️ Lịch sử", "⚙️ Cài đặt"]) 

# ------------------------- TAB 1 – CREATE -------------------------
with tab1:
    left, right = st.columns([0.46, 0.54], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧪 Nhập thông tin")
        title = st.text_input("Tiêu đề bài hát", placeholder="Chú ong chăm chỉ")
        desc  = st.text_area("Miêu tả / Bối cảnh / Title", height=90, placeholder="Vui tươi, dễ nhớ, giáo dục, có điệp khúc…")
        keywords = st.text_input("Từ khóa chính (phân tách dấu phẩy)", placeholder="chăm học, chăm làm, ai cũng quý")
        colx, coly = st.columns([1,1])
        with colx:
            verses = st.number_input("Số verse", 1, 6, 2, step=1)
        with coly:
            bridge = st.toggle("Thêm Bridge", value=False)
        lang = st.selectbox("Ngôn ngữ", ["Vi", "En"], index=0)
        style = st.selectbox("Phong cách nhạc", [
            DEFAULT_SUNOSTYLE,
            "Kids, gentle lullaby, warm, simple instruments",
            "Kids, cheerful, upbeat, clapping, ukulele",
            "Kids, playful, march, percussion",
            "Kids, cute pop, claps, ukulele",
            "Kids, upbeat, bright, classroom sing-along",
            "Kids lullaby, gentle, calm, emotional, soft piano and strings, warm female vocal",
            "Instrumental lullaby, gentle, soothing, soft piano + strings, warm and calm",
            "Kids, bright clear child voice (boy), boy soprano, youthful, innocent tone, light airy timbre"
            "Childlike male vocal, tender and gentle, warm, no heavy drums, clean mix",
            "Boy soprano, soft pop for kids, cheerful and sweet, ukulele + glockenspiel",
            "Young boy vocal, cute and playful, bright and clear, soft dynamics, classroom sing-along"
    
        ], index=0)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("✍️ TIẾN TRÌNH SÁNG TÁC")
        c1, c2 = st.columns([1,1])
        with c1:
            gen_lyrics_btn = st.button("✨ Tạo lời bài hát", use_container_width=True)
        with c2:
            refine_hint = st.text_input("Chỉ dẫn refine (tuỳ chọn)", placeholder="VD: tăng tính lặp ở điệp khúc, câu 5–8 từ…")
            refine_btn = st.button("🪄 Refine", use_container_width=True)
        lyrics = st.text_area("Soạn thảo/Chỉnh sửa trước khi tạo nhạc:", height=220, key="lyrics_box")
        instrumental = st.toggle("🎼 Chỉ giai điệu (instrumental)", value=False)
        make_music_btn = st.button("🎧 Tạo nhạc", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Actions ---
    if gen_lyrics_btn:
        try:
            with st.spinner("Đang tạo lời…"):
                lyrics_new = openai_generate_lyrics(title, desc, keywords, lang, int(verses), bool(bridge), style)
            st.session_state["lyrics_box"] = lyrics_new
            st.success("Đã tạo lời bài hát ✨")
        except Exception as e:
            st.error(f"Lỗi tạo lời: {e}")

    if refine_btn and st.session_state.get("lyrics_box"):
        try:
            with st.spinner("Đang refine…"):
                refined = openai_refine_lyrics(st.session_state["lyrics_box"], refine_hint or "Giữ ngắn gọn, dễ hát, rõ Chorus")
            st.session_state["lyrics_box"] = refined
            st.success("Đã refine lời ✨")
        except Exception as e:
            st.error(f"Lỗi refine: {e}")

    if make_music_btn:
        if not SUNO_API_KEY:
            st.error("Chưa cấu hình SUNO_API_KEY trong Secrets → không thể tạo nhạc.")
        elif not st.session_state.get("lyrics_box"):
            st.warning("Chưa có lời bài hát để tạo nhạc.")
        else:
            lyr = st.session_state["lyrics_box"]
            with st.spinner("Gửi yêu cầu tạo nhạc tới Suno…"):
                create_res = suno_create_music(lyr, title, style, instrumental)
            # Accept both sync & async providers
            audio_url = create_res.get("audio_url")
            image_url = create_res.get("image_url")
            job_id    = create_res.get("job_id") or create_res.get("id")

            if (not audio_url) and job_id:
                with st.spinner("Đang chờ Suno xử lý…"):
                    done = suno_poll_result(job_id)
                if "error" in done:
                    st.error(f"Tạo nhạc thất bại: {done['error']}")
                else:
                    audio_url = done.get("audio_url") or done.get("audio")
                    image_url = done.get("image_url") or done.get("image")

            if audio_url:
                # Save locally
                base = slugify(title or ("kids-song-" + uuid.uuid4().hex[:6]))
                cover_path = os.path.join(COVERS_DIR, base + ".png")
                mp3_path   = os.path.join(MP3_DIR,    base + ".mp3")
                if image_url:
                    download_to_path(image_url, cover_path)
                download_to_path(audio_url, mp3_path)

                # UI
                st.success("Đã tạo nhạc 🎧")
                if os.path.exists(cover_path):
                    st.image(cover_path, width=280)
                audio_bytes = open(mp3_path, "rb").read() if os.path.exists(mp3_path) else _download_bytes(audio_url)
                st.audio(audio_bytes, format="audio/mp3")
                col_d1, col_d2 = st.columns([1,1])
                with col_d1:
                    st.download_button("⬇️ Tải MP3", data=audio_bytes, file_name=f"{base}.mp3")
                if os.path.exists(cover_path):
                    with open(cover_path, "rb") as f:
                        st.download_button("⬇️ Tải ảnh bìa", data=f.read(), file_name=f"{base}.png")

                # Save history
                append_history([
                    uuid.uuid4().hex,
                    title,
                    lyr,
                    style,
                    lang,
                    verses,
                    bridge,
                    instrumental,
                    datetime.now().isoformat(timespec="seconds"),
                    cover_path if os.path.exists(cover_path) else "",
                    mp3_path if os.path.exists(mp3_path) else "",
                    image_url or "",
                    audio_url or "",
                ])
            else:
                st.error("Không nhận được audio_url từ Suno – kiểm tra lại endpoint hoặc credit.")

# ------------------------- TAB 2 – LIBRARY -------------------------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📚 Thư viện (gallery)")
    cover_paths = sorted(glob.glob(os.path.join(COVERS_DIR, "*.png"))) + \
                  sorted(glob.glob(os.path.join(COVERS_DIR, "*.jpg")))
    if not cover_paths:
        st.info("Chưa có bài nào. Hãy tạo bài hát ở tab đầu tiên nhé!")
    else:
        cols = st.columns(3, gap="large")
        for i, cover in enumerate(cover_paths):
            base = os.path.splitext(os.path.basename(cover))[0]
            mp3_candidate = os.path.join(MP3_DIR, base + ".mp3")
            with cols[i % 3]:
                st.image(cover, use_column_width=True)
                st.caption(f"**{base}**")
                btn_cols = st.columns([1,1])
                if os.path.exists(mp3_candidate):
                    with open(mp3_candidate, "rb") as f:
                        btn_cols[0].download_button("⬇️ MP3", f, file_name=f"{base}.mp3", use_container_width=True)
                with open(cover, "rb") as f:
                    btn_cols[1].download_button("⬇️ Ảnh bìa", f, file_name=f"{base}.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------- TAB 3 – HISTORY -------------------------
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🗂️ Lịch sử")
    try:
        df = pd.read_csv(HISTORY_CSV)
        st.dataframe(
            df[["created_at","title","style","lang","verses","bridge","instrumental","audio_url"]].sort_values("created_at", ascending=False),
            use_container_width=True,
        )
    except Exception as e:
        st.info("Chưa có lịch sử hoặc file rỗng.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------- TAB 4 – SETTINGS -------------------------
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚙️ Cài đặt nhanh")
    st.write("• App dùng OpenAI để tạo **lời bài hát** và Suno **V4_5** để tạo **nhạc**.")
    st.write("• Lưu ý: kho `outputs/` trên Cloud là **tạm thời** – khi redeploy có thể mất.")
    st.caption(f"OPENAI key length: {len(OPENAI_API_KEY) if OPENAI_API_KEY else 0} · SUNO: {'OK' if SUNO_API_KEY else 'missing'}")

    st.markdown("""
    **Mẹo sử dụng**
    - Điền *Miêu tả* và *Từ khóa* thật rõ → lời hát chất lượng hơn.
    - Dùng **Refine** để điều chỉnh: nhấn mạnh điệp khúc, giữ mỗi câu 5–8 từ, thêm vần, v.v.
    - Bật *Instrumental* nếu chỉ muốn giai điệu không lời.
    """)
    st.markdown('</div>', unsafe_allow_html=True)






