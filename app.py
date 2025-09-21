import os, time, json, requests, datetime as dt, csv, re, io, uuid
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ============ 1) CONFIG & ENV (CLOUD + LOCAL) ===========
load_dotenv()  # Local: đọc .env; Cloud: sẽ ưu tiên st.secrets

def get_secret(name, default=None):
    # Ưu tiên secrets trên Streamlit Cloud; nếu không có thì lấy từ biến môi trường (.env)
    try:
        import streamlit as st
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

OPENAI_API_KEY    = get_secret("OPENAI_API_KEY")
SUNO_API_KEY      = get_secret("SUNO_API_KEY")
SUNO_API_BASE     = get_secret("SUNO_API_BASE", "https://api.sunoapi.org")
SUNO_MODEL        = get_secret("SUNO_MODEL", "V4_5")
SUNO_CALLBACK_URL = get_secret("SUNO_CALLBACK_URL")
DEFAULT_SUNOSTYLE = get_secret("DEFAULT_SUNOSTYLE", "Kids, cheerful, playful, educational")

# --- Supabase (mới): dùng để KHÔNG MẤT thư viện & lịch sử ---
SUPABASE_URL      = get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = get_secret("SUPABASE_ANON_KEY")  # dùng anon key là đủ cho đọc/ghi nếu bucket public và có policy phù hợp
SUPABASE_BUCKET   = get_secret("SUPABASE_BUCKET", "kids-songs")
SUPABASE_TABLE    = get_secret("SUPABASE_TABLE", "tracks")

# Client OpenAI (SDK >= 1.40)
if not OPENAI_API_KEY:
    st.error("Thiếu OPENAI_API_KEY — hãy vào ‘⋯ → Settings → Secrets’ để thêm.")
    st.stop()
if not SUNO_API_KEY:
    st.error("Thiếu SUNO_API_KEY — thêm trong Secrets.")
    st.stop()
if not SUNO_CALLBACK_URL:
    st.warning("Chưa có SUNO_CALLBACK_URL — tạm dùng webhook.site để demo.")
    # Không stop vì app đang poll; có thể vẫn chạy

# Cho SDK/requests đọc từ ENV nếu cần
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
HEADERS = {"Authorization": f"Bearer {SUNO_API_KEY}", "Content-Type": "application/json"}

# Kết nối Supabase (nếu cung cấp URL & KEY)
supabase = None
supabase_status = "❌"
if SUPABASE_URL and SUPABASE_ANON_KEY:
    try:
        from supabase import create_client, Client  # pip install supabase
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        supabase_status = "✅"
    except Exception as e:
        st.warning(f"Không khởi tạo được Supabase client: {e}")
        supabase = None

# Thư mục xuất (vẫn giữ lưu local làm cache/phòng hờ)
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/covers", exist_ok=True)
os.makedirs("outputs/mp3", exist_ok=True)
HISTORY_CSV = os.path.join("outputs", "tracks.csv")

# ----------- Schema CSV (giữ nguyên) -----------
EXPECTED_HEADER = [
    "time","title","topic","track_index","audio_url","image_url",
    "style","model","mp3_path","cover_path"
]

# ============ 2) PROMPT HỆ THỐNG ============
DEFAULT_LYRICS_SYSTEM = (
    "Bạn là một nhà thơ và nhạc sĩ viết nhạc thiếu nhi chuyên nghiệp cho giáo dục mầm non. "
    "Hãy sáng tác lời bài hát hoặc dùng câu chuyện, bài thơ thiếu nhi để sáng tác lời bài hát, phù hợp lứa tuổi 3 – 6, tươi vui, tích cực, tình cảm, yêu thương. "
    "Mỗi câu 5–10 từ, vần điệu rõ, từ vựng đơn giản. Có điệp khúc dễ nhớ."
)

# ============ 3) HÀM NGHIỆP VỤ ============
def build_user_prompt(
    topic: str,
    language: str = "vi",
    target_words: Optional[List[str]] = None,
    verses: int = 2,
    include_bridge: bool = True,
    min_lines: int = 12,
    max_lines: int = 18,
) -> str:
    tw = ", ".join(target_words) if target_words else "Không bắt buộc"
    structure = ["- Cấu trúc: [Verse 1] → [Chorus]"]
    for i in range(2, verses + 1):
        structure.append(f"→ [Verse {i}] → [Chorus]")
    if include_bridge:
        structure.append("→ [Bridge] (ngắn 2–4 dòng) → [Chorus] (kết)")
    return (
        f"Chủ đề: {topic}\n"
        f"Ngôn ngữ: {language}\n"
        "Yêu cầu:\n"
        "- Ngôn ngữ đơn giản, an toàn cho trẻ 3–6 tuổi; tích cực, hồn nhiên.\n"
        "- Vần điệu rõ, nhịp vui tươi hoặc tình cảm nhẹ nhàng, câu ngắn.\n"
        "- Nội dung giáo dục nhẹ nhàng; khuyến khích hành vi tốt, hoặc tỏ lòng yêu thương và biết ơn.\n"
        f"{' '.join(structure)}.\n"
        f"- Từ ngữ chính (nếu lồng được): {tw}\n"
        f"- Độ dài ~{min_lines}–{max_lines} dòng.\n"
        "- Định dạng đầu ra có nhãn [Verse]/[Chorus]/[Bridge].\n"
    )

def generate_lyrics(topic: str, target_words: Optional[List[str]] = None, language: str = "vi",
                    verses: int = 2, bridge: bool = True) -> str:
    user_prompt = build_user_prompt(topic, language=language, target_words=target_words,
                                    verses=verses, include_bridge=bridge)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": DEFAULT_LYRICS_SYSTEM},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.9,
        max_tokens=700,
    )
    return resp.choices[0].message.content.strip()

def refine_lyrics(original_text: str, instruction: str = "") -> str:
    if not original_text.strip():
        return original_text
    user_msg = (
        "Hãy chỉnh sửa/đánh bóng lời bài hát thiếu nhi bên dưới, giữ nguyên chủ đề và tinh thần cho trẻ 3–6 tuổi. "
        "Tăng vần điệu, nhịp mượt, chia đoạn rõ [Verse]/[Chorus]/[Bridge]. "
        "Áp dụng nhẹ nhàng chỉ dẫn nếu có, không kéo quá dài.\n\n"
        f"Chỉ dẫn: {instruction or 'Không có'}\n\n"
        "Văn bản cần chỉnh:\n" + original_text
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": DEFAULT_LYRICS_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.6,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()

def suno_generate_song(prompt: str, title: str, style: str, instrumental: bool = False) -> str:
    endpoint = f"{SUNO_API_BASE}/api/v1/generate"
    payload = {
        "prompt": prompt[:1800],
        "title": title[:64],
        "style": style[:200],
        "model": SUNO_MODEL,
        "instrumental": instrumental,
        "customMode": True,
        "callBackUrl": SUNO_CALLBACK_URL,
    }
    r = requests.post(endpoint, headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 200 or not data.get("data", {}).get("taskId"):
        raise RuntimeError("Suno generate failed: " + json.dumps(data, ensure_ascii=False))
    return data["data"]["taskId"]

def suno_poll(task_id: str, timeout_sec: int = 360, interval_sec: int = 8):
    endpoint = f"{SUNO_API_BASE}/api/v1/generate/record-info"
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        r = requests.get(
            endpoint,
            headers={"Authorization": f"Bearer {SUNO_API_KEY}"},
            params={"taskId": task_id},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        try:
            items = data["data"]["response"]["sunoData"]
            ready = [it for it in items if it.get("audioUrl") or it.get("audioUrlHigh")]
            if ready:
                return ready
        except Exception:
            pass
        time.sleep(interval_sec)
    raise TimeoutError("Hết thời gian chờ trả kết quả")

def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def sanitize_filename(name: str) -> str:
    name = re.sub(r"\s+", "_", name.strip())
    return re.sub(r"[^\w\-\.]", "", name)

# ---------- CSV helpers: migrate & load ----------
def ensure_history_schema():
    """Đảm bảo tracks.csv có header EXPECTED_HEADER. Nếu file cũ (9 cột), tự migrate sang 10 cột."""
    if not os.path.exists(HISTORY_CSV):
        # tạo file mới với header chuẩn
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(EXPECTED_HEADER)
        return

    # Đọc header hiện tại an toàn
    with open(HISTORY_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if header == EXPECTED_HEADER:
        return  # đúng rồi

    # Migrate: đọc tất cả cũ -> ghi file mới cùng tên với header mới
    rows_old = []
    with open(HISTORY_CSV, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows_old.append(row)

    tmp_path = HISTORY_CSV + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPECTED_HEADER, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for old in rows_old:
            newrow = {k: old.get(k, "") for k in EXPECTED_HEADER}
            # Một số file cũ có 'track_index' là float => ép về int nếu cần
            if newrow.get("track_index"):
                try:
                    newrow["track_index"] = int(float(newrow["track_index"]))
                except Exception:
                    pass
            w.writerow(newrow)

    os.replace(tmp_path, HISTORY_CSV)  # atomically replace

# --- Supabase helpers (mới) ---
def sb_upload_bytes(bucket: str, path: str, data_bytes: bytes, content_type: str) -> Optional[str]:
    """Upload bytes lên Supabase Storage và trả về public URL (nếu cấu hình bucket public).
    Lưu ý: supabase-py v2 kỳ vọng **bytes** hoặc **đường dẫn file**.
    """
    if not supabase:
        return None
    try:
        # Truyền thẳng bytes cho client (không dùng BytesIO)
        supabase.storage.from_(bucket).upload(
            path,
            data_bytes,
            {"contentType": content_type, "upsert": "true"}
        )
        pub = supabase.storage.from_(bucket).get_public_url(path)
        if isinstance(pub, dict) and "publicUrl" in pub:
            return pub["publicUrl"]
        return str(pub)
    except Exception as e:
        st.warning(f"Upload Supabase thất bại ({path}): {e}")
        return None

def supabase_upsert_track(row: dict) -> None:
    if not supabase:
        return
    try:
        # Thử upsert theo schema mở rộng (CSV cũ)
        supabase.table(SUPABASE_TABLE).upsert(row, on_conflict="time,track_index").execute()
        return
    except Exception as e1:
        # Fallback: schema tối giản như ảnh em gửi (id, title, style, lyrics_url, audio_url, cover_url, created_at, uploader)
        try:
            simple_row = {
                "id": str(uuid.uuid4()),
                "title": row.get("title", ""),
                "style": row.get("style", ""),
                "lyrics_url": row.get("lyrics_url", ""),
                "audio_url": row.get("audio_url", ""),
                "cover_url": row.get("image_url", ""),
                "created_at": dt.datetime.utcnow().isoformat(),
                "uploader": "kids-song-ai",
            }
            supabase.table(SUPABASE_TABLE).insert(simple_row).execute()
            st.info("Đã chèn bản ghi theo schema đơn giản (id/title/style/lyrics_url/audio_url/cover_url/created_at/uploader).")
        except Exception as e2:
            st.warning("Ghi bản ghi Supabase thất bại (cả 2 schema): " + str(e1) + " | " + str(e2) + "Hãy kiểm tra lại cột bảng hoặc đổi SUPABASE_TABLE cho khớp.")


def write_history_row(row: dict) -> None:
    # Ghi CSV local (giữ nguyên)
    ensure_history_schema()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPECTED_HEADER, quoting=csv.QUOTE_MINIMAL)
        w.writerow({k: row.get(k, "") for k in EXPECTED_HEADER})
    # Ghi Supabase (mới)
    supabase_upsert_track(row)


def load_history_df_local():
    """Đọc CSV về DataFrame, có fallback để không vỡ UI nếu dữ liệu lẫn cột."""
    import pandas as pd
    ensure_history_schema()
    try:
        return pd.read_csv(HISTORY_CSV, dtype=str, keep_default_na=False)
    except Exception:
        try:
            # fallback engine python, bỏ dòng xấu
            return pd.read_csv(HISTORY_CSV, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
        except Exception:
            # đọc thủ công -> DataFrame
            with open(HISTORY_CSV, "r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                rows = [row for row in r]
            if not rows:
                import pandas as pd
                return pd.DataFrame(columns=EXPECTED_HEADER)
            import pandas as pd
            # Bảo đảm đủ cột
            for r_ in rows:
                for k in EXPECTED_HEADER:
                    r_.setdefault(k, "")
            return pd.DataFrame(rows, columns=EXPECTED_HEADER)


def load_history_df_supabase():
    """Ưu tiên đọc lịch sử."""
    if not supabase:
        return None
    try:
        res = supabase.table(SUPABASE_TABLE).select("*").execute()
        rows = res.data or []
        import pandas as pd
        for r in rows:
            for k in EXPECTED_HEADER:
                r.setdefault(k, "")
        df = pd.DataFrame(rows, columns=EXPECTED_HEADER)
        return df
    except Exception as e:
        st.warning(f"Không tải được lịch sử từ Supabase: {e}")
        return None

# ============ 4) UI / THEME ============
st.set_page_config(page_title="Kids Song AI", page_icon="🎵", layout="centered")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600&family=Inter:wght@400;600;700&display=swap');
:root { --radius: 16px; }
h1,h2,h3 { font-family: 'Fredoka', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
body, p, div, span { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 980px; }
.card { background:#fff; border-radius:var(--radius); padding:1rem 1.25rem;
        box-shadow:0 10px 18px rgba(15,23,42,.06); border:1px solid rgba(15,23,42,.06); margin-bottom:1rem; }
.badge { display:inline-flex; align-items:center; gap:.4rem; padding:.35rem .7rem; border-radius:999px; background:#ECFEFF; color:#0E7490;
         font-size:.78rem; font-weight:700; letter-spacing:.2px; }
.stButton>button { border-radius:12px; padding:.6rem 1rem; font-weight:700; }
.stButton>button[kind="secondary"] { background:#F8FAFC; border:1px solid #E2E8F0; }
.toolbar { display:flex; gap:.5rem; flex-wrap:wrap; }
.grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
.card-sm { border:1px solid #E2E8F0; border-radius:14px; padding:10px; box-shadow:0 4px 10px rgba(15,23,42,.05); }
.card-sm h4 { margin:.2rem 0 .4rem; font-size:1rem; }
audio { width:100% !important; }
footer { color:#94a3b8; font-size:.85rem; }
.status { font-size:.85rem; color:#0f172a; background:#F1F5F9; border:1px solid #E2E8F0; padding:.25rem .5rem; border-radius:8px; }
</style>
""",
    unsafe_allow_html=True,
)

# ============ 5) STATE ============
if "lyrics" not in st.session_state: st.session_state.lyrics = ""
if "title" not in st.session_state: st.session_state.title = ""
if "topic" not in st.session_state: st.session_state.topic = ""
if "targets" not in st.session_state: st.session_state.targets = []
if "generated" not in st.session_state: st.session_state.generated = False

# — Sidebar
with st.sidebar:
    st.markdown("## 👩‍🏫 Hướng dẫn nhanh")
    st.markdown(
        "- **Bước 1:** Nhập Miêu tả/Từ khóa/Title → **Tạo lời**.\n"
        "- **Bước 2:** Chỉnh tay hoặc **Refine**.\n"
        "- **Bước 3:** **Tạo nhạc**, xem ảnh bìa & tải file.\n"
        "- Xem lại ở **📚 Thư viện** hoặc **🗂️ Lịch sử**."
    )
    st.divider()
    st.caption(f"Model Suno: **{SUNO_MODEL}**")
    st.caption(f"Style mặc định: **{DEFAULT_SUNOSTYLE}**")
    st.caption(f"Supabase: <span class='status'>{supabase_status}</span>", unsafe_allow_html=True)

# — Header
st.title("🎵 Kids Song AI")
st.markdown('<span class="badge">OpenAI Lyrics • Suno Music • Supabase Persist</span>', unsafe_allow_html=True)

# — Tabs
tab_make, tab_library, tab_history, tab_settings = st.tabs(["✨ Tạo bài hát", "📚 Thư viện", "🗂️ Lịch sử", "⚙️ Cài đặt"])

# ============ TAB 1: Tạo bài hát ============
with tab_make:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        topic = st.text_input("Miêu tả bài hát", st.session_state.topic or "Trường mầm non của bé")
        target_str = st.text_input("Từ ngữ gợi ý (phân tách bởi dấu phẩy)", "Đồ chơi, sân trường, lớp học, thân thương")
        title = st.text_input("Tiêu đề bài hát", st.session_state.title or "Trường mầm non của bé")
    with col2:
        verses = st.number_input("Số verse", 1, 4, 2)
        bridge = st.toggle("Thêm Bridge", value=True)
        language = st.selectbox("Ngôn ngữ", ["Vi", "En"], index=0)
        style = st.selectbox(
            "Phong cách nhạc",
            [
                DEFAULT_SUNOSTYLE,
                "Kids, gentle, soothing, lullaby, warm",
                "Kids, cheerful, playful, educational",
                "Kids, cute pop, claps, ukulele",
                "Kids, upbeat, bright, classroom sing-along",
                "Kids lullaby, gentle, calm, emotional, soft piano and strings, warm female vocal",
                "Instrumental lullaby, gentle, soothing, soft piano + strings, warm and calm",
                "Kids, bright clear child voice (boy), boy soprano, youthful, innocent tone, light airy timbre",
                "Young boy vocal, cute and playful, bright and clear, soft dynamics, classroom sing-along",
                "Childlike male vocal, tender and gentle, warm, no heavy drums, clean mix",
                "Boy soprano, soft pop for kids, cheerful and sweet, ukulele + glockenspiel",
                "Lullaby for kids, young boy singer, gentle, soothing, soft piano and strings",
                "Gentle emotional kids ballad, young boy vocal, warm and tender, minimal percussion",
            ],
            index=0,
            help="Chọn style nhạc"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Thanh thao tác
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 TIẾN TRÌNH SÁNG TÁC")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        btn_generate = st.button("✨ Tạo lời bài hát", use_container_width=True)
    with c2:
        refine_hint = st.text_input("Chỉ dẫn refine (tuỳ chọn)", placeholder="Ví dụ: Nhập yêu cầu")
    with c3:
        btn_refine = st.button("🪄 Refine", use_container_width=True,
                               disabled=not bool(st.session_state.lyrics.strip()))

    if btn_generate:
        try:
            targets = [w.strip() for w in target_str.split(",") if w.strip()]
            with st.spinner("Đang sáng tác lời..."):
                lyrics = generate_lyrics(topic, targets, language=language, verses=verses, bridge=bridge)
            st.session_state.lyrics = lyrics
            st.session_state.title = title
            st.session_state.topic = topic
            st.session_state.targets = targets
            st.session_state.generated = True
            st.success("Đã sinh lời. Chỉnh sửa trực tiếp hoặc bấm Refine.")
        except Exception as e:
            st.error(str(e))

    if btn_refine and st.session_state.lyrics.strip():
        try:
            with st.spinner("Đang chỉnh sửa lời..."):
                st.session_state.lyrics = refine_lyrics(st.session_state.lyrics, refine_hint)
            st.success("Đã refine lời bài hát.")
        except Exception as e:
            st.error(str(e))

    # Ô soạn thảo lời (luôn hiển thị)
    st.session_state.lyrics = st.text_area(
        "Soạn thảo/Chỉnh sửa tại đây trước khi tạo nhạc:",
        value=st.session_state.lyrics, height=320
    )

    # Nút sinh nhạc Suno
    st.divider()
    left, right = st.columns([1, 2])
    with left:
        instrumental = st.toggle("Chỉ giai điệu (instrumental)", value=False)
    with right:
        btn_music = st.button("🎧 Tạo nhạc", use_container_width=True,
                              disabled=not bool(st.session_state.lyrics.strip()))

    # ========== KẾT QUẢ + ẢNH BÌA GẮN TRONG TRANG ==========
    if btn_music and st.session_state.lyrics.strip():
        try:
            with st.spinner("Đang tạo bài hát..."):
                task_id = suno_generate_song(
                    st.session_state.lyrics,
                    st.session_state.title or "Kids Song",
                    style=style,
                    instrumental=instrumental
                )
                tracks = suno_poll(task_id)

            st.subheader("🎧 Kết quả")
            ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            base = sanitize_filename(st.session_state.title or "Kids_Song")

            for i, t in enumerate(tracks, 1):
                audio_url_orig = t.get("audioUrlHigh") or t.get("audioUrl")
                image_url_orig = t.get("imageUrl")
                mp3_path = ""
                cover_path = ""

                # Lưu file mp3 & cover vào outputs/
                audio_bytes = b""
                img_bytes = b""
                if audio_url_orig:
                    audio_bytes = download_bytes(audio_url_orig)
                    mp3_path = f"outputs/mp3/{ts}_{i}_{base}.mp3"
                    with open(mp3_path, "wb") as f:
                        f.write(audio_bytes)

                if image_url_orig:
                    img_bytes = download_bytes(image_url_orig)
                    cover_path = f"outputs/covers/{ts}_{i}_{base}.jpg"
                    with open(cover_path, "wb") as f:
                        f.write(img_bytes)

                # --- NEW: Upload lên Supabase Storage (nếu có) ---
                audio_url_pub = None
                image_url_pub = None
                lyrics_url_pub = None
                if st.session_state.get("lyrics"):
                    try:
                        lyrics_url_pub = sb_upload_bytes(SUPABASE_BUCKET, f"lyrics/{ts}_{i}_{base}.txt", st.session_state.lyrics.encode("utf-8"), "text/plain")
                    except Exception:
                        pass
                if audio_bytes:
                    audio_url_pub = sb_upload_bytes(SUPABASE_BUCKET, f"mp3/{ts}_{i}_{base}.mp3", audio_bytes, "audio/mpeg")
                if img_bytes:
                    image_url_pub = sb_upload_bytes(SUPABASE_BUCKET, f"covers/{ts}_{i}_{base}.jpg", img_bytes, "image/jpeg")

                # Ưu tiên sử dụng URL trên Supabase để không mất dữ liệu
                audio_url_final = audio_url_pub or audio_url_orig or ""
                image_url_final = image_url_pub or image_url_orig or ""

                # Hiển thị card kết quả: ảnh bìa + player + nút tải
                k1, k2 = st.columns([1, 2])
                with k1:
                    if cover_path and os.path.exists(cover_path):
                        st.image(cover_path, caption="Ảnh bìa", use_column_width=True)
                    elif image_url_final:
                        st.image(image_url_final, caption="Ảnh bìa", use_column_width=True)
                with k2:
                    st.write(f"**{st.session_state.title or 'Kids Song'} — Bản {i}**")
                    if mp3_path and os.path.exists(mp3_path):
                        with open(mp3_path, "rb") as f:
                            st.audio(f.read(), format="audio/mp3")
                        with open(mp3_path, "rb") as f:
                            st.download_button("⬇️ Tải MP3", data=f, file_name=os.path.basename(mp3_path),
                                               mime="audio/mpeg", use_container_width=True,
                                               key=f"dl_now_{ts}_{i}")
                    elif audio_url_final:
                        st.audio(audio_url_final, format="audio/mp3")

                # Lưu lịch sử (CSV + Supabase table)
                row = {
                    "time": ts,
                    "title": st.session_state.title or "Kids Song",
                    "topic": st.session_state.topic or "",
                    "track_index": i,
                    "audio_url": audio_url_final,
                    "image_url": image_url_final,
                    "style": style,
                    "model": SUNO_MODEL,
                    "mp3_path": mp3_path,
                    "cover_path": cover_path,
                    "lyrics_url": lyrics_url_pub or "",
                }
                write_history_row(row)

            st.balloons()
            st.info("Đã lưu vào Supabase và thư mục local. Xem ở tab 📚 Thư viện.")
        except Exception as e:
            st.error(str(e))

    st.markdown('</div>', unsafe_allow_html=True)

# ============ TAB 2: THƯ VIỆN (GALLERY) ============
with tab_library:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📚 Thư viện (Gallery)")
    data_source = "local"

    # Ưu tiên lấy từ Supabase; nếu không có thì lấy từ CSV local
    df = load_history_df_supabase()
    if df is not None and len(df) > 0:
        data_source = "supabase"
    if (df is None) or (len(df) == 0):
        if os.path.exists(HISTORY_CSV):
            try:
                import pandas as pd
                df = load_history_df_local()
            except Exception as e:
                st.error("Không đọc được thư viện: " + str(e))
                df = None
        else:
            df = None

    if df is None or len(df) == 0:
        st.info("Chưa có dữ liệu. Hãy tạo bài hát ở tab ✨ trước nhé.")
    else:
        try:
            # Mới nhất trước
            if "time" in df.columns and "track_index" in df.columns:
                df = df.sort_values(by=["time","track_index"], ascending=[False, True]).reset_index(drop=True)

            # Bộ lọc nhanh
            colf1, colf2 = st.columns([2,1])
            with colf1:
                q = st.text_input("Tìm theo tiêu đề/chủ đề", "")
            with colf2:
                style_vals = sorted([s for s in df.get("style", []).dropna().unique().tolist()]) if "style" in df.columns else []
                style_pick = st.selectbox("Lọc theo style", ["Tất cả"] + style_vals)
            if q and "title" in df.columns and "topic" in df.columns:
                mask = df["title"].str.contains(q, case=False, na=False) | df["topic"].str.contains(q, case=False, na=False)
                df = df[mask]
            if style_pick and style_pick != "Tất cả" and "style" in df.columns:
                df = df[df["style"] == style_pick]

            # Hiển thị nguồn dữ liệu
            st.caption(f"Nguồn dữ liệu: **{'Supabase' if data_source=='supabase' else 'Local CSV'}**")

            # Grid gallery
            if len(df) == 0:
                st.info("Chưa có bài nào khớp bộ lọc.")
            else:
                cols = st.columns(4)
                for idx, row in df.iterrows():
                    col = cols[idx % 4]
                    with col:
                        st.markdown('<div class="card-sm">', unsafe_allow_html=True)
                        cover = (row.get("cover_path") or "").strip()
                        image_url = (row.get("image_url") or "").strip()
                        title = row.get("title") or "Kids Song"
                        subtitle = f"{row.get('time','')} • Bản {int(float(row.get('track_index', 1)))}" if row.get('track_index') else f"{row.get('time','')}"

                        # ảnh bìa
                        if cover and os.path.exists(cover):
                            st.image(cover, use_column_width=True)
                        elif image_url:
                            st.image(image_url, use_column_width=True)
                        else:
                            st.image("https://picsum.photos/seed/kidsmusic/600/400", use_column_width=True)

                        st.markdown(f"<h4>{title}</h4><div style='color:#64748b'>{subtitle}</div>", unsafe_allow_html=True)

                        # audio
                        mp3_path = (row.get("mp3_path") or "").strip()
                        audio_url = (row.get("audio_url") or "").strip()
                        if mp3_path and os.path.exists(mp3_path):
                            with open(mp3_path, "rb") as f:
                                st.audio(f.read(), format="audio/mp3")
                            with open(mp3_path, "rb") as f:
                                st.download_button("⬇️ Tải MP3", data=f, file_name=os.path.basename(mp3_path),
                                                   mime="audio/mpeg", use_container_width=True,
                                                   key=f"dl_lib_{row.get('time','')}_{int(float(row.get('track_index', idx%4+1)))}_{idx}")
                        elif audio_url:
                            st.audio(audio_url, format="audio/mp3")

                        st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error("Không đọc được thư viện: " + str(e))

    st.markdown('</div>', unsafe_allow_html=True)

# ============ TAB 3: LỊCH SỬ ============
with tab_history:
    # Ưu tiên lấy từ Supabase; nếu không có thì lấy CSV
    df_hist = load_history_df_supabase()
    if (df_hist is None or len(df_hist) == 0) and os.path.exists(HISTORY_CSV):
        df_hist = load_history_df_local()

    if df_hist is not None and len(df_hist) > 0:
        try:
            import pandas as pd
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🗂️ Lịch sử tạo nhạc (bảng)")
            st.dataframe(df_hist, use_container_width=True, height=360)
            st.download_button(
                "⬇️ Tải CSV lịch sử",
                df_hist.to_csv(index=False).encode("utf-8"),
                file_name="tracks_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error("Không đọc được bài trước: " + str(e))
    else:
        st.info("Chưa có bài hát nào. Tạo bài hát ở tab ✨ trước nhé.")

# ============ TAB 4: CÀI ĐẶT ============
with tab_settings:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✅ Kiểm tra Supabase")
    colsb1, colsb2 = st.columns(2)
    with colsb1:
        if supabase:
            try:
                # Đếm số bản ghi trong bảng tracks
                res = supabase.table(SUPABASE_TABLE).select("count()", count='exact').execute()
                total_rows = (res.count or 0)
            except Exception:
                total_rows = None
            st.metric(label="Số bản ghi trong bảng", value= total_rows if total_rows is not None else "—")
        else:
            st.info("Chưa cấu hình Supabase URL/KEY")
    with colsb2:
        if supabase:
            try:
                # Liệt kê vài file mới trong bucket mp3/
                files = supabase.storage.from_(SUPABASE_BUCKET).list("mp3", {"limit":5, "offset":0, "sortBy":{"column":"created_at","order":"desc"}})
                file_names = [f.get('name') if isinstance(f, dict) else getattr(f, 'name', '') for f in (files or [])]
                if file_names:
                    st.write("**5 tệp MP3 mới nhất (bucket):**")
                    for nm in file_names:
                        st.write("- ", nm)
                else:
                    st.write("Chưa có tệp trong mp3/ hoặc không lấy được danh sách.")
            except Exception as e:
                st.warning(f"Không thể liệt kê bucket: {e}")
        else:
            st.empty()

    st.divider()
    st.markdown("### 🎨 Preset chủ đề nhanh")
    preset = st.selectbox(
        "Chọn nhanh",
        [
            "Màu sắc cơ bản","Hình tròn – vuông – tam giác","Số đếm 1 – 10","Vệ sinh răng miệng",
            "Chào hỏi & phép lịch sự","An toàn giao thông","Con vật","Gia đình","Nghề nghiệp",
            "Trường mầm non","Bản thân bé","Thầy cô và bạn bè"
        ],
    )
    st.caption("Chọn preset rồi copy sang tab ✨.")

    st.divider()
    st.markdown("### ℹ️ Ghi chú")
    st.markdown(
        "- **Refine** chỉ chỉnh lời hiện tại, không đổi chủ đề.\n"
        "- **Instrumental** yêu cầu Suno tạo giai điệu không lời.\n"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ===========  FOOTER ===========
st.markdown("""
<hr style="margin:24px 0; border:none; border-top:1px solid #e6e8f5;">
<div style="text-align:center; margin-top:8px; line-height:1.7;">
  <div style="font-weight:800; font-size:18px;">
    © Kids Song AI • OpenAI Lyrics + Suno Music – Dành cho Giáo viên mầm non
  </div>
  <div style="font-size:15px; color:#64748b;">
    Ngọc Thảo – <a href="mailto:ms.nthaotran@gmail.com">ms.nthaotran@gmail.com</a>
  </div>
</div>
""", unsafe_allow_html=True)















