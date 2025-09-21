import os, time, json, requests, datetime as dt, csv, re, uuid
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ================== 1) CONFIG & ENV ==================
load_dotenv()  # Local: đọc .env; Cloud: ưu tiên st.secrets

def get_secret(name, default=None):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

OPENAI_API_KEY    = get_secret("OPENAI_API_KEY")
SUNO_API_KEY      = get_secret("SUNO_API_KEY")
SUNO_API_BASE     = get_secret("SUNO_API_BASE", "https://api.sunoapi.org")
SUNO_MODEL        = get_secret("SUNO_MODEL", "V4_5")
SUNO_CALLBACK_URL = get_secret("SUNO_CALLBACK_URL")
DEFAULT_SUNOSTYLE = get_secret("DEFAULT_SUNOSTYLE", "Kids, cheerful, playful, educational")

# Supabase
SUPABASE_URL    = get_secret("SUPABASE_URL")
SUPABASE_KEY    = get_secret("SUPABASE_KEY") 
SUPABASE_BUCKET = get_secret("SUPABASE_BUCKET", "Kids_songs")
SUPABASE_TABLE  = get_secret("SUPABASE_TABLE", "tracks")

if not OPENAI_API_KEY:
    st.error("Thiếu OPENAI_API_KEY — thêm trong Secrets.")
    st.stop()
if not SUNO_API_KEY:
    st.error("Thiếu SUNO_API_KEY — thêm trong Secrets.")
    st.stop()
if not SUNO_CALLBACK_URL:
    st.warning("Chưa có SUNO_CALLBACK_URL — có thể bỏ qua (app sẽ poll).")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)
HEADERS = {"Authorization": f"Bearer {SUNO_API_KEY}", "Content-Type": "application/json"}

# Kết nối Supabase
supabase = None
supabase_status = "❌"
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        supabase_status = "✅"
    except Exception as e:
        st.warning(f"Không khởi tạo được Supabase client: {e}")

# Local output (cache/phòng hờ)
os.makedirs("outputs/mp3", exist_ok=True)
os.makedirs("outputs/covers", exist_ok=True)
HISTORY_CSV = os.path.join("outputs", "tracks.csv")
EXPECTED_HEADER = [
    "time","title","topic","track_index","audio_url","image_url",
    "style","model","mp3_path","cover_path"
]

# ================== 2) PROMPT HỆ THỐNG ==================
DEFAULT_LYRICS_SYSTEM = (
    "Bạn là một nhà thơ và nhạc sĩ viết nhạc thiếu nhi chuyên nghiệp cho giáo dục mầm non. "
    "Hãy sáng tác lời bài hát hoặc dùng câu chuyện, bài thơ thiếu nhi để sáng tác lời bài hát, phù hợp lứa tuổi 3–6, tươi vui, tích cực, tình cảm, yêu thương. "
    "Mỗi câu 5–10 từ, vần điệu rõ, từ vựng đơn giản. Có điệp khúc dễ nhớ."
)

# ================== 3) HÀM NGHIỆP VỤ ==================
def build_user_prompt(topic: str, language: str = "vi", target_words: Optional[List[str]] = None,
                      verses: int = 2, include_bridge: bool = True,
                      min_lines: int = 12, max_lines: int = 18) -> str:
    tw = ", ".join(target_words) if target_words else "Không bắt buộc"
    structure = ["- Cấu trúc: [Verse 1] → [Chorus]"]
    for i in range(2, verses + 1):
        structure.append(f"→ [Verse {i}] → [Chorus]")
    if include_bridge:
        structure.append("→ [Bridge] (ngắn 2–4 dòng) → [Chorus] (kết)")
    return (
        f"Chủ đề: {topic}\nNgôn ngữ: {language}\nYêu cầu:\n"
        "- Ngôn ngữ đơn giản, an toàn cho trẻ 3–6 tuổi; tích cực, hồn nhiên.\n"
        "- Vần điệu rõ, nhịp vui tươi hoặc tình cảm nhẹ nhàng, câu ngắn.\n"
        f"{' '.join(structure)}.\n- Từ ngữ chính (nếu lồng được): {tw}\n- Độ dài ~{min_lines}–{max_lines} dòng.\n"
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
        f"Chỉ dẫn: {instruction or 'Không có'}\n\nVăn bản cần chỉnh:\n" + original_text
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
        r = requests.get(endpoint, headers={"Authorization": f"Bearer {SUNO_API_KEY}"}, params={"taskId": task_id}, timeout=60)
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

# Tên file ASCII an toàn (tránh InvalidKey)
def ascii_slugify(text: str) -> str:
    import unicodedata
    text = (text or "").strip().replace(" ", "_")
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    text = text.strip("._-") or "file"
    return text[:80]

# ---------- CSV helpers ----------
def ensure_history_schema():
    if not os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(EXPECTED_HEADER)
        return
    with open(HISTORY_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f); header = next(reader, None)
    if header == EXPECTED_HEADER:
        return
    rows_old = []
    with open(HISTORY_CSV, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_old.append(row)
    tmp = HISTORY_CSV + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPECTED_HEADER); w.writeheader()
        for old in rows_old:
            newrow = {k: old.get(k, "") for k in EXPECTED_HEADER}
            if newrow.get("track_index"):
                try: newrow["track_index"] = int(float(newrow["track_index"]))
                except Exception: pass
            w.writerow(newrow)
    os.replace(tmp, HISTORY_CSV)

# ---------- Supabase helpers ----------
def sb_upload_bytes(bucket: str, path: str, data_bytes: bytes, content_type: str) -> Optional[str]:
    if not supabase:
        return None
    try:
        supabase.storage.from_(bucket).upload(path, data_bytes, {"contentType": content_type, "upsert": "true"})
        pub = supabase.storage.from_(bucket).get_public_url(path)
        return pub.get("publicUrl") if isinstance(pub, dict) else str(pub)
    except Exception as e:
        st.warning(f"Upload Supabase thất bại ({path}): {e}")
        return None

def supabase_upsert_track(row: dict) -> None:
    if not supabase:
        return
    try:
        supabase.table(SUPABASE_TABLE).upsert(row, on_conflict="time,track_index").execute()
    except Exception as e1:
        # Fallback cho bảng tối giản (id,title,style,lyrics_url,audio_url,cover_url,created_at,uploader)
        try:
            simple = {
                "id": str(uuid.uuid4()),
                "title": row.get("title", ""),
                "style": row.get("style", ""),
                "lyrics_url": row.get("lyrics_url", ""),
                "audio_url": row.get("audio_url", ""),
                "cover_url": row.get("image_url", ""),
                "created_at": dt.datetime.utcnow().isoformat(),
                "uploader": "kids-song-ai",
            }
            supabase.table(SUPABASE_TABLE).insert(simple).execute()
        except Exception as e2:
            st.warning("Ghi Supabase thất bại (cả 2 schema): " + str(e1) + " | " + str(e2))

def write_history_row(row: dict) -> None:
    ensure_history_schema()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPECTED_HEADER); w.writerow({k: row.get(k, "") for k in EXPECTED_HEADER})
    supabase_upsert_track(row)

def load_history_df_local():
    import pandas as pd
    ensure_history_schema()
    try:
        return pd.read_csv(HISTORY_CSV, dtype=str, keep_default_na=False)
    except Exception:
        return pd.DataFrame(columns=EXPECTED_HEADER)

def load_history_df_supabase():
    if not supabase:
        return None
    try:
        res = supabase.table(SUPABASE_TABLE).select("*").execute()
        rows = res.data or []
        import pandas as pd
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Không tải được lịch sử từ Supabase: {e}")
        return None

# ================== 4) UI / THEME ==================
st.set_page_config(page_title="Kids Song AI", page_icon="🎵", layout="centered")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600&family=Inter:wght@400;600;700&display=swap');
:root { --radius: 16px; }
h1,h2,h3 { font-family: 'Fredoka', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
body, p, div, span { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 980px; }
.card { background:#fff; border-radius:var(--radius); padding:1rem 1.25rem; box-shadow:0 10px 18px rgba(15,23,42,.06); border:1px solid rgba(15,23,42,.06); margin-bottom:1rem; }
.badge { display:inline-flex; align-items:center; gap:.4rem; padding:.35rem .7rem; border-radius:999px; background:#ECFEFF; color:#0E7490; font-size:.78rem; font-weight:700; letter-spacing:.2px; }
.stButton>button { border-radius:12px; padding:.6rem 1rem; font-weight:700; }
.card-sm { border:1px solid #E2E8F0; border-radius:14px; padding:10px; box-shadow:0 4px 10px rgba(15,23,42,.05); }
.status { font-size:.85rem; color:#0f172a; background:#F1F5F9; border:1px solid #E2E8F0; padding:.25rem .5rem; border-radius:8px; }
</style>
""",
    unsafe_allow_html=True,
)

# State
if "lyrics" not in st.session_state: st.session_state.lyrics = ""
if "title" not in st.session_state: st.session_state.title = ""
if "topic" not in st.session_state: st.session_state.topic = ""
if "targets" not in st.session_state: st.session_state.targets = []
if "generated" not in st.session_state: st.session_state.generated = False

# Sidebar
with st.sidebar:
    st.markdown("## 👩‍🏫 Hướng dẫn nhanh")
    st.markdown("- **Bước 1:** Nhập Miêu tả/Từ khóa/Title → **Tạo lời**.\n- **Bước 2:** Chỉnh tay hoặc **Refine**.\n- **Bước 3:** **Tạo nhạc**, xem ảnh bìa & tải file.\n- Xem lại ở **📚 Thư viện** hoặc **🗂️ Lịch sử**.")
    st.divider()
    st.caption(f"Model Suno: **{SUNO_MODEL}**")
    st.caption(f"Style mặc định: **{DEFAULT_SUNOSTYLE}**")
    st.caption(f"Supabase: <span class='status'>{supabase_status}</span>", unsafe_allow_html=True)

# Header & Tabs
st.title("🎵 Kids Song AI")
st.markdown('<span class="badge">OpenAI Lyrics • Suno Music *Preschool Education</span>', unsafe_allow_html=True)

tab_make, tab_library, tab_history, tab_settings = st.tabs(["✨ Tạo bài hát", "📚 Thư viện", "🗂️ Lịch sử", "⚙️ Cài đặt"])

# ================== TAB 1: TẠO BÀI HÁT ==================
with tab_make:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        topic = st.text_input("Miêu tả bài hát", st.session_state.topic or "Trường mầm non của bé")
        target_str = st.text_input("Từ ngữ gợi ý (phân tách bởi dấu phẩy)", "Đồ chơi, sân trường, lớp học, thân thương")
        title = st.text_input("Tiêu đề bài hát", st.session_state.title or "Trường mầm non của bé")
    with col2:
        verses = st.number_input("Số verse", 1, 4, 2)
        bridge = st.toggle("Thêm Bridge", value=True)
        language = st.selectbox("Ngôn ngữ", ["Vi", "En"], index=0)
        style = st.selectbox("Phong cách nhạc", [
            DEFAULT_SUNOSTYLE,
            "Kids, gentle, soothing, lullaby, warm",
            "Kids, cute pop, claps, ukulele",
            "Kids, upbeat, bright, classroom sing-along",
            "Instrumental lullaby, gentle, soft piano + strings",
        ], index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 TIẾN TRÌNH SÁNG TÁC")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        btn_generate = st.button("✨ Tạo lời bài hát", use_container_width=True)
    with c2:
        refine_hint = st.text_input("Chỉ dẫn refine (tuỳ chọn)", placeholder="Ví dụ: nhịp nhanh hơn, thêm điệp khúc…")
    with c3:
        btn_refine = st.button("🪄 Refine", use_container_width=True, disabled=not bool(st.session_state.lyrics.strip()))

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

    st.session_state.lyrics = st.text_area("Soạn thảo/Chỉnh sửa tại đây trước khi tạo nhạc:", value=st.session_state.lyrics, height=320)

    st.divider()
    left, right = st.columns([1,2])
    with left:
        instrumental = st.toggle("Chỉ giai điệu (instrumental)", value=False)
    with right:
        btn_music = st.button("🎧 Tạo nhạc", use_container_width=True, disabled=not bool(st.session_state.lyrics.strip()))

    if btn_music and st.session_state.lyrics.strip():
        try:
            with st.spinner("Đang tạo bài hát..."):
                task_id = suno_generate_song(st.session_state.lyrics, st.session_state.title or "Kids Song", style=style, instrumental=instrumental)
                tracks = suno_poll(task_id)

            st.subheader("🎧 Kết quả")
            ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            base = ascii_slugify(st.session_state.title or "Kids_Song")

            for i, t in enumerate(tracks, 1):
                audio_url_orig = t.get("audioUrlHigh") or t.get("audioUrl")
                image_url_orig = t.get("imageUrl")
                mp3_path = cover_path = ""

                audio_bytes = img_bytes = b""
                if audio_url_orig:
                    audio_bytes = download_bytes(audio_url_orig)
                    mp3_path = f"outputs/mp3/{ts}_{i}_{base}.mp3"
                    with open(mp3_path, "wb") as f: f.write(audio_bytes)
                if image_url_orig:
                    img_bytes = download_bytes(image_url_orig)
                    cover_path = f"outputs/covers/{ts}_{i}_{base}.jpg"
                    with open(cover_path, "wb") as f: f.write(img_bytes)

                # Upload Storage
                lyrics_url_pub = None
                if st.session_state.get("lyrics"):
                    try:
                        lyrics_url_pub = sb_upload_bytes(SUPABASE_BUCKET, f"lyrics/{ts}_{i}_{base}.txt", st.session_state.lyrics.encode("utf-8"), "text/plain")
                    except Exception: pass
                audio_url_pub = sb_upload_bytes(SUPABASE_BUCKET, f"mp3/{ts}_{i}_{base}.mp3", audio_bytes, "audio/mpeg") if audio_bytes else None
                image_url_pub = sb_upload_bytes(SUPABASE_BUCKET, f"covers/{ts}_{i}_{base}.jpg", img_bytes, "image/jpeg") if img_bytes else None

                audio_url_final = audio_url_pub or audio_url_orig or ""
                image_url_final = image_url_pub or image_url_orig or ""

                # UI kết quả
                k1, k2 = st.columns([1,2])
                with k1:
                    if cover_path and os.path.exists(cover_path):
                        st.image(cover_path, caption="Ảnh bìa", use_container_width=True)
                    elif image_url_final:
                        st.image(image_url_final, caption="Ảnh bìa", use_container_width=True)
                with k2:
                    st.write(f"**{st.session_state.title or 'Kids Song'} — Bản {i}**")
                    if mp3_path and os.path.exists(mp3_path):
                        with open(mp3_path, "rb") as f: st.audio(f.read(), format="audio/mp3")
                        with open(mp3_path, "rb") as f:
                            st.download_button("⬇️ Tải MP3", data=f, file_name=os.path.basename(mp3_path), mime="audio/mpeg", use_container_width=True, key=f"dl_now_{ts}_{i}")
                    elif audio_url_final:
                        st.audio(audio_url_final, format="audio/mp3")

                # Lưu lịch sử
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

            st.balloons(); st.info("Đã lưu vào Supabase và thư mục local. Xem ở tab 📚 Thư viện.")
        except Exception as e:
            st.error(str(e))

    st.markdown('</div>', unsafe_allow_html=True)

# ================== TAB 2: THƯ VIỆN ==================
with tab_library:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📚 Thư viện (Gallery)")
    data_source = "local"

    df = load_history_df_supabase()
    if df is not None and len(df) > 0:
        data_source = "supabase"
    if (df is None) or (len(df) == 0):
        if os.path.exists(HISTORY_CSV):
            import pandas as pd
            df = load_history_df_local()
        else:
            df = None

    if df is None or len(df) == 0:
        st.info("Chưa có dữ liệu. Hãy tạo bài hát ở tab ✨ trước nhé.")
    else:
        try:
            if "time" in df.columns and "track_index" in df.columns:
                df = df.sort_values(by=["time","track_index"], ascending=[False, True]).reset_index(drop=True)

            colf1, colf2 = st.columns([2,1])
            with colf1:
                q = st.text_input("Tìm theo tiêu đề/chủ đề", "")
            with colf2:
                if "style" in df.columns:
                    style_vals = sorted([s for s in df["style"].dropna().unique().tolist()])
                else:
                    style_vals = []
                style_pick = st.selectbox("Lọc theo style", ["Tất cả"] + style_vals)

            if q and "title" in df.columns and "topic" in df.columns:
                mask = df["title"].str.contains(q, case=False, na=False) | df["topic"].str.contains(q, case=False, na=False)
                df = df[mask]
            if style_pick and style_pick != "Tất cả" and "style" in df.columns:
                df = df[df["style"] == style_pick]

            st.caption(f"Nguồn dữ liệu: **{'Supabase' if data_source=='supabase' else 'Local CSV'}**")

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

                        if cover and os.path.exists(cover):
                            st.image(cover, use_container_width=True)
                        elif image_url:
                            st.image(image_url, use_container_width=True)
                        else:
                            st.image("https://picsum.photos/seed/kidsmusic/600/400", use_container_width=True)

                        st.markdown(f"<h4>{title}</h4><div style='color:#64748b'>{subtitle}</div>", unsafe_allow_html=True)

                        mp3_path = (row.get("mp3_path") or "").strip()
                        audio_url = (row.get("audio_url") or "").strip()
                        if mp3_path and os.path.exists(mp3_path):
                            with open(mp3_path, "rb") as f: st.audio(f.read(), format="audio/mp3")
                            with open(mp3_path, "rb") as f:
                                st.download_button("⬇️ Tải MP3", data=f, file_name=os.path.basename(mp3_path), mime="audio/mpeg", use_container_width=True, key=f"dl_lib_{row.get('time','')}_{int(float(row.get('track_index', idx%4+1)))}_{idx}")
                        elif audio_url:
                            st.audio(audio_url, format="audio/mp3")

                        st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error("Không đọc được thư viện: " + str(e))

    st.markdown('</div>', unsafe_allow_html=True)

# ================== TAB 3: LỊCH SỬ ==================
with tab_history:
    df_hist = load_history_df_supabase()
    if (df_hist is None or len(df_hist) == 0) and os.path.exists(HISTORY_CSV):
        df_hist = load_history_df_local()

    if df_hist is not None and len(df_hist) > 0:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🗂️ Lịch sử tạo nhạc (bảng)")
        st.dataframe(df_hist, use_container_width=True, height=360)
        try:
            st.download_button("⬇️ Tải CSV lịch sử", df_hist.to_csv(index=False).encode("utf-8"), file_name="tracks_history.csv", mime="text/csv", use_container_width=True)
        except Exception:
            pass
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Chưa có bài hát nào. Tạo bài hát ở tab ✨ trước nhé.")

# ================== TAB 4: CÀI ĐẶT ==================
with tab_settings:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✅ Kiểm tra Supabase")
    colsb1, colsb2 = st.columns(2)
    with colsb1:
        if supabase:
            try:
                res = supabase.table(SUPABASE_TABLE).select('*', count='exact').range(0,0).execute()
                total_rows = res.count or 0
            except Exception:
                total_rows = None
            st.metric(label="Số bản ghi trong bảng", value= total_rows if total_rows is not None else "—")
        else:
            st.info("Chưa cấu hình Supabase URL/KEY")
    with colsb2:
        if supabase:
            btn_list  = st.button("🔎 Liệt kê Storage (mp3/ & covers/)", use_container_width=True)
            btn_probe = st.button("🧪 Upload file test", use_container_width=True)
            if btn_probe:
                try:
                    ts = dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
                    p = f"tests/{ts}_hello.txt"
                    puburl = sb_upload_bytes(SUPABASE_BUCKET, p, b"hello from Kids Song AI", "text/plain")
                    if puburl:
                        st.success("Đã upload file test:")
                        st.markdown(f"- `{p}` → [Mở file]({puburl})")
                    else:
                        st.warning("Upload test không thành công (xem cảnh báo ở trên nếu có).")
                except Exception as e:
                    st.warning(f"Lỗi upload test: {e}")
            if btn_list:
                try:
                    mp3_files = supabase.storage.from_(SUPABASE_BUCKET).list("mp3") or []
                    cov_files = supabase.storage.from_(SUPABASE_BUCKET).list("covers") or []
                    st.write(f"**mp3/**: {len(mp3_files)} tệp")
                    for f in mp3_files[:10]:
                        name = f.get('name') if isinstance(f, dict) else getattr(f, 'name', '')
                        url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(f"mp3/{name}")
                        show = url['publicUrl'] if isinstance(url, dict) else str(url)
                        st.markdown(f"- `{name}` → [mở]({show})")
                    st.write(f"**covers/**: {len(cov_files)} tệp")
                    for f in cov_files[:10]:
                        name = f.get('name') if isinstance(f, dict) else getattr(f, 'name', '')
                        url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(f"covers/{name}")
                        show = url['publicUrl'] if isinstance(url, dict) else str(url)
                        st.markdown(f"- `{name}` → [mở]({show})")
                except Exception as e:
                    st.warning(f"Không thể liệt kê bucket: {e}")
        else:
            st.empty()

    st.divider()
    st.markdown("### 🎨 Preset chủ đề nhanh")
    preset = st.selectbox("Chọn nhanh", [
        "Màu sắc cơ bản","Hình tròn – vuông – tam giác","Số đếm 1 – 10","Vệ sinh răng miệng",
        "Chào hỏi & phép lịch sự","An toàn giao thông","Con vật","Gia đình","Nghề nghiệp",
        "Trường mầm non","Bản thân bé","Thầy cô và bạn bè"
    ])
    st.caption("Chọn preset rồi copy sang tab ✨.")

    st.divider()
    st.markdown("### ℹ️ Ghi chú")
    st.markdown(
        "- **Refine** chỉ chỉnh lời hiện tại, không đổi chủ đề.\n"
        "- **Instrumental** yêu cầu Suno tạo giai điệu không lời.\n"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown("""
<hr style="margin:24px 0; border:none; border-top:1px solid #e6e8f5;">
<div style="text-align:center; margin-top:8px; line-height:1.7;">
  <div style="font-weight:800; font-size:18px;">© Kids Song AI • OpenAI Lyrics + Suno Music - Dành cho Giáo viên mầm non</div>
  <div style="font-size:15px; color:#64748b;">Ngọc Thảo – <a href=\"mailto:ms.nthaotran@gmail.com\">ms.nthaotran@gmail.com</a></div>
</div>
""", unsafe_allow_html=True)





