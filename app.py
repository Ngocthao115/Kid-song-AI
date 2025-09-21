import os, re, time, json, datetime as dt, mimetypes, uuid
from pathlib import Path
from typing import Optional, Dict, List

import requests
import streamlit as st
import pandas as pd

# =========================
# 1) CONFIG / SECRETS

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.environ.get(key, default)

OPENAI_API_KEY    = get_secret("OPENAI_API_KEY", "")
SUNO_API_KEY      = get_secret("SUNO_API_KEY", "")
SUNO_API_BASE     = get_secret("SUNO_API_BASE", "https://api.sunoapi.org").rstrip("/")
SUNO_MODEL        = get_secret("SUNO_MODEL", "V4_5")
SUNO_CALLBACK_URL = get_secret("SUNO_CALLBACK_URL", "")
DEFAULT_SUNOSTYLE = get_secret("DEFAULT_SUNOSTYLE", "Kids, cheerful, playful, educational")

# ----- Supabase (persistence)
try:
    from supabase import create_client
except Exception:
    create_client = None

SUPABASE_URL      = get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = get_secret("SUPABASE_ANON_KEY")
BUCKET_NAME       = get_secret("BUCKET_NAME", "kids-songs")
SUPABASE_READY    = bool(SUPABASE_URL and SUPABASE_ANON_KEY and create_client is not None)

@st.cache_resource
def _sb():
    if not SUPABASE_READY:
        raise RuntimeError("Supabase chưa sẵn sàng (thiếu secrets hoặc gói 'supabase').")
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def _sb_public_url(path: str) -> str:
    return _sb().storage.from_(BUCKET_NAME).get_public_url(path)

def _sb_upload_bytes(data: bytes, path: str, content_type: Optional[str] = None) -> str:
    if content_type is None:
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    _sb().storage.from_(BUCKET_NAME).upload(
        path, data, file_options={"contentType": content_type, "upsert": True}
    )
    return _sb_public_url(path)

def _sb_insert_track(title: str, style: str, lyrics_url: str, audio_url: str, cover_url: str, uploader: str = "") -> None:
    _sb().table("tracks").insert({
        "title": title, "style": style,
        "lyrics_url": lyrics_url, "audio_url": audio_url, "cover_url": cover_url,
        "uploader": uploader
    }).execute()

def save_track_to_supabase(
    title: str,
    style: str,
    lyrics_text: Optional[str],
    audio_bytes: Optional[bytes] = None,
    audio_url: Optional[str] = None,
    cover_bytes: Optional[bytes] = None,
    uploader: str = ""
) -> Dict[str, str]:
    """Upload lyrics/audio/cover lên Storage + ghi metadata vào bảng tracks."""
    if not SUPABASE_READY:
        return {}
    uid = str(uuid.uuid4())

    lyr_url = ""
    if lyrics_text and lyrics_text.strip():
        lyr_url = _sb_upload_bytes(lyrics_text.encode("utf-8"), f"{uid}/lyrics.txt", "text/plain; charset=utf-8")

    if audio_bytes is None and audio_url:
        try:
            r = requests.get(audio_url, timeout=120)
            r.raise_for_status()
            audio_bytes = r.content
        except Exception:
            pass
    if not audio_bytes:
        raise RuntimeError("Không có audio để lưu lên Supabase.")

    aud_url = _sb_upload_bytes(audio_bytes, f"{uid}/audio.mp3", "audio/mpeg")

    cov_url = ""
    if cover_bytes:
        cov_url = _sb_upload_bytes(cover_bytes, f"{uid}/cover.jpg")

    _sb_insert_track(title=title, style=style, lyrics_url=lyr_url, audio_url=aud_url, cover_url=cov_url, uploader=uploader)
    return {"lyrics_url": lyr_url, "audio_url": aud_url, "cover_url": cov_url}

def list_tracks_from_supabase(search_title: str = "", style: str = "", limit: int = 200) -> List[dict]:
    if not SUPABASE_READY:
        return []
    q = _sb().table("tracks").select("*").order("created_at", desc=True).limit(limit)
    if search_title:
        q = q.ilike("title", f"%{search_title}%")
    if style and style.lower() != "tất cả":
        q = q.ilike("style", f"%{style}%")
    return q.execute().data

# =========================
# 2) UTILITIES / LOCAL IO

HISTORY_CSV = Path("data/tracks.csv")

def ensure_dirs():
    Path("outputs/mp3").mkdir(parents=True, exist_ok=True)
    Path("outputs/covers").mkdir(parents=True, exist_ok=True)
    Path("outputs/lyrics").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    s = re.sub(r"[^\w\s\-]", "", name or "")
    s = re.sub(r"\s+", "_", s.strip())
    return s[:80] or "Kids_Song"

def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def write_history_row(row: dict) -> None:
    ensure_dirs()
    df = pd.DataFrame([row])
    if HISTORY_CSV.exists():
        old = pd.read_csv(HISTORY_CSV)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False, encoding="utf-8")

def load_history_df() -> pd.DataFrame:
    if HISTORY_CSV.exists():
        try:
            return pd.read_csv(HISTORY_CSV)
        except Exception:
            return pd.read_csv(HISTORY_CSV, engine="python", on_bad_lines="skip")
    return pd.DataFrame(columns=["time","title","topic","track_index","audio_url","image_url","style","model","mp3_path","cover_path"])

# =========================
# 3) OPENAI LYRICS
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

def openai_generate_lyrics(topic: str, style_hint: str) -> str:
    if not _OPENAI_OK or not OPENAI_API_KEY:
        raise RuntimeError("Chưa cấu hình OPENAI_API_KEY.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    sys = ("Bạn là một nhà thơ và nhạc sĩ viết nhạc thiếu nhi chuyên nghiệp cho giáo dục mầm non."
           "Hãy sáng tác lời bài hát hoặc dùng câu chuyện, bài thơ thiếu nhi để sáng tác lời bài hát thiếu nhi, phù hợp với trẻ 3 - 6 tuổi, tươi vui, tích cực, tình cảm và yêu thương."
           "Mỗi câu ngắn từ 5 - 10 từ ngữ, vần điệu rõ, từ vựng đơn giản, có điệp khúc dễ nhớ.")
    prompt = f"Chủ đề: {topic}\nPhong cách: {style_hint}\nViết theo bố cục Verse/Chorus/Bridge (nếu cần)."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
        temperature=0.8
    )
    return resp.choices[0].message.content.strip()

# =========================
# 4) SUNO API (v1 của sunoapi.org)

def suno_create_task(title: str, lyrics: str, style: str) -> str:
    url = f"{SUNO_API_BASE}/api/v1/generate"
    payload = {
        "prompt": lyrics[:1800],
        "title": title[:64],
        "style": style[:200],
        "model": SUNO_MODEL,
        "customMode": True,
    }
    if SUNO_CALLBACK_URL:
        payload["callBackUrl"] = SUNO_CALLBACK_URL
    r = requests.post(url, headers={"Authorization": f"Bearer {SUNO_API_KEY}"}, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    task_id = data.get("data", {}).get("taskId")
    if not task_id:
        raise RuntimeError(f"Suno generate failed: {json.dumps(data, ensure_ascii=False)}")
    return str(task_id)

def suno_poll(task_id: str, timeout_sec: int = 420, interval_sec: int = 6) -> List[dict]:
    url = f"{SUNO_API_BASE}/api/v1/generate/record-info"
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        r = requests.get(url, headers={"Authorization": f"Bearer {SUNO_API_KEY}"}, params={"taskId": task_id}, timeout=60)
        r.raise_for_status()
        data = r.json()
        try:
            items = data["data"].get("sunoData") or data["data"].get("list") or []
            ready = [it for it in items if it.get("audioUrl") or it.get("audioUrlHigh")]
            if ready:
                return ready
        except Exception:
            pass
        time.sleep(interval_sec)
    raise TimeoutError("Hết thời gian chờ trả kết quả")

# =========================
# 5) UI
# =========================

st.set_page_config(page_title="Kids Song AI", page_icon="🎵", layout="wide")
ensure_dirs()

st.title("Kids Song AI")
st.caption("OpenAI Lyrics • Suno Music • Lưu local + Supabase")

tab_create, tab_library, tab_history, tab_admin = st.tabs(["Tạo bài hát", "Thư viện", "Lịch sử", "Cài đặt"])

# ---- Tab 1: Create ----
with tab_create:
    c1, c2 = st.columns([2,1])

    with c1:
        st.session_state.title = st.text_input("Tiêu đề", value=st.session_state.get("title",""))
        st.session_state.topic = st.text_input("Miêu tả/chủ đề", value=st.session_state.get("topic",""))
        style = st.selectbox("Style", [
            "Kids, cheerful, playful, educational",
            "Kids, cute pop, claps, ukulele",
            "Kids, upbeat, bright, classroom sing-along",
            "Kids lullaby, gentle, calm, emotional, soft piano and strings, warm female vocal",
            "Instrumental lullaby, gentle, soothing, soft piano + strings, warm and calm",
        ], index=0)

        if st.button("Sinh lời (OpenAI)"):
            try:
                lyrics = openai_generate_lyrics(st.session_state.topic or st.session_state.title, style)
                st.session_state.lyrics = lyrics
                st.success("Đã sinh lời.")
            except Exception as e:
                st.error(f"Lỗi OpenAI: {e}")

        st.session_state.lyrics = st.text_area("Lời bài hát", value=st.session_state.get("lyrics",""), height=260)

        if st.button("Tạo nhạc"):
            try:
                task_id = suno_create_task(st.session_state.title or "Kids Song", st.session_state.lyrics or "", style or DEFAULT_SUNOSTYLE)
                tracks = suno_poll(task_id)

                st.subheader("Kết quả")
                ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                base = sanitize_filename(st.session_state.title or "Kids_Song")

                for i, t in enumerate(tracks, 1):
                    audio_url = t.get("audioUrlHigh") or t.get("audioUrl")
                    image_url = t.get("imageUrl")
                    mp3_path = ""
                    cover_path = ""

                    # Lưu local
                    audio_bytes = None
                    if audio_url:
                        audio_bytes = download_bytes(audio_url)
                        mp3_path = f"outputs/mp3/{ts}_{i}_{base}.mp3"
                        with open(mp3_path, "wb") as f:
                            f.write(audio_bytes)

                    img_bytes = None
                    if image_url:
                        img_bytes = download_bytes(image_url)
                        cover_path = f"outputs/covers/{ts}_{i}_{base}.jpg"
                        with open(cover_path, "wb") as f:
                            f.write(img_bytes)

                    # Lưu bền Supabase (bổ sung)
                    if SUPABASE_READY:
                        try:
                            save_track_to_supabase(
                                title      = st.session_state.get("title","Kids Song"),
                                style      = style,
                                lyrics_text= st.session_state.get("lyrics",""),
                                audio_bytes= audio_bytes,
                                audio_url  = audio_url,
                                cover_bytes= img_bytes,
                                uploader   = "streamlit-app",
                            )
                            st.caption("Đã sao lưu lên Supabase")
                        except Exception as e:
                            st.warning(f"Lưu Supabase lỗi: {e}")

                    # Hiển thị
                    k1, k2 = st.columns([1,2])
                    with k1:
                        if cover_path and os.path.exists(cover_path):
                            st.image(cover_path, caption="Ảnh bìa", use_container_width=True)
                        elif image_url:
                            st.image(image_url, caption="Ảnh bìa", use_container_width=True)
                    with k2:
                        st.write(f"**{st.session_state.title or 'Kids Song'} - Bản {i}**")
                        if mp3_path and os.path.exists(mp3_path):
                            with open(mp3_path, "rb") as f:
                                mp3_bytes = f.read()
                            st.audio(mp3_bytes, format="audio/mp3", key=f"audio-{ts}-{i}")
                            st.download_button(
                                label="Tải MP3",
                                data=mp3_bytes,
                                file_name=os.path.basename(mp3_path),
                                mime="audio/mpeg",
                                use_container_width=True,
                                key=f"dl-{ts}-{i}",
                            )
                        elif audio_url:
                            st.audio(audio_url, format="audio/mp3", key=f"audio-url-{ts}-{i}")

                    # Ghi CSV lịch sử
                    write_history_row({
                        "time": ts,
                        "title": st.session_state.title or "Kids Song",
                        "topic": st.session_state.topic or "",
                        "track_index": i,
                        "audio_url": audio_url or "",
                        "image_url": image_url or "",
                        "style": style,
                        "model": SUNO_MODEL,
                        "mp3_path": mp3_path,
                        "cover_path": cover_path,
                    })

            except Exception as e:
                st.error(f"Lỗi tạo nhạc: {e}")

    with c2:
        st.info("Gợi ý: Sau khi tạo, file sẽ được lưu.")

# ---- Tab 2: Library ----
with tab_library:
    st.subheader("Thư viện")
    if SUPABASE_READY:
        with st.expander("Xem từ Supabase", expanded=False):
            q1, q2 = st.columns([2,1])
            with q1:
                sb_q = st.text_input("Tìm theo tiêu đề", key="sb-q")
            with q2:
                sb_style = st.selectbox("Lọc style", ["Tất cả",
                    "Kids, cheerful, playful, educational",
                    "Kids, cute pop, claps, ukulele",
                    "Kids, upbeat, bright, classroom sing-along",
                    "Kids lullaby, gentle, calm, emotional, soft piano and strings, warm female vocal",
                    "Instrumental lullaby, gentle, soothing, soft piano + strings, warm and calm"], key="sb-style")
            try:
                rows = list_tracks_from_supabase(search_title=sb_q, style=sb_style)
                if not rows:
                    st.info("Chưa có bài trên Supabase.")
                else:
                    cols = st.columns(4)
                    for idx, r in enumerate(rows):
                        col = cols[idx % 4]
                        with col:
                            if r.get("cover_url"):
                                st.image(r["cover_url"], use_container_width=True)
                            st.markdown(f"**{r.get('title','(Không tiêu đề)')}**")
                            st.caption(f"{(r.get('created_at','') or '')[:19]} • {r.get('style','')}")
                            if r.get("audio_url"):
                                st.audio(r["audio_url"], format="audio/mp3", key=f"sb-audio-{idx}")
                                st.markdown(f"[Tải MP3]({r['audio_url']})")
                            if r.get("lyrics_url"):
                                st.markdown(f"[Lời bài hát]({r['lyrics_url']})")
            except Exception as e:
                st.error(f"Không đọc Supabase: {e}")

    st.markdown("---")
    st.subheader("Thư viện (Local CSV + outputs/)")
    df = load_history_df()
    if df.empty:
        st.info("Chưa có dữ liệu local.")
    else:
        q = st.text_input("Tìm theo tiêu đề (Local)", "")
        style_pick = st.selectbox("Lọc style (Local)", ["Tất cả"] + sorted(list(set(df["style"].dropna().tolist()))))
        show = df.copy()
        if q:
            show = show[show["title"].fillna("").str.contains(q, case=False)]
        if style_pick and style_pick != "Tất cả":
            show = show[show["style"] == style_pick]
        show = show.sort_values("time", ascending=False)

        cols = st.columns(4)
        for idx, row in show.iterrows():
            col = cols[idx % 4]
            with col:
                cover_path = row.get("cover_path","")
                if cover_path and os.path.exists(cover_path):
                    st.image(cover_path, use_container_width=True)
                st.markdown(f"**{row.get('title','(Không tiêu đề)')}**")
                st.caption(f"{row.get('time','')} • {row.get('style','')}")
                mp3_path = row.get("mp3_path","")
                audio_url = row.get("audio_url","")
                if mp3_path and os.path.exists(mp3_path):
                    with open(mp3_path, "rb") as f:
                        mp3_bytes = f.read()
                    st.audio(mp3_bytes, format="audio/mp3", key=f"lib-a-{row.get('time','')}-{idx}")
                    st.download_button(
                        label="Tải MP3",
                        data=mp3_bytes,
                        file_name=os.path.basename(mp3_path),
                        mime="audio/mpeg",
                        use_container_width=True,
                        key=f"lib-dl-{row.get('time','')}-{idx}",
                    )
                elif audio_url:
                    st.audio(audio_url, format="audio/mp3", key=f"lib-url-{row.get('time','')}-{idx}")

# ---- Tab 3: History ----
with tab_history:
    st.subheader("Lịch sử (Local)")
    df = load_history_df()
    if df.empty:
        st.info("Chưa có dữ liệu.")
    else:
        st.dataframe(df.sort_values("time", ascending=False), use_container_width=True, height=380)

# ---- Tab 4: Admin ----
with tab_admin:
    st.subheader("Cài đặt / Kiểm tra")
    st.write("OpenAI:", "OK" if OPENAI_API_KEY else "—")
    st.write("Suno:", "OK" if SUNO_API_KEY else "—")
    st.write("Supabase:", "OK" if SUPABASE_READY else "—")
    st.write("Bucket:", BUCKET_NAME)

    st.markdown("---")
    if SUPABASE_READY and st.button("Import tất cả outputs/ → Supabase"):
        mp3s = list(Path("outputs/mp3").glob("*.mp3"))
        n = 0
        for mp3 in mp3s:
            uid = str(uuid.uuid4())
            aurl = _sb_upload_bytes(mp3.read_bytes(), f"{uid}/audio.mp3", "audio/mpeg")
            stem = Path(mp3).stem
            lyr = Path("outputs/lyrics")/f"{stem}.txt"
            lurl = _sb_upload_bytes(lyr.read_bytes(), f"{uid}/lyrics.txt", "text/plain; charset=utf-8") if lyr.exists() else ""
            cov = Path("outputs/covers")/f"{stem}.jpg"
            curl = _sb_upload_bytes(cov.read_bytes(), f"{uid}/cover.jpg") if cov.exists() else ""
            _sb_insert_track(title=stem, style="", lyrics_url=lurl, audio_url=aurl, cover_url=curl, uploader="import")
            n += 1
        st.success(f"Đã import {n} bài từ outputs/")


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

































