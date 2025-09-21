import streamlit as st
from pathlib import Path
from datetime import datetime
import uuid, io, mimetypes, time, os

import requests

# OpenAI (lyrics)
from openai import OpenAI

# Supabase
from supabase import create_client

# ───────────────────────── Config ─────────────────────────

st.set_page_config(page_title="Kid-song-AI", page_icon="🎵", layout="wide")

OPENAI_OK = "OPENAI_API_KEY" in st.secrets
SUPABASE_OK = "SUPABASE_URL" in st.secrets and "SUPABASE_ANON_KEY" in st.secrets
SUNO_OK = "SUNO_API_BASE" in st.secrets and "SUNO_API_KEY" in st.secrets

BUCKET = st.secrets.get("BUCKET_NAME", "kids-songs")
UPLOADER_DEFAULT = st.secrets.get("APP_UPLOADER_NAME", "")

# ──────────────────────── Helpers ─────────────────────────

@st.cache_resource
def sb_client():
    if not SUPABASE_OK:
        raise RuntimeError("Chưa cấu hình Supabase trong Secrets.")
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])

def sb_public_url(path: str) -> str:
    return sb_client().storage.from_(BUCKET).get_public_url(path)

def sb_upload_bytes(data: bytes, path: str, content_type: str | None = None) -> str:
    if content_type is None:
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    sb_client().storage.from_(BUCKET).upload(
        path, data, file_options={"contentType": content_type, "upsert": True}
    )
    return sb_public_url(path)

def sb_insert_track(title, style, lyrics_url, audio_url, cover_url, uploader=""):
    r = sb_client().table("tracks").insert({
        "title": title, "style": style, "lyrics_url": lyrics_url,
        "audio_url": audio_url, "cover_url": cover_url, "uploader": uploader
    }).execute()
    return r

def sb_list_tracks(search_title="", style="", limit=200):
    q = sb_client().table("tracks").select("*").order("created_at", desc=True).limit(limit)
    if search_title:
        q = q.ilike("title", f"%{search_title}%")
    if style and style.lower() != "tất cả":
        q = q.ilike("style", f"%{style}%")
    return q.execute().data

def fetch_url_bytes(url: str, timeout=60) -> bytes | None:
    try:
        r = requests.get(url, timeout=timeout)
        if r.ok:
            return r.content
    except Exception:
        return None
    return None

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ───────────────────── OpenAI Lyrics ──────────────────────

def openai_generate_lyrics(topic: str, style_hint: str) -> str:
    if not OPENAI_OK:
        raise RuntimeError("Chưa cấu hình OPENAI_API_KEY.")
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    sys = ("Bạn là nhà thơ văn và là nhạc sĩ chuyên nghiệp sáng tác lời bài hát Thiếu nhi hoặc dùng câu chuyện, bài thơ thiếu nhi để sáng tác ra lời bài hát bằng tiếng Việt: vui tươi, an toàn, dễ hát. "
           "Câu ngắn 5–9 âm tiết, điệp khúc rõ ràng, vần điệu dễ nhớ.")
    prompt = f"Viết lời bài hát thiếu nhi theo chủ đề: {topic}\nPhong cách: {style_hint}\nBố cục: Verse/Chorus/Bridge (nếu cần)."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
        temperature=0.8,
    )
    return resp.choices[0].message.content.strip()

# ────────────────────── Suno API ─────────────────────────

def suno_headers():
    return {
        "Authorization": f"Bearer {st.secrets['SUNO_API_KEY']}",
        "Content-Type": "application/json",
    }

def suno_create_track(title: str, prompt: str, style: str) -> str:
    """
    Gửi yêu cầu tạo bài hát. Return: track_id (string).
    LƯU Ý: Endpoint có thể khác. Sửa theo tài liệu/adapter của chị nếu cần.
    """
    base = st.secrets["SUNO_API_BASE"].rstrip("/")
    payload = {
        "title": title,
        "prompt": prompt,          # lyrics text
        "style": style,            # optional style tag
        "make_instrumental": False # hoặc True nếu cần không lời
    }
    # Ví dụ endpoint:
    url = f"{base}/generate"
    r = requests.post(url, json=payload, headers=suno_headers(), timeout=60)
    r.raise_for_status()
    data = r.json()
    # tuỳ API trả về: lấy id đầu tiên
    track_id = data.get("id") or data.get("clip_id") or (data["data"][0]["id"] if "data" in data else None)
    if not track_id:
        raise RuntimeError(f"Không lấy được track id từ Suno: {data}")
    return str(track_id)

def suno_get_track(track_id: str) -> dict:
    """
    Lấy trạng thái bài hát. Return dict có các keys: status, audio_url, image_url.
    """
    base = st.secrets["SUNO_API_BASE"].rstrip("/")
    # Ví dụ endpoint:
    url = f"{base}/tracks/{track_id}"
    r = requests.get(url, headers=suno_headers(), timeout=60)
    r.raise_for_status()
    data = r.json()
    # Chuẩn hoá field
    status = data.get("status") or data.get("state") or "unknown"
    audio_url = data.get("audio_url") or data.get("audio") or data.get("mp3_url")
    image_url = data.get("image_url") or data.get("cover_url")
    return {"status": status, "audio_url": audio_url, "image_url": image_url, "raw": data}

def suno_generate_and_wait(title: str, lyrics: str, style: str, poll_secs=6, max_wait=420) -> tuple[str, str]:
    """
    Tạo bài hát và chờ hoàn tất.
    Return: (audio_url, cover_url). Có thể là URL tạm của Suno; sẽ sao lưu sang Supabase.
    """
    track_id = suno_create_track(title=title, prompt=lyrics, style=style)
    t0 = time.time()
    last_status = "queued"
    with st.status("Đang tạo nhạc…", expanded=True) as s:
        s.write(f"Track ID: `{track_id}`")
        while time.time() - t0 < max_wait:
            info = suno_get_track(track_id)
            status = (info.get("status") or "").lower()
            if status != last_status:
                s.write(f"Trạng thái: **{status}**")
                last_status = status
            if status in ("completed", "complete", "succeeded", "success"):
                s.update(label="✅ Đã tạo xong nhạc", state="complete")
                return info.get("audio_url"), info.get("image_url")
            if status in ("failed", "error", "canceled"):
                s.update(label="❌ Suno báo lỗi", state="error")
                raise RuntimeError(f"Suno lỗi: {info.get('raw')}")
            time.sleep(poll_secs)
        s.update(label="⌛ Hết thời gian chờ", state="error")
        raise TimeoutError("Chờ quá lâu, vui lòng thử lại.")
        
# ───────────────────────── UI ────────────────────────────

st.markdown(
    "<h1 style='margin-bottom:.25rem'>OpenAI Lyrics • Suno Music</h1>"
    "<p style='color:#64748b;margin-top:0'>Kho tàng âm nhạc Thiếu nhi từ AI</p>",
    unsafe_allow_html=True
)

tabs = st.tabs(["🎹 Tạo bài hát", "🖼️ Thư viện (Gallery)", "🕘 Lịch sử", "⚙️ Cài đặt / Admin"])

# ───── Tab 1: Create ─────
with tabs[0]:
    st.subheader("Tạo bài hát")
    c1, c2 = st.columns([2,1])

    with c1:
        title = st.text_input("Miêu tả", placeholder="Ví dụ: Chuyện cậu bé Tích Chu", key="t-title")
        topic = st.text_input("Từ khóa chính", placeholder="Ví dụ: uống nước nhớ nguồn, yêu gia đình...", key="t-topic")
        style = st.selectbox("Style", ["Cheerful","Gentle","Lullaby","Kids Pop","Instrumental","Other"], key="t-style")
        if st.button("✨ Tạo lời bài hát", key="btn-lyr"):
            try:
                st.session_state["lyrics"] = openai_generate_lyrics(topic or title, style)
                st.success("Đã sinh lời! Chỉnh sửa bên dưới trước khi tạo nhạc hoặc lưu.")
            except Exception as e:
                st.error(f"Lỗi OpenAI: {e}")
        lyrics = st.text_area("Lời bài hát (có thể chỉnh sửa)", value=st.session_state.get("lyrics",""), height=260, key="lyrics")

    with c2:
        st.markdown("**Ảnh bìa (tuỳ chọn)**")
        up_cover = st.file_uploader("Chọn ảnh PNG/JPG", type=["png","jpg","jpeg"], key="u-cover")
        st.markdown("---")
        st.markdown("**Âm thanh** — 3 cách:")
        up_audio = st.file_uploader("1) Upload file MP3/WAV", type=["mp3","wav"], key="u-audio")
        audio_url_input = st.text_input("2) Dán URL MP3 (nếu đã có từ Suno/Udio…)", key="u-audio-url")
        suno_will_run = st.checkbox("3) Tạo bằng Suno ngay trong app", value=True, key="u-suno")

        uploader = st.text_input("Người tạo/uploader (tuỳ chọn)", value=UPLOADER_DEFAULT, key="uploader")

        if st.button("💾 Lưu vào Thư viện (Supabase)", use_container_width=True, key="btn-save"):
            try:
                if not SUPABASE_OK:
                    st.warning("Chưa cấu hình Supabase Secrets.")
                    st.stop()
                if not title:
                    st.warning("Hãy nhập tiêu đề.")
                    st.stop()

                # 1) Nếu chọn chạy Suno, tạo nhạc trước
                final_audio_bytes = None
                final_cover_bytes = None
                if suno_will_run:
                    if not SUNO_OK:
                        st.warning("Chưa cấu hình Suno API trong Secrets.")
                        st.stop()
                    if not lyrics.strip():
                        st.warning("Cần có lời bài hát để gửi Suno.")
                        st.stop()
                    au_url, img_url = suno_generate_and_wait(title=title, lyrics=lyrics, style=style)
                    # cố gắng tải bytes để lưu bản copy vào Supabase
                    if au_url:
                        final_audio_bytes = fetch_url_bytes(au_url)
                    if img_url:
                        final_cover_bytes = fetch_url_bytes(img_url)

                # 2) Ưu tiên audio từ upload nếu có; nếu không có, dùng audio đã tải từ URL hoặc do Suno trả
                if up_audio is not None:
                    final_audio_bytes = up_audio.read()
                    audio_ext = Path(up_audio.name).suffix.lower() or ".mp3"
                elif audio_url_input.strip() and final_audio_bytes is None:
                    # có URL do người dùng dán
                    final_audio_bytes = fetch_url_bytes(audio_url_input.strip())
                    audio_ext = ".mp3"
                else:
                    audio_ext = ".mp3"  # mặc định cho Suno

                if final_audio_bytes is None:
                    st.warning("Chưa có âm thanh (upload/URL/Suno).")
                    st.stop()

                # 3) Chuẩn bị dữ liệu để upload Supabase
                uid = str(uuid.uuid4())
                # lyrics
                lyrics_url = ""
                if lyrics.strip():
                    lyrics_url = sb_upload_bytes(lyrics.encode("utf-8"), f"{uid}/lyrics.txt", "text/plain; charset=utf-8")
                # cover: ưu tiên upload ảnh người dùng; nếu không, lấy từ Suno; nếu không có thì bỏ qua
                cover_bytes_to_save = None
                cover_ext = ".png"
                if up_cover is not None:
                    cover_ext = Path(up_cover.name).suffix.lower() or ".png"
                    cover_bytes_to_save = up_cover.read()
                elif final_cover_bytes:
                    cover_bytes_to_save = final_cover_bytes
                cover_url = ""
                if cover_bytes_to_save:
                    cover_url = sb_upload_bytes(cover_bytes_to_save, f"{uid}/cover{cover_ext}")

                # audio
                audio_url = sb_upload_bytes(final_audio_bytes, f"{uid}/audio{audio_ext}", "audio/mpeg")

                # 4) Ghi metadata vào DB
                sb_insert_track(title=title.strip(), style=style, lyrics_url=lyrics_url,
                                audio_url=audio_url, cover_url=cover_url, uploader=uploader.strip())
                st.success("Đã lưu bền vững vào Thư viện ✅")
                st.session_state["lyrics"] = ""
            except Exception as e:
                st.error(f"Lỗi lưu: {e}")

# ───── Tab 2: Library ─────
with tabs[1]:
    st.subheader("Thư viện (Gallery)")
    s1, s2 = st.columns([2,1])
    with s1:
        q = st.text_input("Tìm theo tiêu đề/chủ đề", key="lib-q")
    with s2:
        flt = st.selectbox("Lọc theo style", ["Tất cả","Cheerful","Gentle","Lullaby","Kids Pop","Instrumental","Other"], key="lib-style")

    rows = []
    try:
        rows = sb_list_tracks(search_title=q, style=flt)
    except Exception as e:
        st.error(f"Không đọc được thư viện: {e}")

    if not rows:
        st.info("Chưa có bài hát nào trong Thư viện.")
    else:
        for i, r in enumerate(rows):
            tid = str(r.get("id", f"row-{i}"))
            st.markdown("---")
            st.markdown(f"### {r.get('title','(Không tiêu đề)')}")
            st.caption(f"{r.get('created_at','')[:19]} • {r.get('style','')} • {r.get('uploader','')}")
            if r.get("cover_url"):
                st.image(r["cover_url"], use_container_width=True)
            if r.get("audio_url"):
                st.audio(r["audio_url"], start_time=0, key=f"audio-{tid}")
            c1, c2, c3 = st.columns(3)
            with c1:
                if r.get("audio_url"):
                    st.markdown(f"[⬇ MP3]({r['audio_url']})")
            with c2:
                if r.get("lyrics_url"):
                    st.markdown(f"[📄 Lời bài hát]({r['lyrics_url']})")
            with c3:
                if r.get("cover_url"):
                    st.markdown(f"[🖼 Ảnh bìa]({r['cover_url']})")

# ───── Tab 3: History ─────
with tabs[2]:
    st.subheader("Lịch sử gần đây")
    try:
        hist = sb_list_tracks(limit=100)
    except Exception as e:
        hist = []
        st.error(f"Lỗi tải lịch sử: {e}")
    for i, r in enumerate(hist):
        tid = str(r.get("id", f"h-{i}"))
        cols = st.columns([6,2,3,3])
        cols[0].markdown(f"**{r.get('title','(Không tiêu đề)')}**")
        cols[1].write(r.get("style",""))
        cols[2].write((r.get("created_at","") or "")[:19])
        cols[3].write(r.get("uploader",""))

# ───── Tab 4: Admin ─────
with tabs[3]:
    st.subheader("Cài đặt / Admin")
    st.write("OpenAI:", "✅" if OPENAI_OK else "—")
    st.write("Supabase:", "✅" if SUPABASE_OK else "❌")
    st.write("Bucket:", BUCKET)
    st.write("Suno:", "✅" if SUNO_OK else "—")

    st.markdown("---")
    st.markdown("### ⬆️ Import nhiều file (đẩy lên Supabase)")
    up_audios = st.file_uploader("Audio MP3/WAV (nhiều file)", type=["mp3","wav"], accept_multiple_files=True, key="adm-audios")
    up_lyrics = st.file_uploader("TXT lời (nhiều file)", type=["txt"], accept_multiple_files=True, key="adm-lyrics")
    up_covers = st.file_uploader("Ảnh bìa (nhiều file)", type=["png","jpg","jpeg"], accept_multiple_files=True, key="adm-covers")

    def _to_map(files):
        return {Path(f.name).stem: f for f in files} if files else {}

    if st.button("Import lên Thư viện", key="btn-import"):
        if not SUPABASE_OK:
            st.warning("Chưa cấu hình Supabase.")
        else:
            aud_map, lyr_map, cov_map = _to_map(up_audios), _to_map(up_lyrics), _to_map(up_covers)
            if not aud_map:
                st.warning("Cần tối thiểu 1 audio.")
            else:
                n = 0
                for stem, af in aud_map.items():
                    uid = str(uuid.uuid4())
                    aurl = sb_upload_bytes(af.read(), f"{uid}/audio{Path(af.name).suffix.lower() or '.mp3'}", "audio/mpeg")
                    lurl = ""
                    curl = ""
                    if stem in lyr_map:
                        lurl = sb_upload_bytes(lyr_map[stem].read(), f"{uid}/lyrics.txt", "text/plain; charset=utf-8")
                    if stem in cov_map:
                        curl = sb_upload_bytes(cov_map[stem].read(), f"{uid}/cover{Path(cov_map[stem].name).suffix.lower() or '.png'}")
                    sb_insert_track(title=stem, style="", lyrics_url=lurl, audio_url=aurl, cover_url=curl, uploader="import")
                    n += 1
                st.success(f"Đã import {n} bài.")

    st.markdown("---")
    st.markdown("### ⬇️ Export danh sách (CSV URL)")
    if st.button("Tạo CSV & tải xuống", key="btn-export"):
        try:
            rows = sb_list_tracks(limit=10000)
            import csv
            mem = io.StringIO()
            writer = csv.writer(mem)
            writer.writerow(["id","title","style","lyrics_url","audio_url","cover_url","created_at","uploader"])
            for r in rows:
                writer.writerow([r.get("id"), r.get("title"), r.get("style"), r.get("lyrics_url"),
                                 r.get("audio_url"), r.get("cover_url"), r.get("created_at"), r.get("uploader")])
            st.download_button("Tải CSV", data=mem.getvalue().encode("utf-8"),
                               file_name="kids_songs_urls.csv", mime="text/csv", key="dl-csv")
        except Exception as e:
            st.error(f"Lỗi export: {e}")

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

















