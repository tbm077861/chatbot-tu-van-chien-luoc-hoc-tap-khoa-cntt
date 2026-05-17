"""Streamlit chat UI — RAG Chatbot tư vấn học phần (Giai đoạn 7 multi-intent).

Luồng đúng (sau 3 vòng user clarify):
    1. Sidebar: chọn ngành + HK chuẩn bị đăng ký.
    2. Main top: bảng tự sinh chứa môn (bắt buộc + tự chọn) HK1..HK(target-1)
       của ngành đã chọn. User điền điểm môn nào đã học, bỏ trống môn chưa học.
    3. User chat tự nhiên — bot trả 3 loại câu hỏi:
       A. Tư vấn môn tự chọn HK target (recommend).
       B. Quy định học vụ (regulation).
       C. Khả năng đăng ký 1 môn cụ thể / prereq (prereq).
    4. Multi-turn — bot nhớ context qua session.

Backend URL: env `STREAMLIT_API_URL` (mặc định http://127.0.0.1:8000).

Chạy:
    venv/Scripts/python.exe -m streamlit run frontend/app.py
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("STREAMLIT_API_URL", "http://127.0.0.1:8000")

NGANH_OPTIONS = {
    "CS": "CS — Khoa học Máy tính",
    "IS": "IS — Hệ thống Thông tin",
    "DS": "DS — Khoa học Dữ liệu",
    "SE": "SE — Kỹ thuật Phần mềm",
    "IT": "IT — Công nghệ Thông tin",
}

INTENT_LABEL = {
    "recommend": "🎯 Tư vấn môn tự chọn",
    "regulation": "📜 Quy định học vụ",
    "prereq": "🔗 Điều kiện tiên quyết",
}


@st.cache_data(ttl=30)
def fetch_info() -> dict:
    try:
        r = requests.get(f"{API_URL}/info", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"error": str(e)}


@st.cache_data(ttl=600)
def fetch_grade_table(nganh: str, hk_target: int) -> list[dict]:
    """Cache grade table (curriculum không đổi)."""
    r = requests.get(
        f"{API_URL}/curriculum/grade-table",
        params={"nganh": nganh, "hk_target": hk_target},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def call_chat_v2(payload: dict) -> dict:
    r = requests.post(f"{API_URL}/chat/v2", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


# ===== Page setup =====

st.set_page_config(
    page_title="Chatbot tư vấn học phần",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Chatbot tư vấn đăng ký học phần")
st.caption(
    "Chọn ngành + HK ở sidebar → điền điểm môn đã học → chat tự nhiên. "
    "Bot tư vấn môn tự chọn / trả lời quy định học vụ / kiểm tra tiên quyết."
)

# ===== Session state init =====

defaults = {
    "messages": [],
    "grade_table": None,
    "grades_dict": {},
    "nganh": "CS",
    "hk_target": 3,
    "last_response_meta": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===== Sidebar: chọn ngành + HK + reset =====

with st.sidebar:
    st.header("Cài đặt")
    nganh = st.selectbox(
        "Ngành đào tạo",
        options=list(NGANH_OPTIONS.keys()),
        format_func=lambda k: NGANH_OPTIONS[k],
        index=list(NGANH_OPTIONS.keys()).index(st.session_state.nganh),
    )
    hk_target = st.number_input(
        "Học kỳ chuẩn bị đăng ký",
        min_value=1,
        max_value=9,
        value=int(st.session_state.hk_target),
        step=1,
    )

    # Khi user đổi ngành/HK → reset grade table + chat (giữ UX nhất quán).
    changed = (
        nganh != st.session_state.nganh
        or hk_target != st.session_state.hk_target
    )
    if changed:
        st.session_state.nganh = nganh
        st.session_state.hk_target = int(hk_target)
        st.session_state.grade_table = None
        st.session_state.grades_dict = {}
        st.session_state.messages = []
        st.session_state.last_response_meta = None
        st.rerun()

    if st.button("📋 Tạo bảng điểm", type="primary", use_container_width=True):
        try:
            rows = fetch_grade_table(nganh, int(hk_target))
            st.session_state.grade_table = rows
            st.session_state.grades_dict = {}
            st.session_state.messages = []
            st.session_state.last_response_meta = None
            st.success(f"Đã tạo bảng {len(rows)} môn cần nhập điểm.")
        except requests.RequestException as e:
            st.error(f"Lỗi: {e}")

    if st.button("🗑 Reset chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_response_meta = None
        st.rerun()

    st.divider()
    st.subheader("Backend")
    info = fetch_info()
    if "error" in info:
        st.error(f"Không kết nối được API {API_URL}\n{info['error']}")
    else:
        st.success(f"Mode: **{info.get('mode', '?')}**")
        st.caption(
            f"Corpus: {info.get('corpus_size', '?')} doc · "
            f"top_k={info.get('top_k_context', '?')}"
        )

# ===== Main area: grade table → chat =====

if st.session_state.grade_table is None:
    st.info(
        "👈 Chọn ngành + HK rồi bấm **Tạo bảng điểm** ở sidebar. "
        "Mặc định CS HK3."
    )
    st.stop()

st.subheader(f"Bảng điểm — {nganh} (HK1 → HK{int(hk_target) - 1})")
st.caption(
    "Điền điểm thang 10 cho môn đã học. Bỏ trống môn chưa học/rớt. "
    f"Bot sẽ dùng dữ liệu này để tư vấn môn tự chọn cho HK{int(hk_target)}."
)

# Build DataFrame từ grade_table + grades_dict hiện tại.
rows = st.session_state.grade_table
df_init = pd.DataFrame(
    [
        {
            "Mã môn": r["ma_mon"],
            "Tên môn": r["ten_mon"],
            "HK": r["hk_chuan"],
            "Loại": "Bắt buộc" if r["loai"] == "bat_buoc" else "Tự chọn",
            "TC": r["so_tc"],
            "Điểm (0-10)": st.session_state.grades_dict.get(r["ma_mon"], None),
        }
        for r in rows
    ]
)

edited_df = st.data_editor(
    df_init,
    use_container_width=True,
    hide_index=True,
    disabled=["Mã môn", "Tên môn", "HK", "Loại", "TC"],
    column_config={
        "Điểm (0-10)": st.column_config.NumberColumn(
            min_value=0.0,
            max_value=10.0,
            step=0.01,
            format="%g",
            help="Nhập điểm thập phân tự do trong khoảng 0.0–10.0 (vd 7.25, 8.5).",
        ),
    },
    height=min(400, 45 * (len(rows) + 1)),
    key="grade_editor",
)

# Sync editor → session grades_dict (bỏ NaN).
st.session_state.grades_dict = {
    row["Mã môn"]: float(row["Điểm (0-10)"])
    for _, row in edited_df.iterrows()
    if pd.notna(row["Điểm (0-10)"])
}

# Stats nhanh.
n_filled = len(st.session_state.grades_dict)
n_pass = sum(1 for d in st.session_state.grades_dict.values() if d >= 5.0)
cols = st.columns(3)
cols[0].metric("Đã nhập điểm", f"{n_filled}/{len(rows)}")
cols[1].metric("Đạt", n_pass)
cols[2].metric("Không đạt", n_filled - n_pass)

st.divider()

# ===== Chat =====

st.subheader("💬 Hỏi bot")
st.caption(
    "Vd: *Em định hướng AI/ML, chọn môn tự chọn nào?* · "
    "*Số TC tối đa mỗi HK?* · "
    "*Tôi có thể học môn Máy học không?*"
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

meta = st.session_state.last_response_meta
if meta:
    intent = meta.get("intent", "recommend")
    st.caption(f"_Lượt trả lời gần nhất:_ {INTENT_LABEL.get(intent, intent)}")

    if intent == "recommend" and meta.get("recommendations"):
        with st.expander("📋 Bảng môn gợi ý", expanded=True):
            df = pd.DataFrame(meta["recommendations"])[
                ["doc_id", "ten_mon", "ma_mon", "so_tc", "loai", "score"]
            ]
            st.dataframe(df, use_container_width=True, hide_index=True)
        if meta.get("context_docs"):
            with st.expander(
                f"🔍 Pool {len(meta['context_docs'])} môn tự chọn HK{int(hk_target)} bot đã xét"
            ):
                df_ctx = pd.DataFrame(meta["context_docs"])[
                    ["doc_id", "ten_mon", "ma_mon", "so_tc"]
                ]
                st.dataframe(df_ctx, use_container_width=True, hide_index=True)

    if intent == "prereq" and meta.get("target_courses"):
        st.caption(
            f"Bot đã nhận diện môn: **{', '.join(meta['target_courses'])}**"
        )

prompt = st.chat_input("Hỏi bot (vd: 'Em định hướng AI/ML, chọn môn nào?')")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "nganh": nganh,
        "hk_target": int(hk_target),
        "grades": st.session_state.grades_dict,
        "messages": st.session_state.messages,
    }

    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):
            try:
                data = call_chat_v2(payload)
            except requests.HTTPError as e:
                st.error(f"API lỗi {e.response.status_code}: {e.response.text}")
                st.session_state.messages.pop()
                st.stop()
            except requests.RequestException as e:
                st.error(f"Không gọi được API: {e}")
                st.session_state.messages.pop()
                st.stop()
        st.markdown(data["response"])

    st.session_state.messages.append(
        {"role": "assistant", "content": data["response"]}
    )
    st.session_state.last_response_meta = data
    st.rerun()
