# ======================= ui.py (local model + YT/Reddit + 4 chế độ) =======================
import streamlit as st
import time
import matplotlib.pyplot as plt

# begin: imports phụ trợ suy luận & crawl 
import os, sys
import numpy as np
from collections import Counter
from pathlib import Path
from wordcloud import WordCloud
from dotenv import load_dotenv; load_dotenv()

# Paths & imports cho cấu trúc phẳng (không còn frontend/backend) ====
ROOT = Path(__file__).resolve().parent       # .../sentiment-project
sys.path.append(str(ROOT))                   # để import được crawl_data/*

# Crawler modules nằm ngay trong folder crawl_data/
try:
    from crawl_data.crawl_cmt_from_ytb import Crawler
    from crawl_data.crawl_reddit import CrawlReddit
    _CRAWLER_OK = True
except Exception:
    _CRAWLER_OK = False

# begin: load tokenizer + model .h5 (cache 1 lần)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Đường dẫn model/tokenizer trong notebook/models/
MODEL_PATH     = ROOT / "notebook" / "models" / "btlpython3.h5"
TOKENIZER_PATH = ROOT / "notebook" / "models" / "tokenizer.json"
MAX_LEN = 30  # khớp lúc train

@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not TOKENIZER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model/tokenizer:\n- {MODEL_PATH}\n- {TOKENIZER_PATH}\n"
            "Hãy kiểm tra lại đường dẫn!"
        )
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())
    mdl = load_model(MODEL_PATH)
    return tok, mdl

def predict_local(comments):
    tok, mdl = load_artifacts()
    seq = tok.texts_to_sequences(comments)
    pad = pad_sequences(seq, padding="post", truncating="post", maxlen=MAX_LEN)
    proba = mdl.predict(pad, verbose=0)
    idx = np.argmax(proba, axis=1)
    idx2vi = {0: "Tiêu cực", 1: "Trung lập", 2: "Tích cực"}  # theo thứ tự training của bạn
    return [idx2vi[int(i)] for i in idx]
# --- end ---

# --- Cấu hình trang ---
st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊", layout="centered")

# --- CSS tuỳ chỉnh ---
st.markdown("""
<style>
body {background: linear-gradient(135deg, #f8f9fa, #e3f2fd);}
.stTextArea textarea, .stTextInput input {
    border: 2px solid #4e9af1 !important; border-radius: 10px !important; font-size: 16px !important;
}
.stButton>button {
    background-color: #4e9af1; color: white; border-radius: 8px; height: 3em; width: 100%;
    font-size: 18px; font-weight: 600; transition: 0.3s;
}
.stButton>button:hover {background-color: #1976d2; transform: scale(1.03);}
.result-box {border-radius: 12px; padding: 1.5em; text-align: center; font-size: 20px; font-weight: 600; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4727/4727425.png", width=80)
st.sidebar.title("🎓 Nhóm 10 - ML Project")
st.sidebar.write("**Ứng dụng Phân tích cảm xúc từ Nhiều Nền Tảng** 🤖  \nFrontend hiển thị kết quả cảm xúc và thống kê trực quan.")
st.sidebar.markdown("---")
st.sidebar.info("💡 Chỉ hỗ trợ YouTube / Reddit. Mô hình chạy local (không cần API).")

# --- Tiêu đề ---
st.title("🧠 Phân tích cảm xúc bình luận đa nền tảng")

# --- Nhập link ---
link = st.text_input(f"🔗 Dán link Youtube hoặc Reddit tại đây:", placeholder="Ví dụ: https://www.youtube.com/watch?v=... hoặc https://www.reddit.com/r/...")

# --- Radio 4 chế độ ---
option = st.radio(
    "📊 Chọn chức năng bạn muốn hiển thị:",
    ["Tất cả thống kê", "Thống kê cơ bản", "Phân bố & Tỷ lệ", "WordCloud theo cảm xúc"],
    horizontal=True
)

# --- các hàm vẽ từng cụm ---
def draw_basic_stats(content, authors, labels):
    comments_by = {"Tích cực": [], "Trung lập": [], "Tiêu cực": []}
    for c, lab in zip(content, labels):
        comments_by[lab].append(c)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # 1) Count
    counts = {k: len(v) for k, v in comments_by.items()}
    axes[0].bar(counts.keys(), counts.values(), color=["#66BB6A", "#BDBDBD", "#EF5350"])
    axes[0].set_title("Số lượng comment theo cảm xúc"); axes[0].set_ylabel("Số lượng")

    # 2) Avg length
    avg_len = {k: (np.mean([len(x) for x in v]) if v else 0) for k, v in comments_by.items()}
    axes[1].bar(avg_len.keys(), avg_len.values(), color=["#66BB6A", "#BDBDBD", "#EF5350"])
    axes[1].set_title("Độ dài bình luận trung bình"); axes[1].set_ylabel("Ký tự")

    # 3) Top 5 tác giả
    from collections import Counter
    top_authors = Counter(authors).most_common(5)
    names, values = zip(*top_authors) if top_authors else ([], [])
    axes[2].bar(names, values, color="#42A5F5")
    axes[2].set_title("Top 5 tác giả bình luận"); axes[2].set_ylabel("Số comment")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

def draw_distribution_and_share(content, authors, labels):
    comments_by = {"Tích cực": [], "Trung lập": [], "Tiêu cực": []}
    for c, lab in zip(content, labels):
        comments_by[lab].append(c)

    col1, col2, col3 = st.columns(3)

    with col1:
        fig2, ax2 = plt.subplots(figsize=(5.8, 5))
        sizes = [len(v) for v in comments_by.values()]
        ax2.pie(sizes, labels=list(comments_by.keys()), autopct="%1.1f%%",
                colors=["#66BB6A", "#BDBDBD", "#EF5350"], startangle=90, textprops={"fontsize": 10})
        ax2.set_title("Tỷ lệ cảm xúc"); st.pyplot(fig2); plt.close(fig2)

    with col2:
        fig3, ax3 = plt.subplots(figsize=(5.8, 5.2))
        for name, color in zip(["Tích cực", "Trung lập", "Tiêu cực"], ["#66BB6A", "#BDBDBD", "#EF5350"]):
            lengths = [len(c) for c in comments_by[name]]
            if lengths:
                ax3.hist(lengths, bins=10, alpha=0.55, label=name, color=color)
        ax3.set_xlabel("Độ dài (ký tự)"); ax3.set_ylabel("Số lượng"); ax3.legend()
        ax3.set_title("Phân bố độ dài bình luận"); st.pyplot(fig3); plt.close(fig3)

    with col3:
        from collections import Counter
        top = Counter(authors).most_common(5)
        names = [x[0] for x in top]
        pos_vals, neu_vals, neg_vals = [], [], []
        for name in names:
            pos = sum(1 for a, lab in zip(authors, labels) if a == name and lab == "Tích cực")
            neu = sum(1 for a, lab in zip(authors, labels) if a == name and lab == "Trung lập")
            neg = sum(1 for a, lab in zip(authors, labels) if a == name and lab == "Tiêu cực")
            pos_vals.append(pos); neu_vals.append(neu); neg_vals.append(neg)
        fig4, ax4 = plt.subplots(figsize=(5.8, 5))
        ax4.bar(names, pos_vals, label="Tích cực", color="#66BB6A")
        ax4.bar(names, neu_vals, bottom=pos_vals, label="Trung lập", color="#BDBDBD")
        ax4.bar(names, neg_vals, bottom=[i + j for i, j in zip(pos_vals, neu_vals)], label="Tiêu cực", color="#EF5350")
        ax4.set_ylabel("Số comment"); ax4.set_title("Top 5 tác giả theo nhãn")
        plt.xticks(rotation=30, ha="right"); ax4.legend()
        st.pyplot(fig4); plt.close(fig4)

def draw_wordclouds(content, labels):
    comments_by = {"Tích cực": [], "Trung lập": [], "Tiêu cực": []}
    for c, lab in zip(content, labels):
        comments_by[lab].append(c)

    c1, c2, c3 = st.columns(3)
    for (name, cmap, col) in [("Tích cực", "Greens", c1), ("Trung lập", "Greys", c2), ("Tiêu cực", "Reds", c3)]:
        with col:
            text = " ".join(comments_by[name])
            if text.strip():
                wc = WordCloud(width=420, height=300, background_color="white", colormap=cmap).generate(text)
                fig, ax = plt.subplots(figsize=(6.5, 4))
                ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); ax.set_title(f"WordCloud: {name}")
                st.pyplot(fig); plt.close(fig)
            else:
                st.info(f"Không có comment {name}.")
# ---------------------------------------------------------------

# --- Nút chính ---
if st.button("🚀 Bắt đầu phân tích"):
    if not link.strip():
        st.warning("⚠️ Vui lòng dán link trước khi phân tích.")
    else:
        with st.spinner("🔄 Đang tải và phân tích bình luận từ link..."):
            time.sleep(0.5)

            content, authors = [], []
            link_lc = link.lower()
            is_yt = ("youtube.com" in link_lc) or ("youtu.be" in link_lc)
            is_reddit = "reddit.com" in link_lc

            if not (is_yt or is_reddit):
                st.error("Chỉ hỗ trợ YouTube hoặc Reddit. Vui lòng kiểm tra link.")
                st.stop()

            if _CRAWLER_OK:
                try:
                    if is_yt:
                        cmts = Crawler(link)
                        cmts.get_youtube_comments()
                        content = [c.get("text", "") for c in cmts.comments if isinstance(c, dict)]
                        authors = [c.get("author", "Unknown") for c in cmts.comments if isinstance(c, dict)]
                    else:
                        cr = CrawlReddit()
                        rows = cr.get_comments(link)  # list (author, text)
                        for a, t in rows:
                            authors.append(a or "Unknown")
                            content.append(t or "")
                except Exception as e:
                    st.error(f"Lỗi crawl: {e}")
                    content, authors = [], []
            else:
                st.warning("Không import được crawler từ dự án. Bạn có thể bật crawler (cài lib & .env) hoặc dán comment thủ công ở bản demo khác.")
                # vẫn chặn nếu rỗng
            if not content:
                st.warning("Không có comment hợp lệ để phân tích! (Thiếu API key hoặc link không hợp lệ)")
                st.stop()

            # Suy luận local bằng model .h5
            try:
                labels = predict_local(content)  # ["Tích cực" / "Trung lập" / "Tiêu cực"]
            except Exception as e:
                st.error(f"Lỗi suy luận model local: {e}")
                st.stop()

            # Hiển thị theo 4 chế độ
            if option == "Tất cả thống kê":
                st.success(f"💬 Đã phân tích {len(content)} bình luận từ link.")
                draw_basic_stats(content, authors, labels)
                st.markdown("---")
                draw_distribution_and_share(content, authors, labels)
                st.markdown("---")
                st.subheader("WordCloud theo cảm xúc")
                draw_wordclouds(content, labels)

            elif option == "Thống kê cơ bản":
                st.success(f"💬 {len(content)} bình luận · Thống kê cơ bản")
                draw_basic_stats(content, authors, labels)

            elif option == "Phân bố & Tỷ lệ":
                st.success(f"💬 {len(content)} bình luận · Phân bố & Tỷ lệ")
                draw_distribution_and_share(content, authors, labels)

            elif option == "WordCloud theo cảm xúc":
                st.success(f"💬 {len(content)} bình luận · WordCloud")
                draw_wordclouds(content, labels)

# --- Footer ---
st.markdown("---")
st.markdown("🧩 *Dự án Machine Learning - Phân tích cảm xúc bình luận đa nền tảng Nhóm 10 (2025)*")
# ======================
