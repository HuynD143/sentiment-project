# Sentiment Project — Hướng dẫn chạy dự án

Dự án phân tích cảm xúc (tiếng Anh) cho bình luận từ YouTube và Reddit. Ứng dụng giao diện chạy bằng Streamlit, mô hình TensorFlow (.h5) và tokenizer đã được đóng gói sẵn trong thư mục `notebook/models/`.

## 1) Yêu cầu hệ thống
- Python 3.10 hoặc 3.11 (khuyến nghị 3.11)
- pip, venv (hoặc Conda)
- Internet (nếu muốn crawl bình luận từ YouTube/Reddit)

Tùy chọn:
- Docker (nếu chạy bằng container)

## 2) Tạo môi trường và cài đặt

Unix/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Bạn cũng có thể dùng file hỗ trợ (Windows):
- `setup_env.bat` để tạo môi trường
- `run_ui.bat` để chạy nhanh giao diện

## 3) Cấu hình biến môi trường (.env)
Để crawl bình luận YouTube, bạn cần API key của YouTube Data API v3.

Tạo file `crawl_data/.env` (hoặc `.env` ở thư mục gốc, miễn sao biến đọc được), nội dung ví dụ:

```env
API_KEY=YOUR_YOUTUBE_DATA_API_KEY
```

- Với Reddit: code hiện tại cấu hình sẵn giá trị mẫu trong `crawl_data/crawl_reddit.py`. Bạn có thể thay `client_id`, `client_secret`, `user_agent` của bạn nếu cần.

## 4) Chạy ứng dụng giao diện
Có 2 cách chạy chính:

Cách A — Dùng lệnh tiện ích trong `main.py`:
```bash
# Giao diện chính
python main.py

# Chế độ eval (nếu bạn có trang đánh giá riêng)
python main.py -n eval
```

Cách B — Chạy trực tiếp bằng Streamlit:
```bash
streamlit run ui.py
```

Sau khi chạy, trình duyệt sẽ mở ứng dụng tại địa chỉ dạng:
- http://localhost:8501 (mặc định của Streamlit)

Dockerfile của dự án thiết lập cổng 8502. Khi chạy bằng Docker, ứng dụng sẽ lắng nghe ở 8502 (xem phần Docker bên dưới).

## 5) Sử dụng trên giao diện
- Dán link YouTube hoặc Reddit vào ô nhập
- Chọn chế độ hiển thị: 
  - Tất cả thống kê
  - Thống kê cơ bản
  - Phân bố & Tỷ lệ
  - WordCloud theo cảm xúc
- Ứng dụng sẽ crawl bình luận (nếu là link hợp lệ), chạy mô hình cảm xúc cục bộ và hiển thị kết quả.

Lưu ý: Mặc định mô hình và tokenizer được đọc từ:
- `notebook/models/btlpython3.h5`
- `notebook/models/tokenizer.json`

Hãy đảm bảo các file này tồn tại. Bạn có thể thay đường dẫn trong `ui.py` nếu muốn đổi model.

## 6) Crawl dữ liệu bằng script
- YouTube: `crawl_data/crawl_cmt_from_ytb.py`
  - Ví dụ sử dụng trong Python:
    ```
    from crawl_data.crawl_cmt_from_ytb import Crawler
    c = Crawler(url="https://www.youtube.com/watch?v=XXXXXXXXXXX")
    c.get_youtube_comments()
    # Lưu ra CSV (mặc định ./youtube_comments.csv)
    import asyncio; asyncio.run(c.save_cmt())
    ```
- Reddit: `crawl_data/crawl_reddit.py`
  - Ví dụ sử dụng trong Python:
    ```
    from crawl_data.crawl_reddit import CrawlReddit
    reddit = CrawlReddit()
    comments = reddit.get_comments("https://www.reddit.com/r/.../comments/.../")
    ```

Dữ liệu mẫu có tại thư mục `data/`.

## 7) Chạy bằng Docker (tùy chọn)
Build image:
```bash
docker build -t sentiment-app .
```

Chạy container (mặc định app listen ở 8502 trong container):
```bash
docker run --rm -p 8502:8502 \
  --env-file ./crawl_data/.env \
  sentiment-app
```
Mở trình duyệt: http://localhost:8502

Ghi chú:
- Nếu muốn mount thư mục models/data từ host, thêm `-v` tương ứng.
- Thư mục làm việc trong container là `/app`.

## 8) Cấu trúc thư mục chính
```
.
├─ main.py                 # Điểm vào: chạy UI hoặc eval
├─ ui.py                   # Ứng dụng Streamlit chính
├─ eval.py                 # Trang eval (nếu dùng)
├─ crawl_data/
│  ├─ crawl_cmt_from_ytb.py
│  ├─ crawl_reddit.py
│  └─ .env                 # API_KEY cho YouTube (tự tạo)
├─ notebook/models/        # Mô hình .h5 và tokenizer.json
├─ data/                   # Dữ liệu mẫu
├─ requirements.txt
├─ Dockerfile
├─ run_ui.bat, setup_env.bat (Windows)
└─ backend_api/            # (đang để trống)
```

## 9) Lỗi thường gặp
- Không có model/tokenizer: đảm bảo 2 file `notebook/models/btlpython3.h5` và `notebook/models/tokenizer.json` tồn tại. Có thể chỉnh đường dẫn trong `ui.py` (biến `MODEL_PATH`, `TOKENIZER_PATH`).
- Lỗi YouTube 403/404 khi crawl: kiểm tra `API_KEY` trong `.env` và quota API YouTube Data.
- Lỗi Reddit quyền hạn: thay `client_id`, `client_secret`, `user_agent` hợp lệ trong `crawl_data/crawl_reddit.py`.
- Port bị chiếm: đổi cổng khi chạy Streamlit, ví dụ `streamlit run ui.py --server.port=8503`.
- TensorFlow không tương thích: dùng Python 3.10/3.11 và đúng phiên bản `tensorflow` từ `requirements.txt`.

## 10) Liên hệ / Góp ý
Mọi góp ý vui lòng tạo issue trên repository hoặc liên hệ nhóm phát triển.
# sentiment-project
