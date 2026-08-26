# Tổng quan kỹ thuật — Sentiment Analysis cho bình luận YouTube & Reddit

Tài liệu này mô tả toàn bộ kỹ thuật đã dùng trong dự án, tập trung vào phần **NLP và mô hình**, kèm theo số liệu thật lấy từ `notebook/scripts/train_model.ipynb`. Mục tiêu: dùng làm tài liệu chuẩn bị phỏng vấn — bao gồm cả những điểm làm tốt lẫn những điểm còn hạn chế và cách khắc phục.

---

## 1. Bài toán và kiến trúc tổng thể

**Bài toán:** Phân loại cảm xúc 3 lớp (negative / neutral / positive) cho bình luận tiếng Anh, sau đó tổng hợp thành thống kê trực quan cho một video YouTube hoặc một post Reddit bất kỳ.

Đây không phải bài toán "train một model" đơn thuần, mà là **một pipeline end-to-end**: từ thu thập dữ liệu thô ngoài đời thực → tiền xử lý → suy luận → trực quan hóa → đóng gói triển khai.

```
URL người dùng dán vào
        │
        ▼
┌───────────────────┐
│  Định tuyến URL   │  youtube.com / youtu.be → YouTube Data API v3
│  (ui.py)          │  reddit.com            → PRAW
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Crawl bình luận  │  → chuẩn hóa về 2 list song song:
│  (crawl_data/)    │     content[] (text) + authors[] (tên tác giả)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  NLP inference    │  tokenizer.json → texts_to_sequences → pad_sequences
│  (predict_local)  │  → model.predict → argmax → nhãn
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Trực quan hóa    │  matplotlib: bar / pie / histogram / stacked bar
│  (draw_* funcs)   │  wordcloud: 3 wordcloud theo từng nhãn
└───────────────────┘
```

**Tầng triển khai:** Streamlit (web UI thuần Python), đóng gói bằng Docker (`python:3.11-slim`, cổng 8502).

---

## 2. Dữ liệu

### 2.1. Dữ liệu huấn luyện

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | HuggingFace Hub: `Sp1786/multiclass-sentiment-analysis-dataset` |
| Cách load | `pd.read_csv("hf://datasets/...")` — đọc trực tiếp từ Hub, không cần tải thủ công |
| Split | train / validation / test có sẵn trong dataset |
| Kích thước train | ~31,232 mẫu (suy ra từ 244 steps × batch 128) |
| Kích thước tập đánh giá | 5,205 mẫu |
| Nhãn | `0 = negative`, `1 = neutral`, `2 = positive` (số nguyên, không phải one-hot) |

**Phân bố nhãn tập đánh giá:** negative 1,517 — neutral 1,928 — positive 1,760. Dữ liệu **tương đối cân bằng**, hơi lệch về neutral. Đây là lý do `accuracy` vẫn là một metric có ý nghĩa ở đây (nếu lệch nặng thì phải dùng macro-F1 làm metric chính).

### 2.2. Dữ liệu thực tế (inference)

Khác hoàn toàn với dữ liệu train — đây là điểm quan trọng nhất về mặt kỹ thuật:

- Bình luận YouTube/Reddit có emoji, viết tắt, tiếng lóng, ngôn ngữ hỗn hợp, spam, link.
- Model được train trên tweet/review tiếng Anh đã tương đối sạch.
- → **Domain shift**: hiệu năng thực tế sẽ thấp hơn con số trên tập đánh giá.

### 2.3. Lưu ý về thư mục `data/`

`data/train.csv` và `data/test.csv` **không** được dùng để train model hiện tại. Đó là bộ tweet của Kaggle (có cột `sentiment` dạng chuỗi + metadata quốc gia), còn sót lại từ giai đoạn thử nghiệm ban đầu. `data/cmt_ytb.csv` là output mẫu của crawler.

---

## 3. Kỹ thuật NLP — tiền xử lý văn bản

Đây là phần cần nói kỹ nhất trong phỏng vấn, vì nó thể hiện hiểu biết về cách máy tính "đọc" được chữ.

### 3.1. Tokenization (từ → số)

```python
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(X)
```

`keras.preprocessing.text.Tokenizer` thực hiện các bước sau khi `fit_on_texts`:

1. **Lowercase** toàn bộ text (mặc định `lower=True`).
2. **Loại bỏ dấu câu** theo bộ filter mặc định `!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n`.
3. **Tách token bằng khoảng trắng** (whitespace tokenization).
4. **Xây từ điển** `word_index`, sắp xếp theo **tần suất giảm dần** — từ phổ biến nhất có index 1, hiếm nhất có index lớn nhất.

**Kết quả:** `vocab_size = 29,766` từ.

**`oov_token='<OOV>'` là quyết định thiết kế quan trọng.** Nếu không có nó, mọi từ chưa từng thấy khi train sẽ bị **loại bỏ im lặng** khỏi câu lúc inference. Với dữ liệu YouTube/Reddit đầy tiếng lóng, một câu có thể mất gần hết từ mà không báo lỗi gì. Có `<OOV>`, những từ lạ được ánh xạ về một index chung — model ít nhất biết "có một từ ở đây mà tôi không biết", giữ được cấu trúc và độ dài câu.

### 3.2. Sequencing và Padding

```python
seq = tokenizer.texts_to_sequences(X)
padding = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
```

Mạng nơ-ron cần input có **shape cố định**, nhưng câu thì dài ngắn khác nhau → phải chuẩn hóa về `maxlen`.

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `maxlen` | 100 | Độ dài chuẩn của mọi chuỗi |
| `padding='post'` | Chèn số 0 **vào cuối** | Câu ngắn được đệm phía sau |
| `truncating='post'` | Cắt **phần đuôi** | Câu dài bị bỏ phần cuối |

**Vì sao chọn `post` chứ không phải `pre`?** Với LSTM một chiều, `pre-padding` thường tốt hơn (số 0 nằm ở đầu, thông tin thật nằm gần output cuối cùng nên ít bị "quên"). Nhưng ở đây dùng **Bidirectional** LSTM — có một chiều đọc ngược lại — nên vị trí padding ít ảnh hưởng hơn nhiều. Đây là một trade-off nên chủ động nêu ra khi được hỏi.

**Ví dụ thật từ notebook** — câu 4 từ được pad thành vector 100 chiều:
```
[2677, 8640, 5778, 1286, 0, 0, 0, ... , 0]
 └──── 4 token thật ────┘ └── 96 số 0 padding ──┘
```

### 3.3. Vấn đề `mask_zero` — điểm hạn chế đã nhận diện

Model cuối cùng dùng `Embedding(..., mask_zero=False)`. Nghĩa là **LSTM vẫn xử lý cả 96 bước padding** như thể chúng là từ thật. Về lý thuyết nên bật `mask_zero=True` để Keras tự truyền masking xuống LSTM, giúp mạng bỏ qua các bước padding.

Thực tế mạng vẫn học được (nó tự học rằng embedding của index 0 không mang thông tin), nhưng đây là chỗ có thể cải thiện — vừa tăng độ chính xác vừa tiết kiệm tính toán.

---

## 4. Kiến trúc mô hình

```python
model = tf.keras.Sequential([
    Embedding(input_dim=29766, output_dim=100, input_length=100, mask_zero=False),
    Bidirectional(LSTM(128, return_sequences=False)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
```

### 4.1. Bảng tham số thật

| Layer | Output Shape | Param # | Vai trò |
|---|---|---|---|
| `Embedding` | (None, 100, 100) | **2,976,600** | Mỗi từ → vector 100 chiều học được |
| `Bidirectional(LSTM 128)` | (None, 256) | **234,496** | Đọc câu 2 chiều, nén thành 1 vector ngữ nghĩa |
| `Dense(32, relu)` | (None, 32) | **8,224** | Tầng ẩn phi tuyến |
| `Dense(3, softmax)` | (None, 3) | **99** | Phân phối xác suất trên 3 lớp |
| | **Tổng** | **3,219,419** (12.28 MB) | |

**Quan sát đáng nói:** hơn **92% tham số nằm ở tầng Embedding** (2.97M / 3.22M). Điều này giải thích vì sao file `.h5` nặng ~38 MB dù model "chỉ" có 3.2 triệu tham số, và vì sao overfitting xảy ra — phần lớn năng lực học của mạng dồn vào việc ghi nhớ từ vựng.

### 4.2. Giải thích từng tầng

**Embedding layer.** Chuyển mỗi index từ thành một vector dày (dense) 100 chiều. Khác với one-hot encoding (29,766 chiều, thưa, mọi từ cách đều nhau), embedding học được **quan hệ ngữ nghĩa** — "good" và "great" hội tụ về vùng gần nhau trong không gian vector vì chúng xuất hiện trong ngữ cảnh tương tự. Ở đây embedding được **học từ đầu (trained from scratch)** cùng với model, không dùng pretrained (GloVe/Word2Vec).

**Bidirectional LSTM.** LSTM là mạng hồi quy có cơ chế **cổng (gate)** — forget gate, input gate, output gate — giúp giữ thông tin qua nhiều bước thời gian và giải quyết vấn đề *vanishing gradient* của RNN thuần.

`Bidirectional` chạy **hai** LSTM song song: một đọc câu từ trái sang phải, một từ phải sang trái, rồi nối (concatenate) hai trạng thái cuối → `128 × 2 = 256` chiều.

Vì sao 2 chiều quan trọng với sentiment? Ví dụ câu *"The movie was great, until the ending ruined everything"*. LSTM một chiều đọc tới "great" sẽ nghiêng về positive và có thể bị ảnh hưởng bởi thứ tự. Chiều ngược lại bắt được "ruined everything" sớm, giúp mô hình cân bằng ngữ cảnh hai đầu.

`return_sequences=False` → chỉ lấy trạng thái ẩn cuối cùng, tức là **một vector 256 chiều tóm tắt toàn câu**, thay vì output cho từng token (dùng cho bài toán gán nhãn chuỗi như NER).

**Dense + Softmax.** `Dense(32, relu)` học tổ hợp phi tuyến của các đặc trưng. `Dense(3, softmax)` biến logits thành phân phối xác suất tổng bằng 1 trên 3 lớp.

### 4.3. Kiến trúc Transformer đã thử và loại bỏ

Trong notebook có hai class được **tự viết tay** rồi comment lại:

```python
class PositionEncode(Layer):   # Token embedding + learned positional embedding
class EncodeBlock(Layer):      # MultiHeadAttention → Add & Norm → FFN → Add & Norm
```

`EncodeBlock` là một **Transformer encoder block đầy đủ**: self-attention đa đầu, hai kết nối residual, hai lớp LayerNormalization, và feed-forward network 2 tầng — đúng theo kiến trúc trong *Attention Is All You Need*.

Đây là điểm mạnh nên nêu trong phỏng vấn: **đã tự cài đặt Transformer từ đầu**, hiểu vì sao cần positional encoding (self-attention không có khái niệm thứ tự, khác LSTM vốn tuần tự theo bản chất), và vì sao cần residual + LayerNorm (giúp gradient chảy qua mạng sâu). Cuối cùng chọn BiLSTM vì với ~31K mẫu, Transformer train từ đầu dễ overfit và cần nhiều dữ liệu hơn để phát huy.

---

## 5. Quá trình huấn luyện

| Cấu hình | Giá trị | Lý do |
|---|---|---|
| Optimizer | `AdamW(learning_rate=0.0001)` | AdamW tách **weight decay** khỏi bước cập nhật gradient (khác Adam gộp chung vào L2 loss) → regularize đúng hơn |
| Loss | `sparse_categorical_crossentropy` | Nhãn ở dạng số nguyên `0/1/2`, **không cần one-hot** — tiết kiệm bộ nhớ so với `categorical_crossentropy` |
| Epochs | 7 | |
| Batch size | 128 | 244 steps/epoch |
| Metric | `accuracy` | |
| Phần cứng | Tesla P100 16GB | ~6 giây/epoch |

### 5.1. Đường cong huấn luyện (số liệu thật)

| Epoch | train_acc | train_loss | val_acc | val_loss |
|---|---|---|---|---|
| 1 | 0.3805 | 1.0905 | 0.5016 | 1.0189 |
| 2 | 0.5292 | 0.9445 | 0.5958 | 0.8496 |
| 3 | 0.6394 | 0.7760 | 0.6363 | 0.7890 |
| 4 | 0.7034 | 0.6890 | 0.6636 | 0.7642 |
| **5** | 0.7540 | 0.6192 | **0.6699** | **0.7578** ← thấp nhất |
| 6 | 0.7812 | 0.5591 | 0.6755 | 0.7658 ↑ |
| 7 | 0.8038 | 0.5100 | 0.6734 | 0.7788 ↑ |

**Phân tích — đây là phần thể hiện tư duy ML rõ nhất:**

- Epoch 1 accuracy 0.38 ≈ mức đoán ngẫu nhiên (1/3). Mạng khởi động từ con số 0.
- Từ epoch 5 trở đi: **train_loss tiếp tục giảm (0.62 → 0.51) nhưng val_loss bắt đầu tăng (0.7578 → 0.7788)**. Đây là dấu hiệu kinh điển của **overfitting** — model chuyển từ *học quy luật tổng quát* sang *ghi nhớ tập train*.
- Khoảng cách cuối cùng: train 0.80 vs val 0.67 → **gap 13 điểm phần trăm**.
- **Điểm dừng tối ưu là epoch 5.** Nếu có `EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)` thì đã tự động dừng và lấy lại trọng số tốt nhất.

### 5.2. Regularization — thiếu sót đã nhận diện

Trong notebook, `Dropout(0.2)` có mặt nhưng bị comment. Model cuối **không có regularization nào ngoài weight decay của AdamW**. Với tầng Embedding chiếm 92% tham số, đây chính là nguyên nhân trực tiếp của overfitting ở mục 5.1.

---

## 6. Đánh giá mô hình

### 6.1. Classification report (5,205 mẫu)

```
              precision    recall  f1-score   support

  0 negative     0.66      0.70      0.68      1517
  1 neutral      0.60      0.61      0.61      1928
  2 positive     0.77      0.72      0.74      1760

    accuracy                         0.67      5205
   macro avg      0.68      0.68      0.68      5205
weighted avg      0.68      0.67      0.67      5205
```

### 6.2. Đọc kết quả

**Neutral là lớp khó nhất (F1 = 0.61).** Điều này hợp lý về mặt ngôn ngữ học: negative và positive có **từ khóa tín hiệu mạnh** ("terrible", "amazing"), còn neutral được định nghĩa bằng sự *vắng mặt* của tín hiệu cảm xúc — khó hơn nhiều cho model dựa trên từ vựng. Neutral cũng là lớp mà **con người gán nhãn kém nhất quán nhất**, nên bản thân nhãn đã có nhiễu.

**Positive là lớp tốt nhất (F1 = 0.74)**, với precision 0.77 — khi model nói "positive" thì thường đúng.

**Baseline để so sánh:** đoán ngẫu nhiên = 33%, luôn đoán lớp đông nhất (neutral) = 1928/5205 = **37%**. Model đạt **67%**, tức **cao hơn baseline khoảng 30 điểm phần trăm**. Đây là cách trình bày kết quả thuyết phục hơn nhiều so với chỉ nói "accuracy 67%".

### 6.3. Lỗi trong quy trình đánh giá — cần trung thực khi trình bày

Trong notebook, cell tạo tập test có lỗi:

```python
testx, testy = df_val['text'], df_val['label']
testx_seq = tokenizer.texts_to_sequences(valx)   # ← dùng valx, không phải testx
```

Cộng với việc `df_val` được load lại từ split **validation**, kết quả là: **classification report ở trên được tính trên tập validation, không phải tập test độc lập.**

Tập validation này đã được dùng để theo dõi trong lúc train, nên con số 0.67 **có khả năng lạc quan hơn** hiệu năng thật trên dữ liệu chưa từng thấy. Cách sửa: dùng đúng `df_test` (đã được load sẵn ở cell đầu nhưng chưa dùng đến).

Nêu chủ động lỗi này trong phỏng vấn là một điểm cộng lớn — nó cho thấy hiểu **vì sao** cần tách 3 tập train/val/test, chứ không chỉ làm theo công thức.

---

## 7. Triển khai (Deployment)

### 7.1. Đồng bộ tokenizer giữa train và serve

Đây là kỹ thuật MLOps cốt lõi của dự án. Model chỉ hiểu các **chỉ số từ** đúng theo từ điển lúc train. Nếu lúc serve tạo tokenizer mới, mọi index sẽ lệch và dự đoán trở thành vô nghĩa — **mà không hề báo lỗi**.

Giải pháp: serialize tokenizer cùng với model.

```python
# Lúc train
with open("../models/tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())

# Lúc serve (ui.py)
tok = tokenizer_from_json(open(TOKENIZER_PATH, encoding="utf-8").read())
```

File `tokenizer.json` nặng ~2.7 MB — chính là từ điển 29,766 từ.

**Nguyên tắc rút ra:** *artifact của một model NLP không chỉ là file trọng số, mà là cặp (weights + tokenizer) — chúng phải được version cùng nhau.*

### 7.2. Cache model bằng `@st.cache_resource`

```python
@st.cache_resource(show_spinner=False)
def load_artifacts():
    ...
    return tok, mdl
```

Streamlit **chạy lại toàn bộ script từ đầu mỗi lần người dùng tương tác**. Không có cache, model 38 MB sẽ được load lại từ đĩa sau mỗi cú click — mất vài giây mỗi lần.

`@st.cache_resource` giữ object trong bộ nhớ, dùng chung cho mọi session. Đây là decorator đúng cho **tài nguyên không serialize được** (model, kết nối DB); `@st.cache_data` là cho dữ liệu có thể copy được (DataFrame). Phân biệt được hai cái này là câu hỏi phỏng vấn Streamlit rất hay gặp.

### 7.3. Batch inference

```python
proba = mdl.predict(pad, verbose=0)      # (N, 3) — toàn bộ N bình luận trong 1 lần gọi
idx = np.argmax(proba, axis=1)           # vectorized argmax theo hàng
```

Toàn bộ bình luận được đưa vào **một lần gọi `predict` duy nhất**, không lặp từng câu. TensorFlow tận dụng phép nhân ma trận trên toàn batch — nhanh hơn vòng lặp Python hàng chục lần.

### 7.4. Docker

```dockerfile
FROM python:3.11-slim
RUN apt-get install -y build-essential libglib2.0-0 libsm6 ... libfreetype6 libpng16-16
COPY requirements.txt ./ && RUN pip install -r requirements.txt --no-cache-dir
COPY . .
CMD ["streamlit", "run", "ui.py", "--server.address=0.0.0.0", "--server.port=8502"]
```

Hai chi tiết đáng nói:
- **Thứ tự COPY**: copy `requirements.txt` và cài đặt **trước** khi copy source code → tận dụng **Docker layer cache**, sửa code không phải cài lại toàn bộ thư viện.
- **`--server.address=0.0.0.0`**: bắt buộc, vì mặc định Streamlit chỉ bind `localhost` — trong container thì port mapping ra host sẽ không hoạt động.
- Các thư viện `libfreetype6`, `libpng16-16` là **system dependency của matplotlib/wordcloud**, không cài được qua pip.

---

## 8. Kỹ thuật thu thập dữ liệu

### 8.1. YouTube Data API v3

```python
request = self.youtube.commentThreads().list(
    part='snippet', videoId=self.VIDEO_ID, maxResults=100,
    pageToken=next_page_token, textFormat='plainText'
)
```

- **Cursor-based pagination**: API trả tối đa 100 kết quả/lần kèm `nextPageToken`. Vòng `while True` lặp cho tới khi không còn token → lấy được **toàn bộ** bình luận chứ không chỉ 100 cái đầu.
- **Xử lý lỗi phân biệt theo HTTP status**: `403` (hết quota hoặc comment bị tắt) và `404` (video không tồn tại) được bắt riêng — không gộp chung thành một `except Exception`.
- **Trích video ID bằng regex đa dạng format** — hỗ trợ 3 dạng URL:
  ```python
  r'(?<=v=)[a-zA-Z0-9_-]{11}'          # youtube.com/watch?v=  (lookbehind)
  r'youtu\.be/([a-zA-Z0-9_-]{11})'     # link rút gọn
  r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})'
  ```
  Dùng **positive lookbehind** `(?<=v=)` cho pattern đầu để match ID mà không nuốt tiền tố, và xử lý phân biệt `group()` vs `group(1)` qua `match.lastindex`.
- **API key nạp từ `.env`** qua `python-dotenv`, `cache_discovery=False` để tránh cảnh báo cache của thư viện Google.

### 8.2. Reddit qua PRAW

```python
self.submission.comments.replace_more(limit=0)
comments = [(c.author.name if c.author else "[deleted]", c.body)
            for c in self.submission.comments.list()]
```

- **`replace_more(limit=0)`** là mấu chốt: cây bình luận Reddit chứa các node "load more comments" giả. `limit=0` **loại bỏ** các node đó thay vì gọi thêm API để mở rộng — đánh đổi độ phủ lấy tốc độ và số lần gọi API.
- **`.list()`** làm phẳng cây bình luận lồng nhau thành danh sách một chiều.
- **Xử lý tài khoản đã xóa**: `c.author` trả về `None` với comment của user đã xóa → phải kiểm tra trước khi `.name`, nếu không sẽ `AttributeError`.

### 8.3. Adapter pattern

Hai nguồn trả về cấu trúc khác nhau — YouTube trả `dict`, Reddit trả `tuple` — nhưng `ui.py` chuẩn hóa cả hai về cùng một dạng `(content[], authors[])` trước khi vào pipeline NLP. Nhờ đó phần inference và visualization **không cần biết dữ liệu đến từ đâu**, và thêm nguồn mới (TikTok, Facebook…) chỉ cần viết thêm một adapter.

---

## 9. Trực quan hóa

Ba nhóm biểu đồ, chọn qua radio button:

| Chức năng | Biểu đồ | Kỹ thuật |
|---|---|---|
| Thống kê cơ bản | Số lượng theo nhãn, độ dài TB, top 5 tác giả | `plt.subplots(1, 3)`, `collections.Counter.most_common(5)` |
| Phân bố & Tỷ lệ | Pie chart tỷ lệ, histogram độ dài, stacked bar top tác giả | `ax.pie(autopct)`, `ax.hist(alpha=0.55)` để chồng 3 phân phối, `bottom=` để xếp chồng cột |
| WordCloud | 3 wordcloud riêng cho từng nhãn | `WordCloud(colormap=...)` với `Greens/Greys/Reds` |

**Chi tiết đáng nói:** `plt.close(fig)` được gọi sau mỗi `st.pyplot(fig)`. Không làm vậy, matplotlib giữ figure trong bộ nhớ toàn cục và **rò rỉ bộ nhớ** sau nhiều lần rerun của Streamlit.

**Bảng màu nhất quán** — xanh `#66BB6A` / xám `#BDBDBD` / đỏ `#EF5350` cho positive/neutral/negative, dùng thống nhất qua mọi biểu đồ để người xem không phải học lại quy ước.

---

## 10. Những hạn chế đã biết và hướng cải thiện

Trình bày được phần này quan trọng ngang phần "làm được gì" — nó cho thấy khả năng đánh giá phê phán chính công việc của mình.

| # | Hạn chế | Ảnh hưởng | Cách khắc phục |
|---|---|---|---|
| 1 | **Train/serve skew về `maxlen`**: notebook pad `maxlen=100`, nhưng `ui.py`/`eval.py` pad `MAX_LEN=30` | Không báo lỗi (BiLSTM chấp nhận độ dài biến thiên) nhưng model đang suy luận ở độ dài chưa từng train → giảm chính xác thầm lặng | Đồng bộ về 100, hoặc **lưu `maxlen` vào file config cạnh model** để không thể lệch |
| 2 | Không có `EarlyStopping`, không có `Dropout` | Overfitting từ epoch 5 (mục 5.1) | `EarlyStopping(patience=2, restore_best_weights=True)` + `Dropout(0.3)` sau LSTM + `recurrent_dropout` |
| 3 | Đánh giá trên tập validation thay vì test (mục 6.3) | Con số 0.67 có thể lạc quan | Dùng `df_test` đã load sẵn |
| 4 | Embedding train từ đầu | 92% tham số phải học từ 31K mẫu — quá ít | Dùng **GloVe/FastText pretrained**, hoặc fine-tune **DistilBERT** (thường đạt 85–90% cho bài toán này) |
| 5 | Không có tiền xử lý riêng cho comment mạng xã hội | Emoji, URL, `@mention`, ký tự lặp ("sooo goood") bị bỏ hoặc tách sai | Thêm bước clean: chuẩn hóa ký tự lặp, chuyển emoji thành text, strip URL. **Lưu ý: emoji là tín hiệu cảm xúc mạnh — bỏ đi là mất thông tin quý** |
| 6 | Whitespace tokenization | Không xử lý được từ chưa thấy ngoài `<OOV>` | **Subword tokenization** (BPE/WordPiece) — chia từ lạ thành mảnh đã biết, giảm mạnh tỷ lệ OOV |
| 7 | Không xử lý phủ định | "not good" bị hiểu qua từng từ rời | BiLSTM có bắt được một phần nhờ ngữ cảnh, nhưng attention/transformer xử lý tốt hơn hẳn |
| 8 | Reddit credentials hardcode trong source | Rủi ro bảo mật, không đổi được theo môi trường | Đưa vào biến môi trường như YouTube API key |
| 9 | Không có confidence threshold | Dự đoán 0.34/0.33/0.33 được coi ngang với 0.99 | Hiện thêm `max(proba)`, hoặc gắn nhãn "không chắc chắn" khi dưới ngưỡng |
| 10 | Không có unit test | Không tự tin khi refactor | Test cho regex trích video ID, cho `predict_local` với input đã biết trước |

---

## 11. Tóm tắt "tôi học được gì" — dùng để trả lời phỏng vấn

**Về NLP:**
- Toàn bộ chuỗi biến đổi từ **văn bản thô → số → vector ngữ nghĩa**: tokenization, xây từ điển theo tần suất, xử lý OOV, padding/truncating, embedding.
- Vì sao **embedding tốt hơn one-hot**, và vì sao tầng embedding lại chiếm phần lớn tham số của model NLP nhỏ.
- Cơ chế **LSTM và gating** giải quyết vanishing gradient của RNN; vì sao **Bidirectional** có ích cho phân loại câu nhưng vô dụng cho bài toán sinh văn bản thời gian thực.
- Tự cài đặt **Transformer encoder block** (multi-head attention + residual + LayerNorm + positional encoding) và hiểu vì sao nó cần positional encoding còn LSTM thì không.

**Về Machine Learning:**
- **Đọc đường cong train/val để phát hiện overfitting** — biết rằng dấu hiệu là *val_loss tăng trong khi train_loss giảm*, chứ không phải nhìn accuracy.
- Chọn metric phù hợp: vì sao **macro-F1 và so sánh với baseline** có ý nghĩa hơn accuracy trần trụi.
- Phân tích lỗi theo lớp: hiểu **vì sao neutral khó** về mặt ngôn ngữ học, chứ không chỉ nhận xét "lớp này điểm thấp".
- Vì sao **train/val/test phải tách bạch**, qua chính lỗi mình mắc phải trong notebook.
- `sparse_categorical_crossentropy` vs `categorical_crossentropy`; AdamW vs Adam.

**Về Kỹ thuật phần mềm & MLOps:**
- **Artifact của model NLP = weights + tokenizer**, phải version cùng nhau; và **train/serve skew** là loại bug nguy hiểm nhất vì nó *không báo lỗi*.
- Caching tài nguyên nặng trong ứng dụng web (`@st.cache_resource`), batch inference thay vì vòng lặp.
- Làm việc với **API bên thứ ba thực tế**: pagination, quota, rate limit, xử lý lỗi theo status code, quản lý secret qua biến môi trường.
- **Adapter pattern** để nhiều nguồn dữ liệu đổ về một pipeline chung.
- Đóng gói Docker: layer caching, system dependency, network binding trong container.
- Vệ sinh tài nguyên (`plt.close`) để tránh memory leak trong ứng dụng chạy dài.

**Bài học lớn nhất:** phần khó của một dự án ML thực tế **không nằm ở model**. Model chỉ là ~30 dòng code. Phần khó nằm ở việc lấy được dữ liệu thật, giữ cho tiền xử lý lúc train và lúc serve **giống hệt nhau**, và trình bày kết quả sao cho người không biết ML vẫn hiểu được.

---

## Phụ lục — bảng tra nhanh số liệu

| Chỉ số | Giá trị |
|---|---|
| Vocab size | 29,766 |
| Embedding dim | 100 |
| Sequence length (train) | 100 |
| Sequence length (serve) | 30 ⚠️ lệch |
| LSTM units | 128 (×2 = 256 do Bidirectional) |
| Tổng tham số | 3,219,419 (12.28 MB) |
| Kích thước file `.h5` | ~38 MB |
| Kích thước `tokenizer.json` | ~2.7 MB |
| Train samples | ~31,232 |
| Eval samples | 5,205 |
| Epochs / batch size | 7 / 128 |
| Optimizer / LR | AdamW / 1e-4 |
| Train accuracy (cuối) | 0.8038 |
| Val accuracy (cuối) | 0.6734 |
| Val accuracy (tốt nhất, epoch 5) | 0.6699 · val_loss 0.7578 |
| Macro F1 | 0.68 |
| F1 theo lớp | neg 0.68 · neu 0.61 · pos 0.74 |
| Majority baseline | 0.37 |
