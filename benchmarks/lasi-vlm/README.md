# So sánh VLM trên video LASI: 8B Instruct, 8B Thinking (BF16, FP8), 30B-A3B Thinking (AWQ)

Báo cáo ngày 08/10/2026. Câu hỏi: model nào mô tả video thao tác tốt nhất cho MentorMind (đúng bước, đúng lúc,
đúng người làm), chạy nhanh tới đâu trên máy chủ của nhóm, và tối ưu được tới đâu.

**Chỉ so danh sách cuối**: các bước nháp chuyên gia thấy trên giao diện sau cả quy trình (lượt A, lượt B lọc, lượt C
gộp), không so kết quả trung gian.

## Kết luận

- **Dùng 8B Thinking FP8**, kèm `KNOWHOW_VLM_CONCURRENCY=4` ở MentorMind và speculative decoding ngram ở máy chủ:
  cả job video LASI chạy **23 phút**. Chạy lần lượt từng request thì khoảng 73 phút: 66 phút đo được tới khi job dừng
  ở bước lý do, cộng khoảng 7 phút cho các lượt lý do còn lại. Tức là nhanh hơn khoảng 3.2 lần.
- **8B Thinking tốt hơn Instruct rõ rệt**: ba danh sách cuối của 8B Thinking tìm ra 43–56% bước đúng (Instruct 19%),
  tìm ra bước chính 56–69% (Instruct 31%), chính xác 69–88% (Instruct 38%), bịa 4–8% (Instruct 25%). Đổi lại,
  Instruct nhanh nhất (9 phút).
- **FP8 không kém BF16 về số bước tìm ra** (0.44 so với 0.43), sinh token nhanh hơn 1.6 lần (59 so với 37.5 token/giây)
  và nhẹ hơn 6.3 GiB VRAM. Lần FP8 đầu có độ chính xác thấp hơn BF16 (0.69 so với 0.88), lần FP8 thứ hai thì ngang
  (0.84). Với một lần BF16 và hai lần FP8 thì chưa khẳng định chắc được, nhưng không thấy FP8 kém một cách nhất quán.
- **30B-A3B AWQ chưa hơn 8B Thinking trên video này**: tìm ra 0.33, chính xác 0.63, bịa 0.13, đúng người làm tốt nhất
  (0.80). Mỗi lượt gọi nhanh hơn FP8 khoảng 2.5 lần (20 so với 50 giây), cả job 31 phút khi chạy lần lượt. Cần bật
  JSON gọn (`JSON_WHITESPACE=compact`). Nếu không, 30B lặp khoảng trắng tới hết `max_tokens` ở 10/44 câu trả lời
  (8B: 2/242). Bật rồi thì hết lặp khoảng trắng, nhưng 3/63 câu vẫn chạm hạn mức vì lặp chữ số.
- **Tối ưu ở máy chủ**: gộp nhiều request (4 cùng lúc → tổng 167 token/giây, gấp 3 lần) là đòn bẩy chính.
  Speculative decoding chỉ giúp khi chạy một request: ngram +32%, EAGLE3 +24% (đầu dự đoán huấn luyện cho
  Instruct). Khi đã gộp 4 request thì cả hai không còn khác biệt.
- **Điểm yếu chung** của mọi model: gọi cốc là "linh kiện", nhầm máy với robot, gần như không ghi được điểm mấu chốt
  (đầu gắp nhiều cốc, khung dẫn hướng giữ vạt thùng).

## 1. Cách làm

### Video và nhãn đúng

- **Video**: `LASI Success Case - Food Industry` (YouTube `m4icg6FgTBU`, công khai), 3 phút 14 giây, 1080p, có lời
  nói tiếng Thái. Dây chuyền đóng thùng cốc thức ăn thú cưng nắp đỏ: công nhân ở bàn và băng tải, robot gắp cụm cốc
  vào thùng carton, máy dựng thùng.
- **Nhãn đúng**: `data/video/groundtruth.json` của MentorMind, **27 bước**, trong đó **16 bước thuộc công đoạn chính**
  (đóng thùng, đậy nắp). Mỗi bước có thời điểm bắt đầu/kết thúc, người làm (công nhân, robot, máy, kỹ sư), có bước
  kèm điểm mấu chốt. Nhãn do Claude (Opus 5.5) gán tay từ khung hình (1 khung/giây cả video, 2 khung/giây ở đoạn có
  thao tác; không dùng output của Qwen3-VL), **chưa được người duyệt** (file ghi `draft`): mọi số liệu dưới đây là so
  với bộ nhãn đó.

### Mọi model chạy đúng luồng người dùng: tải video lên giao diện MentorMind

Trang "Tải dữ liệu" → `POST /videos` → `POST /videos/{id}/process`, rồi chờ job xong. Đổi model ở máy chủ và
`KNOWHOW_VLM_MODEL` ở client; mọi thứ khác giữ nguyên, trừ cấu hình JSON của 30B (dòng cuối bảng):

| Giai đoạn | Cấu hình |
|---|---|
| Cắt đoạn | Theo cảnh (PySceneDetect) → **30 đoạn**, giống hệt nhau ở mọi lần chạy (đã so ranh giới) |
| Nhận dạng lời nói | Whisper `small` trên máy chủ |
| Lượt A | Mỗi đoạn một lượt gọi VLM: lấy mẫu theo chuyển động 2/4/8 khung/giây (mặc định 4), bỏ khung gần trùng, tối đa 40 khung, 512 token/khung (448×252). Trên LASI mọi lần chạy gửi cùng 429 khung (14.3 khung/đoạn, trung bình 2.2 khung/giây). Trả JSON theo schema |
| Lượt B | Chạy lượt A lần nữa, chỉ giữ bước cả hai lần cùng thấy (lọc bước bịa, spec 04 §3) |
| Lượt C | Gộp bước bị cắt ngang ranh giới đoạn (chính model đó, chỉ chữ) |
| Đề xuất lý do | Cho bước có lời chuyên gia (chính model đó) |
| Thời điểm bước | Các bước phủ kín đoạn theo thứ tự (`KNOWHOW_VLM_STEP_TIMES=segment`) |
| Giới hạn trả lời | Instruct 8192 token; Thinking 16 384 token |
| JSON ở máy chủ | Các lần chạy 8B: khoảng trắng tự do (`JSON_WHITESPACE=any`); 30B: JSON gọn (`compact`, mục 5) |

Mỗi model có **một bản sao riêng** của video (`ffmpeg -c copy`, chỉ đổi metadata `title`: khung hình và tiếng giống
hệt, `video_id` khác) và **mã công đoạn riêng**, nên kết quả nằm cạnh nhau trong giao diện (trang Duyệt, chọn mã):

| Mã công đoạn trên UI | Model | `VLM_VARIANT` | Hugging Face | Trọng số |
|---|---|---|---|---|
| `LASI-8B-INSTRUCT` | 8B Instruct | `instruct` | `Qwen/Qwen3-VL-8B-Instruct` | BF16, 17 GB |
| `LASI-8B-THINKING` | 8B Thinking | `thinking` | `Qwen/Qwen3-VL-8B-Thinking` | BF16, 17 GB |
| `LASI-8B-THINKING-FP8` | 8B Thinking | `thinking-fp8` | `Qwen/Qwen3-VL-8B-Thinking-FP8` | FP8 theo khối, 10.6 GB (trên Ampere chạy Marlin W8A16) |
| `LASI-8B-THINKING-FP8-FAST` | 8B Thinking FP8, tối ưu (mục 4) | `thinking-fp8` + ngram | `Qwen/Qwen3-VL-8B-Thinking-FP8` | như trên, `KNOWHOW_VLM_CONCURRENCY=4` |
| `LASI-30B-THINKING` | 30B-A3B Thinking | `30b-thinking` | `QuantTrio/Qwen3-VL-30B-A3B-Thinking-AWQ` | AWQ 4-bit, 17.9 GB (MoE: 30 tỉ tham số, mỗi token dùng 3 tỉ) |

### Máy chủ

Vast.ai, 1× RTX A5000 24 GB (Ampere), vLLM 0.31.0, context 65 536 cho mọi model, KV cache FP8,
`gpu-memory-utilization` 0.94, mỗi lần một model (24 GB không chứa được hai). Client ở Việt Nam gọi qua ngrok →
Caddy (token) → gateway → vLLM.

| Model | VRAM trọng số (vLLM báo) | KV cache còn lại | Ghi chú |
|---|---|---|---|
| 8B BF16 (Instruct, Thinking) | 16.6 GiB | 72 624 token | |
| 8B Thinking FP8 | 10.3 GiB | 164 672 token | |
| 30B-A3B AWQ | 17.0 GiB | 100 528 token (1.5 request 64K cùng lúc) | cần `JSON_WHITESPACE=compact` (mục 5) |

### Cách chấm danh sách cuối

1. **Giám khảo ngữ nghĩa, mù** (`judge_prompt.md`, `judge.py`): một model ngôn ngữ (Claude) đọc 27 bước đúng và
   **một** danh sách cuối, **không biết model nào** sinh ra nó, rồi quyết định với từng bước dự đoán: mô tả bước đúng
   nào (cùng hành động trên cùng loại vật, thời gian chồng nhau hoặc lệch ≤ 2 giây; diễn đạt khác vẫn tính), người làm
   đúng không, điểm mấu chốt đúng không, và nếu không khớp thì có phải **bịa** không (hành động không hề diễn ra lúc
   đó). **Hai giám khảo độc lập** chấm mỗi danh sách; số liệu là trung bình hai giám khảo.

   | Chỉ số | Nghĩa |
   |---|---|
   | Tìm ra | số bước đúng (khác nhau) mà danh sách mô tả được / 27 |
   | Tìm ra bước chính | như trên, chỉ tính 16 bước công đoạn chính |
   | Chính xác | số bước dự đoán mô tả một bước đúng / số bước dự đoán |
   | Bịa | số bước dự đoán mô tả việc không xảy ra / số bước dự đoán |
   | Đúng người làm | trên các bước khớp, tỉ lệ ghi đúng công nhân / robot / máy |

2. **Chỉ số tự động của repo** (spec 04 §6), để đối chiếu: khớp khi tên giống nhau theo `token_set_ratio ≥ 70` **và**
   thời gian chồng nhau (rất chặt: "Gắp cụm cốc nắp đỏ và đặt vào thùng carton" và "Robot gắp cốc bỏ vào thùng" không
   khớp), và **phủ thời gian** (bước đúng nào có ít nhất một bước dự đoán trùng thời gian).
3. **Tốc độ**: thời gian cả job trên UI (đồng hồ monotonic, không tính lúc máy client ngủ); trên mọi lượt gọi của lượt
   A và B: giây mỗi lượt (trung vị), token sinh ra mỗi lượt (trung vị), token sinh chia thời gian gọi.

## 2. Độ chính xác (danh sách cuối)

Danh sách cuối của mỗi lần chạy, chấm bởi hai giám khảo mù (trung bình). Hai giám khảo cho kết quả gần như trùng
nhau: "tìm ra" lệch nhau tối đa một bước (1/27). Cột "khớp chữ chặt" và "phủ thời gian" là chỉ số tự động của repo.

| Model | Trọng số | Danh sách | Số bước | Tìm ra (recall) | Tìm ra bước chính | Chính xác (precision) | Bịa | Đúng người làm | Khớp chữ chặt (repo) | Phủ thời gian |
|---|---|---|---|---|---|---|---|---|---|---|
| 8B Instruct | BF16 | cuối (UI) | 16 | 0.19 | 0.31 | 0.38 | 0.25 | 0.67 | 0.07 | 0.78 |
| 8B Thinking | BF16 | cuối (UI) | 20 | 0.43 | 0.66 | 0.88 | 0.08 | 0.77 | 0.19 | 0.48 |
| 8B Thinking | FP8 | cuối (UI) | 24 | 0.44 | 0.56 | 0.69 | 0.08 | 0.73 | 0.11 | 0.82 |
| 30B-A3B Thinking | AWQ 4-bit | cuối (UI) | 16 | 0.33 | 0.44 | 0.63 | 0.13 | 0.80 | 0.11 | 0.63 |
| 8B Thinking, tối ưu (4 request cùng lúc + ngram) | FP8 | cuối (UI) | 25 | 0.56 | 0.69 | 0.84 | 0.04 | 0.76 | 0.22 | 0.74 |

- **8B Thinking so với Instruct**: ba lần chạy 8B Thinking tìm ra gấp 2.3–3.0 lần số bước đúng (11.5–15 so với 5
  bước), chính xác khoảng gấp đôi, bịa ít hơn 3–6 lần. Instruct sót gần hết các bước của công nhân và nhiều cảnh
  robot. 30B Thinking hơn Instruct ít hơn: tìm ra gấp 1.8 lần, chính xác gấp 1.7 lần, bịa ít hơn 2 lần.
- **BF16 so với FP8**: số bước tìm ra bằng nhau (0.43 và 0.44). Lần FP8 đầu chính xác thấp hơn (0.69 so với 0.88) và
  tìm ra bước chính ít hơn (0.56 so với 0.66). Lần FP8 thứ hai (bản tối ưu, đầu ra của model không đổi) được 0.84 và
  0.69, tức chênh lệch đó cỡ mức dao động giữa các lần chạy cùng model. Với một lần BF16 và hai lần FP8 thì chưa khẳng
  định chắc được.
- **Lượt B lọc mất cả bước đúng**. Đo bằng giám khảo ở hai lần chạy: Instruct tìm ra 0.26 sau lượt A còn 0.19 ở danh
  sách cuối, FP8 từ 0.59 còn 0.44 (verdict của lượt A nằm trong `results/verdicts/`). Phủ thời gian giảm ở cả năm lần
  chạy (Instruct 0.82→0.78, BF16 0.96→0.48, FP8 1.00→0.82, 30B 0.67→0.63, FP8 tối ưu 1.00→0.74). Lượt B là chỗ nên
  xem lại tiếp theo.
- 30B đưa ra ít bước nhất, ngang Instruct (16), và bỏ sót cảnh dựng thùng (51–53 giây) và hai chu kỳ robot gắp cốc
  (55–60 giây, 92–98 giây).

## 3. Tốc độ

| Model | Trọng số | Lượt gọi (A+B) | Giây mỗi lượt (trung vị) | Token sinh mỗi lượt (trung vị) | Token sinh / giây gọi (gồm prefill) | Cả job trên UI |
|---|---|---|---|---|---|---|
| 8B Instruct | BF16 | 60 | 5 | 55 | 18.1 | 9 phút |
| 8B Thinking | BF16 | 61 | 77 | 2920 | 37.5 | 103 phút |
| 8B Thinking | FP8 | 60 | 50 | 2870 | 59.3 | 66 phút |
| 30B-A3B Thinking | AWQ 4-bit | 63 | 20 | 1799 | 121.6 | 31 phút |
| 8B Thinking, tối ưu (4 request cùng lúc + ngram) | FP8 | 61 | 66 | 3022 | 51.2 | 23 phút |

- Số liệu theo **từng** lượt gọi. Lần chạy tối ưu có 4 request cùng lúc, nên mỗi lượt chậm hơn một chút nhưng tổng
  thông lượng gấp khoảng 3 lần (mục 4).
- "Token sinh / giây gọi" gồm cả xử lý prompt (khoảng 8 000 token ảnh mỗi lượt) và mạng. Với Thinking (khoảng 3 000
  token ra mỗi lượt) con số này gần tốc độ sinh thật. Với Instruct (khoảng 55 token ra mỗi lượt) phần lớn thời gian là
  xử lý prompt, nên 18 token/giây không phải tốc độ sinh của nó.
- Thời gian cả job: Instruct, 30B và bản tối ưu là **đủ cả job** (cắt đoạn, lời nói, lượt A, B, C, đề xuất lý do).
  BF16 tính tới lúc lưu bước nháp (bước lý do bị dừng để lấy GPU cho phần tối ưu). FP8 chạy lần lượt dừng ở bước lý do
  (lỗi đã sửa, mục 5). 66 phút của nó gồm cắt đoạn và lời nói 2.4, lượt A 25.9, lượt B 25.0, gộp 4.1 và 8.5 phút lý do
  tới lúc dừng. Các lượt lý do còn lại (khoảng 52 giây mỗi lượt) thêm khoảng 7 phút, nên cả job khoảng 73 phút.

## 4. Tối ưu tốc độ

Đo trên **6 đoạn LASI thật** (pass A, `speed.py`), 8B Thinking FP8, cùng máy chủ; mỗi cấu hình đo khi chạy một
request và khi chạy 4 request cùng lúc. Số liệu gốc: `results/engine/`.

| Cấu hình máy chủ | 1 request (tổng token/giây) | 4 request cùng lúc | Token sinh mỗi bước giải mã (acceptance length, gồm 1 token của model chính) |
|---|---|---|---|
| Không dùng speculative decoding | 56 | 167 | – |
| ngram (`num_speculative_tokens` 4, `prompt_lookup` 2–4) | **74 (+32%)** | 169 | 2.1–2.4 |
| EAGLE3, đầu `AQ-MedAI/Qwen3-VL-8B-Instruct-eagle3` | 70 (+24%) | 176 | 1.6–1.7 |
| EAGLE3, đầu `taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3` | 55 (±0) | 171 | 1.3–1.4 |
| 6 request cùng lúc (không spec) | | 136 (chỉ có 6 request nên cuối lượt còn ít request chạy) | – |

1. **Gộp request (MentorMind `KNOWHOW_VLM_CONCURRENCY`)**: A5000 giải mã bị giới hạn bởi băng thông bộ nhớ. Mỗi bước
   đọc trọng số một lần cho cả lô, nên 4 request cùng lúc cho tổng gấp 3 lần. MentorMind trước đây gửi từng request
   một, nên phải thêm cài đặt ở client (spec 04 AC-30): lượt A và lượt B chạy chồng nhau trên một pool N request, đề
   xuất lý do cũng chạy N cùng lúc, kết quả giống hệt chạy lần lượt.
2. **Speculative decoding** (không đổi kết quả): chỉ có ích khi ít request. Phần suy nghĩ của Thinking hay lặp cụm từ
   nên ngram đoán trúng nhiều hơn EAGLE3. Hai đầu EAGLE3 có sẵn được huấn luyện cho bản Instruct; không có đầu nào cho
   Qwen3-VL-8B-Thinking. Không dùng được model nháp nhỏ (Qwen3-VL-2B) vì vLLM 0.31 không hỗ trợ M-RoPE cho model nháp.
3. **Prefix caching** (bật sẵn): lượt B gửi lại đúng prompt của lượt A, nên phần ảnh đã tính được dùng lại khi còn
   trong cache. Tỉ lệ trúng trong `results/engine/spec_metrics.txt` (45–67%) chỉ phản ánh cách đo (`speed.py` gửi lại
   cùng 6 prompt ở mỗi chế độ), không phải một job thật.
4. **Chưa dùng**: `thinking_token_budget` của vLLM (cắt phần suy nghĩ sau N token) làm đổi kết quả. Không cần, vì
   mục tiêu 40 phút đã đạt.

**Kết quả trên UI**: FP8 + ngram + `KNOWHOW_VLM_CONCURRENCY=4`, cả job **22.9 phút** (trạng thái hoàn tất, 30 đoạn,
25 bước nháp, 10 bước có lời chuyên gia, 1 lý do đề xuất), so với khoảng 73 phút khi chạy lần lượt. Lượt A và B mất
18 phút thay vì 51. Chạy song song và ngram không đổi đầu ra của model; danh sách cuối không kém (dòng "tối ưu" ở mục 2).

Cấu hình đề nghị cho máy chủ của nhóm:

```bash
VLM_VARIANT=thinking-fp8 \
SPEC_CONFIG='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
bash /root/run.sh
# .env của MentorMind: KNOWHOW_VLM_MODEL=Qwen/Qwen3-VL-8B-Thinking-FP8, KNOWHOW_VLM_MAX_TOKENS=16384,
# KNOWHOW_LLM_MAX_TOKENS=16384, KNOWHOW_VLM_CONCURRENCY=4
```

## 5. Sự cố trong lúc đo và cách xử lý

| Sự cố | Ảnh hưởng | Xử lý |
|---|---|---|
| ngrok free trả **503** khi một response im lặng khoảng 5 phút | Lần chạy 8B Thinking BF16 đầu tiên hỏng ở lượt C sau 117 phút (lượt gộp suy luận hơn 5 phút) | **Gateway heartbeat** (repo này, `INFERENCE_HEARTBEAT_S=15`): câu trả lời chưa có sau 15 giây → gửi ngay header 200 rồi một dấu cách mỗi 15 giây. Đã thử thật: một lượt 325 giây qua ngrok trả 200, JSON hợp lệ |
| Máy Mac chạy client **ngủ đông** (gập nắp / nút nguồn) | Hai lần chạy dừng ở giữa (socket chết, request treo tới timeout 1800 giây) | Bỏ các lần đó, chạy lại; các lần sau đặt `KNOWHOW_VLM_TIMEOUT_S=120`: nhờ heartbeat, request khoẻ không bao giờ im lặng quá 15 giây, nên timeout 120 giây chỉ cắt kết nối chết và client tự thử lại |
| **Máy Vast tự khởi động lại** (lúc 02:02 và 12:22, phía host) | ngrok mất kết nối khoảng 2 phút, job đang chạy báo lỗi 404 | Chạy lại; dịch vụ tự bật lại nhờ supervisor |
| 30B-A3B **lặp khoảng trắng** trong JSON (`\n\n  ` tới hết 16 384 token) | 10/44 câu trả lời của lần chạy bị bỏ hỏng, phải hỏi lại, có đoạn hỏng cả hai lần (8B: 2/242 câu, một ở BF16, một ở FP8 tối ưu) | `run.sh` thêm `JSON_WHITESPACE=compact` (mặc định): vLLM chỉ cho JSON gọn (`disable_any_whitespace`, backend xgrammar). Lần chạy 30B dùng trong báo cáo: 0/63 câu lặp khoảng trắng, nhưng 3/63 câu vẫn chạm hạn mức vì lặp chữ số, phải hỏi lại |
| 8B Thinking FP8 suy luận hết 16 384 token ở **một** bước đề xuất lý do, vLLM trả `content = null` | Cả job FP8 báo lỗi ở bước lý do (5/14), dù 24 bước nháp đã lưu | MentorMind (spec 04 AC-29): câu trả lời rỗng là câu trả lời không đọc được của riêng bước đó, job chạy tiếp. Danh sách cuối của FP8 không bị ảnh hưởng (lưu trước bước lý do) |
| Xoá `/venv/main` của template Vast để lấy chỗ trống | Không giải phóng được gì (nằm ở lớp image chỉ đọc); Jupyter của template không còn chạy | `run.sh` dùng `hf`/`python` của venv vLLM. Thuê máy mới để thử nhiều model: chọn đĩa 100 GB |

## 6. Chạy lại

Máy chủ (SSH vào instance Vast), mỗi model một lần:

```bash
VLM_VARIANT=thinking-fp8 bash /root/run.sh       # instruct | thinking | thinking-fp8 | 30b-thinking
JSON_WHITESPACE=any VLM_VARIANT=thinking bash /root/run.sh    # cấu hình đã dùng cho 3 lần chạy 8B
```

Máy chạy UI (một checkout `mentormind-knowhow-ai` có `.env` trỏ tới máy chủ):

```bash
export MENTORMIND=~/mentormind-knowhow-ai B=<repo này>/benchmarks/lasi-vlm
ffmpeg -i "$MENTORMIND/data/video/LASI Success Case - Food Industry (Asian Alliance Internation Co.,Ltd.) [m4icg6FgTBU].mkv" \
  -map 0 -c copy -metadata title="LASI (thinking)" LASI-thinking.mkv          # một bản sao cho mỗi model
KNOWHOW_VLM_TIMEOUT_S=120 KNOWHOW_LLM_TIMEOUT_S=120 bash $B/api.sh Qwen/Qwen3-VL-8B-Thinking 16384 &
VITE_API_BASE_URL=http://localhost:8010 npm --prefix $MENTORMIND/mentormind/frontend run dev -- --port 5175 &
uv run --no-project --with playwright python $B/upload.py LASI-thinking.mkv LASI-8B-THINKING shots 8b-thinking
cd $MENTORMIND && uv run python $B/score.py data/work 8b-thinking=<video_id> > scores.json
python3 $B/judge.py render scores.json 8b-thinking final > prompt.txt   # cho 2 giám khảo, mỗi người một file verdict
python3 $B/judge.py metrics scores.json 8b-thinking final verdict_1.json verdict_2.json
python3 $B/overview.py runs.json                                        # bảng ở mục 2 và 3
```

Số liệu của báo cáo này nằm trong `results/`: điểm của từng lần chạy (`score_*.json`, kèm thời gian và token từng
lượt gọi), verdict của hai giám khảo cho mọi danh sách đã chấm (`verdicts/`, cả lượt A), `runs.json`, và số đo engine
(`engine/`).

## 7. Giới hạn

- Một video (3 phút 14 giây), 27 bước đúng; nhãn là bản nháp do Claude gán, chưa có người duyệt.
- **Mỗi model một danh sách cuối** (8B Thinking FP8 hai). Chỉ Instruct, 30B và FP8 tối ưu chạy đủ cả bước lý do.
  Model sinh có yếu tố ngẫu nhiên (lượt A nhiệt độ > 0): các lượt A độc lập của 8B Thinking cho "tìm ra" từ 0.52 đến
  0.59 (lần BF16 đầu tiên `score_8b-thinking-try1.json`, hỏng ở lượt C, và lần FP8). Hai lần chạy FP8 cho danh sách
  cuối 0.44 và 0.56 (chênh 0.11). Chênh lệch cỡ đó giữa hai cấu hình chưa đủ để kết luận cấu hình nào hơn.
- Cấu hình JSON khác nhau giữa 30B (gọn) và các lần 8B (khoảng trắng tự do): đây là một khác biệt ngoài model khi so
  30B với 8B.
- Giám khảo là một model ngôn ngữ, mù với model nhưng không xem video: chỉ so chữ và thời gian với nhãn.
- Tốc độ đo trên một RTX A5000 qua ngrok từ Việt Nam; máy khác hoặc gọi trong mạng LAN sẽ khác.
