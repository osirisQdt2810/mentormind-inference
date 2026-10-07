# Chạy VLM server (Ollama + Qwen3-VL-8B Q4) trên máy của bạn

Một lệnh, chạy được trên **macOS** (Apple Silicon hoặc Intel) và **Linux** (NVIDIA, AMD/ROCm hoặc CPU).

## Cần có
- `curl`; trên Linux cần thêm `zstd`.
- Muốn mở ra internet: [ngrok](https://ngrok.com/download), đã chạy `ngrok config add-authtoken <token>` một lần.
- Khoảng 8 GB trống trên đĩa (Ollama + model 6 GB).

## Chạy
```bash
cd 3rdparty/vlm_server            # hoặc thư mục clone mentormind-inference
bash scripts/serve-ollama.sh      # serve tại http://127.0.0.1:11434
```
Lần đầu chạy, script tải Ollama 0.35.1 (macOS khoảng 150 MB, Linux khoảng 1,4 GB) và model `qwen3-vl:8b` (6 GB). Những lần sau chỉ mất vài giây.

**Mở cho người khác dùng qua ngrok** (mật khẩu tối thiểu 8 ký tự):
```bash
NGROK_BASIC_AUTH=admin:MatKhauDai123 bash scripts/serve-ollama.sh --ngrok
```
Khi xong, script in ra một khối `# ---- mentormind .env ----` gồm URL ngrok (đã kèm user:pass) và tên model. **Gửi nguyên khối đó** cho người chạy app, họ chép vào `.env` là dùng được.

**Dừng:** `bash scripts/serve-ollama.sh --stop`. Nếu lúc chạy có `--port N` thì thêm `--port N`.

## Tùy chỉnh (nếu cần)
| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--image-min-tokens N` | 512 | Số token tối thiểu cho mỗi ảnh. Ollama gốc ép 1024 (frame 448×252 thật ra chỉ 112). 512 cho chất lượng ngang bản BF16 trên video LASI, ít hơn 1024 khoảng 29% token |
| `--context N` | macOS 49152, Linux 98304 | Ngân sách token prompt + trả lời cho mỗi request |
| `--port N` | 11434 | Đổi cổng nếu 11434 đã bị chiếm |
| `--gpu N` | (tự chọn) | Linux: chỉ dùng GPU số N |

Mỗi tham số đều đặt được bằng biến môi trường, ví dụ `VLM_SERVER_IMAGE_MIN_TOKENS=512`. Xem hết bằng `--help`.

## Kiểm tra
Script tự gửi một ảnh thử và in ra dòng `one 448x252 frame = N tokens`. Con số này phải xấp xỉ `--image-min-tokens`; nếu ra khoảng 1024 thì bước bọc `image-min-tokens` chưa có tác dụng. Log nằm ở `~/.cache/vlm-engine/ollama/run-<cổng>/` (`ollama.log`, `ngrok.log`).

## Lỗi hay gặp
- **"something already serves http://127.0.0.1:11434"**: app Ollama đang chạy và chiếm cổng. Thoát app (biểu tượng trên thanh menu → Quit Ollama), hoặc chạy với `--port 11435`. Model dùng chung thư mục `~/.ollama/models` với app, nên không phải tải lại.
- **Chạy rất chậm** (máy ít VRAM, hoặc Mac 16 GB): context lớn không vừa GPU nên một phần chạy bằng CPU. Giảm `--context` kèm `--image-min-tokens` (ví dụ `--context 32768 --image-min-tokens 256`: một đoạn 40 frame khoảng 11K token).
- **Model luôn "nghĩ" lâu trước khi trả lời**: đây là đặc tính của `qwen3-vl:8b`. Phía app cần đặt `KNOWHOW_VLM_MAX_TOKENS` và `KNOWHOW_LLM_MAX_TOKENS` đủ lớn; `.env.example` của mentormind đã đặt sẵn.
