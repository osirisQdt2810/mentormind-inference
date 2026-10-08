# mentormind-inference — máy chủ suy luận của MentorMind

Repo này chạy mọi phần suy luận mà `mentormind-knowhow-ai` gọi qua HTTP:

- **VLM** Qwen3-VL-8B, API tương thích OpenAI (`/v1/chat/completions`), chạy trên GPU bằng vLLM hoặc Ollama.
- **Gateway CPU** (FastAPI): nhận dạng giọng nói (faster-whisper), embeddings (bge-m3), chuyển đổi tài
  liệu (Docling). Mọi đường dẫn `/v1/*` khác được reverse-proxy sang VLM, nên **một URL public** (một domain
  ngrok, sau lớp Caddy kiểm tra token) phục vụ tất cả.

Tên cũ của repo là `vlm-engine`. Gói Python vẫn là `vlm_server`, CLI vẫn là `python -m vlm_server <lệnh>`.
MentorMind gắn repo này làm submodule tại `3rdparty/vlm_server`. Chạy tại máy mình (loopback, IP private) thì
đây là provider duy nhất được nhận dữ liệu `factory_confidential` (constitution Điều 2).

```
mentormind-knowhow-ai                                    mentormind-inference (repo này)
KNOWHOW_VLM_BASE_URL  ─┐                              ┌─ POST /v1/audio/transcriptions ─► faster-whisper (CPU)
KNOWHOW_INFERENCE_URL ─┴► https://<domain>/v1 ─► Caddy :10100 ─► gateway :18080 ─┼─ POST /v1/embeddings ──────────► bge-m3 (CPU)
                          (Authorization: Bearer)    (kiểm tra token)             ├─ POST /v1/documents/convert ───► Docling (CPU)
                                                                                  └─ /v1/* còn lại ──────────────► vLLM :18000 (GPU)
```

VLM có hai backend (`VLM_SERVER_BACKEND`):

- **`ollama` (mặc định)**: `qwen3-vl:8b` Q4_K_M trên Ollama 0.35.1, chạy được trên macOS và Linux, cổng 11434.
  - Một lệnh: `bash scripts/serve-ollama.sh` (thêm `--ngrok` để mở cho máy khác). Hướng dẫn cho người dùng ở **[scripts/README.md](scripts/README.md)**.
  - Docker: `docker/Dockerfile.ollama`.
  - Ollama tự ép mỗi ảnh Qwen-VL tốn tối thiểu 1024 token. `scripts/ollama/llama-server-wrapper.sh` đổi được mức này qua `VLM_SERVER_IMAGE_MIN_TOKENS` (mặc định 512; quét trên video LASI ngày 4/10, mức này cho chất lượng ngang bản BF16 và ít hơn 1024 khoảng 29% token).
- **`vllm`**: Qwen3-VL-8B-Instruct BF16 trên vLLM, đúng 1 GPU NVIDIA hoặc AMD, cổng 8100 (mục 3 đến 5). Bản deploy
  trên Vast.ai (mục 2) dùng backend này, cổng 18000, phía sau gateway.

| Thư mục | Nhiệm vụ |
|---|---|
| `vlm_server/config.py` | Cấu hình `VLM_SERVER_*` (`ServerConfig`) |
| `vlm_server/gpu/` | Phát hiện nền tảng và chọn đúng 1 GPU: `nvidia.py` (nvidia-smi), `rocm.py` (amd-smi/rocm-smi), `select.py` |
| `vlm_server/serve/` | `ollama.py` dựng lệnh `scripts/serve-ollama.sh`; `command.py` dựng lệnh và biến môi trường vLLM theo nền tảng; `launch.py` chạy backend đã chọn |
| `vlm_server/gateway/` | Gateway CPU: `config.py` (`INFERENCE_*`), `app.py` (`create_app`), `asr.py`, `embeddings.py`, `documents.py` (engine nạp thư viện nặng khi cần), `proxy.py` (proxy sang VLM), `cli.py` (`python -m vlm_server gateway`) |
| `requirements/` | Thư viện theo nền tảng: gateway (`gateway-common.txt` + một file cho mỗi nền tảng) và venv vLLM trên máy CUDA mới (`vllm-linux-cuda.txt`), xem mục 1 |
| `scripts/` | `serve-ollama.sh` (macOS và Linux, một lệnh), `ollama/llama-server-wrapper.sh`, `README.md` hướng dẫn; `vast/run.sh` deploy lên Vast.ai (`run.sh` ở gốc repo gọi lại script này); `lib/platform.sh` nhận diện nền tảng và chọn file `requirements/` |
| `vlm_server/tools/smoke.py` | Kiểm tra một server VLM đang chạy |
| `docker-compose.yml`, `docker/` | Chạy VLM riêng (không cần repo chính): image CUDA và ROCm, override CDI `docker/docker-compose.cdi.yml`; container giữ sẵn môi trường, server bật khi cần |

## 1. Gateway CPU: ASR, embeddings, tài liệu và proxy VLM

`python -m vlm_server gateway` nghe ở `INFERENCE_HOST:INFERENCE_PORT` (mặc định `127.0.0.1:18080`). Gateway
không tự kiểm tra token: Caddy đứng trước làm việc đó (`Authorization: Bearer`). Mọi body đều là JSON.

### Hợp đồng API

Base URL của client là `<URL public>/v1`. Tên trường và cấu trúc dưới đây là cố định: client trong
`mentormind-knowhow-ai` được viết theo đúng hợp đồng này.

| Endpoint | Đầu vào | Đầu ra |
|---|---|---|
| `GET /health` | | `{"status":"ok","vlm_upstream":"<url>","services":{"asr":"<model>","embeddings":"<model>","documents":"docling"}}` |
| `POST /v1/audio/transcriptions` | multipart: `file` (audio; nên là WAV 16 kHz mono, file nào ffmpeg đọc được cũng nhận); `language` tùy chọn (trống, thiếu hoặc `auto` = tự nhận); `model` tùy chọn; `response_format` chỉ nhận `verbose_json` (mặc định); `timestamp_granularities[]` bị bỏ qua (luôn có words) | `{"text","language","duration","segments":[{"id","start","end","text","words":[…]}],"words":[{"word","start","end","probability"}]}` |
| `POST /v1/embeddings` | JSON `{"model": tùy chọn, "input": str \| [str]}` | `{"object":"list","model":"BAAI/bge-m3","data":[{"object":"embedding","index":i,"embedding":[…]}],"usage":{"prompt_tokens":n,"total_tokens":n}}` |
| `POST /v1/documents/convert` | multipart: `file` (`.pdf`, `.docx`, `.xlsx`, `.pptx`) | `{"filename","num_pages","document": <DoclingDocument.export_to_dict()>}` |
| `GET /v1/models`, `POST /v1/chat/completions` và mọi `/v1/*` khác | nguyên văn | proxy sang `INFERENCE_VLM_UPSTREAM`: giữ method, query, body, header (bỏ header hop-by-hop và `Host`), trả lại nguyên status, header và body; `"stream": true` (SSE) được chuyển tiếp theo từng chunk; câu trả lời không stream chậm hơn `INFERENCE_HEARTBEAT_S` thì có heartbeat (bảng dưới) |

| Lỗi | Khi nào |
|---|---|
| 400 `{"error": …}` | File rỗng, `response_format` khác `verbose_json`, `input` rỗng, audio không giải mã được |
| 422 `{"error": …}` | Thiếu trường/sai kiểu; tài liệu không thuộc 4 định dạng trên hoặc Docling không chuyển được |
| 502 `{"error": …}` | VLM upstream không trả lời |
| 503 `{"error": …}` | Engine chưa cài thư viện (faster-whisper, torch/transformers, docling) hoặc không nạp được model |

Chi tiết cần giữ đúng:

- **ASR**: faster-whisper `word_timestamps=True`, `beam_size=INFERENCE_ASR_BEAM_SIZE`, `vad_filter=INFERENCE_ASR_VAD`.
  `word` là **token thô** đúng như faster-whisper trả về, **giữ khoảng trắng đầu**: client ghép các từ bằng
  cách nối chuỗi, tiếng Thái nối liền không có dấu cách. `segments[].text` đã strip. Thời gian âm được kẹp về 0.
  `language` là mã ISO 639-1 do faster-whisper nhận ra (ví dụ `th`).
- **Embeddings**: tính y hệt embedder local của MentorMind: `AutoTokenizer` + `AutoModel(...).eval()`,
  `padding=True, truncation=True, max_length=1024`, `torch.no_grad()`, vector = `last_hidden_state[:, 0]` (CLS),
  rồi chuẩn hóa L2 bằng Python (chia cho norm nếu norm > 0). Mỗi lượt chạy 16 câu (`INFERENCE_EMBED_BATCH`),
  như phía local. Đã so trên máy dev: cùng danh sách câu thì vector remote **bằng tuyệt đối** vector local.
  Một câu đứng riêng và cùng câu đó trong một batch có thể lệch ở chữ số cuối (padding), ở cả hai phía.
- **Tài liệu**: Docling `DocumentConverter().convert(path)` với cấu hình mặc định (có OCR vùng ảnh).
- Mỗi engine nạp model ở lần gọi đầu, giữ lại, và xử lý tuần tự (một khóa cho mỗi engine). Ba engine chạy song song với nhau.

### Cấu hình (`INFERENCE_*`)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `INFERENCE_HOST` / `INFERENCE_PORT` | `127.0.0.1` / `18080` | Địa chỉ nghe |
| `INFERENCE_VLM_UPSTREAM` | `http://127.0.0.1:18000` | VLM nhận các `/v1/*` còn lại |
| `INFERENCE_PROXY_TIMEOUT_S` | `1800` | Thời gian chờ tối đa một request proxy |
| `INFERENCE_HEARTBEAT_S` | `15` | `POST /v1/chat/completions` không stream mà chưa có câu trả lời sau chừng này giây: gateway gửi ngay header 200 JSON rồi một dấu cách sau mỗi chừng ấy giây tới khi có JSON (JSON bỏ qua khoảng trắng đầu). ngrok free trả 503 cho response im lặng khoảng 5 phút, mà model Thinking hay bước gộp có thể sinh lâu hơn. Lỗi upstream đến sau lúc đó vẫn mang body lỗi nhưng status 200 (header `X-Gateway-Heartbeat`). `0` = tắt |
| `INFERENCE_ASR_MODEL` | `small` | Model faster-whisper (`small` = `Systran/faster-whisper-small`) |
| `INFERENCE_ASR_DEVICE` / `INFERENCE_ASR_COMPUTE_TYPE` | `cpu` / `int8` | |
| `INFERENCE_ASR_BEAM_SIZE` / `INFERENCE_ASR_VAD` | `5` / `true` | |
| `INFERENCE_EMBED_MODEL` / `INFERENCE_EMBED_BATCH` | `BAAI/bge-m3` / `16` | Phải trùng model embedder local của MentorMind |

Docling đọc thêm biến của chính nó, ví dụ `DOCLING_ARTIFACTS_PATH` (thư mục model đã tải sẵn, xem mục 2).

### Thư viện theo nền tảng (`requirements/`)

Mỗi nền tảng có file requirements riêng. Phần chung nằm trong `gateway-common.txt`; mỗi file `gateway-<nền tảng>.txt`
nạp nó bằng `-r gateway-common.txt` rồi thêm đúng bản torch của nền tảng đó. Đầu mỗi file ghi nền tảng, lệnh cài
chính xác và đã kiểm tra tới đâu.

| File | Dùng ở đâu | Cờ `uv pip install` | Đã kiểm tra |
|---|---|---|---|
| `gateway-common.txt` | Phần chung (FastAPI, faster-whisper, PyAV `<19`, Docling, transformers). Không cài riêng | | Như `gateway-linux-cpu.txt` |
| `gateway-linux-cpu.txt` | Gateway trên Linux, chạy CPU. **Mặc định trên mọi máy Linux, kể cả máy Vast có GPU**: gateway chạy CPU theo thiết kế, GPU để cho vLLM | `--torch-backend=cpu` | Cùng gói với `requirements-gateway.txt` cũ, đã deploy trên Vast; chính file này qua `run.sh`: chưa |
| `gateway-macos.txt` | Gateway trên Mac Apple silicon, **macOS 14+ (Sonoma)** (Docling dùng MPS, phần còn lại chạy CPU). macOS 13 thiếu wheel `docling-parse`, uv phải build từ source | không cần | Chỉ resolve (`uv pip compile`, đích macOS 14) |
| `gateway-linux-cuda.txt` | Linux + NVIDIA, chỉ khi muốn Docling chạy trên GPU (tự chọn) | `--torch-backend=auto` | Chỉ resolve |
| `gateway-linux-rocm.txt` | Linux + AMD (ROCm), chỉ khi muốn Docling chạy trên GPU (tự chọn); torch lấy bản ROCm | `--torch-backend=auto` (hoặc `rocm6.4`…) | Chưa: chỉ resolve |
| `vllm-linux-cuda.txt` | venv vLLM trên máy Linux + NVIDIA mới (Vast): vLLM và thư viện của launcher | `--torch-backend=auto` | Cùng gói với lệnh `uv pip install vllm` đã deploy trên Vast (vllm không khóa phiên bản); chính file này qua `run.sh`: chưa |

Không chắc máy mình dùng file nào thì chạy `bash scripts/lib/platform.sh`: script in ra nền tảng (OS, kiến trúc,
GPU) cùng file và lệnh cài cho gateway và vLLM. Máy dev CUDA 12.6 vẫn cài vLLM bằng `uv sync` (mục 3); ROCm lấy
vLLM từ image của AMD (mục 5); macOS chạy VLM bằng Ollama.

### Chạy tại chỗ

Gateway dùng venv riêng, tách khỏi venv vLLM. Linux (CPU, kể cả máy có GPU):

```bash
uv venv .venv-gateway --python 3.12
uv pip install --python .venv-gateway/bin/python -r requirements/gateway-linux-cpu.txt --torch-backend=cpu
INFERENCE_VLM_UPSTREAM=http://127.0.0.1:8100 .venv-gateway/bin/python -m vlm_server gateway
curl -s http://127.0.0.1:18080/health
curl -s http://127.0.0.1:18080/v1/embeddings -H 'Content-Type: application/json' -d '{"input": "xin chào"}'
curl -s http://127.0.0.1:18080/v1/audio/transcriptions -F file=@clip.wav -F language=th
curl -s http://127.0.0.1:18080/v1/documents/convert -F file=@sop.pdf
```

Các nền tảng khác chỉ khác bước `uv pip install` (venv tạo như trên):

```bash
# macOS 14+ (Sonoma) Apple silicon: torch mặc định của PyPI (có MPS); macOS 13 phải build docling-parse từ source
uv pip install --python .venv-gateway/bin/python -r requirements/gateway-macos.txt
# Linux + NVIDIA, Docling trên GPU: uv chọn index CUDA hợp với driver
uv pip install --python .venv-gateway/bin/python -r requirements/gateway-linux-cuda.txt --torch-backend=auto
# Linux + AMD (ROCm), Docling trên GPU: uv đọc GPU qua rocm_agent_enumerator; không được thì ghi rõ
# bản ROCm của máy (cat /opt/rocm/.info/version), ví dụ --torch-backend=rocm6.4
uv pip install --python .venv-gateway/bin/python -r requirements/gateway-linux-rocm.txt --torch-backend=auto
# ROCm bên trong image vLLM của AMD (torch ROCm có sẵn): khóa đúng bản torch đó để uv không tải bản khác
python3 -c "import torch; print('torch==' + torch.__version__)" > /tmp/rocm-torch.txt
uv pip install --system -r requirements/gateway-linux-rocm.txt -c /tmp/rocm-torch.txt
```

Bản CUDA và ROCm chỉ đưa model của Docling lên GPU: bge-m3 luôn chạy CPU, ASR chạy CPU trừ khi đặt
`INFERENCE_ASR_DEVICE=cuda` (CTranslate2 không có bản ROCm). **Bản macOS, CUDA và ROCm mới được resolve bằng
`uv pip compile`, chưa cài và chạy thật**. Trên Vast mới chạy thật các gói của `requirements-gateway.txt` và
`uv pip install vllm` trước đây; `gateway-linux-cpu.txt` và `vllm-linux-cuda.txt` chứa đúng các gói đó nhưng chưa
được cài qua `run.sh` mới.

Lần gọi đầu mỗi endpoint sẽ tải model vào cache Hugging Face (`HF_HOME`): Whisper small khoảng 0,5 GB,
bge-m3 khoảng 2,3 GB, model của Docling khoảng 0,7 GB trở lên.

## 2. Deploy trên Vast.ai: `scripts/vast/run.sh`

Dành cho một instance Vast.ai dùng template PyTorch (chạy bằng root, có Caddy portal và supervisor). Script tự
clone repo này, nên chỉ cần chép riêng file đó lên máy, hoặc chạy `bash run.sh` từ một bản clone. Chạy lại sau
mỗi lần Start:

```bash
curl -fsSLo run.sh https://raw.githubusercontent.com/osirisQdt2810/mentormind-inference/main/scripts/vast/run.sh
bash run.sh
```

Các bước:

1. Clone hoặc cập nhật repo vào `/opt/mentormind-inference` theo `ENGINE_REF` (mặc định `main`; đặt một commit để mọi instance chạy cùng một bản).
   `ENGINE_REF` phải là bản đã có `requirements/` và `scripts/lib/platform.sh`.
2. Nhận diện nền tảng bằng `scripts/lib/platform.sh` của bản clone (`uname -s`, `uname -m`; `nvidia-smi` chạy được → `cuda`,
   có `amd-smi`/`rocm-smi` → `rocm`, còn lại `cpu`) rồi chọn file requirements. Log cho biết file nào và lệnh nào:

   ```
   [run.sh] Nền tảng: Linux x86_64, accelerator cuda.
   [run.sh] vLLM deps: requirements/vllm-linux-cuda.txt (cd /opt/mentormind-inference && uv pip install --python /opt/vllm/bin/python -r requirements/vllm-linux-cuda.txt --torch-backend=auto)
   [run.sh] Gateway deps: requirements/gateway-linux-cpu.txt (cd /opt/mentormind-inference && uv pip install --python /opt/inference-cpu/bin/python -r requirements/gateway-linux-cpu.txt --torch-backend=cpu -q)
   ```

   Lệnh trong ngoặc đúng từng chữ là lệnh script chạy (khi venv cần cài), dán vào terminal ở thư mục nào cũng chạy được.

   Gateway nhận `gateway-linux-cpu.txt` **dù máy có CUDA**: gateway chạy CPU theo thiết kế, GPU dành cho vLLM.
   Đổi bằng `GATEWAY_REQUIREMENTS` / `VLLM_REQUIREMENTS` (đường dẫn tính từ gốc repo); log thêm `[GATEWAY_REQUIREMENTS]`
   khi file đến từ biến này. Máy không có NVIDIA thì script dừng (vLLM ROCm lấy từ image của AMD, không qua script này).
3. Venv vLLM `/opt/vllm`: `uv venv` rồi `uv pip install -r requirements/vllm-linux-cuda.txt --torch-backend=auto`
   (hiện tiến trình của uv, khoảng 5 phút trên máy mới).
   Tải model VLM đã chọn (`VLM_VARIANT`/`VLM_MODEL`, mặc định `Qwen/Qwen3-VL-8B-Instruct`, 17 GB) bằng `hf` của chính venv
   này, một lần cho mỗi model (không cần venv `/venv/main` của template).
4. Venv CPU của gateway `/opt/inference-cpu`: `uv pip install -r requirements/gateway-linux-cpu.txt --torch-backend=cpu`.
   Mỗi venv chỉ cài lại khi chưa có, hoặc khi dòng requirement trong file **hay trong file nó nạp bằng `-r`**
   (`gateway-common.txt`) đổi: hash của các dòng đó lưu trong `<venv>/.requirements` cùng tên file và cờ uv. Hash bỏ
   qua comment và dòng trống, nên sửa phần header (ví dụ dòng `Tested:`) không cài lại, không khởi động lại service. Đổi sang file hoặc cờ khác (một nền tảng
   khác) thì venv được tạo lại từ đầu, vì `uv pip install` giữ nguyên bản torch đã cài nếu nó vẫn thỏa yêu cầu.
5. Tải sẵn model CPU vào `HF_HOME` một lần: `Systran/faster-whisper-small`, `BAAI/bge-m3`, và model của Docling
   (`docling-tools models download` vào `$HF_HOME/docling-models`, gateway đọc qua `DOCLING_ARTIFACTS_PATH`).
6. Hai service supervisor: `vllm` (`python -m vlm_server serve`, `127.0.0.1:18000`) và `gateway`
   (`python -m vlm_server gateway` từ venv CPU, `127.0.0.1:18080`, `INFERENCE_VLM_UPSTREAM=http://127.0.0.1:18000`).
   Mục portal `vLLM` của Caddy trỏ cổng ngoài **10100 → 18080** (gateway). Một service chỉ được khởi động lại khi
   script sinh ra cho nó thay đổi hoặc nó không ở trạng thái RUNNING. Script của mỗi service ghi kèm hash requirements
   của nó, script gateway ghi thêm commit: code mới chỉ khởi động lại gateway (vài giây), không đụng vLLM (vài phút).
7. Mở URL public: domain tĩnh ngrok nếu máy có authtoken ngrok, nếu không thì Cloudflare quick tunnel (URL đổi sau mỗi lần Start).
8. Tự kiểm tra qua URL public: `GET /v1/models` (200 khi có token) và `POST /v1/embeddings` với `"xin chào"` (vector 1024 chiều).
9. In ra các dòng `.env` cho client MentorMind (lưu ở `/root/mentormind-inference.env`):

```
KNOWHOW_VLM_BASE_URL=https://<domain>/v1
KNOWHOW_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
KNOWHOW_VLM_API_KEY=<token>
KNOWHOW_VLM_MAX_FRAMES=40
KNOWHOW_VLM_MAX_TOKENS=8192
KNOWHOW_LLM_MAX_TOKENS=8192
KNOWHOW_INFERENCE_URL=https://<domain>/v1
KNOWHOW_INFERENCE_API_KEY=<token>
KNOWHOW_ASR_PROVIDER=remote
KNOWHOW_EMBEDDER=remote
KNOWHOW_DOC_EXTRACTOR=remote
```

| Biến của script | Mặc định | Ý nghĩa |
|---|---|---|
| `ENGINE_REF` | `main` | Nhánh, tag hoặc commit của repo này |
| `VLM_VARIANT` | `instruct` | Model định sẵn: `instruct` (8B BF16), `thinking` (8B BF16), `thinking-fp8` (8B FP8), `30b-thinking` (30B-A3B AWQ 4-bit); so sánh trên video LASI ở `benchmarks/lasi-vlm/README.md`. Model *Thinking* tự thêm `--reasoning-parser qwen3` và in `KNOWHOW_VLM_MAX_TOKENS`/`KNOWHOW_LLM_MAX_TOKENS=16384`. Đổi model thì đổi `KNOWHOW_VLM_MODEL` ở client theo dòng script in ra. Mỗi lần chạy `run.sh` phải đặt lại biến này (không đặt = `instruct`) |
| `VLM_MODEL` | (theo `VLM_VARIANT`) | Id Hugging Face bất kỳ vLLM chạy được; thắng `VLM_VARIANT` |
| `JSON_WHITESPACE` | `compact` | `compact` = JSON trả lời không có khoảng trắng tuỳ ý (`disable_any_whitespace` của structured outputs): với khoảng trắng tự do, Qwen3-VL-30B-A3B lặp `\n\n  ` tới hết `max_tokens` ở 10/44 câu trả lời trên LASI (8B: 2/242). `any` = mặc định của vLLM |
| `SPEC_CONFIG` | (không) | JSON `--speculative-config` của vLLM, vd. `{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}` |
| `NGROK_DOMAIN` | `tiptop-ritzy-finisher.ngrok-free.dev` | Domain tĩnh ngrok |
| `MAX_MODEL_LEN` / `KV_CACHE_DTYPE` / `GPU_UTIL` | `65536` / `fp8` / `0.94` | Context 64K trên card 24 GB nhờ KV cache FP8 |
| `CPU_OFFLOAD_GB` | `0` | GB trọng số chuyển sang RAM để dành VRAM cho KV cache |
| `ASR_MODEL` / `EMBED_MODEL` | `small` / `BAAI/bge-m3` | Model của gateway |
| `GATEWAY_REQUIREMENTS` | `requirements/gateway-linux-cpu.txt` (Linux) | File requirements của venv gateway; cờ uv đi theo file |
| `VLLM_REQUIREMENTS` | `requirements/vllm-linux-cuda.txt` (Linux + NVIDIA) | File requirements của venv vLLM |

Log: `/var/log/portal/vllm.log`, `/var/log/portal/gateway.log`, `/var/log/portal/ngrok.log`.

## 3. VLM trên máy NVIDIA (uv)

| Thứ | Giá trị đã kiểm tra |
|---|---|
| GPU | 1 × 24 GB (đo trên RTX A5000). Trọng số BF16 chiếm 16.6 GiB |
| Driver | CUDA 12.6 → vLLM **0.16.0** + torch **2.9.1+cu126**. vLLM từ 0.17 chỉ có bản cu128+, **đừng nâng** nếu chưa nâng driver |
| Đĩa | khoảng 9 GB cho `.venv` và 17 GB cho model (`~/.cache/huggingface`) |

```bash
uv sync                                          # một lần
uv run hf download Qwen/Qwen3-VL-8B-Instruct     # tùy chọn; lần chạy đầu cũng tự tải

uv run python -m vlm_server serve --dry-run      # in lệnh, không chạy
uv run python -m vlm_server serve                # tự chọn GPU trống nhất (>= 21 GiB free)
VLM_SERVER_GPU=3 uv run python -m vlm_server serve
uv run python -m vlm_server smoke --frames 48    # ở terminal khác: PASS = hiểu đúng thứ tự khung
```

Từ repo `mentormind-knowhow-ai`: `make vlm-server`, `make vlm-server-smoke`, `uv run knowhow vlm ping`.

**Chỉ 1 GPU**, được chặn ở 3 chỗ: `VLM_SERVER_GPU` chỉ nhận một số (`"0,1"` bị từ chối), chỉ đúng
index đó được đặt visible, và vLLM chạy `--tensor-parallel-size 1`. **Server chạy theo nhu cầu**:
khởi động mất 1–3 phút, dùng xong thì `Ctrl-C` để trả GPU (máy dev dùng chung).

## 4. AMD MI250 (ROCm)

MI250 có 2 GCD, mỗi GCD 64 GB. ROCm coi mỗi GCD là một GPU, và server dùng **một GCD**. Chạy bằng
Docker là đường ngắn nhất, vì vLLM cho ROCm có sẵn trong image của AMD (mục 5). Nếu chạy trên máy đã
cài sẵn vLLM ROCm:

```bash
VLM_SERVER_PLATFORM=rocm VLM_SERVER_GPU=0 python3 -m vlm_server serve --dry-run
```

- Chọn GPU qua `amd-smi metric --mem-usage --json`, nếu không có thì dùng `rocm-smi --showmeminfo vram --json`.
  Khi không đọc được, hãy đặt thẳng `VLM_SERVER_GPU=<index>`.
- Launcher đặt `HIP_VISIBLE_DEVICES=<i>` và `CUDA_VISIBLE_DEVICES=<i>` (vLLM ROCm yêu cầu hai biến khớp nhau),
  kèm `VLLM_ROCM_USE_AITER=0`, vì kernel AITER dành cho MI300 (gfx942+) còn MI250 là gfx90a.
- Context mặc định là **32768** trên GPU hơn 32 GB (16384 trên card 24 GB). Muốn đổi thì đặt `VLM_SERVER_MAX_MODEL_LEN`.
- MI250 không có FP8, nên giữ BF16.
- Thêm biến môi trường cho vLLM qua `VLM_SERVER_EXTRA_ENV='{"VLLM_ATTENTION_BACKEND":"TRITON_ATTN"}'` nếu backend mặc định lỗi.
- **Chưa chạy thật trên MI250.** Các parser amd-smi/rocm-smi được viết theo định dạng JSON đã biết và
  đã có test. Nếu output trên máy bạn khác, hãy gửi lại output của `amd-smi metric --mem-usage --json`.

## 5. Docker: build một lần, `exec` vào là có môi trường

Container chỉ giữ môi trường (`sleep infinity`); server bật khi cần.

```bash
# NVIDIA (nvidia-container-toolkit)
VLM_GPU=3 docker compose --profile cuda up -d --build
docker compose exec vlm-cuda python -m vlm_server serve        # Ctrl-C để dừng server
docker compose exec vlm-cuda python -m vlm_server smoke --frames 48

# Máy dùng CDI thay cho nvidia runtime (vd. rootless Docker)
docker compose -f docker-compose.yml -f docker/docker-compose.cdi.yml --profile cuda up -d --build

# AMD MI250
VLM_GPU=0 docker compose --profile rocm up -d --build
docker compose exec vlm-rocm python3 -m vlm_server serve
```

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `VLM_GPU` | `0` | GPU (hoặc GCD) duy nhất được cấp cho container |
| `HF_CACHE` | `~/.cache/huggingface` | Mount cache model, để khỏi tải lại 17 GB |
| `VLM_SERVER_PORT` | `8100` | Port trên loopback của máy host |
| `VLM_SECRETS_FILE` | `../../.env` | File chứa `VLM_SERVER_API_KEY` (không bắt buộc) |
| `ROCM_VLLM_IMAGE` | `rocm/vllm:rocm7.14.1_cdna_…_vllm_0.23.0` | Image vLLM ROCm. Tag phải hỗ trợ gfx90a, kiểm tra bằng `python3 -c "import torch; print(torch.cuda.get_arch_list())"` |

- Image CUDA dựng lại đúng môi trường uv của máy dev (`uv.lock`), trên `nvidia/cuda:12.6.3-devel`.
- Image ROCm lấy vLLM từ image của AMD, chỉ thêm launcher của repo này.
- **Cả hai image mới được kiểm tra bằng `docker build --check` và `docker compose config`**. Chưa
  build thật trên máy dev vì phân vùng `/home` chỉ còn khoảng 14 GB, trong khi image CUDA khoảng 17 GB.

## 6. Nối client vào server

`mentormind-knowhow-ai` trỏ sẵn tới server, không cần cấu hình: `KNOWHOW_VLM_PROVIDER=local_openai`,
`http://127.0.0.1:8100/v1`, model `Qwen/Qwen3-VL-8B-Instruct`.

Từ máy khác: `ssh -N -L 8100:127.0.0.1:8100 <user>@<gpu-host>`. URL loopback, IP private, IP
Tailscale/CGNAT hay hostname LAN đều được tính là **local**; trỏ sang host public thì client tự coi
là **cloud**, và dữ liệu nhà máy bị chặn.

Token bảo vệ (tùy chọn): đặt `VLM_SERVER_API_KEY` trong file secrets (mặc định là `.env` của repo chính
của repo mẹ). Token được truyền cho vLLM qua biến `VLLM_API_KEY`, không qua command line, nên không lộ trong `ps`.

Bản deploy Vast.ai (mục 2) dùng một URL public cho cả VLM và gateway: chép các dòng `KNOWHOW_*` mà
`scripts/vast/run.sh` in ra vào `.env` của MentorMind (`KNOWHOW_VLM_BASE_URL` và `KNOWHOW_INFERENCE_URL` cùng
là `<URL>/v1`, cùng một token). Đó là host public, nên client coi là **cloud**.

## 7. Số liệu đo (RTX A5000 24 GB, 28/9/2026)

| Thông số | Giá trị |
|---|---|
| Trọng số BF16 | 16.64 GiB |
| KV cache (`gpu_memory_utilization=0.90`) | 2.28 GiB = 16,560 token → `max_model_len=16384` (32768 không khởi động được) |
| 1 khung 448×252 | khoảng 115 token |
| 48 khung (trần spec 04) | 5.5k token vào, 2.9–3.8 s |
| 8 khung | 1.0k token vào, 1.6 s (lần gọi đầu với schema mới mất 10–12 s để biên dịch grammar JSON) |

## 8. Tinh chỉnh chất lượng (ghi nhận khi thử video thật)

- Không dùng `temperature=0` cho danh sách bước dài: decode greedy làm model lặp vòng. Nên dùng
  `presence_penalty=1.5` với `temperature` từ 0.2 trở lên, đặt phía client:
  `KNOWHOW_VLM_EXTRA_BODY={"presence_penalty": 1.5}`.
- Với prompt "chỉ liệt kê bước" (không cho mô tả trước), model trả `{"steps": []}` cho mọi đoạn.
  Prompt lượt A hiện yêu cầu mô tả `scene` trước rồi mới liệt kê `steps` (spec 04 v1.5).

## 9. Sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `No GPU with >= 21 GiB free` | Chọn thẳng một card: `VLM_SERVER_GPU=<index>` |
| `... KV cache is needed, which is larger than the available KV cache memory` | Giảm `VLM_SERVER_MAX_MODEL_LEN`, hoặc tăng `VLM_SERVER_GPU_MEMORY_UTILIZATION` (tối đa khoảng 0.95 trên máy dùng chung) |
| `address already in use` | Đổi `VLM_SERVER_PORT`, và đổi `KNOWHOW_VLM_BASE_URL` phía client cho khớp |
| Engine chết ở `init_device` (CUDA) | torch không phải bản cu126; chạy lại `uv sync` |
| `HTTP 400 At most 64 image(s) may be provided in one prompt.` | Giữ `KNOWHOW_VLM_MAX_FRAMES` ≤ 64 |
| ROCm: `Không đọc được danh sách GPU AMD` | Đặt `VLM_SERVER_GPU=<index>` |
| Gateway trả 503 `... is not installed` | Chạy gateway bằng venv đã cài file `requirements/gateway-*.txt` của nền tảng; thông báo lỗi in sẵn lệnh cài |
| Gateway trả 502 `VLM upstream ... unreachable` | vLLM chưa chạy hoặc sai `INFERENCE_VLM_UPSTREAM`; xem `/var/log/portal/vllm.log` |
| Lần gọi đầu `/v1/embeddings` hoặc `/v1/audio/transcriptions` chậm | Model đang nạp (hoặc đang tải nếu chưa có trong `HF_HOME`); các lần sau nhanh |

## 10. Test

Không cần GPU, cũng không cần torch, faster-whisper hay docling: gateway được test bằng engine giả
(`tests/gateway/`), proxy bằng `httpx.MockTransport`. Nhóm `dev` chỉ chứa thư viện nhẹ:

```bash
uv run --only-group dev python -m pytest     # parser nvidia/amd, chọn 1 GPU, lệnh CUDA/ROCm, gateway, requirements/
uv run --only-group dev ruff check . && uv run --only-group dev ruff format --check .
```
