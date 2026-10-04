# vlm-engine — Qwen3-VL-8B tự host (mặc định: Ollama, Q4)

Server API tương thích OpenAI (`/v1/chat/completions`) phục vụ Qwen3-VL-8B cho repo `mentormind-knowhow-ai`.
Repo đó gắn repo này làm submodule tại `3rdparty/vlm_server` và chỉ gọi HTTP qua provider `local_openai`
(spec 04 FR-08). Vì chạy tại chỗ, đây là provider duy nhất được nhận dữ liệu `factory_confidential`
(constitution Điều 2).

Có hai backend (`VLM_SERVER_BACKEND`):

- **`ollama` (mặc định)**: `qwen3-vl:8b` Q4_K_M trên Ollama 0.35.1, chạy được trên macOS và Linux, cổng 11434.
  - Một lệnh: `bash scripts/serve-ollama.sh` (thêm `--ngrok` để mở cho máy khác). Hướng dẫn cho người dùng ở **[scripts/README.md](scripts/README.md)**.
  - Docker: `docker/Dockerfile.ollama`.
  - Ollama tự ép mỗi ảnh Qwen-VL tốn tối thiểu 1024 token. `scripts/ollama/llama-server-wrapper.sh` đổi được mức này qua `VLM_SERVER_IMAGE_MIN_TOKENS` (mặc định 512; quét trên video LASI ngày 4/10, mức này cho chất lượng ngang bản BF16 và ít hơn 1024 khoảng 29% token).
- **`vllm`**: Qwen3-VL-8B-Instruct BF16 trên vLLM, đúng 1 GPU NVIDIA hoặc AMD, cổng 8100 (phần còn lại của README này).

```
mentormind-knowhow-ai                         vlm-engine (repo này)
provider "local_openai" ── HTTP ──► 127.0.0.1:11434/v1 ──► Ollama (llama-server, image-min-tokens) ──► GPU
                                    127.0.0.1:8100/v1  ──► vLLM (1 GPU, tensor-parallel 1)
```

| Thư mục | Nhiệm vụ |
|---|---|
| `vlm_server/config.py` | Cấu hình `VLM_SERVER_*` (`ServerConfig`) |
| `vlm_server/gpu/` | Phát hiện nền tảng và chọn đúng 1 GPU: `nvidia.py` (nvidia-smi), `rocm.py` (amd-smi/rocm-smi), `select.py` |
| `vlm_server/serve/` | `ollama.py` dựng lệnh `scripts/serve-ollama.sh`; `command.py` dựng lệnh và biến môi trường vLLM theo nền tảng; `launch.py` chạy backend đã chọn |
| `scripts/` | `serve-ollama.sh` (macOS và Linux, một lệnh), `ollama/llama-server-wrapper.sh`, `README.md` hướng dẫn |
| `vlm_server/tools/smoke.py` | Kiểm tra một server đang chạy |
| `docker-compose.yml`, `docker/` | Chạy vlm-engine riêng (không cần repo chính): image CUDA và ROCm, override CDI `docker/docker-compose.cdi.yml`; container giữ sẵn môi trường, server bật khi cần |

## 1. Chạy trực tiếp trên máy NVIDIA (uv)

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

## 2. AMD MI250 (ROCm)

MI250 có 2 GCD, mỗi GCD 64 GB. ROCm coi mỗi GCD là một GPU, và server dùng **một GCD**. Chạy bằng
Docker là đường ngắn nhất, vì vLLM cho ROCm có sẵn trong image của AMD (mục 3). Nếu chạy trên máy đã
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

## 3. Docker: build một lần, `exec` vào là có môi trường

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

## 4. Nối client vào server

`mentormind-knowhow-ai` trỏ sẵn tới server, không cần cấu hình: `KNOWHOW_VLM_PROVIDER=local_openai`,
`http://127.0.0.1:8100/v1`, model `Qwen/Qwen3-VL-8B-Instruct`.

Từ máy khác: `ssh -N -L 8100:127.0.0.1:8100 <user>@<gpu-host>`. URL loopback, IP private, IP
Tailscale/CGNAT hay hostname LAN đều được tính là **local**; trỏ sang host public thì client tự coi
là **cloud**, và dữ liệu nhà máy bị chặn.

Token bảo vệ (tùy chọn): đặt `VLM_SERVER_API_KEY` trong file secrets (mặc định là `.env` của repo chính
của repo mẹ). Token được truyền cho vLLM qua biến `VLLM_API_KEY`, không qua command line, nên không lộ trong `ps`.

## 5. Số liệu đo (RTX A5000 24 GB, 28/9/2026)

| Thông số | Giá trị |
|---|---|
| Trọng số BF16 | 16.64 GiB |
| KV cache (`gpu_memory_utilization=0.90`) | 2.28 GiB = 16,560 token → `max_model_len=16384` (32768 không khởi động được) |
| 1 khung 448×252 | khoảng 115 token |
| 48 khung (trần spec 04) | 5.5k token vào, 2.9–3.8 s |
| 8 khung | 1.0k token vào, 1.6 s (lần gọi đầu với schema mới mất 10–12 s để biên dịch grammar JSON) |

## 6. Tinh chỉnh chất lượng (ghi nhận khi thử video thật)

- Không dùng `temperature=0` cho danh sách bước dài: decode greedy làm model lặp vòng. Nên dùng
  `presence_penalty=1.5` với `temperature` từ 0.2 trở lên, đặt phía client:
  `KNOWHOW_VLM_EXTRA_BODY={"presence_penalty": 1.5}`.
- Với prompt "chỉ liệt kê bước" (không cho mô tả trước), model trả `{"steps": []}` cho mọi đoạn.
  Prompt lượt A hiện yêu cầu mô tả `scene` trước rồi mới liệt kê `steps` (spec 04 v1.5).

## 7. Sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `No GPU with >= 21 GiB free` | Chọn thẳng một card: `VLM_SERVER_GPU=<index>` |
| `... KV cache is needed, which is larger than the available KV cache memory` | Giảm `VLM_SERVER_MAX_MODEL_LEN`, hoặc tăng `VLM_SERVER_GPU_MEMORY_UTILIZATION` (tối đa khoảng 0.95 trên máy dùng chung) |
| `address already in use` | Đổi `VLM_SERVER_PORT`, và đổi `KNOWHOW_VLM_BASE_URL` phía client cho khớp |
| Engine chết ở `init_device` (CUDA) | torch không phải bản cu126; chạy lại `uv sync` |
| `HTTP 400 At most 64 image(s) may be provided in one prompt.` | Giữ `KNOWHOW_VLM_MAX_FRAMES` ≤ 64 |
| ROCm: `Không đọc được danh sách GPU AMD` | Đặt `VLM_SERVER_GPU=<index>` |

## 8. Test

```bash
uv run pytest     # không cần GPU: parser nvidia/amd, chọn 1 GPU, lệnh + env CUDA/ROCm
```
