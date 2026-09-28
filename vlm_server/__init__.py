"""vlm-engine: a self-hosted OpenAI-compatible VLM server — vLLM serving Qwen3-VL-8B-Instruct on
exactly ONE GPU, NVIDIA (CUDA) or AMD (ROCm, e.g. one MI250 GCD).

``config`` (settings), ``gpu`` (vendor discovery + one-GPU selection), ``serve`` (vLLM command
and launcher), ``tools`` (smoke test). Clients call it over HTTP only; mentormind-knowhow-ai uses
it as its ``local_openai`` VLM provider.
"""
